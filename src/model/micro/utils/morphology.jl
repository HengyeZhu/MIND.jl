# Geometry integration.

function arc_lengths(points::Vector{Point3D})
    arcs = zeros(Float64, length(points))
    for index in 2:length(points)
        left, right = points[index - 1], points[index]
        arcs[index] =
            arcs[index - 1] +
            sqrt((right.x - left.x)^2 + (right.y - left.y)^2 + (right.z - left.z)^2)
    end
    return arcs
end

function section_geometry(section::Section)
    if length(section.points) < 2
        points = [
            Point3D(0.0, 0.0, 0.0, section.diameter),
            Point3D(section.length, 0.0, 0.0, section.diameter),
        ]
    else
        points = section.points
    end
    arcs = arc_lengths(points)
    arcs[end] > 0.0 || throw(ArgumentError("section $(section.name) has zero arc length"))
    return points, arcs
end

function integrate_geometry(points, arcs, boundary::Float64, Ra::Float64, cursor)
    area = 0.0
    resistance = 0.0
    diameter_integral = 0.0
    index, previous_arc, previous_diameter = cursor
    # Carry the interpolation cursor through every half-compartment, following
    # NEURON's diam_from_list. Each pt3d span is visited once per section, including
    # coincident samples and diameter jumps on a compartment boundary.
    while true
        next = index + 1
        at_boundary = arcs[next] > boundary || next == length(points)
        if at_boundary
            span = arcs[next] - arcs[index]
            fraction = abs(span) < 1.0e-10 ? 1.0 : (boundary - arcs[index]) / span
            arc = boundary
            diameter = (1.0 - fraction) * points[index].diameter + fraction * points[next].diameter
        else
            arc, diameter = arcs[next], points[next].diameter
        end
        delta = arc - previous_arc
        diameter_integral += (previous_diameter + diameter) * delta
        distance = max(delta, 1.0e-15)
        conductance = previous_diameter * diameter / distance
        resistance += 1.0 / (conductance == 0.0 ? 1.0e-15 : conductance)
        area +=
            (previous_diameter + diameter) *
            sqrt(distance^2 + 0.25 * (diameter - previous_diameter)^2)
        previous_arc, previous_diameter = arc, diameter
        at_boundary && break
        index = next
    end
    return 0.5 * pi * area,
    0.04 * Ra / pi * resistance,
    0.5 * diameter_integral,
    (index, previous_arc, previous_diameter)
end

# Topology and locations.

"""Map a RANGE field to its compartment, including the nearest end compartment."""
function range_node(sections, section, x::Real)
    name = Symbol(section)
    haskey(sections, name) || throw(ArgumentError("unknown section $name"))
    0.0 <= x <= 1.0 || throw(ArgumentError("section location must be in [0, 1]"))
    nodes = sections[name]
    count = length(nodes) - 1
    return nodes[min(floor(Int, x * count) + 1, count)]
end

function section_order(sections::AbstractVector{Section})
    names = Dict(section.name => index for (index, section) in enumerate(sections))
    length(names) == length(sections) || throw(ArgumentError("duplicate section name"))
    marks = zeros(UInt8, length(sections))
    order = Int[]
    sizehint!(order, length(sections))
    pending = Int[]
    # Resolve each parent chain iteratively, preserving the input-order DFS.
    first_section = firstindex(sections)
    for start in 1:length(sections)
        index = start
        while marks[index] != 2
            marks[index] == 1 && throw(ArgumentError("morphology contains a cycle"))
            marks[index] = 1
            push!(pending, index)
            parent = sections[first_section + index - 1].parent
            parent === nothing && break
            index = get(names, parent, 0)
            index != 0 || throw(ArgumentError("unknown parent section $parent"))
        end
        while !isempty(pending)
            index = pop!(pending)
            marks[index] = 2
            push!(order, index)
        end
    end
    return order
end

function location_node(layout::MorphologyLayout, section, x::Real)
    return location_node(layout.section_nodes, layout.section_starts, section, x)
end

function location_node(sections, starts, section, x::Real)
    name = Symbol(section)
    haskey(sections, name) || throw(ArgumentError("unknown section $name"))
    0.0 <= x <= 1.0 || throw(ArgumentError("section location must be in [0, 1]"))
    x == 0.0 && return starts[name]
    nodes = sections[name]
    x == 1.0 && return nodes[end]
    index = min(floor(Int, x * (length(nodes) - 1)) + 1, length(nodes) - 1)
    return nodes[index]
end

# Neurolucida parsing and contours.

struct AscList
    items::Vector{Union{Float64, String, AscList}}
end

struct AscBranch
    points::Vector{Point3D}
    children::Vector{AscBranch}
end

function parse_asc(source::String, filename::AbstractString)
    pattern = r";[^\n]*|\"[^\"]*\"|[()|<>,]|[^\s()|<>,;]+"
    tokens = String[]
    lines = Int[]
    line = 1
    previous = firstindex(source)
    for matched in eachmatch(pattern, source)
        line += count(==('\n'), SubString(source, previous, prevind(source, matched.offset)))
        previous = matched.offset
        token = String(matched.match)
        startswith(token, ';') && continue
        push!(tokens, token)
        push!(lines, line)
    end
    position = Ref(1)
    function expression()
        items = Union{Float64, String, AscList}[]
        start = position[]
        position[] += 1
        while position[] <= length(tokens)
            token = tokens[position[]]
            if token == ")"
                position[] += 1
                return AscList(items)
            elseif token == "("
                push!(items, expression())
            elseif token == "<"
                while position[] <= length(tokens) && tokens[position[]] != ">"
                    position[] += 1
                end
                position[] += 1
            else
                number = tryparse(Float64, token)
                if number !== nothing
                    isfinite(number) || throw(
                        ArgumentError("$filename:$(lines[position[]]): non-finite ASC coordinate"),
                    )
                    push!(items, number)
                elseif startswith(token, '"')
                    push!(
                        items,
                        token[nextind(token, firstindex(token)):prevind(token, lastindex(token))],
                    )
                elseif token != ","
                    push!(items, token)
                end
                position[] += 1
            end
        end
        throw(ArgumentError("$filename:$(lines[start]): unclosed ASC expression"))
    end
    roots = AscList[]
    while position[] <= length(tokens)
        if tokens[position[]] == "("
            push!(roots, expression())
        else
            position[] += 1
        end
    end
    return roots
end

function asc_label(value)
    value isa String || return :unknown
    normalized = replace(uppercase(value), r"\s+" => "")
    return normalized == "CELLBODY" ? :soma :
           normalized == "AXON" ? :axon :
           normalized == "DENDRITE" ? :dend : normalized == "APICAL" ? :apic : :unknown
end

function asc_tree_label(root::AscList)
    label = :unknown
    for item in root.items
        candidate = asc_label(item)
        if item isa AscList && !isempty(item.items)
            candidate = asc_label(first(item.items))
        end
        candidate != :unknown && (label = candidate)
    end
    return label
end

function asc_branches(items)
    branches = [AscBranch(Point3D[], AscBranch[])]
    for item in items
        if item == "|"
            push!(branches, AscBranch(Point3D[], AscBranch[]))
        elseif item isa AscList && !isempty(item.items)
            if first(item.items) isa Float64
                length(item.items) >= 3 && all(x -> x isa Float64, view(item.items, 1:3)) ||
                    throw(ArgumentError("ASC point requires three coordinates"))
                diameter =
                    length(item.items) >= 4 && item.items[4] isa Float64 ? abs(item.items[4]) : 1.0
                diameter <= 1e-6 && (diameter = 1.0)
                push!(
                    last(branches).points,
                    Point3D(item.items[1], item.items[2], item.items[3], diameter),
                )
            elseif first(item.items) isa AscList
                append!(last(branches).children, asc_branches(item.items))
            end
        end
    end
    return filter(branch -> !isempty(branch.points) || !isempty(branch.children), branches)
end

function interpolate_samples(x, y, points)
    return map(points) do point
        point <= first(x) && return first(y)
        point >= last(x) && return last(y)
        right = searchsortedfirst(x, point)
        left = right-1
        fraction = (point-x[left]) / (x[right]-x[left])
        y[left] + fraction * (y[right]-y[left])
    end
end

"""Match MIND_Sim's contour-centroid soma construction, with FP64 input samples."""
function asc_soma_points(contour::Vector{Point3D})
    length(contour) >= 3 || return Point3D[]
    arcs = arc_lengths(contour)
    last(arcs) > 1e-12 || return Point3D[]
    sampled_arcs = collect(range(0.0, last(arcs); length = 101))
    sampled = hcat(
        [
            interpolate_samples(arcs, getfield.(contour, field), sampled_arcs) for
            field in (:x, :y, :z)
        ]...,
    )
    center = vec(sum(sampled; dims = 1)) ./ size(sampled, 1)
    centered = sampled .- transpose(center)
    axes = eigen(Symmetric(transpose(centered) * centered))
    major = axes.vectors[:, 3]
    major[argmax(abs.(major))] < 0.0 && (major .*= -1.0)
    minor = axes.vectors[:, 2]
    minor[3] = 0.0
    if norm(minor) < 1e-6
        minor = [-major[2], major[1], 0.0]
        norm(minor) < 1e-12 && (minor = [1.0, 0.0, 0.0])
    end
    minor ./= norm(minor)
    longitudinal, radial = centered * major, centered * minor
    shift = 1-argmax(longitudinal)
    longitudinal, radial = circshift(longitudinal, shift), circshift(radial, shift)
    split = argmin(longitudinal)
    1 < split < length(longitudinal) || return Point3D[]
    function monotone(x, y)
        keep = [1]
        for index in 2:length(x)
            x[index] > x[last(keep)] && push!(keep, index)
        end
        return x[keep], y[keep]
    end
    x1, y1 = monotone(reverse(longitudinal[1:(split - 1)]), reverse(radial[1:(split - 1)]))
    x2, y2 = monotone(longitudinal[split:end], radial[split:end])
    min(length(x1), length(x2)) >= 2 || return Point3D[]
    merged = sort!([x1; x2])
    merged[end - 1] > merged[2] || return Point3D[]
    coordinates = collect(range(merged[2], merged[end - 1]; length = 21))
    diameters =
        abs.(interpolate_samples(x1, y1, coordinates) .- interpolate_samples(x2, y2, coordinates))
    diameters[1] = 0.5 * (diameters[1]+diameters[2])
    diameters[end] = 0.5 * (diameters[end]+diameters[end - 1])
    return [
        Point3D((center .+ major .* coordinate)..., diameter > 1e-6 ? diameter : 1.0) for
        (coordinate, diameter) in zip(coordinates, diameters)
    ]
end
