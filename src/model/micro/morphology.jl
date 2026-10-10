# Section geometry and morphology queries.

"""A morphology sample in micrometres; all geometry remains Float64."""
struct Point3D
    x::Float64
    y::Float64
    z::Float64
    diameter::Float64
    function Point3D(x::Real, y::Real, z::Real, diameter::Real)
        values = Float64.((x, y, z, diameter))
        all(isfinite, values) || throw(ArgumentError("non-finite morphology coordinate"))
        values[4] >= 0.0 || throw(ArgumentError("sample diameter must be nonnegative"))
        return new(values...)
    end
end

mutable struct Section
    name::Symbol
    label::Symbol
    parent::Union{Nothing, Symbol}
    parent_x::Float64
    child_x::Float64
    nseg::Int
    length::Float64
    diameter::Float64
    points::Vector{Point3D}
    axial_resistivity::Float64
    capacitance::Float64
end

"""Create a section; attach its `child_x` end (0 or 1) at the parent's `parent_x`.

Length and diameter are in μm, Ra in Ω·cm. As in NEURON, pt3d samples run from
the attached end; segment locations and parameter callbacks retain logical x.
"""
function Section(
    name;
    label = :dend,
    parent = nothing,
    parent_x = 1.0,
    child_x = 0.0,
    nseg = 1,
    length = 100.0,
    diameter = 1.0,
    points = Point3D[],
    Ra = 100.0,
    cm = 1.0,
)
    geometry = collect(points)
    if Base.length(geometry) >= 2
        # Only the total is needed here; discretization computes the arc table.
        length = 0.0
        for index in 2:Base.length(geometry)
            left, right = geometry[index - 1], geometry[index]
            length += sqrt((right.x - left.x)^2 + (right.y - left.y)^2 + (right.z - left.z)^2)
        end
        diameter = geometry[1].diameter
    end
    nseg isa Integer && nseg > 0 || throw(ArgumentError("nseg must be a positive integer"))
    0.0 <= parent_x <= 1.0 || throw(ArgumentError("parent_x must be in [0, 1]"))
    child_x in (0, 1) || throw(ArgumentError("child_x must be 0 or 1"))
    all(isfinite, (length, diameter, Ra, cm, parent_x)) ||
        throw(ArgumentError("non-finite section parameter"))
    length > 0 && (diameter > 0 || (!isempty(geometry) && diameter == 0)) && Ra > 0 && cm >= 0 ||
        throw(ArgumentError("invalid section geometry or biophysics"))
    return Section(
        Symbol(name),
        Symbol(label),
        parent === nothing ? nothing : Symbol(parent),
        Float64(parent_x),
        Float64(child_x),
        Int(nseg),
        Float64(length),
        Float64(diameter),
        geometry,
        Float64(Ra),
        Float64(cm),
    )
end

struct MorphologyLayout
    parent::Vector{Int32}
    area::Vector{Float64}
    diameter::Vector{Float64}
    axial_conductance::Vector{Float64}
    capacitance::Vector{Float64}
    section_nodes::Dict{Symbol, Vector{Int32}}
    section_starts::Dict{Symbol, Int32} # Logical x=0; x=1 is section_nodes[name][end].
    edge_length::Vector{Float64}
end

"""Prepare reusable CPU morphology geometry for distances and compartment queries."""
MorphologyLayout(sections::AbstractVector{Section}) = discretize(sections)

"""Cable-path distance in μm between two `(section, x)` compartment locations.

Locations use the same compartment/end-point mapping as the simulator. Points
in disconnected trees have infinite distance.
"""
function path_distance(layout::MorphologyLayout, first::Tuple, second::Tuple)
    left, right = location_node(layout, first...), location_node(layout, second...)
    # Discretization numbers every parent before its children. Ascend the
    # larger index until the paths meet, without constructing an ancestor map.
    distance = 0.0
    while left != right
        if left > right
            distance += layout.edge_length[left]
            left = layout.parent[left]
        else
            distance += layout.edge_length[right]
            right = layout.parent[right]
        end
    end
    return left == 0 ? Inf : distance
end

"""Mean diameter in μm of the segment containing `x`, including endpoint queries."""
function segment_diameter(layout::MorphologyLayout, section, x::Real)
    return layout.diameter[range_node(layout.section_nodes, section, x)]
end

# Morphology import.

"""Read SWC into CPU sections by depth-first traversal of its parent topology.

Same-type unary chains form sections; soma and proximal attachments retain
NEURON's geometry conventions. IDs may have gaps and appear in any order, and
disconnected roots remain separate trees. Geometry remains Float64.
"""
function load_swc(path::AbstractString; nseg::Integer = 1, Ra::Real = 100.0, cm::Real = 1.0)
    # 1. Read FP64 geometry, then resolve external IDs once into dense indices.
    indices = Dict{Int, Int}()
    kinds, parents = Int[], Int[]
    samples = Point3D[]
    for (line_number, line) in enumerate(eachline(path))
        content = strip(first(split(line, '#'; limit = 2)))
        isempty(content) && continue
        values = split(content)
        length(values) >= 7 ||
            throw(ArgumentError("$path:$line_number: expected seven SWC columns"))
        id, kind = parse(Int, values[1]), parse(Int, values[2])
        haskey(indices, id) && throw(ArgumentError("duplicate SWC sample $id"))
        point = Point3D(
            parse(Float64, values[3]),
            parse(Float64, values[4]),
            parse(Float64, values[5]),
            2.0 * parse(Float64, values[6]),
        )
        push!(samples, point)
        push!(kinds, kind)
        push!(parents, parse(Int, values[7]))
        indices[id] = length(samples)
    end
    isempty(samples) && throw(ArgumentError("empty SWC morphology"))
    children = [Int[] for _ in samples]
    roots = Int[]
    for index in eachindex(samples)
        parent = parents[index]
        if parent < 0
            parents[index] = 0
            push!(roots, index)
        else
            haskey(indices, parent) || throw(ArgumentError("SWC has missing parent $parent"))
            parents[index] = indices[parent]
            push!(children[parents[index]], index)
        end
    end
    # Only roots need soma counts for the three-point convention. Reuse the
    # DFS stack rather than retaining a full traversal and per-sample counts.
    root_soma_counts = zeros(Int, length(roots))
    pending = Int[]
    visited = 0
    for (root_index, root) in enumerate(roots)
        push!(pending, root)
        while !isempty(pending)
            index = pop!(pending)
            visited += 1
            root_soma_counts[root_index] += kinds[index] == 1
            append!(pending, Iterators.reverse(children[index]))
        end
    end
    visited == length(samples) || throw(ArgumentError("SWC contains an unrooted cycle"))

    # 2. Keep the original same-type unary chains. Soma cables continue past
    # neurite attachments; a root/proximal bifurcation uses one continuing cable.
    soma_children = [count(child -> kinds[child] == 1, branch) for branch in children]
    continuation = zeros(Int, length(samples))
    proximal = falses(length(samples))
    for index in eachindex(samples)
        branch = children[index]
        kind = kinds[index]
        candidate = 0
        if kind == 1
            if soma_children[index] == 1 || (parents[index] == 0 && soma_children[index] > 1)
                candidate = branch[findfirst(child -> kinds[child] == 1, branch)]
            end
        elseif length(branch) == 1 && kinds[only(branch)] == kind
            candidate = only(branch)
        elseif parents[index] == 0 || kinds[parents[index]] == 1
            position = findfirst(child -> kinds[child] == kind, branch)
            if position !== nothing
                candidate = branch[position]
                for child in branch
                    child != candidate && (proximal[child] = true)
                end
            end
        end
        continuation[index] = candidate
    end
    # NeuroMorpho's three-point soma represents one sphere, not three cables.
    absorbed = falses(length(samples))
    for (root_index, root) in enumerate(roots)
        kinds[root] == 1 && root_soma_counts[root_index] == 3 && soma_children[root] == 2 ||
            continue
        ends = filter(child -> kinds[child] == 1, children[root])
        center = samples[root]
        all(
            child -> isempty(children[child]) && samples[child].diameter == center.diameter,
            ends,
        ) || continue
        span = sum(
            sqrt(
                (samples[child].x-center.x)^2 +
                (samples[child].y-center.y)^2 +
                (samples[child].z-center.z)^2,
            ) for child in ends
        )
        abs(span / center.diameter - 1.0) < 0.01 || continue
        continuation[root] = 0
        absorbed[ends] .= true
    end

    # 3. Form sections and resolve soma/proximal attachments before removing any
    # degenerate sections. Copied parent points retain diameter discontinuities.
    section_of = zeros(Int, length(samples))
    heads, tails, section_parents = Int[], Int[], Int[]
    connections = Float64[]
    geometry = Vector{Point3D}[]
    append!(pending, Iterators.reverse(roots))
    while !isempty(pending)
        head = pop!(pending)
        index = head
        kind, parent_index = kinds[head], parents[head]
        points, branches = Point3D[], Int[]
        while true
            push!(points, samples[index])
            section_of[index] = length(heads) + 1
            for child in children[index]
                child != continuation[index] && !absorbed[child] && push!(branches, child)
            end
            continuation[index] == 0 && break
            index = continuation[index]
        end
        parent = parent_index == 0 ? 0 : section_of[parent_index]
        parent_x = 1.0
        if parent != 0
            soma = kinds[heads[parent]] == 1
            wire_first = false
            if section_parents[parent] == 0 && soma && kind != 1 && length(geometry[parent]) == 1
                parent_x = 0.5
                wire_first = length(points) > 1
            elseif section_parents[parent] == 0 && parent_index == heads[parent]
                parent_x = 0.0
                wire_first = kind != 1 && soma_children[parent_index] > 1
            elseif soma
                if parent_index != tails[parent]
                    parent_x = 0.5
                    wire_first = kind != 1 && length(points) > 1
                elseif soma_children[parent_index] > 1
                    wire_first = kind != 1 && length(points) > 1
                end
            end
            proximal[head] && (parent_x = 0.0)
            if !wire_first
                previous = samples[parent_index]
                if soma && kind != 1
                    previous = Point3D(previous.x, previous.y, previous.z, points[1].diameter)
                end
                pushfirst!(points, previous)
            end
        end
        push!(heads, head)
        push!(tails, index)
        push!(section_parents, parent)
        push!(connections, parent_x)
        push!(geometry, points)
        append!(pending, Iterators.reverse(branches))
    end

    # 4. Remove zero-length one/two-point child sections and reconnect children
    # through their original attachment, as Import3d does before instantiation.
    sections = Section[]
    names = Vector{Symbol}(undef, length(heads))
    removed = falses(length(heads))
    labels = Dict(1 => :soma, 2 => :axon, 3 => :dend, 4 => :apic)
    label_counts = Dict{Symbol, Int}()
    for index in eachindex(heads)
        points = geometry[index]
        parent = section_parents[index]
        if parent != 0 && removed[parent]
            connections[index] = connections[parent]
            parent = section_parents[parent]
            section_parents[index] = parent
        end
        if parent != 0 && length(points) <= 2
            first_point, last_point = points[1], points[end]
            if (first_point.x, first_point.y, first_point.z) ==
               (last_point.x, last_point.y, last_point.z)
                removed[index] = true
                continue
            end
        end
        if length(points) == 1
            point = only(points)
            half = 0.5 * point.diameter
            points = [
                Point3D(point.x - half, point.y, point.z, point.diameter),
                point,
                Point3D(point.x + half, point.y, point.z, point.diameter),
            ]
        end
        kind = kinds[heads[index]]
        label = get(labels, kind, Symbol(string(kind)))
        ordinal = get(label_counts, label, 0)
        name = haskey(labels, kind) ? Symbol(label, :_, ordinal) : Symbol(:sec_, kind, :_, ordinal)
        label_counts[label] = ordinal + 1
        names[index] = name
        push!(
            sections,
            Section(
                name;
                label,
                parent = parent == 0 ? nothing : names[parent],
                parent_x = connections[index],
                points,
                nseg,
                Ra,
                cm,
            ),
        )
    end
    return sections
end

"""Load Neurolucida ASCII morphology; retain branch boundaries and section labels."""
function load_asc(path::AbstractString; nseg::Integer = 1, Ra::Real = 100.0, cm::Real = 1.0)
    roots = parse_asc(read(path, String), path)
    trees = [(asc_tree_label(root), asc_branches(root.items)) for root in roots]
    contours = [
        branch.points for (label, branches) in trees if label == :soma for
        branch in branches if !isempty(branch.points)
    ]
    length(contours) <= 1 || throw(ArgumentError("ASC supports one CellBody contour"))
    sections = Section[]
    parent = nothing
    if !isempty(contours)
        points = asc_soma_points(only(contours))
        if length(points) >= 2
            parent = :soma_0
            push!(sections, Section(parent; label = :soma, points, nseg, Ra, cm))
        end
    end
    counters = Dict(:axon=>0, :dend=>0, :apic=>0)
    function emit(branch, label, parent, terminal, parent_x)
        points = copy(branch.points)
        current_terminal = isempty(points) ? terminal : last(points)
        if terminal !== nothing && !isempty(points)
            first_point = first(points)
            if (terminal.x, terminal.y, terminal.z) != (first_point.x, first_point.y, first_point.z)
                pushfirst!(
                    points,
                    Point3D(terminal.x, terminal.y, terminal.z, first_point.diameter),
                )
            end
        end
        if !isempty(points)
            if length(points) == 1
                point = only(points)
                points = [
                    Point3D(point.x-0.5*point.diameter, point.y, point.z, point.diameter),
                    point,
                    Point3D(point.x+0.5*point.diameter, point.y, point.z, point.diameter),
                ]
            end
            name = Symbol(label, :_, counters[label])
            counters[label] += 1
            push!(sections, Section(name; label, parent, parent_x, points, nseg, Ra, cm))
            parent, parent_x = name, 1.0
        end
        for child in branch.children
            emit(child, label, parent, current_terminal, parent_x)
        end
    end
    for (label, branches) in trees
        haskey(counters, label) || continue
        for branch in branches
            emit(branch, label, parent, nothing, parent === nothing ? 1.0 : 0.5)
        end
    end
    return sections
end

# Compartment discretization.

"""Discretize reusable section templates on the CPU, including zero-capacitance endpoints."""
function discretize(sections::AbstractVector{Section})
    parent = Int32[]
    area = Float64[]
    diameter = Float64[]
    axial = Float64[]
    capacitance = Float64[]
    nodes = Dict{Symbol, Vector{Int32}}()
    starts = Dict{Symbol, Int32}()
    edge_length = Float64[]
    layout =
        MorphologyLayout(parent, area, diameter, axial, capacitance, nodes, starts, edge_length)
    function add_node(
        parent_node,
        node_area,
        node_diameter,
        conductance,
        node_capacitance,
        span = 0.0,
    )
        length(parent) < typemax(Int32) || throw(ArgumentError("template exceeds Int32 node range"))
        push!(parent, parent_node)
        push!(area, node_area)
        push!(diameter, node_diameter)
        push!(axial, conductance)
        push!(capacitance, node_capacitance)
        push!(edge_length, span)
        return Int32(length(parent))
    end
    first_section = firstindex(sections)
    for index in section_order(sections)
        section = sections[first_section + index - 1]
        section.nseg > 0 &&
        isfinite(section.axial_resistivity) &&
        section.axial_resistivity > 0 &&
        isfinite(section.capacitance) &&
        section.capacitance >= 0 ||
            throw(ArgumentError("invalid segmentation or biophysics on section $(section.name)"))
        points, arcs = section_geometry(section)
        if section.parent === nothing
            connection = add_node(Int32(0), 100.0, 0.0, 0.0, 0.0)
        else
            connection = location_node(layout, section.parent, section.parent_x)
        end
        # pt3d runs from the attached end, as in NEURON. Keep the solver's
        # parent-first order while storing public compartment locations by x.
        section_nodes = Vector{Int32}(undef, section.nseg + 1)
        previous = connection
        previous_right = 0.0
        step = arcs[end] / section.nseg
        cursor = (1, arcs[1], points[1].diameter)
        reversed = section.child_x == 1.0
        for ordinal in 1:section.nseg
            middle = (ordinal - 0.5) * step
            right = ordinal * step
            al, rl, dl, cursor =
                integrate_geometry(points, arcs, middle, section.axial_resistivity, cursor)
            ar, rr, dr, cursor =
                integrate_geometry(points, arcs, right, section.axial_resistivity, cursor)
            node_area = al + ar
            segment = reversed ? section.nseg + 1 - ordinal : ordinal
            node = add_node(
                previous,
                node_area,
                (dl + dr) / step,
                1.0 / (previous_right + rl),
                section.capacitance * node_area * 1.0e-5,
                ordinal == 1 ? 0.5 * step : step,
            )
            section_nodes[segment] = node
            previous = node
            previous_right = rr
        end
        endpoint = add_node(previous, 100.0, 0.0, 1.0 / previous_right, 0.0, 0.5 * step)
        starts[section.name] = reversed ? endpoint : connection
        section_nodes[end] = reversed ? connection : endpoint
        nodes[section.name] = section_nodes
    end
    return layout
end
