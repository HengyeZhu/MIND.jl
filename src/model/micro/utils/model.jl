# Instance ownership and selections.

# Display identities, never the entire owning network, at the Julia prompt.
Base.show(io::IO, model::MicroModel) = print(
    io,
    "MicroModel(",
    length(model.cells),
    " cells, ",
    length(model.points),
    " point processes, ",
    length(model.connections),
    " connections)",
)
Base.show(io::IO, cell::Cell) = print(io, "Cell(gid=", cell.gid, ")")
Base.getproperty(cell::Cell, name::Symbol) =
    name == :position ? getfield(cell, :model).cells[getfield(cell, :index)].position :
    getfield(cell, name)
Base.propertynames(::Cell) = (:model, :index, :gid, :position)
function Base.setproperty!(cell::Cell, name::Symbol, value)
    name == :position || return setfield!(cell, name, value)
    model = getfield(cell, :model)
    model.runtime.layout === nothing ||
        throw(ArgumentError("cell position must be set before build!"))
    position = NTuple{3, Float64}(value)
    all(isfinite, position) ||
        throw(ArgumentError("cell position requires three finite coordinates in μm"))
    model.cells[getfield(cell, :index)].position = position
    return value
end
Base.show(io::IO, section::CellSection) =
    print(io, "CellSection(gid=", section.cell.gid, ", name=", repr(section.name), ")")
Base.show(io::IO, group::SectionGroup) =
    print(io, "SectionGroup(gid=", group.cell.gid, ", sections=", group.names, ")")
Base.show(io::IO, point::PointProcess) = print(
    io,
    "PointProcess(",
    point.model.points[point.index].mechanism.ir.name,
    ", index=",
    point.index,
    ")",
)
Base.show(io::IO, source::SpikeSource) = print(io, "SpikeSource(sid=", source.sid, ")")

function parameter_values(values)
    result = Dict{Symbol, Any}()
    for (name, value) in pairs(values)
        result[Symbol(name)] =
            value isa Real ? Float64(value) : value isa AbstractArray ? copy(value) : value
    end
    return result
end

# Object selection overloads share the cell workflow and resolved section names.
insert!(target::Union{SectionGroup, CellSection}, mechanism; kwargs...) =
    (insert!(target.cell, mechanism; sections = target, kwargs...); target)
add_pointprocess!(
    target::CellSection,
    mechanism;
    x = 0.5,
    params = NamedTuple(),
    spike_times = (),
) = create_pointprocess!(
    target.cell.model,
    mechanism,
    target.cell.index,
    target.name,
    Float64(x),
    params,
    spike_times,
)

function create_pointprocess!(
    model::MicroModel,
    mechanism,
    cell::Int,
    section::Symbol,
    x::Float64,
    parameters,
    spike_times,
)
    compiled = resolve_mechanism(mechanism)
    compiled.ir.role == :micro && compiled.ir.kind in (:point, :artificial) ||
        throw(ArgumentError("expected micro point process or artificial cell"))
    cell != 0 ||
        compiled.ir.kind == :artificial ||
        throw(ArgumentError("point process requires a cell"))
    0.0 <= x <= 1.0 || throw(ArgumentError("point location must be in [0, 1]"))
    # An empty tuple avoids allocating an unused schedule for every synapse.
    times = isempty(spike_times) ? () : collect(Float64, spike_times)
    all(isfinite, times) && all(>=(0.0), times) || throw(ArgumentError("invalid spike times"))
    issorted(times) || throw(ArgumentError("spike times must be sorted"))
    push!(
        model.points,
        PointInstance(compiled, cell, section, x, parameter_values(parameters), times),
    )
    return PointProcess(model, length(model.points))
end

register_spike_source!(
    target::CellSection;
    sid::Union{Nothing, Integer} = nothing,
    x = 0.5,
    threshold = 10.0,
) = register_source!(
    target.cell.model,
    target.cell.index,
    target.name,
    Float64(x),
    0,
    Float64(threshold),
    sid,
    nothing,
    :v,
    1,
)

register_spike_source!(target::CellSection, field; kwargs...) =
    register_spike_source!(target.cell, field; section = target.name, kwargs...)

function register_source!(
    model::MicroModel,
    cell::Int,
    section::Symbol,
    x::Float64,
    point::Int,
    threshold::Float64,
    sid,
    mechanism = nothing,
    field::Union{Nothing, Symbol} = nothing,
    element::Int = 1,
)
    0.0 <= x <= 1.0 && isfinite(threshold) || throw(ArgumentError("invalid spike detector"))
    mechanism === nothing && element != 1 && throw(BoundsError(1:1, element))
    # Match source identity to persistent storage, before creating any edges.
    # Endpoint aliases follow parent attachments; section interiors use the same
    # compartment discretization as voltage/RANGE references.
    key = nothing
    if field !== nothing
        kind = mechanism === nothing ? (field == :v ? :voltage : :ion) : :mechanism
        exact_voltage = mechanism === nothing || point != 0
        owner, name, coordinate = cell, section, x
        key_field, key_element = field, element
        digest = mechanism === nothing ? "" : mechanism.ir.source_digest
        if mechanism !== nothing
            info = fieldinfo(mechanism.ir, field)
            1 <= element <= fieldwidth(info) || throw(BoundsError(fieldcolumns(info), element))
            if any(
                binding -> first(binding) == field && last(binding) == VoltageInput,
                mechanism.ir.inputs,
            )
                kind, digest, key_field, key_element = :voltage, "", :v, 1
                if point != 0
                    instance = model.points[point]
                    owner, name, coordinate = instance.cell, instance.section, instance.x
                end
            elseif info.scope == :global
                key = (:global, 0, :none, 0, digest, field, element)
            elseif point != 0
                key = (:point, point, :none, 0, digest, field, element)
            end
        end
        if key === nothing
            sections = model.morphologies[model.cells[owner].morphology]
            while true
                index = findfirst(candidate -> candidate.name == name, sections)
                index === nothing && throw(ArgumentError("unknown section $name"))
                selected = sections[index]
                # A density mechanism's v belongs to its nearest compartment,
                # including at x=0/1; a cable or point voltage uses exact ends.
                if kind == :voltage &&
                   exact_voltage &&
                   coordinate == selected.child_x &&
                   selected.parent !== nothing
                    name, coordinate = selected.parent, selected.parent_x
                    continue
                end
                segment =
                    kind == :voltage && exact_voltage && coordinate in (0.0, 1.0) ?
                    (coordinate == 0.0 ? 0 : selected.nseg + 1) :
                    min(floor(Int, coordinate * selected.nseg) + 1, selected.nseg)
                key = (kind, owner, name, segment, digest, key_field, key_element)
                break
            end
        end
    end
    existing =
        key !== nothing ? get(model.variable_sources, key, 0) :
        point != 0 ? get(model.point_sources, point, 0) : 0
    if existing != 0
        source = model.sources[existing]
        return SpikeSource(model, existing, source.sid)
    end
    identity = sid === nothing ? model.next_source_id : Int64(sid)
    identity >= 0 || throw(ArgumentError("spike-source IDs must be nonnegative"))
    haskey(model.source_indices, identity) &&
        throw(ArgumentError("duplicate spike-source ID $identity"))
    push!(
        model.sources,
        SpikeSourceDefinition(
            identity,
            cell,
            section,
            x,
            point,
            threshold,
            mechanism,
            field,
            element,
        ),
    )
    index = length(model.sources)
    model.source_indices[identity] = index
    if key !== nothing
        model.variable_sources[key] = index
    elseif point != 0
        model.point_sources[point] = index
    end
    model.next_source_id = max(model.next_source_id, identity + 1)
    return SpikeSource(model, index, identity)
end

function selected_sections(cell::Cell, selectors)
    if selectors isa SectionGroup || selectors isa CellSection
        selectors.cell.model === cell.model && selectors.cell.index == cell.index ||
            throw(ArgumentError("sections belong to another cell"))
        return selectors isa SectionGroup ? copy(selectors.names) : [selectors.name]
    end
    model = cell.model
    return selected_sections(model.morphologies[model.cells[cell.index].morphology], selectors)
end

function connection_values(weight, delay, arity::Int)
    values = weight isa Real ? Float64(weight) : Float64.(weight)
    latency = Float64(delay)
    (values isa Float64 ? isfinite(values) : all(isfinite, values)) &&
    isfinite(latency) &&
    latency >= 0.0 || throw(ArgumentError("invalid connection weight or delay"))
    (values isa Float64 ? 1 : length(values)) <= arity ||
        throw(DimensionMismatch("too many NET_RECEIVE weights"))
    return values, latency
end

function selected_sections(morphology::AbstractVector{Section}, selectors)
    names =
        selectors isa Symbol || selectors isa AbstractString ? (Symbol(selectors),) :
        selectors isa AbstractVector{Symbol} ? selectors : Symbol.(selectors)
    selected = if :all in names
        [section.name for section in morphology]
    elseif length(names) == 1
        # One section or group needs no temporary membership sets. Explicit
        # section names retain precedence over an identically named label.
        name = first(names)
        explicit = any(section -> section.name == name, morphology)
        [
            section.name for
            section in morphology if (explicit ? section.name : section.label) == name
        ]
    else
        # Explicit names take precedence over labels, including names returned
        # by section_group. Resolve membership once per selection.
        explicit = Set(names)
        labels = copy(explicit)
        for section in morphology
            delete!(labels, section.name)
        end
        [
            section.name for
            section in morphology if section.name in explicit || section.label in labels
        ]
    end
    isempty(selected) && throw(ArgumentError("section selection is empty"))
    return selected
end

# Initial ion values are captured once for each cell, using the same section selectors as insert!.
function cell_ion_values(morphology::AbstractVector{Section}, ions)
    values = Dict{Tuple{Symbol, Symbol}, Any}()
    styles = Dict{Tuple{Symbol, Symbol}, Union{IonStyle, Symbol}}()
    for (selector, species) in pairs(ions)
        sections = selected_sections(morphology, selector)
        for (ion, configuration) in pairs(species)
            name = Symbol(ion)
            policy = get(configuration, :reversal_policy, nothing)
            style = get(configuration, :style, nothing)
            policy in (nothing, :automatic, :fixed, :initial, :advance, :dynamic) ||
                throw(ArgumentError("unknown ion reversal policy"))
            style === nothing ||
                style isa IonStyle ||
                throw(ArgumentError("style must be an IonStyle"))
            style === nothing ||
                policy === nothing ||
                throw(ArgumentError("choose either an ion style or a reversal policy"))
            for section in sections
                # An explicit species entry also requests allocation when all
                # of its values and style use the normal defaults.
                styles[(section, name)] = something(style, policy, :automatic)
                for (parameter, field) in (
                    (:internal, Symbol(name, :i)),
                    (:external, Symbol(name, :o)),
                    (:reversal, Symbol(:e, name)),
                )
                    value = get(configuration, parameter, nothing)
                    value === nothing && continue
                    if value isa Function
                        values[(section, field)] = value
                    else
                        converted = value isa Real ? Float64(value) : collect(Float64, value)
                        elements = converted isa Real ? (converted,) : converted
                        all(isfinite, elements) || throw(ArgumentError("non-finite ion value"))
                        parameter == :reversal ||
                            all(>=(0.0), elements) ||
                            throw(ArgumentError("negative ion concentration"))
                        values[(section, field)] = converted
                    end
                end
            end
        end
    end
    return values, styles
end

# Mechanism parameter evaluation in logical instance order.

function apply_point_parameters!(
    values::Vector{Float64},
    count::Int,
    physical_row::Int,
    ir::MechanismIR,
    parameters::Dict{Symbol, Any},
    fields::Dict{Symbol, Tuple{Int, Int}},
    ion_parameters::Dict{Tuple{Int32, Symbol}, Float64},
    node::Int32,
)
    # A point's assignments are used once; no tuple, binding vectors or
    # per-instance cache entry is needed. Resolve shared metadata lazily from IR.
    row = view(values, physical_row:count:length(values))
    for (name, value) in parameters
        offset, extent = get!(fields, name) do
            field = checked_parameter_field(ir, name)
            (field.offset, field.extent)
        end
        actual = value isa Function ? value(nothing) : value
        # Resolve the scalar type before crossing the assignment barrier.
        # An Any-valued call would heap-box the row view for every parameter.
        if extent == 0
            scalar = actual isa Float64 ? actual : Float64(actual)::Float64
            assign_parameter!(row, offset, 0, name, scalar)
        else
            assign_parameter!(row, offset, extent, name, actual)
        end
    end
    # Read ion values only after all callbacks, preserving assignment order
    # and writing shared ion storage exactly once per explicit ion parameter.
    collect_ion_parameters!(ion_parameters, row, ir, parameters, node)
    return nothing
end

function fill_density_binding!(
    values::Vector{Float64},
    count::Int,
    physical_rows::Union{UnitRange{Int}, Vector{Int}},
    offset::Int,
    parameters::P,
    gid::Int64,
    position::NTuple{3, Float64},
    sections,
    node_map::Dict{Symbol, Vector{Int32}},
    diameter::Vector{Float64},
    area::Vector{Float64},
    ion_parameters::Dict{Tuple{Int32, Symbol}, Float64},
) where {P <: ParameterBindings}
    for section in sections
        nodes = node_map[section]
        segments = length(nodes) - 1
        for segment in 1:segments
            node = nodes[segment]
            location = (;
                gid,
                position,
                section,
                x = (segment - 0.5) / segments,
                diameter = diameter[node],
                area = area[node],
            )
            row = view(values, physical_rows[offset + segment]:count:length(values))
            apply_parameters!(row, parameters; location)
            collect_ion_parameters!(ion_parameters, row, parameters, node)
        end
        offset += segments
    end
    return offset
end

# DHS ready-node heap.

# Dendritic Hierarchical Scheduling (Zhang et al., Nature Communications, 2023).
# A ready node has no unfinished children. Each round selects the deepest ready
# nodes, up to its lane budget. Negative BFS ranks break ties in NEURON-G order.
# Complete cells stay within one CUDA warp.

function push_dhs_candidate!(heap::Vector{Tuple{Int32, Int32}}, item)
    push!(heap, item)
    child = length(heap)
    while child > 1
        parent = child >> 1
        heap[parent] >= item && break
        heap[child] = heap[parent]
        child = parent
    end
    heap[child] = item
    return nothing
end

function pop_dhs_candidate!(heap::Vector{Tuple{Int32, Int32}})
    result, last = first(heap), pop!(heap)
    isempty(heap) && return result
    parent = 1
    while 2 * parent <= length(heap)
        child = 2 * parent
        child < length(heap) && heap[child + 1] > heap[child] && (child += 1)
        heap[child] <= last && break
        heap[parent] = heap[child]
        parent = child
    end
    heap[parent] = last
    return result
end
