# Population morphology and cell instances.

# Construction metadata and device storage belong to one model lifecycle.
# Concrete runtime functions specialize after reading this CPU ownership slot.
abstract type AbstractModelLayout end
abstract type AbstractSimulationState end

mutable struct ModelRuntime
    layout::Union{Nothing, AbstractModelLayout}
    state::Union{Nothing, AbstractSimulationState}
end

ModelRuntime() = ModelRuntime(nothing, nothing)

mutable struct CellInstance
    const morphology::Int
    const gid::Int64
    position::NTuple{3, Float64}
    const voltage::Float64
    const section_voltages::Dict{Symbol, Float64}
    const bindings::Vector{DensityBinding}
    const ion_values::Dict{Tuple{Symbol, Symbol}, Any}
    const ion_styles::Dict{Tuple{Symbol, Symbol}, Union{IonStyle, Symbol}}
end

struct PointInstance
    mechanism::CompiledMechanism
    cell::Int
    section::Symbol
    x::Float64
    parameters::Dict{Symbol, Any}
    spike_times::Union{Tuple{}, Vector{Float64}}
end

mutable struct MicroModel
    runtime::ModelRuntime
    id::UInt64
    morphologies::Vector{Vector{Section}}
    cells::Vector{CellInstance}
    points::Vector{PointInstance}
    sources::Vector{SpikeSourceDefinition}
    source_indices::Dict{Int64, Int}
    point_sources::Dict{Int, Int}
    variable_sources::Dict{Tuple{Symbol, Int, Symbol, Int, String, Symbol, Int}, Int}
    next_source_id::Int64
    connections::Vector{Connection}
    gids::Set{Int64}
    ions::Dict{Symbol, IonSpecies}
    celsius::Float64
end

"""A cell in its owning CPU model, with per-cell biophysics and shared morphology."""
struct Cell
    model::MicroModel
    index::Int
    gid::Int64
end

"""One point-process instance in its owning CPU model."""
struct PointProcess
    model::MicroModel
    index::Int
end

"""A shared scalar threshold detector or artificial/external event source."""
struct SpikeSource
    model::MicroModel
    index::Int
    sid::Int64
end

"""A section of one cell, independent of its later GPU layout."""
struct CellSection
    cell::Cell
    name::Symbol
end

"""An ordered selection of sections belonging to one cell."""
struct SectionGroup <: AbstractVector{CellSection}
    cell::Cell
    names::Vector{Symbol}
end

Base.size(group::SectionGroup) = size(group.names)
Base.IndexStyle(::Type{SectionGroup}) = IndexLinear()
Base.getindex(group::SectionGroup, index::Int) = CellSection(group.cell, group.names[index])
Base.getindex(group::SectionGroup, indices::AbstractVector{<:Integer}) =
    SectionGroup(group.cell, group.names[indices])
Base.getindex(group::SectionGroup, ::Colon) = SectionGroup(group.cell, copy(group.names))

function MicroModel(; celsius = 6.3, ions = ())
    isfinite(celsius) && celsius > -273.15 || throw(ArgumentError("invalid temperature"))
    species = default_ions()
    for ion in ions
        species[ion.name] = ion
    end
    return MicroModel(
        ModelRuntime(),
        rand(UInt64),
        Vector{Section}[],
        CellInstance[],
        PointInstance[],
        SpikeSourceDefinition[],
        Dict{Int64, Int}(),
        Dict{Int, Int}(),
        Dict{Tuple{Symbol, Int, Symbol, Int, String, Symbol, Int}, Int}(),
        0,
        Connection[],
        Set{Int64}(),
        species,
        Float64(celsius),
    )
end

"""Select one cell's section group by label, preserving morphology order.

The group is a view of the cell for inserting mechanisms with their parameters.
Indexing returns a CellSection; a range returns another SectionGroup.
"""
function section_group(cell::Cell, label)
    name = Symbol(label)
    model = cell.model
    morphology = model.morphologies[model.cells[cell.index].morphology]
    names = [section.name for section in morphology if name == :all || section.label == name]
    isempty(names) && throw(ArgumentError("unknown section group $name"))
    return SectionGroup(cell, names)
end

"""Select one section by its exact name; labels are selected with section_group."""
function section(cell::Cell, name)
    selected = Symbol(name)
    model = cell.model
    morphology = model.morphologies[model.cells[cell.index].morphology]
    any(section -> section.name == selected, morphology) ||
        throw(ArgumentError("unknown section $selected"))
    return CellSection(cell, selected)
end

"""Remove selected sections before cell creation; retained children become roots."""
function delete_sections!(morphology::Vector{Section}; sections)
    removed = Set(selected_sections(morphology, sections))
    for section in morphology
        section.name in removed && continue
        if section.parent in removed
            section.parent = nothing
        end
    end
    filter!(section -> !(section.name in removed), morphology)
    return morphology
end

"""Insert a density mechanism where absent; existing instances keep their initial inputs."""
function insert!(cell::Cell, mechanism; sections = :all, params = NamedTuple())
    model = cell.model
    instance = model.cells[cell.index]
    compiled = resolve_mechanism(mechanism)
    compiled.ir.role == :micro && compiled.ir.kind == :density ||
        throw(ArgumentError("cell insertion requires a distributed micro mechanism"))
    selected = selected_sections(cell, sections)
    for binding in instance.bindings
        binding.mechanism.ir.name == compiled.ir.name || continue
        existing = Set(binding.sections)
        filter!(name -> !(name in existing), selected)
        isempty(selected) && break
    end
    isempty(selected) ||
        push!(instance.bindings, DensityBinding(compiled, selected, parameter_values(params)))
    return cell
end

"""Create one point process at a location on one cell."""
function add_pointprocess!(
    cell::Cell,
    mechanism;
    section = :soma,
    x = 0.5,
    params = NamedTuple(),
    spike_times = (),
)
    return create_pointprocess!(
        cell.model,
        mechanism,
        cell.index,
        Symbol(section),
        Float64(x),
        params,
        spike_times,
    )
end

"""Create one artificial cell without a morphology location."""
function add_pointprocess!(model::MicroModel, mechanism; params = NamedTuple(), spike_times = ())
    return create_pointprocess!(model, mechanism, 0, :none, 0.0, params, spike_times)
end

# Populations and connections.

"""A fixed range of cells in its owning model; population membership does not grow."""
struct Population <: AbstractVector{Cell}
    model::MicroModel
    indices::UnitRange{Int}
end

Base.size(population::Population) = (length(population.indices),)
Base.IndexStyle(::Type{Population}) = IndexLinear()
function Base.getindex(population::Population, index::Int)
    row = population.indices[index]
    return Cell(population.model, row, population.model.cells[row].gid)
end

"""Create a fixed number of cells, sampling origins uniformly within the given μm ranges.

Omitted ranges keep that coordinate at zero without consuming a random draw.
Pass Section inputs directly, including load_swc/load_asc results. Morphology is
captured once for the population. Set individual cell.position values before build!.
"""
function add_population!(
    model::MicroModel,
    sections::AbstractVector{Section},
    count::Integer;
    first_gid = length(model.cells),
    xrange = (0.0, 0.0),
    yrange = (0.0, 0.0),
    zrange = (0.0, 0.0),
    rng::AbstractRNG = Random.default_rng(),
    voltage = -65.0,
    ions = NamedTuple(),
)
    # 1. Fix population size and coordinate ranges, then capture its shared morphology.
    count >= 0 || throw(ArgumentError("population size must be nonnegative"))
    ranges = map((xrange, yrange, zrange)) do bounds
        length(bounds) == 2 && all(isfinite, bounds) && bounds[1] <= bounds[2] ||
            throw(ArgumentError("position ranges require finite lower and upper bounds"))
        (Float64(bounds[1]), Float64(bounds[2]))
    end
    first_cell = length(model.cells) + 1
    count == 0 && return Population(model, first_cell:(first_cell - 1))
    morphology = deepcopy(collect(sections))
    push!(model.morphologies, morphology)
    morphology_index = length(model.morphologies)
    sizehint!(model.cells, length(model.cells) + count; shrink = false)
    sizehint!(model.gids, length(model.gids) + count)

    # 2. Instantiate each cell with its own biophysics and initial spatial position.
    for index in 1:count
        gid = Int64(first_gid + index - 1)
        gid in model.gids && throw(ArgumentError("duplicate cell gid $gid"))
        value = voltage isa Function ? voltage(index) : voltage
        position = map(ranges) do (lower, upper)
            lower == upper ? lower : lower + (upper - lower) * rand(rng)
        end
        initial_voltage = value isa Real ? Float64(value) : Float64(value[:all])
        isfinite(initial_voltage) || throw(ArgumentError("non-finite initial voltage"))
        section_voltages = Dict{Symbol, Float64}()
        if !(value isa Real)
            for (selector, voltage_value) in pairs(value)
                selector == :all && continue
                isfinite(voltage_value) || throw(ArgumentError("non-finite initial voltage"))
                for name in selected_sections(morphology, selector)
                    section_voltages[name] = Float64(voltage_value)
                end
            end
        end
        ion_values, ion_styles = cell_ion_values(morphology, ions)
        push!(
            model.cells,
            CellInstance(
                morphology_index,
                gid,
                position,
                initial_voltage,
                section_voltages,
                DensityBinding[],
                ion_values,
                ion_styles,
            ),
        )
        push!(model.gids, gid)
    end

    # 3. Expose the fixed model range; no duplicate vector of cell handles is stored.
    return Population(model, first_cell:(first_cell + Int(count) - 1))
end

"""Register a cell's voltage detector; SID is independent of cell and node indices."""
function register_spike_source!(
    cell::Cell;
    sid::Union{Nothing, Integer} = nothing,
    section = :soma,
    x = 0.5,
    threshold = 10.0,
)
    return register_source!(
        cell.model,
        cell.index,
        Symbol(section),
        Float64(x),
        0,
        Float64(threshold),
        sid,
        nothing,
        :v,
        1,
    )
end

"""Register an ion field or a density mechanism scalar/array element as a threshold source."""
function register_spike_source!(
    cell::Cell,
    field;
    mechanism = nothing,
    section = :soma,
    x = 0.5,
    index::Integer = 1,
    threshold = 10.0,
    sid::Union{Nothing, Integer} = nothing,
)
    compiled = mechanism === nothing ? nothing : resolve_mechanism(mechanism)
    return register_source!(
        cell.model,
        cell.index,
        Symbol(section),
        Float64(x),
        0,
        Float64(threshold),
        sid,
        compiled,
        Symbol(field),
        Int(index),
    )
end

"""Register a point-process scalar/array element rather than its net_event output."""
function register_spike_source!(
    point::PointProcess,
    field;
    index::Integer = 1,
    threshold = 10.0,
    sid::Union{Nothing, Integer} = nothing,
)
    compiled = point.model.points[point.index].mechanism
    return register_source!(
        point.model,
        0,
        :none,
        0.0,
        point.index,
        Float64(threshold),
        sid,
        compiled,
        Symbol(field),
        Int(index),
    )
end

"""Register a point-process event source or an external input on a micro model."""
function register_spike_source!(
    source::Union{MicroModel, PointProcess};
    sid::Union{Nothing, Integer} = nothing,
)
    model = source isa MicroModel ? source : source.model
    point = source isa PointProcess ? source.index : 0
    threshold = source isa PointProcess ? 10.0 : 0.0
    return register_source!(model, 0, :none, 0.0, point, threshold, sid)
end

"""Retrieve an already registered spike source by its explicit SID."""
function spike_source(model::MicroModel, sid::Integer)
    index = model.source_indices[Int64(sid)]
    return SpikeSource(model, index, Int64(sid))
end

"""Connect one voltage/event source to one point process.

Point processes and artificial cells can be connected directly: their net_event
output reuses one registered source across outgoing connections. NET_RECEIVE may
use vector weights.
"""
function connect!(
    source::Union{SpikeSource, PointProcess},
    target::PointProcess;
    weight = 1.0,
    delay = 0.0,
)
    model = source.model
    target.model === model ||
        throw(ArgumentError("connection endpoints belong to different micro models"))
    block = get(model.points[target.index].mechanism.ir.blocks, :net_receive, nothing)
    block === nothing && throw(ArgumentError("target has no NET_RECEIVE block"))
    arity = length(block.arguments)
    values, latency = connection_values(weight, delay, arity)
    source_index = if source isa SpikeSource
        source.index
    else
        index = get(model.point_sources, source.index, 0)
        index == 0 ? register_spike_source!(source).index : index
    end
    push!(model.connections, Connection(source_index, target.index, values, arity, latency))
    return length(model.connections)
end

# Pack cells, mechanisms and connections.

struct MicroOutputBinding
    roi::Int
    mechanism::CompiledMechanism
    sources::Set{Int64}
    parameters::Dict{Symbol, Any}
    initial_state::Dict{Symbol, Any}
end

struct MicroInputBinding
    roi::Int
    mechanism::CompiledMechanism
    parameters::Dict{Symbol, Any}
    initial_state::Dict{Symbol, Any}
end

"""Pack morphology and mechanism storage in DHS warp/round/lane order."""
function pack_micro(model::MicroModel, inputs::Vector{MicroInputBinding}; dhs_threads::Int = 16)
    # 1. Discretize each morphology once; biophysical configuration belongs to cells.
    layouts = discretize.(model.morphologies)
    cell_count = length(model.cells)
    morphology_cells = [Int[] for _ in model.morphologies]
    for (index, cell) in enumerate(model.cells)
        push!(morphology_cells[cell.morphology], index)
    end
    total = sum(
        length(layouts[index].parent) * length(morphology_cells[index]) for
        index in eachindex(layouts);
        init = 0,
    )
    total < typemax(Int32) || throw(ArgumentError("packed node count exceeds Int32 range"))
    original_nodes, warp_offsets, round_offsets, lane_rounds =
        dhs_permutation(layouts, morphology_cells, dhs_threads)
    permutation = invperm(original_nodes)

    # 2. Copy cell geometry/voltage and retain logical section locations.
    parent = Vector{Int32}(undef, total)
    voltage = Vector{Float64}(undef, total)
    area, diameter, capacitance, axial = ntuple(_ -> Vector{Float64}(undef, total), 4)
    maps = [Dict{Symbol, Vector{Int32}}() for _ in 1:cell_count]
    starts = [Dict{Symbol, Int32}() for _ in 1:cell_count]
    offset = 0
    for morphology_index in eachindex(layouts)
        layout = layouts[morphology_index]
        cells = morphology_cells[morphology_index]
        count = length(cells)
        for (lane, cell_index) in enumerate(cells)
            # Capture this morphology's offset by value. Capturing the outer
            # reassigned offset would box every node-index calculation.
            global_node = let cell_offset = offset
                local_node -> permutation[cell_offset + (local_node - 1) * count + lane]
            end
            for node in eachindex(layout.parent)
                index = global_node(node)
                parent[index] = layout.parent[node] == 0 ? 0 : global_node(layout.parent[node])
                voltage[index] = model.cells[cell_index].voltage
                area[index] = layout.area[node]
                diameter[index] = layout.diameter[node]
                capacitance[index] = layout.capacitance[node]
                axial[index] = layout.axial_conductance[node]
            end
            for (section, nodes) in layout.section_nodes
                maps[cell_index][section] = global_node.(nodes)
                starts[cell_index][section] = global_node(layout.section_starts[section])
                override = get(model.cells[cell_index].section_voltages, section, nothing)
                if override !== nothing
                    voltage[maps[cell_index][section]] .= override
                    for endpoint in (layout.section_starts[section], nodes[end])
                        layout.parent[endpoint] == 0 && (voltage[global_node(endpoint)] = override)
                    end
                end
            end
        end
        offset += length(layout.parent) * count
    end
    # 3. Reserve mechanism storage and map logical density rows to their final layout.
    groups = MechanismRows[]
    group_indices = Dict{String, Int}()
    ion_parameters = Dict{Tuple{Int32, Symbol}, Float64}()
    row_counts = Dict{String, Int}()
    for cell in model.cells
        layout = layouts[cell.morphology]
        for binding in cell.bindings
            nodes = sum(length(layout.section_nodes[section]) - 1 for section in binding.sections)
            digest = binding.mechanism.ir.source_digest
            row_counts[digest] = get(row_counts, digest, 0) + nodes
        end
    end
    for point in model.points
        digest = point.mechanism.ir.source_digest
        row_counts[digest] = get(row_counts, digest, 0) + 1
    end
    function group_for(mechanism)
        return get!(group_indices, mechanism.ir.source_digest) do
            capacity = row_counts[mechanism.ir.source_digest]
            push!(groups, MechanismRows(mechanism; capacity))
            length(groups)
        end
    end
    for (cell_index, cell) in enumerate(model.cells)
        for binding in cell.bindings
            group = groups[group_for(binding.mechanism)]
            for section in binding.sections
                nodes = maps[cell_index][section]
                for segment in 1:(length(nodes) - 1)
                    push!(group.nodes, nodes[segment])
                    push!(group.targets, 0)
                    push!(group.owners, 0)
                    push!(group.logical_ids, UInt64(original_nodes[nodes[segment]]))
                end
            end
        end
    end
    point_count = length(model.points)
    point_group = Vector{Int32}(undef, point_count)
    point_row = Vector{Int32}(undef, point_count)
    for (index, point) in enumerate(model.points)
        digest = point.mechanism.ir.source_digest
        group_index = get(group_indices, digest, 0)
        group_index == 0 && (group_index = group_for(point.mechanism))
        group = groups[group_index]
        node =
            point.cell == 0 ? Int32(0) :
            location_node(maps[point.cell], starts[point.cell], point.section, point.x)
        push!(group.nodes, node)
        push!(group.targets, Int32(index))
        push!(group.owners, Int32(0))
        push!(group.logical_ids, UInt64(index) + UInt64(total))
        point_group[index], point_row[index] = group_index, length(group.nodes)
    end
    # Match independent density and point rows to the cable permutation.
    # Preserve ties, artificial-cell order and logical random identities.
    for group in groups
        ordered = serial_instances(group.ir) || group.ir.kind == :artificial
        count = length(group.nodes)
        sort_keys = ordered ? group.logical_ids : group.nodes
        if issorted(sort_keys)
            group.physical_rows = 1:count
        else
            # Keep only the logical-to-physical map after moving identity arrays.
            # Stable ties preserve point order and logical random identities.
            order = sortperm(sort_keys)
            group.physical_rows = invperm(order)
            group.nodes = group.nodes[order]
            group.targets = group.targets[order]
            group.owners = group.owners[order]
            group.logical_ids = group.logical_ids[order]
        end
        resize!(group.values, count * length(group.defaults))
        for (field, value) in enumerate(group.defaults)
            fill!(view(group.values, ((field - 1) * count + 1):(field * count)), value)
        end
    end
    # Evaluate callbacks once, in the same cell/binding/section/segment order.
    # Resolve each binding once; referenced arrays and callback state are read
    # when their logical instance is evaluated.
    cursors = zeros(Int, length(groups))
    for (cell_index, cell) in enumerate(model.cells)
        for binding in cell.bindings
            index = group_indices[binding.mechanism.ir.source_digest]
            group = groups[index]
            parameters = bind_parameters(group.ir, binding.parameters)
            # One concrete parameter binding covers all selected sections.
            cursors[index] = fill_density_binding!(
                group.values,
                length(group.nodes),
                group.physical_rows,
                cursors[index],
                parameters,
                cell.gid,
                cell.position,
                binding.sections,
                maps[cell_index],
                diameter,
                area,
                ion_parameters,
            )
        end
    end
    # 4. Evaluate point parameters in logical order, directly into final columns.
    # Each point's parameter dictionary is consumed once. Cache only the field
    # offsets/permissions shared by a mechanism, not a binding for every point.
    point_fields = [Dict{Symbol, Tuple{Int, Int}}() for _ in groups]
    pattern_times, pattern_targets = Float64[], Int32[]
    for (index, point) in enumerate(model.points)
        group_index = point_group[index]
        group = groups[group_index]
        logical_row = point_row[index]
        physical_row = group.physical_rows[logical_row]
        node = group.nodes[physical_row]
        apply_point_parameters!(
            group.values,
            length(group.nodes),
            physical_row,
            group.ir,
            point.parameters,
            point_fields[group_index],
            ion_parameters,
            node,
        )
        point_row[index] = physical_row
        if point.mechanism.ir.name == :PatternStim &&
           point.mechanism.ir.source_digest == builtin(:PatternStim).ir.source_digest
            append!(pattern_times, point.spike_times)
            append!(pattern_targets, fill(Int32(index), length(point.spike_times)))
        elseif !isempty(point.spike_times)
            throw(ArgumentError("spike_times applies to PatternStim only"))
        end
    end
    order = sortperm(pattern_times; alg = MergeSort)
    pattern_times, pattern_targets = pattern_times[order], pattern_targets[order]
    source_ids = [source.sid for source in model.sources]
    source_variables = Union{Nothing, VariableRef}[nothing for _ in model.sources]
    source_targets = Int32[]
    source_thresholds = Float64[]
    for source in model.sources
        push!(source_targets, source.field === nothing ? source.point : 0)
        push!(source_thresholds, source.threshold)
    end
    for index in eachindex(inputs)
        virtual_sid = -Int64(index)
        push!(source_ids, virtual_sid)
        push!(source_variables, nothing)
        push!(source_targets, point_count + index)
        push!(source_thresholds, 0.0)
    end
    # 5. Pack source fanout, delays and connection weights without losing API indices.
    # Reuse registered source indices. Virtual inputs follow them in declaration
    # order. Both micro and cross-scale edges retain their scalar creation order.
    source_count = length(model.sources)
    source_index(connection) =
        connection.source_index > 0 ? connection.source_index :
        source_count - connection.source_index
    connections = model.connections
    connection_count = length(connections)
    fanout = ones(Int64, length(source_ids) + 1)
    max_weights = 1
    for connection in connections
        fanout[source_index(connection) + 1] += 1
        max_weights = max(max_weights, connection.arity)
    end
    for index in 2:length(fanout)
        fanout[index] += fanout[index - 1] - 1
    end
    edge_weights = zeros(Float64, connection_count, max_weights)
    edge_targets = Vector{Int32}(undef, connection_count)
    edge_delays = Vector{Float64}(undef, connection_count)
    connection_edges = Vector{Int64}(undef, length(model.connections))
    connection_initializers = [has_event_initializer(group.ir) for group in groups]
    cursors = copy(fanout)
    for (original, connection) in enumerate(connections)
        source = source_index(connection)
        index = cursors[source]
        cursors[source] += 1
        edge_targets[index], edge_delays[index] = connection.target, connection.delay
        weights = connection.weights
        # NEURON clears auxiliary connection weights unless NET_RECEIVE has
        # its own INITIAL. Capture that initialization rule in the fixed snapshot.
        if weights === nothing
            edge_weights[index, 1] = connection.weight
        elseif !connection_initializers[point_group[connection.target]]
            isempty(weights) || (edge_weights[index, 1] = first(weights))
        else
            for field in eachindex(weights)
                edge_weights[index, field] = weights[field]
            end
        end
        connection_edges[original] = index
    end
    # 6. Resolve ion defaults, spatial overrides, writers and initialization styles.
    ion_defaults = copy(model.ions)
    ion_declarations = [ion for group in groups for ion in group.ir.ions]
    needed_ions = Set(ion.name for ion in ion_declarations)
    for cell in model.cells, (_, name) in keys(cell.ion_styles)
        push!(needed_ions, name)
    end
    # A loaded MOD may supply VALENCE even when another mechanism is the only
    # user of that species in this model, matching NEURON's ion registration.
    used_mechanisms = Set(group.ir.source_digest for group in groups)
    lock(MOD_CACHE_LOCK) do
        for compiled in values(LOADED_MODS)
            compiled.ir.source_digest in used_mechanisms && continue
            for ion in compiled.ir.ions
                ion.name in needed_ions && push!(ion_declarations, ion)
            end
        end
    end
    for declaration in ion_declarations
        declaration.valence === nothing && continue
        if haskey(ion_defaults, declaration.name)
            ion_defaults[declaration.name].valence == declaration.valence ||
                throw(ArgumentError("inconsistent valence for ion $(declaration.name)"))
        else
            # NEURON's defaults for a user-defined species: 1 mM on each side.
            ion_defaults[declaration.name] = IonSpecies(
                declaration.name;
                valence = declaration.valence,
                internal = 1.0,
                external = 1.0,
            )
        end
    end
    for group in groups, declaration in group.ir.ions
        haskey(ion_defaults, declaration.name) ||
            throw(ArgumentError("USEION $(declaration.name) needs a VALENCE declaration"))
    end
    ion_names = sort!(collect(keys(ion_defaults)))
    # Allocate only declared/explicitly configured (compartment, species) pairs.
    # Sort by cable node so each node owns a contiguous ion range; mechanism
    # transfers resolve their row once and retain coalesced field columns.
    ion_columns = Dict{Symbol, Tuple{Symbol, Int}}()
    for name in ion_names
        for (column, field) in enumerate((
            Symbol(:e, name),
            Symbol(name, :i),
            Symbol(name, :o),
            Symbol(:i, name),
            Symbol(:di, name, :_dv_),
        ))
            ion_columns[field] = (name, column)
        end
    end
    occupied = Set{Tuple{Int32, Symbol}}()
    for group in groups, ion in group.ir.ions, node in group.nodes
        node != 0 && push!(occupied, (node, ion.name))
    end
    # NEURON need_memb installs a point process's required ion throughout its
    # section, so later density insertion shares the same ion instances.
    point_ion_sections = Set{Tuple{Int, Symbol, Symbol}}()
    for point in model.points
        point.cell == 0 && continue
        for ion in point.mechanism.ir.ions
            push!(point_ion_sections, (point.cell, point.section, ion.name))
        end
    end
    for (cell, section, name) in point_ion_sections
        nodes = maps[cell][section]
        for node in view(nodes, 1:(length(nodes) - 1))
            push!(occupied, (node, name))
        end
    end
    for (cell_index, cell) in enumerate(model.cells)
        for (section, name) in keys(cell.ion_styles)
            haskey(ion_defaults, name) || throw(ArgumentError("unknown ion $name"))
            nodes = maps[cell_index][section]
            for node in view(nodes, 1:(length(nodes) - 1))
                push!(occupied, (node, name))
            end
        end
    end
    locations = sort!(collect(occupied))
    ion_locations = Dict(location => Int32(row) for (row, location) in enumerate(locations))
    ion_offsets = ones(Int32, total + 1)
    ion_species = Vector{Int32}(undef, length(locations))
    ion_data = zeros(Float64, length(locations), 5)
    ion_styles = zeros(UInt8, length(locations))
    valences = [ion_defaults[name].valence for name in ion_names]
    initial_concentrations = zeros(Float64, 2, length(ion_names))
    species_defaults = Dict{Symbol, Tuple{Int32, Float64, Float64, Float64, UInt8}}()
    for (index, name) in enumerate(ion_names)
        ion = ion_defaults[name]
        initial_concentrations[1, index] = ion.internal
        initial_concentrations[2, index] = ion.external
        reversal =
            ion.nernst ? nernst_potential(ion.internal, ion.external, ion.valence, model.celsius) :
            ion.reversal
        style = ion.nernst ? ion_style_bits(IonStyle(1, 2, true, true, false)) : UInt8(0)
        species_defaults[name] = (Int32(index), reversal, ion.internal, ion.external, style)
    end
    for (row, (node, name)) in enumerate(locations)
        species, reversal, internal, external, style = species_defaults[name]
        ion_offsets[node + 1] += 1
        ion_species[row] = species
        ion_styles[row] = style
        ion_data[row, 1] = reversal
        ion_data[row, 2] = internal
        ion_data[row, 3] = external
    end
    for node in 2:length(ion_offsets)
        ion_offsets[node] += ion_offsets[node - 1] - 1
    end
    for ((node, field), value) in ion_parameters
        haskey(ion_columns, field) || throw(ArgumentError("unknown ion field $field"))
        name, column = ion_columns[field]
        ion_data[ion_locations[(node, name)], column] = value
    end
    for (cell_index, cell) in enumerate(model.cells)
        for ((section, field), value) in cell.ion_values
            haskey(ion_columns, field) || throw(ArgumentError("unknown ion field $field"))
            name, column = ion_columns[field]
            nodes = view(maps[cell_index][section], 1:(length(maps[cell_index][section]) - 1))
            value isa AbstractVector &&
                length(value) != length(nodes) &&
                throw(DimensionMismatch("ion field $field expects $(length(nodes)) segment values"))
            for (segment, node) in enumerate(nodes)
                location = (;
                    gid = cell.gid,
                    position = cell.position,
                    section,
                    x = (segment - 0.5) / length(nodes),
                    diameter = diameter[node],
                    area = area[node],
                )
                actual =
                    value isa Function ? value(location) :
                    value isa AbstractVector ? value[segment] : value
                converted = Float64(actual)
                isfinite(converted) && (column == 1 || converted >= 0.0) ||
                    throw(ArgumentError("invalid ion field $field"))
                ion_data[ion_locations[(node, name)], column] = converted
            end
        end
    end
    plans = [finish_group(group; ion_locations) for group in groups]
    concentration_owners = Set{Tuple{Int32, Symbol}}()
    for group in plans
        ir = group.mechanism.ir
        for ion in ir.ions
            concentrations = (Symbol(ion.name, :i), Symbol(ion.name, :o))
            read_concentration = any(name -> name in ion.reads, concentrations)
            written = filter(name -> name in ion.writes, concentrations)
            concentration_style = !isempty(written) ? 3 : read_concentration ? 1 : 0
            reversal = Symbol(:e, ion.name)
            reversal_style = reversal in ion.writes ? 3 : reversal in ion.reads ? 1 : 0
            for node in group.nodes
                node == 0 && continue
                row = ion_locations[(node, ion.name)]
                ion_styles[row] =
                    promote_ion_style(ion_styles[row], concentration_style, reversal_style)
                for name in written
                    key = (node, name)
                    key in concentration_owners &&
                        throw(ArgumentError("multiple writers of $name at node $node"))
                    push!(concentration_owners, key)
                end
            end
        end
    end
    # An explicit species default requests dynamic reversal unless a mechanism
    # owns that reversal. Section-level overrides below take final precedence.
    for (row, (_, name)) in enumerate(locations)
        ion_defaults[name].nernst || continue
        (ion_styles[row] >> 3) & 0x03 == 0x03 && continue
        ion_styles[row] |= 0x60
    end
    for (cell_index, cell) in enumerate(model.cells)
        for ((section, ion), override) in cell.ion_styles
            for node in view(maps[cell_index][section], 1:(length(maps[cell_index][section]) - 1))
                row = ion_locations[(node, ion)]
                ion_styles[row] = ion_style_override(ion_styles[row], override)
            end
        end
    end
    # 7. Assemble the axial geometry and return the immutable topology snapshot.
    axial_diagonal = zeros(Float64, total)
    for node in permutation
        ancestor = parent[node]
        ancestor == 0 && continue
        axial_diagonal[node] += axial[node]
        axial_diagonal[ancestor] += axial[node]
    end
    cable = DHSPlan(warp_offsets, round_offsets, lane_rounds)
    return MicroPlan(
        model.id,
        parent,
        voltage,
        area,
        diameter,
        capacitance,
        axial,
        axial_diagonal,
        cable,
        getfield.(model.cells, :gid),
        maps,
        starts,
        plans,
        point_group,
        point_row,
        source_ids,
        source_variables,
        source_targets,
        source_thresholds,
        fanout,
        edge_targets,
        edge_delays,
        edge_weights,
        connection_edges,
        ion_names,
        ion_locations,
        ion_offsets,
        ion_species,
        ion_data,
        valences,
        ion_styles,
        initial_concentrations,
        pattern_times,
        pattern_targets,
        model.celsius,
    )
end

# DHS warp scheduling and node permutation.

function dhs_rounds(parent::Vector{Int32}, max_lanes::Int)
    # 1. Canonicalize only the solver's child order, as NEURON-G's tree_analysis
    # does. Imported sections and their logical DFS indices remain untouched.
    depth, remaining = zeros(Int32, length(parent)), zeros(Int32, length(parent))
    roots = Int32[]
    for node in eachindex(parent)
        ancestor = parent[node]
        0 <= ancestor < node || throw(ArgumentError("DHS expects a topologically ordered forest"))
        if ancestor == 0
            push!(roots, Int32(node))
        else
            depth[node] = depth[ancestor] + Int32(1)
            remaining[ancestor] += Int32(1)
        end
    end
    offsets = cumsum([Int32(1); remaining])
    children = Vector{Int32}(undef, offsets[end] - 1)
    cursors = copy(offsets)
    for node in eachindex(parent)
        ancestor = parent[node]
        ancestor == 0 && continue
        children[cursors[ancestor]] = Int32(node)
        cursors[ancestor] += Int32(1)
    end
    sizes, hashes = ones(Int32, length(parent)), zeros(UInt64, length(parent))
    tree_order(node) = (sizes[node], hashes[node], node)
    for node in reverse(eachindex(parent))
        descendants = view(children, offsets[node]:(offsets[node + 1] - 1))
        sort!(descendants; by = tree_order)
        hash = UInt64(length(descendants))
        for child in descendants
            hash ⊻= hashes[child] + UInt64(0x9e3779b9) + (hash << 6) + (hash >> 2)
            sizes[node] += sizes[child]
        end
        hashes[node] = hash
    end
    sort!(roots; by = tree_order)
    tree_hash = length(roots) == 1 ? hashes[only(roots)] : UInt64(length(roots))
    if length(roots) > 1
        for root in roots
            tree_hash ⊻= hashes[root] + UInt64(0x9e3779b9) + (tree_hash << 6) + (tree_hash >> 2)
        end
    end
    breadth_first = copy(roots)
    sizehint!(breadth_first, length(parent))
    cursor = 1
    while cursor <= length(breadth_first)
        node = breadth_first[cursor]
        append!(breadth_first, view(children, offsets[node]:(offsets[node + 1] - 1)))
        cursor += 1
    end

    # 2. Small trees use one lane and reverse BFS elimination. Split trees
    # choose deepest ready nodes, breaking ties by increasing BFS rank.
    split = count(iszero, remaining) >= max_lanes
    lanes = split ? max_lanes : 1
    if !split
        rounds = [Int32[node] for node in Iterators.reverse(breadth_first)]
        return (; rounds, lanes, tree_hash)
    end
    rank = invperm(breadth_first)
    heap = Tuple{Int32, Int32}[]
    for node in eachindex(parent)
        remaining[node] == 0 && push_dhs_candidate!(heap, (depth[node], -rank[node]))
    end
    rounds = Vector{Int32}[]
    while !isempty(heap)
        selected =
            Int32[breadth_first[-pop_dhs_candidate!(heap)[2]] for _ in 1:min(lanes, length(heap))]
        push!(rounds, selected)
        # Newly ready parents belong to the next round, never this round.
        for node in selected
            ancestor = parent[node]
            ancestor == 0 && continue
            remaining[ancestor] -= Int32(1)
            remaining[ancestor] == 0 &&
                push_dhs_candidate!(heap, (depth[ancestor], -rank[ancestor]))
        end
    end
    return (; rounds, lanes, tree_hash)
end

"""Lay out DHS rounds in shared warps across morphology templates."""
function dhs_permutation(layouts, morphology_cells, max_lanes::Int)
    # 1. Reuse each template's node schedule and original storage coordinates.
    original_nodes, warp_offsets, round_offsets = Int32[], Int32[1], Int32[1]
    lane_rounds = Int32[]
    total = sum(
        length(layout.parent) * length(cells) for (layout, cells) in zip(layouts, morphology_cells);
        init = 0,
    )
    sizehint!(original_nodes, total)
    schedules = NamedTuple{
        (:offset, :cells, :nodes, :lanes, :rounds, :sizes, :tree_hash),
        Tuple{Int, Int, Int, Int, Vector{Vector{Int32}}, Vector{Int32}, UInt64},
    }[]
    offset = 0
    for (layout, cells) in zip(layouts, morphology_cells)
        cell_count = length(cells)
        cell_count == 0 && continue
        rounds, lanes, tree_hash = dhs_rounds(layout.parent, max_lanes)
        sizes = zeros(Int32, lanes)
        for round in rounds, lane in eachindex(round)
            sizes[lane] += Int32(1)
        end
        push!(
            schedules,
            (;
                offset,
                cells = cell_count,
                nodes = length(layout.parent),
                lanes,
                rounds,
                sizes,
                tree_hash,
            ),
        )
        offset += length(layout.parent) * cell_count
    end
    # Match NEURON-G's size/topology grouping, including equal-size templates.
    # A template boundary does not force a partially empty warp.
    sort!(schedules; by = schedule -> (schedule.nodes, schedule.tree_hash), alg = MergeSort)

    # 2. Sort each warp's branches by work, as in dynamic_interleave2. Ready
    # counts never increase: selecting k leaves exposes at most k new parents.
    # Every branch is therefore active from round one until its cached limit;
    # sorting by that limit keeps active lanes contiguous in every round.
    pending = Tuple{Int, Int}[]
    sizehint!(pending, 32)
    function flush_warp!()
        isempty(pending) && return nothing
        branches = Tuple{Int, Int, Int}[]
        sizes = Int32[]
        for (index, cell) in pending
            for lane in 1:schedules[index].lanes
                push!(branches, (index, cell, lane))
                push!(sizes, schedules[index].sizes[lane])
            end
        end
        # NEURON-G breaks equal-size ties in reverse original lane order.
        order = sortperm(eachindex(branches); by = lane -> (sizes[lane], lane), rev = true)
        append!(lane_rounds, sizes[order])
        append!(lane_rounds, zeros(Int32, 32 - length(branches)))
        round_count = maximum(sizes)
        for round in 1:round_count
            for branch in order
                round > sizes[branch] && break
                index, cell, lane = branches[branch]
                pattern = schedules[index]
                node = pattern.rounds[round][lane]
                push!(original_nodes, Int32(pattern.offset + (node - 1) * pattern.cells + cell))
            end
            push!(round_offsets, Int32(length(original_nodes) + 1))
        end
        push!(warp_offsets, Int32(length(round_offsets)))
        empty!(pending)
        return nothing
    end

    # 3. Assign complete cells to at most 32 lanes, continuing across templates.
    used_lanes = 0
    for (index, schedule) in enumerate(schedules), cell in 1:schedule.cells
        if used_lanes + schedule.lanes > 32
            flush_warp!()
            used_lanes = 0
        end
        push!(pending, (index, cell))
        used_lanes += schedule.lanes
    end
    flush_warp!()
    return original_nodes, warp_offsets, round_offsets, lane_rounds
end
