# Upload, configure and initialize a simulation.

"""Upload the model's layout and allocate its sole device state."""
function allocate_simulation(plan::ModelLayout; bindings = ())
    # 1. Validate bindings before allocating any device state.
    aliases = link_aliases(plan, bindings)
    stream_count = plan.cuda_streams
    CUDA.functional(true) ||
        throw(ArgumentError("MIND runtime requires a functional NVIDIA CUDA device"))
    CUDA.allowscalar(false)
    if any(
        group.mechanism.ir.recursive for kind in MECHANISM_REFERENCE_KINDS for
        group in reference_groups(plan, kind)
    )
        # CUDA cannot infer the maximum stack of a recursive device call graph.
        # Reserve a normal native-call stack only for models which need it;
        # retain a larger limit already selected through CUDA.limit!.
        CUDA.limit!(CUDA.LIMIT_STACK_SIZE, max(CUDA.limit(CUDA.LIMIT_STACK_SIZE), 65536))
    end

    # 2. Upload shared arrays and allocate device queues, clocks and numerical state.
    micro_plan, macro_plan = plan.micro, plan.macro_model
    node_count = micro_plan === nothing ? 0 : length(micro_plan.voltage)
    point_count = micro_plan === nothing ? 0 : length(micro_plan.point_row)
    target_count = point_count + sum(input -> size(input.group.data, 1), plan.inputs; init = 0)
    source_count = micro_plan === nothing ? 0 : length(micro_plan.source_ids)
    region_count = macro_plan === nothing ? 0 : length(macro_plan.labels)
    exposure_count = macro_plan === nothing ? 0 : length(macro_plan.exposures)
    target_snapshot =
        macro_plan !== nothing &&
        any(edge -> any(==(Int32(2)), view(edge.group.exposure_reads, 3, :)), macro_plan.edges)
    voltage = micro_plan === nothing ? CUDA.zeros(Float64, 0) : CuArray(micro_plan.voltage)
    copy_float(field) =
        micro_plan === nothing ? CUDA.zeros(Float64, 0) : CuArray(getfield(micro_plan, field))
    copy_index(field) =
        micro_plan === nothing ? CUDA.zeros(Int32, 0) : CuArray(getfield(micro_plan, field))
    # Counts delimit valid slots. Producers write every payload field before
    # delivery; empty slots need no initial fill, just as on reinitialization.
    event_capacity = max(plan.event_capacity, source_count, 1)
    outbox = EventOutbox(
        CuArray{ScheduledEvent}(undef, event_capacity),
        CUDA.zeros(Int64, 1),
        CUDA.zeros(Int64, target_count),
        device_pages(ScheduledEvent, 1),
    )
    spikes = device_spikes(source_count, plan.spike_capacity)
    macro_capacity =
        macro_plan === nothing ? plan.spike_capacity :
        max(plan.spike_capacity, plan.spike_capacity * round(Int, plan.dt_macro / plan.dt_micro))
    macro_spikes = device_spikes(source_count, macro_capacity)
    targets = micro_plan === nothing ? Int32[] : micro_plan.source_targets
    target_offsets, target_sources = target_source_index(targets, target_count)
    # Nernst/current assembly and cable assembly overwrite their scratch arrays
    # before the first read; only persistent state and counters need initial values.
    state = SimulationState(
        plan,
        RuntimeConfiguration(
            plan.dt_micro,
            plan.dt_macro,
            micro_plan === nothing ? 6.3 : micro_plan.celsius,
            plan.secondorder,
        ),
        DeviceMechanism[],
        DeviceMechanism[],
        DeviceMechanism[],
        DeviceMechanism[],
        DeviceMechanism[],
        voltage,
        copy(voltage),
        copy_float(:area),
        copy_float(:diameter),
        copy_float(:capacitance),
        copy_float(:axial),
        copy_index(:parent),
        CuArray{Float64}(undef, node_count),
        CuArray{Float64}(undef, node_count),
        CuArray{Float64}(undef, node_count),
        CuArray{Float64}(undef, node_count),
        micro_plan === nothing ? CUDA.zeros(Float64, 0, 0) : CuArray(micro_plan.ion_data),
        outbox,
        spikes,
        macro_spikes,
        micro_plan === nothing ? CUDA.zeros(Float64, 0, 1) : CuArray(micro_plan.edge_weights),
        CUDA.fill(false, 0),
        CuArray(target_offsets),
        CuArray(target_sources),
        CUDA.zeros(Float64, region_count, exposure_count),
        CUDA.zeros(Float64, region_count, exposure_count),
        target_snapshot ? CUDA.zeros(Float64, region_count, exposure_count) :
        CUDA.zeros(Float64, 0, 0),
        macro_plan === nothing ? CUDA.zeros(Float64, 0, 0, 1) : CuArray(macro_plan.history),
        CUDA.zeros(Int32, 1),
        0,
        0,
        false,
        nothing,
        Dict{Symbol, Any}(),
    )
    # 3. Bind shared GLOBAL/POINTER storage before creating mechanism contexts.
    state.work[:globals] = [CuArray(group.values) for group in plan.globals]
    prepare_alias_storage!(state, aliases)
    # Resolve every detector to the same storage banks used by managed POINTER.
    # Aliases of one scalar share one comparison and first-registration threshold.
    detector_indices = Dict{VariableRef, Int}()
    detector_sources = Vector{Int32}[]
    detector_banks, detector_offsets = Int32[], Int64[]
    detector_thresholds = Float64[]
    bank_indices = state.work[:alias_bank_indices]::Dict{Tuple{Symbol, Int}, Int}
    bank_rows = state.work[:alias_bank_rows]::Vector{Int}
    if micro_plan !== nothing
        for (source, variable) in enumerate(micro_plan.source_variables)
            variable === nothing && continue
            target = alias_target(variable, aliases)
            detector = get!(detector_indices, target) do
                push!(detector_sources, Int32[])
                push!(detector_thresholds, micro_plan.source_thresholds[source])
                bank = bank_indices[(target.kind, target.group)]
                push!(detector_banks, bank)
                push!(detector_offsets, reference_offset(bank_rows[bank], target))
                length(detector_sources)
            end
            push!(detector_sources[detector], source)
        end
    end
    offsets = Int32[1]
    sources = Int32[]
    for group in detector_sources
        append!(sources, group)
        push!(offsets, length(sources) + 1)
    end
    state.source_above = CUDA.fill(false, length(detector_sources))
    state.work[:detectors] = (;
        banks = state.work[:alias_descriptors],
        bank_ids = CuArray(detector_banks),
        indices = CuArray(detector_offsets),
        thresholds = CuArray(detector_thresholds),
        offsets = CuArray(offsets),
        sources = CuArray(sources),
    )
    delete!(state.work, :alias_bank_indices)
    delete!(state.work, :alias_bank_rows)
    delete!(state.work, :alias_descriptors)
    # 4. Prepare micro groups, ion metadata, cable workspaces and density batches.
    if micro_plan !== nothing
        state.work[:cable_workspace] =
            CableWorkspace(plan.secondorder == 0 ? CUDA.zeros(Float64, 0) : copy(voltage))
        append!(
            state.groups,
            [allocate_group(group, state, plan.dt_micro) for group in micro_plan.groups],
        )
        state.work[:ion_valences] = CuArray(micro_plan.ion_valences)
        state.work[:ion_offsets] = CuArray(micro_plan.ion_offsets)
        state.work[:ion_species] = CuArray(micro_plan.ion_species)
        state.work[:ion_styles] = CuArray(micro_plan.ion_styles)
        state.work[:ion_initial_concentrations] = CuArray(micro_plan.ion_initial_concentrations)
        state.work[:axial_diagonal] = CuArray(micro_plan.axial_diagonal)
        cable = micro_plan.cable
        state.work[:cable] = (;
            warp_offsets = CuArray(cable.warp_offsets),
            round_offsets = CuArray(cable.round_offsets),
            lane_rounds = CuArray(cable.lane_rounds),
        )
        state.work[:ordered_micro_groups] = sort(
            state.groups;
            by = group ->
                !any(
                    ion ->
                        Symbol(ion.name, :i) in ion.writes || Symbol(ion.name, :o) in ion.writes,
                    group.mechanism.ir.ions,
                ),
            alg = MergeSort,
        )
        # Fuse launch-limited populations; larger groups get separate kernels
        # to retain occupancy and avoid scanning sparse mechanism/node maps.
        # Alias-bearing groups stay separate; the access DAG orders their
        # resolved targets against contiguous ordinary density batches.
        multiprocessors = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        fuse_density =
            node_count <= 512 * multiprocessors &&
            count(group -> group.mechanism.ir.kind == :density, state.groups) > 1
        for (phase, groups) in
            ((:current, state.groups), (:advance, state.work[:ordered_micro_groups]))
            batches, separate, jobs =
                micro_phase_operations(groups, node_count, phase, fuse_density)
            state.work[Symbol(:density_, phase)] = batches
            state.work[Symbol(:separate_, phase)] = separate
            state.work[Symbol(:micro_jobs_, phase)] = jobs
        end
    end
    # 5. Prepare macro regions, delayed edges and cross-scale transform storage.
    if macro_plan !== nothing
        append!(
            state.region_groups,
            [allocate_group(group, state, plan.dt_macro) for group in macro_plan.groups],
        )
        append!(
            state.edge_groups,
            [allocate_group(edge.group, state, plan.dt_macro) for edge in macro_plan.edges],
        )
        state.work[:edge_sources] = [CuArray(edge.sources) for edge in macro_plan.edges]
        state.work[:edge_targets] = [CuArray(edge.targets) for edge in macro_plan.edges]
        state.work[:edge_weights] = [CuArray(edge.weights) for edge in macro_plan.edges]
        state.work[:edge_delays] = [CuArray(edge.delay_steps) for edge in macro_plan.edges]
        state.work[:edge_macro_targets] = [
            CuArray(Bool[macro_plan.region_group[target] != 0 for target in edge.targets]) for
            edge in macro_plan.edges
        ]
        for (key, is_macro) in ((:edge_macro_rows, true), (:edge_micro_rows, false))
            state.work[key] = [
                CuArray(
                    Int32[
                        row for (row, target) in enumerate(edge.targets) if
                        (macro_plan.region_group[target] != 0) == is_macro
                    ],
                ) for edge in macro_plan.edges
            ]
        end
        state.work[:history_provided] = CuArray(Matrix{Bool}(macro_plan.history_provided))
    end
    append!(
        state.output_groups,
        [allocate_group(output.group, state, plan.dt_macro) for output in plan.outputs],
    )
    append!(
        state.input_groups,
        [allocate_group(input.group, state, plan.dt_macro) for input in plan.inputs],
    )
    state.work[:output_sources] = [CuArray(output.sources) for output in plan.outputs]
    state.work[:input_stream_ids] = [CuArray(input.stream_ids) for input in plan.inputs]
    state.work[:output_source_offsets] = [CuArray(output.source_offsets) for output in plan.outputs]
    state.work[:output_merge] = [
        (;
            cursors = CuArray{Int64}(undef, length(output.sources)),
            times = CuArray{Float64}(undef, length(output.sources)),
            winners = CuArray{Int64}(undef, 2 * length(output.sources)),
        ) for output in plan.outputs
    ]
    # 6. Build publication/event/stream schedules and freeze runtime metadata.
    prepare_macro_publication!(state)
    prepare_event_schedule!(state, target_offsets, target_sources)
    prepare_phase_schedules!(state, stream_count)
    # Device groups retain their arrays and alias descriptors. Drop upload-only maps.
    delete!(state.work, :mechanism_storage)
    delete!(state.work, :group_aliases)
    # The dictionary is temporary during construction. The runtime receives a
    # concrete NamedTuple, so metadata lookups do not dispatch through Any.
    work = (; state.work...)
    fields = ntuple(index -> getfield(state, index), Base.fieldcount(typeof(state)) - 1)
    return SimulationState(fields..., work)
end

"""Initialize the model's device state; subsequent calls restore the same initial values.

POINTER bindings are supplied only for the first device allocation.
"""
function initialize!(model::Union{MicroModel, MacroModel}; bindings = ())
    runtime = model.runtime
    fresh = runtime.state === nothing
    if fresh
        runtime.state = allocate_simulation(runtime.layout; bindings)
    elseif !isempty(bindings)
        throw(ArgumentError("POINTER bindings are fixed at initial allocation"))
    end
    # First allocation already uploaded initial values; later calls reuse storage.
    initialize!(runtime.state; restore = !fresh)
    return model
end

"""Reset clocks and device state, then execute the model's INITIAL blocks."""
function initialize!(state::SimulationState; restore::Bool = true)
    # 1. Restart clocks, queues and detectors from the built snapshot.
    own_phase_storage!(state, CUDA.stream())
    state.initialized = false
    state.recording = nothing
    state.step_micro = 0
    state.step_macro = 0
    copyto!(state.voltage, state.initial_voltage)
    fill!(state.errors, 0)
    clear_event_counts!(state.outbox.count)
    clear_event_counts!(state.outbox.sequences)
    reset_events!(state.work[:event_schedule])
    fill!(state.source_above, false)
    clear_event_counts!(state.spikes.counts)
    clear_event_counts!(state.macro_spikes.counts)
    fill!(state.exposures, 0.0)
    fill!(state.input_exposures, 0.0)
    if restore
        for (index, globals) in enumerate(state.layout.globals)
            copyto!(state.work[:globals][index], globals.values)
        end
        if state.layout.micro !== nothing
            copyto!(state.weights, state.layout.micro.edge_weights)
            copyto!(state.ions, state.layout.micro.ion_data)
        end
    end
    # 2. Reset each group's random streams.
    micro_groups = state.layout.micro === nothing ? state.groups : state.work[:ordered_micro_groups]
    all_groups = (
        micro_groups,
        state.region_groups,
        state.edge_groups,
        state.output_groups,
        state.input_groups,
    )
    for groups in all_groups
        for group in groups
            restore && copyto!(group.data, group.plan.data)
            if !isempty(group.context.rng_counters)
                digest = parse(UInt64, first(group.mechanism.ir.source_digest, 16); base = 16)
                launch_kernel!(
                    initialize_random_kernel!,
                    size(group.data, 1),
                    group.context,
                    digest,
                )
            end
        end
    end
    # 3. Apply ion initialization, INITIAL blocks and connection initialization.
    if state.layout.micro !== nothing
        launch_kernel!(
            initialize_ions_kernel!,
            size(state.ions, 1),
            state.ions,
            state.work[:ion_styles],
            state.work[:ion_species],
            state.work[:ion_initial_concentrations],
        )
        launch_kernel!(
            prepare_currents_kernel!,
            max(length(state.voltage), size(state.ions, 1)),
            state.ions,
            state.work[:ion_valences],
            state.work[:ion_species],
            state.work[:ion_styles],
            state.configuration.celsius,
            state.current,
            state.conductance,
            Val(true),
        )
    end
    for groups in all_groups
        for (index, group) in enumerate(groups)
            weights = groups === state.edge_groups ? state.work[:edge_weights][index] : nothing
            initialize_group!(group, state; weights)
            if state.layout.micro !== nothing && any(
                ion -> Symbol(ion.name, :i) in ion.writes || Symbol(ion.name, :o) in ion.writes,
                group.mechanism.ir.ions,
            )
                slots = Tuple(
                    unique(
                        slot for
                        (_, column, slot) in eachcol(group.plan.ion_writes) if column in (2, 3)
                    ),
                )
                launch_kernel!(
                    initialize_concentration_reversals_kernel!,
                    size(group.data, 1),
                    state.ions,
                    group.context.ion_rows,
                    Val(slots),
                    state.work[:ion_species],
                    state.work[:ion_styles],
                    state.work[:ion_valences],
                    state.configuration.celsius,
                )
            end
        end
    end
    initialization = state.work[:connection_initialization]
    if initialization !== nothing
        launch_kernel!(initialize_connections_kernel!, initialization.count, initialization.payload)
    end
    # 4. Deliver time-zero events and establish the initial membrane currents.
    if state.layout.micro !== nothing
        # INITIAL and connection INITIAL may emit before any receiver executes.
        # Threshold detection starts with the first micro step, after time-zero
        # reception and current evaluation, as in NEURON's fixed-step workflow.
        flush_spikes!(state)
        deliver_events!(state, 0.0)
        micro_currents!(state, 0.0)
        flush_spikes!(state)
    end
    # 5. Publish ROI state and seed delayed history.
    for group in state.region_groups
        launch_kernel!(
            write_exposures_kernel!,
            size(group.data, 1),
            group.data,
            group.owners,
            state.exposures,
            group.exposure_writes,
        )
    end
    # Micro-owned exposures start from their supplied history (zero otherwise).
    # Transform INITIAL state is private until the first exchange step.
    if state.layout.macro_model !== nothing
        # Packing already resolved the history ring on the fixed macro grid.
        restore && copyto!(state.history, state.layout.macro_model.history)
        launch_kernel!(
            initialize_history_kernel!,
            length(state.exposures),
            state.history,
            state.exposures,
            state.work[:history_provided],
        )
        macro_coupling!(state; targets = :macro)
        for group in state.region_groups
            launch_kernel!(
                read_exposures_kernel!,
                size(group.data, 1),
                group.data,
                group.owners,
                state.exposures,
                group.exposure_reads,
            )
        end
    end
    # 6. Expose a usable state only after the complete device state is valid.
    validate_device_state!(state)
    state.initialized = true
    return state
end

# Device-group allocation and INITIAL execution are concrete-type boundaries.
function allocate_group(plan::MechanismPlan, state::SimulationState, dt::Float64)
    # 1. Allocate instance identity and stochastic state.
    count = size(plan.data, 1)
    nodes, targets, owners = CuArray(plan.nodes), CuArray(plan.targets), CuArray(plan.owners)
    logical_ids = CuArray(plan.logical_ids)
    ir = plan.mechanism.ir
    uses_random = any(block -> block.effects.random, values(ir.blocks))
    streams = uses_random ? max(1, length(ir.randoms)) : 0
    counters, keys = CUDA.zeros(UInt64, count, streams), CUDA.zeros(UInt64, count, 3 * streams)
    words = CUDA.zeros(UInt32, count, 4 * streams)
    temperature = state.configuration.celsius
    shared_index = findfirst(
        group -> group.mechanism.ir.source_digest == plan.mechanism.ir.source_digest,
        state.layout.globals,
    )
    shared = shared_index === nothing ? nothing : state.work[:globals][shared_index]
    # NET_RECEIVE imports mentioned concentrations and publishes STATE values.
    # Resolve this once from IR; no event kernel interprets field names.
    event_block = get(ir.blocks, :net_receive, nothing)
    event_names =
        event_block === nothing ? Set{Symbol}() :
        union(event_block.effects.reads, event_block.effects.writes)
    ion_events = Tuple(
        (Int32(field), Int32(column), Int32(slot), fieldinfo(ir, field).category == :state) for
        (field, column, slot) in eachcol(plan.ion_reads) if
        column in (2, 3) && event_block !== nothing && fieldinfo(ir, field).name in event_names
    )
    # 2. Capture the group's concrete device storage and shared alias context.
    context = DeviceContext(
        state.voltage,
        state.initial_voltage,
        state.area,
        state.diameter,
        nodes,
        targets,
        logical_ids,
        counters,
        keys,
        words,
        state.outbox,
        state.spikes,
        state.macro_spikes,
        state.weights,
        state.ions,
        CuArray(plan.ion_rows),
        ion_events,
        state.target_offsets,
        state.target_sources,
        state.errors,
        shared,
        has_pointers(plan.mechanism.ir) ? state.work[:group_aliases][plan.data] : nothing,
        Val(serial_group_instances(state, plan)),
        0.0,
        dt,
        temperature,
        -65.0,
        0.0,
        dt,
        NaN,
        plan.mechanism.ir.role == :macro2micro ? UInt64(0) : state.layout.seed,
        Int64(0),
    )
    # 3. Resolve active numerical phases and choose safe context/fusion variants.
    currents = copy(ir.currents)
    for ion in ir.ions
        name = Symbol(:i, ion.name)
        name in ion.writes && push!(currents, name)
    end
    current_fields = Tuple(Int32(fieldindex(ir, name)) for name in unique(currents))
    electrode_fields = Tuple(Int32(fieldindex(ir, name)) for name in ir.electrode_currents)
    breakpoint = get(ir.blocks, :breakpoint, nothing)
    effects = breakpoint === nothing ? nothing : breakpoint.effects
    compact =
        effects === nothing || !(
            effects.random ||
            effects.schedules_events ||
            effects.emits_spikes ||
            effects.discontinuous
        )
    phases = (;
        current_fields = Val(current_fields),
        electrode_fields = Val(electrode_fields),
        context = compact ? NumericalContext(context) : context,
        current = breakpoint !== nothing &&
                  ir.kind != :artificial &&
                  (!isempty(current_fields) || !isempty(electrode_fields)),
        advance = breakpoint !== nothing && (
            ir.role != :micro || !has_membrane_current(ir) || contains_solve(breakpoint.body)
        ),
        deferred_projection = !has_pointers(ir),
    )
    return DeviceMechanism(
        plan,
        state.work[:mechanism_storage][plan.data],
        CuArray(plan.initial_overrides),
        owners,
        Val(Tuple(Tuple(column) for column in eachcol(plan.ion_reads))),
        Val(Tuple(Tuple(column) for column in eachcol(plan.ion_writes))),
        Val(Tuple(Tuple(column) for column in eachcol(plan.exposure_reads))),
        Val(Tuple(Tuple(column) for column in eachcol(plan.exposure_writes))),
        context,
        phases,
        plan.mechanism,
    )
end

function initialize_group!(group::DeviceMechanism, state::SimulationState; weights = nothing)
    count = size(group.data, 1)
    launch_kernel!(
        initialize_mechanism_kernel!,
        group.context.serial isa Val{true} ? min(1, count) : count,
        group.mechanism.initialize!,
        group.data,
        group.context,
        state.ions,
        group.ion_reads,
        group.ion_writes,
        group.initial_overrides,
        weights,
    )
    return nothing
end

function density_operations(groups, node_count, phase)
    # micro_phase_operations already selected contiguous independent density
    # groups. Partition equal memberships, then upload each fused operation.
    isempty(groups) && return NamedTuple[]
    rows = zeros(Int32, node_count, length(groups))
    for (column, group) in enumerate(groups), (row, node) in enumerate(group.plan.nodes)
        rows[node, column] = row
    end
    # Nodes with identical mechanisms share one specialized kernel. Omitting
    # absent mechanisms and boundary nodes reduces register pressure and scans.
    partitions = Dict{Tuple{Vararg{Int}}, Vector{Int32}}()
    for node in 1:node_count
        signature = Tuple(column for column in eachindex(groups) if rows[node, column] != 0)
        isempty(signature) && continue
        push!(get!(Vector{Int32}, partitions, signature), Int32(node))
    end
    batches = NamedTuple[]
    for (signature, nodes) in sort!(collect(partitions); by = first)
        operations = Tuple(
            begin
                group = groups[column]
                DensityOperation(
                    phase == :current ? group.mechanism.current! : group.mechanism.advance!,
                    group.data,
                    group.phases.context,
                    group.ion_reads,
                    group.ion_writes,
                    group.phases.current_fields,
                    group.phases.electrode_fields,
                )
            end for column in signature
        )
        columns = collect(signature)
        host_rows = rows[nodes, columns]
        push!(
            batches,
            (;
                operations,
                nodes = CuArray(nodes),
                rows = CuArray(host_rows),
                host_rows,
                groups = groups[columns],
            ),
        )
    end
    return batches
end

function prepare_event_schedule!(state::SimulationState, target_offsets, target_sources)
    # 1. Resolve targets once; CPU queues use integers, never runtime MOD names.
    groups = [state.groups; state.input_groups]
    count = length(state.outbox.sequences)
    target_groups, target_rows = zeros(Int32, count), zeros(Int32, count)
    feedback = falses(length(groups))
    for (index, group) in enumerate(groups)
        block = get(group.mechanism.ir.blocks, :net_receive, nothing)
        block === nothing && continue
        feedback[index] = block.effects.schedules_events || block.effects.emits_spikes
        for (row, target) in enumerate(group.plan.targets)
            target == 0 && continue
            target_groups[target], target_rows[target] = index, row
        end
    end

    # 2. Independent target levels batch safely; cycles/shared aliases retain
    # one globally chronological queue and stop at each feedback-producing event.
    children = [Int32[] for _ in 1:count]
    indegree = zeros(Int, count)
    if state.layout.micro !== nothing
        plan = state.layout.micro
        for (source, target) in enumerate(plan.source_targets)
            target == 0 && continue
            append!(
                children[target],
                view(
                    plan.edge_targets,
                    plan.fanout_offsets[source]:(plan.fanout_offsets[source + 1] - 1),
                ),
            )
        end
        for outgoing in children
            sort!(outgoing)
            unique!(outgoing)
            for target in outgoing
                indegree[target] += 1
            end
        end
    end
    frontier = findall(iszero, indegree)
    target_levels = zeros(Int, count)
    processed, level = 0, 0
    while !isempty(frontier)
        level += 1
        next = Int[]
        for target in frontier
            processed += 1
            target_groups[target] != 0 && (target_levels[target] = level)
            for child in children[target]
                indegree[child] -= 1
                indegree[child] == 0 && push!(next, child)
            end
        end
        frontier = next
    end
    ordered =
        processed < count || any(groups) do group
            (serial_instances(group.mechanism.ir) || any(last, group.context.ion_events)) &&
            haskey(group.mechanism.ir.blocks, :net_receive)
        end
    if ordered
        target_levels .= Int.(target_groups .!= 0)
        level = 1
    end

    # 3. Retain one compact outbox and reusable host/device delivery buffers.
    capacity = length(state.outbox.values)
    transferred = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    CUDA.record(transferred)
    state.work[:event_schedule] = EventSchedule(
        [ScheduledEvent[] for _ in 1:level],
        ScheduledEvent[],
        target_groups,
        target_rows,
        target_levels,
        target_offsets,
        target_sources,
        feedback,
        ordered,
        1,
        CUDA.pin(zeros(Int64, 1)),
        CUDA.pin(Vector{ScheduledEvent}(undef, capacity)),
        CuArray{ScheduledEvent}(undef, capacity),
        ScheduledEvent[],
        ScheduledEvent[],
        falses(count),
        CuArray{ScheduledEvent}(undef, capacity),
        CUDA.pin(Vector{ScheduledEvent}(undef, capacity)),
        Int64[],
        Int32[],
        CuArray{Int64}(undef, 1),
        CuArray{Int32}(undef, 1),
        Tuple{Int, Int, Int}[],
        transferred,
    )
    state.work[:event_groups] = groups
    initialized_groups = findall(group -> has_event_initializer(group.mechanism.ir), groups)
    dispatch_needed = ordered || !isempty(initialized_groups)
    state.work[:event_target_groups], state.work[:event_target_rows] =
        dispatch_needed ? (CuArray(target_groups), CuArray(target_rows)) : (nothing, nothing)
    arrays = Tuple(group.data for group in groups)
    contexts = Tuple(group.context for group in groups)
    arguments = (
        Tuple(group.mechanism.receive! for group in groups),
        arrays,
        contexts,
        state.work[:event_target_groups],
        state.work[:event_target_rows],
        state.weights,
    )
    state.work[:event_payload] = ordered ? CuArray([CUDA.cudaconvert(arguments)]) : nothing
    # Connection INITIAL follows scalar creation order. Independent targets can
    # execute in parallel, but shared fields/aliases retain one ordered batch.
    state.work[:connection_initialization] = nothing
    if !isempty(initialized_groups) && state.layout.micro !== nothing
        plan = state.layout.micro
        edges = Int64[
            edge for edge in plan.connection_edges if
            target_groups[plan.edge_targets[edge]] in initialized_groups
        ]
        offsets = Int64[1]
        ordered = any(initialized_groups) do index
            ir = groups[index].mechanism.ir
            has_shared_writes(ir) || has_pointers(ir)
        end
        if !ordered
            sort!(edges; by = edge -> plan.edge_targets[edge], alg = MergeSort)
            for index in 2:length(edges)
                plan.edge_targets[edges[index]] != plan.edge_targets[edges[index - 1]] &&
                    push!(offsets, index)
            end
        end
        isempty(edges) || push!(offsets, length(edges) + 1)
        device_offsets, device_edges = CuArray(offsets), CuArray(edges)
        edge_targets = CuArray(plan.edge_targets)
        arguments = (
            Tuple(group.mechanism.event_initialize! for group in groups),
            arrays,
            contexts,
            device_offsets,
            device_edges,
            edge_targets,
            state.work[:event_target_groups],
            state.work[:event_target_rows],
            state.weights,
        )
        # The descriptor lives in global memory; heterogeneous mechanisms do
        # not enlarge the kernel's parameter area. Retain its index allocations.
        state.work[:connection_initialization] = (;
            count = length(offsets) - 1,
            payload = CuArray([CUDA.cudaconvert(arguments)]),
            offsets = device_offsets,
            edges = device_edges,
            targets = edge_targets,
        )
    end
    return nothing
end

# Resolve numerical phases and stream dependencies.

"""Resolve storage effects into device stream schedules and retained lifetimes."""
function prepare_phase_schedules!(state, stream_count::Int)
    pool = [CUDA.CuStream(; flags = CUDA.STREAM_NON_BLOCKING) for _ in 1:stream_count]
    state.work[:storage_identity] = IdDict(
        group.data => (kind, index) for kind in MECHANISM_REFERENCE_KINDS for
        (index, group) in enumerate(reference_groups(state.layout, kind))
    )
    merge!(state.work, build_phase_schedules(state, pool))
    state.work[:phase_storage] = phase_storage(state)
    state.work[:phase_owner] = Ref{Union{Nothing, CUDA.CuStream}}(nothing)
    return nothing
end

# Numerical phase analysis: micro currents/states, then macro regions/edges.
function build_phase_schedules(state, pool)
    schedules = Dict{Symbol, PhaseSchedule}()
    for phase in (state.layout.micro === nothing ? () : (:current, :advance))
        jobs = state.work[Symbol(:micro_jobs_, phase)]
        accesses = PhaseAccess[]
        for job in jobs
            access = PhaseAccess()
            if job.density
                batch = state.work[Symbol(:density_, phase)][job.index]
                for (column, group) in enumerate(batch.groups)
                    mechanism_access!(
                        access,
                        state,
                        group,
                        phase,
                        view(batch.host_rows, :, column);
                        exclusive = true,
                    )
                end
            else
                group = state.work[Symbol(:separate_, phase)][job.index]
                mechanism_access!(access, state, group, phase, axes(group.data, 1))
            end
            push!(accesses, access)
        end
        schedules[Symbol(phase, :_schedule)] = phase_schedule(jobs, accesses, pool)
    end
    for (phase, groups) in ((:region, state.region_groups), (:edge, state.edge_groups))
        jobs, accesses = PhaseJob[], PhaseAccess[]
        for (index, group) in enumerate(groups)
            access = mechanism_access!(PhaseAccess(), state, group, phase, axes(group.data, 1))
            kind, group_index = state.work[:storage_identity][group.plan.data]
            rows = BitSet(axes(group.data, 1))
            for binding in axes(group.plan.exposure_reads, 2)
                field = Int(group.plan.exposure_reads[1, binding])
                exposure = Int(group.plan.exposure_reads[2, binding])
                add_access!(access.writes, (kind, group_index, field), rows)
                # Edge reads use a frozen snapshot/history. REGION reads use
                # live storage and must wait for any preceding alias writer.
                phase == :region &&
                    add_access!(access.reads, (:exposure, 0, exposure), group.plan.owners)
            end
            for binding in axes(group.plan.exposure_writes, 2)
                field = Int(group.plan.exposure_writes[1, binding])
                exposure = Int(group.plan.exposure_writes[2, binding])
                add_access!(access.reads, (kind, group_index, field), rows)
                if phase == :region
                    add_access!(access.writes, (:exposure, 0, exposure), group.plan.owners)
                else
                    add_access!(access.writes, (kind, group_index, field), rows)
                    edge = state.layout.macro_model.edges[index]
                    owners =
                        group.plan.exposure_writes[3, binding] == 1 ? edge.sources : edge.targets
                    add_access!(access.adds, (:exposure, 0, exposure), owners)
                end
            end
            push!(jobs, PhaseJob(false, index))
            push!(accesses, access)
        end
        schedules[Symbol(phase, :_schedule)] = phase_schedule(jobs, accesses, pool)
    end
    return schedules
end

# Build the predecessor DAG, assign independent jobs to streams, and allocate events.
function phase_schedule(jobs, accesses, stream_pool)
    predecessors = [Int[] for _ in jobs]
    levels = ones(Int, length(jobs))
    for next in eachindex(jobs), previous in 1:(next - 1)
        if phase_dependency(accesses[previous], accesses[next])
            push!(predecessors[next], previous)
            levels[next] = max(levels[next], levels[previous] + 1)
        end
    end
    counters = zeros(Int, maximum(levels; init = 0))
    for level in levels
        counters[level] += 1
    end
    width = maximum(counters; init = 1)
    stream_count = min(length(stream_pool), width)
    streams = stream_pool[1:stream_count]
    indices, tails = ones(Int, length(jobs)), zeros(Int, stream_count)
    fill!(counters, 0)
    for index in eachindex(jobs)
        level = levels[index]
        indices[index] = mod(counters[level], stream_count) + 1
        counters[level] += 1
        tails[indices[index]] = index
    end
    filter!(!iszero, tails)
    # A completion is needed only at a cross-stream dependency or a stream tail.
    # Successive jobs on the same stream already execute in launch order.
    completed = Union{Nothing, CUDA.CuEvent}[nothing for _ in jobs]
    event() = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    if stream_count > 1
        for tail in tails
            completed[tail] = event()
        end
        for next in eachindex(jobs), previous in predecessors[next]
            indices[previous] == indices[next] && continue
            completed[previous] === nothing && (completed[previous] = event())
        end
    end
    return PhaseSchedule(jobs, predecessors, indices, streams, event(), completed, tails)
end

function mechanism_access!(access, state, group, phase, rows; exclusive = false)
    ir = group.mechanism.ir
    block = get(ir.blocks, :breakpoint, nothing)
    block === nothing && return access
    effects = block.effects
    access.barrier |= effects.schedules_events || effects.emits_spikes || effects.discontinuous
    kind, index = state.work[:storage_identity][group.plan.data]
    global_index = findfirst(
        candidate -> candidate.mechanism.ir.source_digest == ir.source_digest,
        state.layout.globals,
    )
    row_set = BitSet(rows)
    nodes = BitSet(node for row in rows for node in (group.plan.nodes[row],) if node != 0)
    for field in ir.fields, column in fieldcolumns(field)
        read = field.name in effects.reads
        write = field.name in effects.writes
        if field.scope == :pointer
            for row in rows
                source = VariableRef(state.layout.identity, kind, index, row, column)
                target = state.work[:aliases][source]
                read && add_reference_access!(access.reads, target)
                write && add_reference_access!(access.writes, target)
            end
        elseif field.scope == :global
            key = (:global, something(global_index), column)
            read && add_access!(access.reads, key, (1,))
            write && add_access!(access.writes, key, (1,))
        else
            key = (kind, index, column)
            read && add_access!(access.reads, key, row_set)
            write && add_access!(access.writes, key, row_set)
        end
    end
    # Execution inputs and ion transfers address storage outside the mechanism.
    for (name, input) in ir.inputs
        input == VoltageInput &&
            name in effects.reads &&
            add_access!(access.reads, (:voltage, 0, 0), nodes)
    end
    ion_rows = [
        BitSet(
            group.plan.ion_rows[row, slot] for row in rows if group.plan.ion_rows[row, slot] != 0
        ) for slot in axes(group.plan.ion_rows, 2)
    ]
    for binding in axes(group.plan.ion_reads, 2)
        field = group.plan.ion_reads[1, binding]
        ion = group.plan.ion_reads[2, binding]
        slot = group.plan.ion_reads[3, binding]
        add_access!(access.reads, (:ion, 0, Int(ion)), ion_rows[slot])
        add_access!(access.writes, (kind, index, Int(field)), row_set)
    end
    if phase == :current
        # The current wrapper reads NODEV for both voltage probes, even when
        # the MOD body does not explicitly name its voltage input.
        add_access!(access.reads, (:voltage, 0, 0), nodes)
        sums = exclusive ? access.writes : access.adds
        for field in (
            binding_fields(group.phases.current_fields)...,
            binding_fields(group.phases.electrode_fields)...,
        )
            add_access!(access.reads, (kind, index, Int(field)), row_set)
        end
        add_access!(sums, (:current, 0, 0), nodes)
        add_access!(sums, (:conductance, 0, 0), nodes)
        for binding in axes(group.plan.ion_writes, 2)
            column = Int(group.plan.ion_writes[2, binding])
            slot = group.plan.ion_writes[3, binding]
            field = Int(group.plan.ion_writes[1, binding])
            add_access!(access.reads, (kind, index, field), row_set)
            if column == 4
                add_access!(sums, (:ion, 0, column), ion_rows[slot])
                add_access!(sums, (:ion, 0, column + 1), ion_rows[slot])
            else
                add_access!(access.writes, (:ion, 0, column), ion_rows[slot])
            end
        end
    else
        for binding in axes(group.plan.ion_writes, 2)
            column = Int(group.plan.ion_writes[2, binding])
            slot = group.plan.ion_writes[3, binding]
            column == 4 && continue
            field = Int(group.plan.ion_writes[1, binding])
            add_access!(access.reads, (kind, index, field), row_set)
            add_access!(access.writes, (:ion, 0, column), ion_rows[slot])
        end
    end
    return access
end

"""Keep shared writers and cross-row aliases ordered; allow independent rows to use GPU lanes."""
function serial_group_instances(state, plan::MechanismPlan)
    ir = plan.mechanism.ir
    (has_shared_writes(ir) || has_voltage_reset(ir)) && return true
    # Pure current additions commute. A same-node read of an accumulated ion
    # current, or any ordinary ion write, gives rows an observable dependency.
    ion_reads = Set((column, slot) for (_, column, slot) in eachcol(plan.ion_reads))
    if any(
        (column != 4 || (column, slot) in ion_reads) for
        (_, column, slot) in eachcol(plan.ion_writes)
    )
        occupied = BitSet()
        for node in plan.nodes
            node == 0 && continue
            node in occupied && return true
            push!(occupied, node)
        end
    end
    has_pointers(ir) || return false
    identity = nothing
    for kind in MECHANISM_REFERENCE_KINDS,
        (index, candidate) in enumerate(reference_groups(state.layout, kind))

        candidate.data === plan.data && (identity = (kind, index))
    end
    kind, index = something(identity)
    readers, writers = Dict{VariableRef, Int}(), Dict{VariableRef, Int}()
    function remember!(access, reference, row)
        access[reference] = get(access, reference, row) == row ? row : -1
    end
    # Include ordinary shared-storage accesses before comparing aliases. A
    # pointer into another row's voltage, ion or ROI may conflict even when
    # every pointer in the group has a different target.
    read_names = Set{Symbol}()
    foreach(block -> union!(read_names, block.effects.reads), values(ir.blocks))
    reads_voltage =
        has_membrane_current(ir) ||
        any(binding -> last(binding) == VoltageInput && first(binding) in read_names, ir.inputs)
    global_group = findfirst(
        group -> group.mechanism.ir.source_digest == ir.source_digest,
        state.layout.globals,
    )
    for row in axes(plan.data, 1)
        node = plan.nodes[row]
        if node != 0
            reads_voltage &&
                remember!(readers, VariableRef(state.layout.identity, :voltage, 0, node, 0), row)
            for (access, bindings) in ((readers, plan.ion_reads), (writers, plan.ion_writes))
                for (_, column, slot) in eachcol(bindings)
                    ion_row = plan.ion_rows[row, slot]
                    remember!(
                        access,
                        VariableRef(state.layout.identity, :ion, 0, ion_row, column),
                        row,
                    )
                    # Ionic current evaluation also accumulates di/dv. A
                    # POINTER in another row can observe or overwrite it.
                    if access === writers && column == 4
                        remember!(
                            writers,
                            VariableRef(state.layout.identity, :ion, 0, ion_row, 5),
                            row,
                        )
                    end
                end
            end
        end
        for field in ir.fields, column in fieldcolumns(field)
            field.scope == :global && field.name in read_names || continue
            remember!(
                readers,
                VariableRef(state.layout.identity, :global, something(global_group), 0, column),
                row,
            )
        end
        if ir.role == :region
            for (access, bindings) in
                ((readers, plan.exposure_reads), (writers, plan.exposure_writes))
                for column in view(bindings, 2, :)
                    remember!(
                        access,
                        VariableRef(state.layout.identity, :exposure, 0, plan.owners[row], column),
                        row,
                    )
                end
            end
        elseif ir.role == :macro2macro
            edge = state.layout.macro_model.edges[index]
            # READ bindings use frozen history/snapshots; only macro-owned
            # writes touch the live exposure bank addressable by POINTER.
            state.layout.macro_model.region_group[edge.targets[row]] == 0 && continue
            for binding in axes(plan.exposure_writes, 2)
                column, direction =
                    plan.exposure_writes[2, binding], plan.exposure_writes[3, binding]
                owner = direction == 1 ? edge.sources[row] : edge.targets[row]
                remember!(
                    writers,
                    VariableRef(state.layout.identity, :exposure, 0, owner, column),
                    row,
                )
            end
        end
    end
    for field in ir.fields
        field.scope == :pointer || continue
        read = any(block -> field.name in block.effects.reads, values(ir.blocks))
        write = any(block -> field.name in block.effects.writes, values(ir.blocks))
        for column in fieldcolumns(field), row in axes(plan.data, 1)
            source = VariableRef(state.layout.identity, kind, index, row, column)
            target = state.work[:aliases][source]
            if target.kind == kind && target.group == index
                target.row != row && (read || write) && return true
                continue
            end
            if write
                get(readers, target, row) != row && return true
                get(writers, target, row) != row && return true
                writers[target] = row
            end
            if read
                get(writers, target, row) != row && return true
                # -1 represents multiple independent readers. A later writer
                # then conflicts regardless of which reader was encountered last.
                readers[target] = get(readers, target, row) == row ? row : -1
            end
        end
    end
    return false
end

# Retain every allocation referenced by descriptor-only stream launches.
function phase_storage(state)
    collector = PhaseStorageCollector(CuArray[], IdDict{Any, Nothing}())
    # Ordered reception and connection INITIAL carry raw descriptors in their
    # payloads. Their index allocations must follow the same task/stream handoff.
    Adapt.adapt(
        collector,
        (
            state.work[:event_target_groups],
            state.work[:event_target_rows],
            state.work[:connection_initialization],
        ),
    )
    Adapt.adapt(
        collector,
        (
            state.voltage,
            state.ions,
            state.current,
            state.conductance,
            state.exposures,
            state.input_exposures,
            state.exposure_snapshot,
            state.history,
            state.errors,
            state.work[:detectors],
            state.source_above,
            state.spikes,
            state.macro_spikes,
        ),
    )
    for bank in state.work[:alias_banks]
        Adapt.adapt(collector, bank)
    end
    if state.layout.macro_model !== nothing
        Adapt.adapt(collector, state.work[:coupling_reset_masks])
        for publication in state.work[:edge_publication]
            Adapt.adapt(collector, publication)
        end
        Adapt.adapt(collector, state.work[:output_publication])
    end
    for groups in (
            state.groups,
            state.region_groups,
            state.edge_groups,
            state.output_groups,
            state.input_groups,
        ),
        group in groups

        Adapt.adapt(collector, (group.data, group.context, group.phases.context, group.owners))
    end
    if state.layout.micro !== nothing
        for batches in (state.work[:density_current], state.work[:density_advance]),
            batch in batches

            Adapt.adapt(collector, (batch.operations, batch.nodes, batch.rows))
        end
    end
    for key in (
        :edge_sources,
        :edge_targets,
        :edge_weights,
        :edge_delays,
        :edge_macro_targets,
        :edge_macro_rows,
        :edge_micro_rows,
        :input_stream_ids,
    )
        haskey(state.work, key) || continue
        for array in state.work[key]
            Adapt.adapt(collector, array)
        end
    end
    return collector.arrays
end

# Precompute stable publication destinations and retained field descriptors.

"""Prepare unique reset locations and output contribution indices before time stepping."""
function prepare_macro_publication!(state)
    plan = state.layout.macro_model
    plan === nothing && return nothing
    state.work[:coupling_reset_masks] = CuArray(coupling_reset_masks(plan))
    state.work[:edge_publication] = [projection_publication(edge, plan) for edge in plan.edges]
    # Group storage is owned by SimulationState/alias_banks; raw descriptors
    # avoid copies while retaining the fixed binding graph.
    descriptor_type = typeof(CUDA.cudaconvert(state.voltage))
    banks = descriptor_type[CUDA.cudaconvert(vec(group.data)) for group in state.output_groups]
    contributions = Dict{Int64, Vector{Tuple{Int32, Int64}}}()
    owners = length(plan.labels)
    for (bank, output) in enumerate(state.layout.outputs)
        group = output.group
        for row in axes(group.data, 1), index in axes(group.exposure_writes, 2)
            field = group.exposure_writes[1, index]
            exposure = group.exposure_writes[2, index]
            destination = group.owners[row] + (exposure - 1) * owners
            value_offset = row + (field - 1) * size(group.data, 1)
            push!(
                get!(Vector{Tuple{Int32, Int64}}, contributions, destination),
                (Int32(bank), value_offset),
            )
        end
    end
    # Empty output groups need no device descriptors or publication launches.
    state.work[:output_publication] = nothing
    isempty(banks) && return nothing
    bank_ids, value_offsets, offsets = Int32[], Int64[], Int64[1]
    # Dense CSR rows address the exposure bank directly. Empty rows retain the
    # macro-owned value; no second destination map is needed at the boundary.
    for destination in 1:length(state.exposures)
        for (bank, offset) in get(contributions, destination, ())
            push!(bank_ids, bank)
            push!(value_offsets, offset)
        end
        push!(offsets, length(bank_ids) + 1)
    end
    state.work[:output_publication] = OutputPublication(
        CuArray(banks),
        CuArray(bank_ids),
        CuArray(value_offsets),
        CuArray(offsets),
    )
    return nothing
end
