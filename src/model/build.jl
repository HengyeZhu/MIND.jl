# Pack the model's construction layout and initial values.

struct MechanismPlan
    mechanism::CompiledMechanism
    data::Matrix{Float64}
    # Zero rows indicate that INITIAL has no explicit per-instance override.
    initial_overrides::Matrix{Float64}
    nodes::Vector{Int32}
    targets::Vector{Int32}
    owners::Vector{Int32}
    logical_ids::Vector{UInt64}
    ion_rows::Matrix{Int32}
    ion_reads::Matrix{Int32}
    ion_writes::Matrix{Int32}
    exposure_reads::Matrix{Int32}
    exposure_writes::Matrix{Int32}
end

"""DHS rounds packed by warp, with each lane's active-round count cached."""
struct DHSPlan
    warp_offsets::Vector{Int32}
    round_offsets::Vector{Int32}
    lane_rounds::Vector{Int32}
end

struct MicroPlan
    model_id::UInt64
    parent::Vector{Int32}
    voltage::Vector{Float64}
    area::Vector{Float64}
    diameter::Vector{Float64}
    capacitance::Vector{Float64}
    axial::Vector{Float64}
    axial_diagonal::Vector{Float64}
    cable::DHSPlan
    cell_gids::Vector{Int64}
    location_maps::Vector{Dict{Symbol, Vector{Int32}}}
    location_starts::Vector{Dict{Symbol, Int32}}
    groups::Vector{MechanismPlan}
    point_group::Vector{Int32}
    point_row::Vector{Int32}
    source_ids::Vector{Int64}
    source_variables::Vector{Union{Nothing, VariableRef}}
    source_targets::Vector{Int32}
    source_thresholds::Vector{Float64}
    fanout_offsets::Vector{Int64}
    edge_targets::Vector{Int32}
    edge_delays::Vector{Float64}
    edge_weights::Matrix{Float64}
    connection_edges::Vector{Int64}
    ion_names::Vector{Symbol}
    ion_locations::Dict{Tuple{Int32, Symbol}, Int32}
    ion_offsets::Vector{Int32}
    ion_species::Vector{Int32}
    ion_data::Matrix{Float64}
    ion_valences::Vector{Float64}
    ion_styles::Vector{UInt8}
    ion_initial_concentrations::Matrix{Float64}
    pattern_times::Vector{Float64}
    pattern_targets::Vector{Int32}
    celsius::Float64
end

struct MacroEdgePlan
    group::MechanismPlan
    sources::Vector{Int32}
    targets::Vector{Int32}
    weights::Vector{Float64}
    delay_steps::Vector{Int64}
end

struct MacroPlan
    model_id::UInt64
    labels::Vector{String}
    exposures::Vector{Symbol}
    groups::Vector{MechanismPlan}
    edges::Vector{MacroEdgePlan}
    history::Array{Float64, 3}
    history_provided::BitMatrix
    region_group::Vector{Int32}
    region_row::Vector{Int32}
end

struct OutputTransformPlan
    group::MechanismPlan
    source_offsets::Vector{Int64}
    sources::Vector{Int32}
end

struct InputTransformPlan
    group::MechanismPlan
    stream_ids::Vector{UInt64}
end

struct GlobalPlan
    mechanism::CompiledMechanism
    values::Vector{Float64}
end

"""The model's packed CPU layout and initial values for its device state."""
struct ModelLayout <: AbstractModelLayout
    identity::UInt64
    micro::Union{Nothing, MicroPlan}
    macro_model::Union{Nothing, MacroPlan}
    outputs::Vector{OutputTransformPlan}
    inputs::Vector{InputTransformPlan}
    globals::Vector{GlobalPlan}
    dt_micro::Float64
    dt_macro::Float64
    seed::UInt64
    event_capacity::Int
    spike_capacity::Int
    secondorder::Int
    cuda_streams::Int
end

# CPU construction workflow.

"""Pack the model's instances and connections on the CPU; initialize! uploads them."""
function build!(
    model::Union{MicroModel, MacroModel};
    dt_micro = 0.025,
    dt_macro = 0.1,
    seed::Integer = 1,
    event_capacity::Integer = 64,
    spike_capacity::Integer = 16,
    secondorder::Integer = 0,
    dhs_threads::Integer = 16,
    cuda_streams::Integer = 4,
    global_parameters = (),
)
    micro = model isa MicroModel ? model : model.micro
    macro_model = model isa MacroModel ? model : nothing
    input_bindings = macro_model === nothing ? MicroInputBinding[] : macro_model.inputs

    # 1. Validate the shared time grid and construction settings.
    micro_dt, macro_dt = Float64(dt_micro), Float64(dt_macro)
    isfinite(micro_dt) && micro_dt > 0.0 ||
        throw(ArgumentError("dt_micro must be positive and finite"))
    isfinite(macro_dt) && macro_dt > 0.0 ||
        throw(ArgumentError("dt_macro must be positive and finite"))
    event_capacity > 0 && spike_capacity > 0 ||
        throw(ArgumentError("event and spike capacities must be positive"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    secondorder in 0:2 || throw(ArgumentError("secondorder must be 0, 1 or 2"))
    dhs_threads in (1, 2, 4, 8, 16, 32) ||
        throw(ArgumentError("dhs_threads must be a power of two from 1 to 32"))
    cuda_streams > 0 || throw(ArgumentError("cuda_streams must be positive"))
    if micro !== nothing && macro_model !== nothing
        integer_ratio(macro_dt, micro_dt, "dt_macro")
    end

    # 2. Pack micro morphology/mechanisms and macro regions independently.
    micro_plan =
        micro === nothing ? nothing :
        pack_micro(micro, input_bindings; dhs_threads = Int(dhs_threads))
    macro_plan = macro_model === nothing ? nothing : pack_macro(macro_model, macro_dt)

    # 3. Link both scales for per-macro-step device exchange.
    if micro_plan === nothing || macro_plan === nothing
        outputs, inputs = OutputTransformPlan[], InputTransformPlan[]
    else
        outputs, inputs = pack_transforms(macro_model, micro_plan)
    end

    # 4. Resolve shared GLOBAL values and install this model's construction snapshot.
    layout = ModelLayout(
        rand(RandomDevice(), UInt64),
        micro_plan,
        macro_plan,
        outputs,
        inputs,
        pack_globals(micro_plan, macro_plan, outputs, inputs, global_parameters),
        micro_dt,
        macro_dt,
        UInt64(seed),
        Int(event_capacity),
        Int(spike_capacity),
        Int(secondorder),
        Int(cuda_streams),
    )
    if micro_plan !== nothing
        for (index, source) in enumerate(micro.sources)
            source.field === nothing && continue
            reference = if source.point != 0
                mechanism_ref(
                    layout,
                    PointProcess(micro, source.point),
                    source.field;
                    index = source.index,
                )
            else
                cell = Cell(micro, source.cell, micro.cells[source.cell].gid)
                if source.mechanism !== nothing
                    mechanism_ref(
                        layout,
                        cell,
                        source.mechanism,
                        source.field;
                        section = source.section,
                        x = source.x,
                        index = source.index,
                    )
                elseif source.field == :v
                    voltage_ref(layout, cell; section = source.section, x = source.x)
                else
                    ion_ref(layout, cell, source.field; section = source.section, x = source.x)
                end
            end
            micro_plan.source_variables[index] = reference
        end
    end
    if model.runtime.state !== nothing
        own_phase_storage!(model.runtime.state, CUDA.stream())
        CUDA.synchronize()
    end
    model.runtime.state = nothing
    model.runtime.layout = layout
    return model
end

# Declare cross-scale exchange.

"""Bind one micro source to an ROI through a MICRO2MACRO mechanism.

As in the original MIND_Sim, calls with the same ROI, compiled mechanism,
parameters and initial state share one transform state. Each call adds one source
and returns the state's identity for recording.
"""
function micro2macro!(
    region::ROI,
    mechanism;
    source::SpikeSource,
    params = NamedTuple(),
    initial_state = NamedTuple(),
)
    validate_micro_region(region)
    model = region.model
    source.model === model.micro ||
        throw(ArgumentError("spike source belongs to another micro network"))
    compiled = resolve_mechanism(mechanism)
    compiled.ir.role == :micro2macro || throw(ArgumentError("expected MICRO2MACRO mechanism"))
    parameters, state = parameter_values(params), parameter_values(initial_state)
    for (index, output) in enumerate(model.outputs)
        if output.roi == region.index &&
           output.mechanism.ir.source_digest == compiled.ir.source_digest &&
           output.parameters == parameters &&
           output.initial_state == state
            push!(output.sources, source.sid)
            return Micro2Macro(model, index)
        end
    end
    push!(
        model.outputs,
        MicroOutputBinding(region.index, compiled, Set{Int64}((source.sid,)), parameters, state),
    )
    return Micro2Macro(model, length(model.outputs))
end

"""Bind ROI exposures and one point process through a stateful MACRO2MICRO MOD."""
function macro2micro!(
    region::ROI,
    mechanism;
    target::PointProcess,
    weight = 1.0,
    delay = 0.0,
    params = NamedTuple(),
    initial_state = NamedTuple(),
)
    model = region.model
    target.model === model.micro ||
        throw(ArgumentError("point process belongs to another micro network"))
    compiled = resolve_mechanism(mechanism)
    compiled.ir.role == :macro2micro || throw(ArgumentError("expected MACRO2MICRO mechanism"))
    block = get(model.micro.points[target.index].mechanism.ir.blocks, :net_receive, nothing)
    block === nothing && throw(ArgumentError("macro2micro target has no NET_RECEIVE"))
    arity = length(block.arguments)
    value, latency = connection_values(weight, delay, arity)
    push!(
        model.inputs,
        MicroInputBinding(
            region.index,
            compiled,
            parameter_values(params),
            parameter_values(initial_state),
        ),
    )
    # Create this edge now, on the same path as ordinary micro connections.
    # Its negative input index is rebased after the final micro source count.
    push!(
        model.micro.connections,
        Connection(-length(model.inputs), target.index, value, arity, latency),
    )
    return target
end

# Pack cross-scale transforms.

function pack_transforms(model::MacroModel, micro_plan::MicroPlan)
    # 1. Resolve micro spike sources for each MICRO2MACRO output.
    available, names = roi_exposures(model)
    source_map = Dict(sid => index for (index, sid) in enumerate(micro_plan.source_ids))
    output_groups = MechanismRows[]
    output_offsets, output_sources = Vector{Int64}[], Vector{Int32}[]
    for (index, binding) in enumerate(model.outputs)
        validate_exposure_bindings(binding.mechanism, available, binding.roi, binding.roi)
        # Contiguous batches preserve cross-transform effects and alias order.
        if isempty(output_groups) ||
           last(output_groups).ir.source_digest != binding.mechanism.ir.source_digest
            push!(output_groups, MechanismRows(binding.mechanism))
            push!(output_offsets, Int64[1])
            push!(output_sources, Int32[])
        end
        group = last(output_groups)
        add_mechanism_row!(
            group;
            owner = binding.roi,
            logical_id = UInt64(index),
            params = binding.parameters,
            initial_state = binding.initial_state,
        )
        sources = last(output_sources)
        # The previous frontend kept this set sorted after every insertion.
        # Sort once here to retain the identical source/reduction order.
        for sid in sort!(collect(binding.sources))
            haskey(source_map, sid) ||
                throw(ArgumentError("unknown micro2macro spike-source ID $sid"))
            push!(sources, source_map[sid])
        end
        push!(last(output_offsets), length(sources) + 1)
    end
    outputs = [
        OutputTransformPlan(
            finish_group(group; exposures = names),
            output_offsets[index],
            output_sources[index],
        ) for (index, group) in enumerate(output_groups)
    ]
    # 2. Resolve random identities independently of physical transform batches.
    point_count = length(model.micro.points)
    identities = Dict{Tuple{String, Int, Dict{Symbol, Any}, Dict{Symbol, Any}}, UInt64}()
    stream_ids = Vector{UInt64}(undef, length(model.inputs))
    for (index, binding) in enumerate(model.inputs)
        validate_exposure_bindings(binding.mechanism, available, binding.roi, binding.roi)
        key = (
            binding.mechanism.ir.source_digest,
            binding.roi,
            binding.parameters,
            binding.initial_state,
        )
        stream_ids[index] = get!(identities, key) do
            UInt64(length(identities))
        end
    end
    # 3. Pack contiguous IR-compatible inputs with row-specific ROI and RNG keys.
    input_groups = MechanismRows[]
    input_indices = UnitRange{Int}[]
    for (index, binding) in enumerate(model.inputs)
        if isempty(input_groups) ||
           last(input_groups).ir.source_digest != binding.mechanism.ir.source_digest
            push!(input_groups, MechanismRows(binding.mechanism))
            push!(input_indices, index:index)
        else
            input_indices[end] = first(last(input_indices)):index
        end
        group = last(input_groups)
        add_mechanism_row!(
            group;
            owner = binding.roi,
            target = point_count + index,
            logical_id = UInt64(index - 1),
            params = binding.parameters,
            initial_state = binding.initial_state,
        )
    end
    inputs = [
        InputTransformPlan(
            finish_group(group; exposures = names),
            stream_ids[input_indices[batch]],
        ) for (batch, group) in enumerate(input_groups)
    ]
    return outputs, inputs
end

# Resolve shared GLOBAL values.

"""Resolve one set of shared MOD values per mechanism in the model layout."""
function pack_globals(micro, macro_model, outputs, inputs, parameters)
    # 1. Gather all mechanism roles and validate explicit GLOBAL overrides.
    groups = MechanismPlan[]
    micro !== nothing && append!(groups, micro.groups)
    if macro_model !== nothing
        append!(groups, macro_model.groups)
        append!(groups, (edge.group for edge in macro_model.edges))
    end
    append!(groups, (binding.group for binding in outputs))
    append!(groups, (binding.group for binding in inputs))
    overrides = Dict{String, Dict{Symbol, Any}}()
    for (mechanism, values) in parameters
        compiled = resolve_mechanism(mechanism)
        supplied = parameter_values(values)
        for name in keys(supplied)
            haskey(compiled.ir.fieldmap, name) && fieldinfo(compiled.ir, name).scope == :global ||
                throw(ArgumentError("$(compiled.ir.name).$name is not a shared GLOBAL"))
        end
        digest = compiled.ir.source_digest
        haskey(overrides, digest) &&
            throw(ArgumentError("duplicate GLOBAL parameters for $(compiled.ir.name)"))
        overrides[digest] = supplied
    end
    # 2. Evaluate each shared initializer once and reject conflicting instance values.
    plans = GlobalPlan[]
    indices = Dict{String, Int}()
    evaluated = Dict{String, Vector{Float64}}()
    for group in groups
        ir = group.mechanism.ir
        fields = filter(field -> field.scope == :global, ir.fields)
        isempty(fields) && continue
        supplied = get(overrides, ir.source_digest, NamedTuple())
        values = copy(get!(evaluated, ir.source_digest) do
            initial_values = defaults(ir)
            apply_parameters!(initial_values, ir, supplied)
        end)
        for field in fields, column in fieldcolumns(field)
            all(isnan, view(group.initial_overrides, :, column)) || throw(
                ArgumentError(
                    "GLOBAL $(ir.name).$(field.name) cannot have per-instance initial_state overrides",
                ),
            )
            field.name in keys(supplied) && continue
            isempty(group.data) || (values[column] = group.data[1, column])
            for row in axes(group.data, 1)
                group.data[row, column] == values[column] || throw(
                    ArgumentError(
                        "GLOBAL $(ir.name).$(field.name) has conflicting instance values; declare RANGE or use build!(global_parameters=...)",
                    ),
                )
            end
        end
        if haskey(indices, ir.source_digest)
            existing = plans[indices[ir.source_digest]].values
            for field in fields, column in fieldcolumns(field)
                existing[column] == values[column] || throw(
                    ArgumentError(
                        "GLOBAL $(ir.name).$(field.name) differs between mechanism groups",
                    ),
                )
            end
        else
            push!(plans, GlobalPlan(group.mechanism, values))
            indices[ir.source_digest] = length(plans)
        end
    end
    # 3. Reject overrides for mechanisms absent from this model layout.
    all(digest -> haskey(indices, digest), keys(overrides)) ||
        throw(ArgumentError("GLOBAL parameters supplied for a mechanism absent from the model"))
    return plans
end
