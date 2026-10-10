# Connectome data.

"""Directed connectome matrices use [target, source] indexing and millisecond delays."""
struct Connectivity
    labels::Vector{String}
    weights::Matrix{Float64}
    delays::Matrix{Float64}
    function Connectivity(labels, weights::AbstractMatrix, delays::AbstractMatrix)
        names = String.(labels)
        count = length(names)
        count > 0 && all(!isempty, names) && length(unique(names)) == count ||
            throw(ArgumentError("ROI labels must be nonempty and unique"))
        size(weights) == size(delays) == (count, count) ||
            throw(DimensionMismatch("connectivity matrices must match ROI labels"))
        all(isfinite, weights) && all(isfinite, delays) && all(>=(0.0), delays) ||
            throw(ArgumentError("connectivity must be finite with nonnegative delays"))
        return new(names, Matrix{Float64}(weights), Matrix{Float64}(delays))
    end
end

"""Read MIND_Sim's labelled matrix,weights / matrix,delays_ms CSV format."""
function load_connectivity(path::AbstractString)
    rows = filter(row -> any(!isempty, row), csv_fields.(readlines(path)))
    function section(name)
        marker = findfirst(row -> length(row) >= 2 && row[1] == "matrix" && row[2] == name, rows)
        marker === nothing && throw(ArgumentError("missing matrix,$name in $path"))
        marker < length(rows) || throw(ArgumentError("empty matrix,$name"))
        labels = rows[marker + 1][2:end]
        count = length(labels)
        marker + count + 1 <= length(rows) || throw(ArgumentError("incomplete matrix,$name"))
        values = Matrix{Float64}(undef, count, count)
        for index in 1:count
            row = rows[marker + index + 1]
            length(row) == count + 1 && row[1] == labels[index] ||
                throw(ArgumentError("matrix,$name labels or dimensions differ"))
            for column in 1:count
                values[index, column] = parse(Float64, row[column + 1])
            end
        end
        return labels, values
    end
    labels, weights = section("weights")
    delay_labels, delays = section("delays_ms")
    labels == delay_labels || throw(ArgumentError("weight and delay labels differ"))
    for target in eachindex(labels), source in eachindex(labels)
        target == source &&
            (weights[target, source] != 0.0 || delays[target, source] != 0.0) &&
            throw(ArgumentError("connectivity CSV diagonal must be zero"))
        weights[target, source] <= 0.0 &&
            delays[target, source] != 0.0 &&
            throw(ArgumentError("CSV delay must be zero when weight is nonpositive"))
    end
    return Connectivity(labels, weights, delays)
end

# ROI ownership and projections.

struct RegionDefinition
    mechanism::CompiledMechanism
    parameters::Dict{Symbol, Any}
    initial_state::Dict{Symbol, Any}
end

struct MacroProjection
    source::Int
    target::Int
    mechanism::CompiledMechanism
    parameters::Dict{Symbol, Any}
end

mutable struct MacroModel
    runtime::ModelRuntime
    connectivity::Connectivity
    regions::Vector{Union{Nothing, RegionDefinition}}
    micro_exposures::Dict{Int, Vector{Symbol}}
    projections::Vector{MacroProjection}
    initial_histories::Dict{Tuple{Int, Symbol}, Vector{Float64}}
    micro::Union{Nothing, MicroModel}
    outputs::Vector{MicroOutputBinding}
    inputs::Vector{MicroInputBinding}
end

function MacroModel(connectivity::Connectivity)
    return MacroModel(
        ModelRuntime(),
        connectivity,
        Union{Nothing, RegionDefinition}[nothing for _ in connectivity.labels],
        Dict{Int, Vector{Symbol}}(),
        MacroProjection[],
        Dict{Tuple{Int, Symbol}, Vector{Float64}}(),
        nothing,
        MicroOutputBinding[],
        MicroInputBinding[],
    )
end

struct ROI
    model::MacroModel
    index::Int
end

function roi(model::MacroModel, label)
    index = label isa Integer ? Int(label) : findfirst(==(String(label)), model.connectivity.labels)
    index !== nothing && 1 <= index <= length(model.regions) ||
        throw(ArgumentError("unknown ROI $label"))
    return ROI(model, index)
end

"""Attach a MOD REGION mechanism and initial state to an ROI."""
function use_macro!(
    region::ROI,
    mechanism;
    params = NamedTuple(),
    initial_state = NamedTuple(),
    history = NamedTuple(),
)
    compiled = resolve_mechanism(mechanism)
    compiled.ir.role == :region || throw(ArgumentError("expected a MIND REGION mechanism"))
    region.model.regions[region.index] === nothing ||
        throw(ArgumentError("ROI already has a macro owner"))
    region.model.regions[region.index] =
        RegionDefinition(compiled, parameter_values(params), parameter_values(initial_state))
    capture_history!(region, history)
    return region
end

"""Attach an independently constructed micro network to the ROI graph.

ROI exposures define its macro-facing variables. Cells and point processes retain
their identities. Both scales share one model runtime; binding does not copy cells.
"""
function use_micro!(region::ROI, micro::MicroModel; exposures, history = NamedTuple())
    model = region.model
    haskey(model.micro_exposures, region.index) &&
        throw(ArgumentError("ROI already has a micro owner"))
    model.micro === nothing ||
        model.micro === micro ||
        throw(ArgumentError("ROIs in one simulation must share the same micro network"))
    model.micro = micro
    model.runtime = micro.runtime
    model.micro_exposures[region.index] = unique(Symbol.(exposures))
    capture_history!(region, history)
    return region
end

"""The transform state selected by micro2macro!, for recording."""
struct Micro2Macro
    model::MacroModel
    index::Int
end

"""Attach a macro coupling rule; its weight and delay come from the connectome."""
function connect!(source::ROI, target::ROI, mechanism; params = NamedTuple())
    source.model === target.model || throw(ArgumentError("ROIs belong to different models"))
    compiled = resolve_mechanism(mechanism)
    compiled.ir.role == :macro2macro || throw(ArgumentError("expected MIND MACRO2MACRO mechanism"))
    push!(
        source.model.projections,
        MacroProjection(source.index, target.index, compiled, parameter_values(params)),
    )
    return target
end

# Pack regions, projections and histories.

"""Pack ROI ownership, delayed projections and initial history on one host thread."""
function pack_macro(model::MacroModel, dt::Float64)
    # 1. Resolve every ROI's owner and available exposures.
    available, names = roi_exposures(model)
    count = length(model.regions)
    for index in 1:count
        model.regions[index] === nothing &&
            !haskey(model.micro_exposures, index) &&
            throw(ArgumentError("ROI $(model.connectivity.labels[index]) has no owner"))
    end
    # 2. Reserve storage, then pack individually declared REGION instances in order.
    groups = MechanismRows[]
    indices = Dict{String, Int}()
    region_counts = Dict{String, Int}()
    for definition in model.regions
        definition === nothing && continue
        digest = definition.mechanism.ir.source_digest
        region_counts[digest] = get(region_counts, digest, 0) + 1
    end
    region_group, region_row = zeros(Int32, count), zeros(Int32, count)
    for (owner, definition) in enumerate(model.regions)
        definition === nothing && continue
        index = get!(indices, definition.mechanism.ir.source_digest) do
            rows = region_counts[definition.mechanism.ir.source_digest]
            push!(groups, MechanismRows(definition.mechanism; capacity = rows))
            length(groups)
        end
        region_group[owner] = index
        region_row[owner] = add_mechanism_row!(
            groups[index];
            owner,
            logical_id = UInt64(owner),
            params = definition.parameters,
            initial_state = definition.initial_state,
        )
    end
    plans = [finish_group(group; exposures = names) for group in groups]
    # 3. Resolve projection inputs and pack delay steps against the macro clock.
    edge_groups = MechanismRows[]
    empty!(indices)
    sources, targets = Vector{Int32}[], Vector{Int32}[]
    weights = Vector{Float64}[]
    delay_steps = Vector{Int64}[]
    projection_counts = Dict{String, Int}()
    for projection in model.projections
        digest = projection.mechanism.ir.source_digest
        projection_counts[digest] = get(projection_counts, digest, 0) + 1
    end
    capacity = 1
    for projection in model.projections
        validate_exposure_bindings(
            projection.mechanism,
            available,
            projection.source,
            projection.target,
        )
        weight = model.connectivity.weights[projection.target, projection.source]
        weight == 0.0 && continue
        delay = model.connectivity.delays[projection.target, projection.source]
        steps = round(Int64, delay / dt, RoundNearest)
        capacity = max(capacity, steps + 1)
        index = get!(indices, projection.mechanism.ir.source_digest) do
            rows = projection_counts[projection.mechanism.ir.source_digest]
            push!(edge_groups, MechanismRows(projection.mechanism; capacity = rows))
            push!(sources, sizehint!(Int32[], rows))
            push!(targets, sizehint!(Int32[], rows))
            push!(weights, sizehint!(Float64[], rows))
            push!(delay_steps, sizehint!(Int64[], rows))
            length(edge_groups)
        end
        row = add_mechanism_row!(
            edge_groups[index];
            owner = projection.target,
            logical_id = UInt64(length(weights[index]) + 1),
            params = projection.parameters,
        )
        # MIND coupling rules expose discrete delay steps and zero-based ROI
        # IDs in MOD fields; connectome delays themselves remain FP64 ms.
        row_values = mechanism_row(edge_groups[index], row)
        for (field, input) in projection.mechanism.ir.construction_inputs
            value = if input == ConnectionWeight
                weight
            elseif input == ConnectionDelaySteps
                Float64(steps)
            elseif input == SourceROI
                Float64(projection.source - 1)
            elseif input == TargetROI
                Float64(projection.target - 1)
            else
                throw(ArgumentError("unsupported construction input $input"))
            end
            row_values[field] = value
        end
        push!(sources[index], projection.source)
        push!(targets[index], projection.target)
        push!(weights[index], weight)
        push!(delay_steps[index], steps)
    end
    edges = [
        MacroEdgePlan(
            finish_group(group; exposures = names),
            sources[index],
            targets[index],
            weights[index],
            delay_steps[index],
        ) for (index, group) in enumerate(edge_groups)
    ]
    # 4. Fill the history ring from newest to oldest, extending the oldest sample.
    # Preserve the supplied samples on the fixed macro time grid.
    capacity = max(capacity, maximum(length, values(model.initial_histories); init = 1))
    history = zeros(Float64, count, length(names), capacity)
    provided = falses(count, length(names))
    lengths = unique(length(samples) for samples in values(model.initial_histories))
    length(lengths) <= 1 ||
        throw(ArgumentError("all initial histories must have the same sample count"))
    for ((owner, name), values) in model.initial_histories
        # History has the shared ROI/output layout, including outputs with no
        # active writer on this particular ROI (for example CA3's initial z).
        name in names || throw(ArgumentError("initial history names an unknown exposure $name"))
        column = findfirst(==(name), names)
        provided[owner, column] = true
        for age in 0:(capacity - 1)
            sample = max(length(values) - age, 1)
            history[owner, column, mod(-age, capacity) + 1] = values[sample]
        end
    end
    return MacroPlan(
        UInt64(objectid(model)),
        copy(model.connectivity.labels),
        names,
        plans,
        edges,
        history,
        provided,
        region_group,
        region_row,
    )
end
