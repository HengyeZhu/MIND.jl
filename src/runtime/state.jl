# Owned simulation storage.

"""Geometrically growing device pages, owned by the enclosing simulation buffer."""
mutable struct DevicePages{T, H}
    const heads::H
end

struct DevicePageView{T, H}
    heads::H
end

function Adapt.adapt_structure(to, pages::DevicePages{T}) where {T}
    heads = Adapt.adapt(to, pages.heads)
    return DevicePageView{T, typeof(heads)}(heads)
end

"""A scheduled target event; a negative target in device feedback identifies a source spike."""
struct ScheduledEvent
    time::Float64
    flag::Float64
    edge::Int64
    order::Int64
    target::Int32
end

struct RecordedSpike
    time::Float64
    source::Int32
end

"""Append-only GPU feedback, drained into the CPU priority queues at phase boundaries."""
struct EventOutbox{D, C, P}
    values::D
    count::C
    sequences::C
    pages::P
end
Adapt.@adapt_structure EventOutbox

"""CPU event ordering and reusable transfers; MOD state stays in its owning mechanism storage."""
mutable struct EventSchedule{D, L, I}
    heap::Vector{ScheduledEvent}
    target_groups::Vector{Int32}
    target_rows::Vector{Int32}
    target_ranks::Vector{Int}
    source_ranks::Vector{Int}
    target_source_offsets::Vector{Int64}
    target_sources::Vector{Int32}
    feedback::BitVector
    ordered::Bool
    next_order::Int64
    count::Vector{Int64}
    requests::Vector{ScheduledEvent}
    collected::D
    pending::Vector{ScheduledEvent}
    events::D
    staging::Vector{ScheduledEvent}
    offsets::Vector{Int64}
    rows::Vector{Int32}
    device_offsets::L
    device_rows::I
    groups::Vector{Tuple{Int, Int, Int}}
    transferred::CUDA.CuEvent
end

"""Numerical settings fixed by model construction."""
struct RuntimeConfiguration
    dt_micro::Float64
    dt_macro::Float64
    celsius::Float64
    secondorder::Int
end

struct SpikeBuffer{F, C, P}
    times::F
    counts::C
    pages::P
end
Adapt.@adapt_structure SpikeBuffer

struct SpikeRecording{F, I, C, P}
    times::F
    sources::I
    count::C
    pages::P
end
Adapt.@adapt_structure SpikeRecording

"""Owned device descriptors and CSR contributors for publishing micro outputs."""
struct OutputPublication{B, I, L}
    banks::B
    bank_ids::I
    value_offsets::L
    offsets::L
end
Adapt.@adapt_structure OutputPublication

"""Stable field contributions grouped by region or micro-input destination."""
struct ProjectionPublication{L}
    value_offsets::L
    destinations::L
    offsets::L
end
Adapt.@adapt_structure ProjectionPublication

"""One device spike chunk with burst growth, flushed synchronously into the host result."""
mutable struct SpikeRecordingBuffer{R}
    device::R
    times::Vector{Float64}
    sources::Vector{Int32}
    capacity::Int
    flush_bound::Int
    reserved::Int
end

"""GPU storage and compiled phases for one group of user-defined MOD instances.

Columns of data follow plan.mechanism.ir.fields; this struct has no fixed list
of model states. GLOBAL fields use the shared storage carried by the context.
"""
struct DeviceMechanism{M, D, I, R, W, E, O, C, K, T}
    plan::M
    data::D
    initial_overrides::D
    owners::I
    ion_reads::R
    ion_writes::W
    exposure_reads::E
    exposure_writes::O
    context::C
    phases::K
    mechanism::T
end

"""Only device-callable fields of a density mechanism's numerical phase."""
struct DensityOperation{F, D, C, R, W, T, U}
    function_::F
    data::D
    context::C
    ion_reads::R
    ion_writes::W
    current_fields::T
    electrode_fields::U
end
Adapt.@adapt_structure DensityOperation

"""One bounded device recording buffer and the complete host observation array."""
struct RecordingBuffer{N, D}
    device::D
    values::Array{Float64, N}
    capacity::Int
    width::Int
end

struct DeviceRecording{M, H, S, I, G}
    selections::Vector{RecordSpec}
    micro_records::M
    macro_records::H
    spikes::S
    spike_source_count::Int
    exposure_fields::I
    exposure_indices::Vector{Int}
    exposure_owners::I
    exposure_regions::Vector{Int}
    gather::G
    start_micro_step::Int64
    start_macro_step::Int64
    macro_times::Vector{Float64}
end

"""The model's device resources and clocks, separate from MOD STATE fields.

Mechanism equations operate on the compiled layouts in groups. Voltage and ions
are shared cable-system storage; exposures and history provide ROI coupling.
work is a concrete NamedTuple after allocation, not a runtime symbol table.
"""
mutable struct SimulationState{V, I, L, B, M, H, Q, S, W} <: AbstractSimulationState
    layout::ModelLayout
    configuration::RuntimeConfiguration
    groups::Vector{DeviceMechanism}
    region_groups::Vector{DeviceMechanism}
    edge_groups::Vector{DeviceMechanism}
    output_groups::Vector{DeviceMechanism}
    input_groups::Vector{DeviceMechanism}
    voltage::V
    initial_voltage::V
    area::V
    diameter::V
    capacitance::V
    axial::V
    parent::I
    diagonal::V
    rhs::V
    ions::M
    outbox::Q
    spikes::S
    macro_spikes::S
    weights::M
    source_above::B
    target_offsets::L
    target_sources::I
    exposures::M
    input_exposures::M
    exposure_snapshot::M
    history::H
    errors::I
    step_micro::Int64
    step_macro::Int64
    initialized::Bool
    recording::Union{Nothing, DeviceRecording}
    work::W
end

# Resolved device execution inputs.

# The device execution interface for compiled MOD phases. Input tags come from
# the compiler IR; this file does not interpret user variable names or equations.

struct DeviceContext{V, I, U, Q, S, W, O, E, R, B, G, P, X, T, J}
    voltage::V
    initial_voltage::V
    area::V
    diameter::V
    nodes::I
    targets::I
    logical_ids::U
    rng_counters::R
    rng_keys::R
    rng_words::B
    outbox::Q
    spikes::S
    macro_spikes::S
    weights::W
    ions::W
    ion_rows::J
    ion_events::T
    target_offsets::O
    target_sources::I
    errors::E
    shared::G
    references::X
    serial::P
    t::Float64
    dt::Float64
    secondorder::Int
    celsius::Float64
    v_init::Float64
    start_time::Float64
    stop_time::Float64
    voltage_probe::Float64
    seed::UInt64
    edge::Int64
end
Adapt.@adapt_structure DeviceContext

"""Compact CUDA arguments for deterministic numerical phases without events.

Keeping queue, connectivity and RNG arrays out of these kernels avoids spilling
large context tuples into per-thread local memory when mechanisms are fused.
"""
struct NumericalContext{V, I, G, P, X, J}
    voltage::V
    initial_voltage::V
    area::V
    diameter::V
    nodes::I
    ion_rows::J
    errors::I
    shared::G
    references::X
    serial::P
    t::Float64
    dt::Float64
    secondorder::Int
    celsius::Float64
    v_init::Float64
    start_time::Float64
    stop_time::Float64
    voltage_probe::Float64
end
Adapt.@adapt_structure NumericalContext

NumericalContext(context::DeviceContext) = NumericalContext(
    context.voltage,
    context.initial_voltage,
    context.area,
    context.diameter,
    context.nodes,
    context.ion_rows,
    context.errors,
    context.shared,
    context.references,
    context.serial,
    context.t,
    context.dt,
    context.secondorder,
    context.celsius,
    context.v_init,
    context.start_time,
    context.stop_time,
    context.voltage_probe,
)

@inline function context_at(
    context::NumericalContext;
    t = context.t,
    dt = context.dt,
    voltage_probe = context.voltage_probe,
    start_time = context.start_time,
    stop_time = context.stop_time,
)
    return NumericalContext(
        context.voltage,
        context.initial_voltage,
        context.area,
        context.diameter,
        context.nodes,
        context.ion_rows,
        context.errors,
        context.shared,
        context.references,
        context.serial,
        t,
        dt,
        context.secondorder,
        context.celsius,
        context.v_init,
        start_time,
        stop_time,
        voltage_probe,
    )
end

const KernelContext = Union{DeviceContext, NumericalContext}

@inline function context_at(
    context::DeviceContext;
    t = context.t,
    dt = context.dt,
    edge = context.edge,
    voltage_probe = context.voltage_probe,
    start_time = context.start_time,
    stop_time = context.stop_time,
)
    return DeviceContext(
        context.voltage,
        context.initial_voltage,
        context.area,
        context.diameter,
        context.nodes,
        context.targets,
        context.logical_ids,
        context.rng_counters,
        context.rng_keys,
        context.rng_words,
        context.outbox,
        context.spikes,
        context.macro_spikes,
        context.weights,
        context.ions,
        context.ion_rows,
        context.ion_events,
        context.target_offsets,
        context.target_sources,
        context.errors,
        context.shared,
        context.references,
        context.serial,
        t,
        dt,
        context.secondorder,
        context.celsius,
        context.v_init,
        start_time,
        stop_time,
        voltage_probe,
        context.seed,
        edge,
    )
end

@inline context_value(context::KernelContext, ::Val{TimeInput}, row) = context.t
@inline context_value(context::KernelContext, ::Val{StepInput}, row) = context.dt
@inline context_value(context::KernelContext, ::Val{TemperatureInput}, row) = context.celsius
@inline function context_value(context::KernelContext, ::Val{InitialVoltageInput}, row)
    node = @inbounds context.nodes[row]
    return @inbounds node == 0 ? context.v_init : context.initial_voltage[node]
end
@inline context_value(context::KernelContext, ::Val{WindowStartInput}, row) = context.start_time
@inline context_value(context::KernelContext, ::Val{WindowStopInput}, row) = context.stop_time
@inline function context_value(context::KernelContext, ::Val{VoltageInput}, row)
    node = @inbounds context.nodes[row]
    return @inbounds node == 0 ? context.v_init :
                     isnan(context.voltage_probe) ? context.voltage[node] : context.voltage_probe
end
@inline function context_value(context::KernelContext, ::Val{AreaInput}, row)
    node = @inbounds context.nodes[row]
    return @inbounds node == 0 ? 100.0 : context.area[node]
end
@inline function context_value(context::KernelContext, ::Val{DiameterInput}, row)
    node = @inbounds context.nodes[row]
    return @inbounds node == 0 ? 0.0 : context.diameter[node]
end

@inline function mechanism_error!(context, code::Int32)
    CUDA.atomic_max!(pointer(context.errors, 1), code)
    return nothing
end

@inline function event_weight(context, first_weight::Float64, index::Integer)
    context.edge == 0 && return index == 1 ? first_weight : 0.0
    return context.weights[context.edge, index]
end

@inline function set_event_weight!(context, index::Integer, value::Float64)
    context.edge != 0 && (context.weights[context.edge, index] = value)
    return nothing
end

# Managed numerical aliases.

struct DeviceAliases{B, I, O}
    banks::B
    bank_ids::I
    offsets::O
end

Adapt.@adapt_structure DeviceAliases

"""An immediate field access: aliases never use independent entry/exit copies."""
struct DirectField{D}
    data::D
    row::Int
    offset::Int
end

DirectField(data, row::Integer, offset::Integer) =
    DirectField{typeof(data)}(data, Int(row), Int(offset))

@inline Base.getindex(field::DirectField, index::Integer) =
    field.row == 0 ? field.data[field.offset + index - 1] :
    field.data[field.row, field.offset + index - 1]

@inline function Base.setindex!(field::DirectField, value, index::Integer)
    if field.row == 0
        field.data[field.offset + index - 1] = value
    else
        field.data[field.row, field.offset + index - 1] = value
    end
    return value
end

struct AliasField{A}
    aliases::A
    row::Int
    offset::Int
end

AliasField(aliases, row::Integer, offset::Integer) =
    AliasField{typeof(aliases)}(aliases, Int(row), Int(offset))

@inline function Base.getindex(field::AliasField, index::Integer)
    column = field.offset + index - 1
    bank = field.aliases.bank_ids[field.row, column]
    offset = field.aliases.offsets[field.row, column]
    # POINTER names the live target. NEURON's current probe changes its local
    # v argument, not NODEV or another mechanism's pointer target.
    return field.aliases.banks[bank][offset]
end

@inline function Base.setindex!(field::AliasField, value, index::Integer)
    column = field.offset + index - 1
    bank = field.aliases.bank_ids[field.row, column]
    offset = field.aliases.offsets[field.row, column]
    field.aliases.banks[bank][offset] = value
    return value
end

function prepare_alias_storage!(state::SimulationState, aliases::Dict{VariableRef, VariableRef})
    state.work[:aliases] = aliases
    storage = IdDict{Matrix{Float64}, typeof(state.ions)}()
    banks = CuVector{Float64}[state.voltage, vec(state.ions), vec(state.exposures)]
    bank_rows = Int[length(state.voltage), size(state.ions, 1), size(state.exposures, 1)]
    indices = Dict((:voltage, 0)=>1, (:ion, 0)=>2, (:exposure, 0)=>3)
    for (index, values) in enumerate(state.work[:globals])
        push!(banks, values)
        push!(bank_rows, length(values))
        indices[(:global, index)] = length(banks)
    end
    for kind in MECHANISM_REFERENCE_KINDS,
        (index, group) in enumerate(reference_groups(state.layout, kind))

        data = CuArray(group.data)
        storage[group.data] = data
        push!(banks, vec(data))
        push!(bank_rows, size(data, 1))
        indices[(kind, index)] = length(banks)
    end
    state.work[:mechanism_storage] = storage
    state.work[:alias_banks] = banks # Own every backing allocation for the device descriptors.
    state.work[:alias_bank_indices] = indices
    state.work[:alias_bank_rows] = bank_rows
    descriptors = CuArray(CUDA.cudaconvert.(banks))
    state.work[:alias_descriptors] = descriptors
    isempty(aliases) && return nothing
    group_aliases = IdDict{Matrix{Float64}, Any}()
    for kind in MECHANISM_REFERENCE_KINDS,
        (index, group) in enumerate(reference_groups(state.layout, kind))

        has_pointers(group.mechanism.ir) || continue
        ids = zeros(Int32, size(group.data))
        offsets = zeros(Int64, size(group.data))
        # Visit this group's resolved bindings once to locate their device storage.
        for field in group.mechanism.ir.fields
            field.scope == :pointer || continue
            for column in fieldcolumns(field), row in axes(group.data, 1)
                source = VariableRef(state.layout.identity, kind, index, row, column)
                target = aliases[source]
                bank = indices[(target.kind, target.group)]
                ids[row, column] = bank
                offsets[row, column] = reference_offset(bank_rows[bank], target)
            end
        end
        group_aliases[group.data] = DeviceAliases(descriptors, CuArray(ids), CuArray(offsets))
    end
    state.work[:group_aliases] = group_aliases
    return nothing
end

resolved_reference(state::SimulationState, ref::VariableRef) = get(state.work[:aliases], ref, ref)

# Stream dependency representation.

# Actual-storage analysis preserves ordinary read/write dependencies; only
# explicitly atomic sums may overlap across CUDA streams.
const ResourceKey = Tuple{Symbol, Int, Int}
const ResourceAccess = Dict{ResourceKey, BitSet}

struct PhaseStorageCollector
    arrays::Vector{CuArray}
    seen::IdDict{Any, Nothing}
end

mutable struct PhaseAccess
    reads::ResourceAccess
    writes::ResourceAccess
    adds::ResourceAccess
    barrier::Bool
end

struct PhaseJob
    density::Bool
    index::Int
end

struct PhaseSchedule
    jobs::Vector{PhaseJob}
    predecessors::Vector{Vector{Int}}
    stream_indices::Vector{Int}
    streams::Vector{CUDA.CuStream}
    ready::CUDA.CuEvent
    completed::Vector{Union{Nothing, CUDA.CuEvent}}
    tails::Vector{Int}
end

PhaseAccess() = PhaseAccess(ResourceAccess(), ResourceAccess(), ResourceAccess(), false)

"""The current physical simulation time."""
simulation_time(state::SimulationState) =
    state.layout.macro_model === nothing ? micro_time(state) : macro_time(state)

"""Return a value snapshot of this state's current numerical settings."""
function runtime_configuration(state::SimulationState)
    configuration = state.configuration
    return (;
        dt_micro = configuration.dt_micro,
        dt_macro = state.configuration.dt_macro,
        celsius = configuration.celsius,
        secondorder = configuration.secondorder,
    )
end

# Explicit observation and event input.

"""Read a resolved scalar explicitly from the GPU at a user observation point."""
function read_value(state::SimulationState, selection::VariableRef)
    validate_reference(state.layout, selection)
    selection = resolved_reference(state, selection)
    data = reference_storage(state, selection)
    index = reference_offset(data, selection)
    own_phase_storage!(state, CUDA.stream())
    try
        return only(Array(view(vec(data), index:index)))
    finally
        # Host observation may clear CUDA.jl's dirty marker. Reacquire before
        # subsequent descriptor-only launches, even on this same task stream.
        state.work[:phase_owner][] = nothing
    end
end

read_value(state::SimulationState, selection::RecordSpec) = read_value(state, selection.variable)

"""Read a connection's NET_RECEIVE weights, including its mutable weight state."""
function connection_weights(state::SimulationState, connection::Integer)
    edge = connection_edge(state, connection)
    plan = state.layout.micro
    target = plan.edge_targets[edge]
    mechanism = plan.groups[plan.point_group[target]].mechanism
    count = length(mechanism.ir.blocks[:net_receive].arguments)
    own_phase_storage!(state, CUDA.stream())
    try
        return vec(Array(view(state.weights, edge:edge, 1:count)))
    finally
        state.work[:phase_owner][] = nothing
    end
end

"""Schedule one connection event at an absolute arrival time, without its delay.

The host priority queue orders this event; its compiled NET_RECEIVE phase
executes with the ordinary mechanism batch. The call corresponds to the original NetCon.event API.
"""
function inject_event!(state::SimulationState, connection::Integer, time::Real)
    edge = connection_edge(state, connection)
    state.initialized || throw(ArgumentError("simulation is not initialized"))
    arrival = Float64(time)
    now = micro_time(state)
    isfinite(arrival) && arrival >= now ||
        throw(ArgumentError("event arrival precedes the current time"))
    enqueue_event!(
        state.work[:event_schedule],
        ScheduledEvent(arrival, 0.0, edge, edge, state.layout.micro.edge_targets[edge]),
    )
    return state
end

"""Schedule an external source spike at absolute time, applying every outgoing delay.

Create the source with `register_spike_source!(model; sid)` before building.
CPU delivery/fanout and device recording/micro2macro consumption use the
ordinary event pipeline. Pending input survives split runs and is cleared by initialize!.
"""
function inject_spike!(state::SimulationState, input::SpikeSource, time::Real)
    state.initialized || throw(ArgumentError("simulation is not initialized"))
    plan = state.layout.micro
    plan !== nothing && plan.model_id == input.model.id ||
        throw(ArgumentError("spike source belongs to another model"))
    source = input.index
    1 <= source <= length(plan.source_ids) &&
    plan.source_ids[source] == input.sid &&
    plan.source_variables[source] === nothing &&
    plan.source_targets[source] == 0 ||
        throw(ArgumentError("source $(input.sid) is not a built external spike input"))
    arrival = Float64(time)
    isfinite(arrival) && arrival >= micro_time(state) ||
        throw(ArgumentError("spike time precedes the current time"))
    enqueue_event!(state.work[:event_schedule], ScheduledEvent(arrival, 0.0, 0, 0, -Int32(source)))
    return state
end
