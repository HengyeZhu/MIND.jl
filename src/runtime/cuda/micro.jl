# Device events and spike routing.

@inline function refresh_random_words!(context, row, column)
    counter = context.rng_counters[row, column]
    offset = 3 * (column - 1)
    a, b = (counter >> 2) % UInt32, context.rng_keys[row, offset + 3] % UInt32
    c, d = context.rng_keys[row, offset + 1] % UInt32, context.rng_keys[row, offset + 2] % UInt32
    key0 = context.seed % UInt32
    key1 = (context.seed >> 32) % UInt32
    for _ in 1:10
        product0 = UInt64(0xd2511f53) * UInt64(a)
        product1 = UInt64(0xcd9e8d57) * UInt64(c)
        a, b, c, d = ((product1 >> 32) % UInt32) ⊻ b ⊻ key0,
        product1 % UInt32,
        ((product0 >> 32) % UInt32) ⊻ d ⊻ key1,
        product0 % UInt32
        key0 += UInt32(0x9e3779b9)
        key1 += UInt32(0xbb67ae85)
    end
    offset = 4 * (column - 1)
    context.rng_words[row, offset + 1] = a
    context.rng_words[row, offset + 2] = b
    context.rng_words[row, offset + 3] = c
    context.rng_words[row, offset + 4] = d
    return nothing
end

@inline function uniform_random!(context, row, stream)
    column = Int(stream)
    counter = context.rng_counters[row, column]
    which = Int(counter & UInt64(3))
    word = context.rng_words[row, 4 * (column - 1) + which + 1]
    next_counter = (counter + UInt64(1)) & UInt64(0x3ffffffff)
    context.rng_counters[row, column] = next_counter
    if which == 3
        # Random123 increments its full counter and prepares the next four words
        # when consuming the fourth, even if MOD changes the IDs before drawing.
        if next_counter == 0
            offset = 3 * (column - 1)
            for index in (offset + 3, offset + 1, offset + 2)
                value = (context.rng_keys[row, index] + UInt64(1)) % UInt32
                context.rng_keys[row, index] = UInt64(value)
                value != 0 && break
            end
        end
        refresh_random_words!(context, row, column)
    end
    # Match NEURON Random123's open-interval conversion in Float64. The
    # 32-bit words are discrete PRNG output, never Float32 numerical state.
    return (Float64(word) + 1.0) * (1.0 / 4294967297.0)
end

@inline function normal_random!(context, row, stream)
    while true
        first = 2.0 * uniform_random!(context, row, stream) - 1.0
        second = 2.0 * uniform_random!(context, row, stream) - 1.0
        radius = first * first + second * second
        0.0 < radius <= 1.0 && return first * sqrt(-2.0 * log(radius) / radius)
    end
end

@inline function set_random_sequence!(context, row, stream, sequence)
    bounded = 0.0 <= sequence <= Float64(0xffffffffff) ? sequence : 0.0
    context.rng_counters[row, Int(stream)] = UInt64(trunc(bounded)) & UInt64(0x3ffffffff)
    refresh_random_words!(context, row, Int(stream))
    return 0.0
end

@inline function set_random_ids!(context, row, stream, first, second, third = 0.0)
    offset = 3 * (Int(stream) - 1)
    context.rng_keys[row, offset + 1] = UInt64(trunc(first))
    context.rng_keys[row, offset + 2] = UInt64(trunc(second))
    context.rng_keys[row, offset + 3] = UInt64(trunc(third))
    # Match nrnran123_setids: retain the sequence and the already generated words.
    # A subsequent random_setseq regenerates them immediately with the new IDs.
    return 0.0
end

function initialize_random_kernel!(context, digest)
    row = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    row > length(context.logical_ids) && return nothing
    identity = context.logical_ids[row]
    for stream in axes(context.rng_counters, 2)
        offset = 3 * (stream - 1)
        context.rng_keys[row, offset + 1] = UInt64(identity % UInt32)
        context.rng_keys[row, offset + 2] = identity >> 32
        context.rng_keys[row, offset + 3] = UInt64(digest % UInt32) ⊻ UInt64(stream - 1)
        set_random_sequence!(context, row, stream, 0.0)
    end
    return nothing
end

@inline function emit_event!(outbox, errors, event::ScheduledEvent)
    slot = CUDA.atomic_add!(pointer(outbox.count, 1), Int64(1)) + Int64(1)
    if slot <= length(outbox.values)
        outbox.values[slot] = event
    else
        address = page_pointer(outbox.pages, 1, slot - length(outbox.values), true, errors)
        UInt(address) != 0 && unsafe_store!(address, event)
    end
    return nothing
end

@inline function schedule_self!(context, row, delay::Float64, flag::Float64)
    if delay < 0.0 || !isfinite(delay) || context.targets[row] == 0
        mechanism_error!(context, Int32(5))
        return 0.0
    end
    target = context.targets[row]
    sequence = context.outbox.sequences[target] + 1
    context.outbox.sequences[target] = sequence
    emit_event!(
        context.outbox,
        context.errors,
        ScheduledEvent(context.t + delay, flag, context.edge, sequence, Int32(target)),
    )
    return 0.0
end

function initialize_connections_kernel!(payload)
    functions, arrays, contexts, offsets, edges, targets, groups, rows, weights =
        @inbounds payload[1]
    batch = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    batch >= length(offsets) && return nothing
    for index in offsets[batch]:(offsets[batch + 1] - 1)
        edge = edges[index]
        target = targets[edge]
        dispatch_event!(
            functions,
            arrays,
            contexts,
            groups[target],
            rows[target],
            0.0,
            edge,
            weights[edge, 1],
            0.0,
        )
    end
    return nothing
end

@inline function append_spike!(buffer, errors, source::Integer, time::Float64)
    position = CUDA.atomic_add!(pointer(buffer.counts, source), Int64(1)) + Int64(1)
    store_spike!(buffer, source, position, time, errors)
    return nothing
end

@inline function emit_spike!(context, row, time::Float64)
    target = context.targets[row]
    target == 0 && return 0.0
    sequence = context.outbox.sequences[target] + 1
    context.outbox.sequences[target] = sequence
    for offset in context.target_offsets[target]:(context.target_offsets[target + 1] - 1)
        source = context.target_sources[offset]
        append_spike!(context.spikes, context.errors, source, time)
        append_spike!(context.macro_spikes, context.errors, source, time)
        # Publish at the MOD call site, preserving its order relative to
        # net_send. A later sweep of the spike buffer would lose that order.
        emit_event!(
            context.outbox,
            context.errors,
            ScheduledEvent(time, 0.0, context.edge, sequence, -Int32(source)),
        )
    end
    return 0.0
end

"""Execute a CPU-ordered batch: one worker per target, preserving its event order."""
function receive_batch_kernel!(
    function_::F,
    data,
    context,
    events,
    offsets,
    rows,
    first,
    count,
) where {F}
    for index in instance_indices(count, context.serial)
        batch = first + index - 1
        row = rows[batch]
        @inbounds for slot in offsets[batch]:(offsets[batch + 1] - 1)
            event = events[slot]
            weight = event.edge == 0 ? 0.0 : context.weights[event.edge, 1]
            function_(
                data,
                row,
                context_at(context; t = event.time, edge = event.edge),
                weight,
                event.flag,
            )
        end
    end
    return nothing
end

@inline function dispatch_event!(
    functions::Tuple,
    arrays::Tuple,
    contexts::Tuple,
    group,
    row,
    time,
    edge,
    weight,
    flag,
)
    if group == 1
        first(functions)(
            first(arrays),
            row,
            context_at(first(contexts); t = time, edge),
            weight,
            flag,
        )
    else
        dispatch_event!(
            Base.tail(functions),
            Base.tail(arrays),
            Base.tail(contexts),
            group - 1,
            row,
            time,
            edge,
            weight,
            flag,
        )
    end
    return nothing
end

@inline dispatch_event!(::Tuple{}, ::Tuple{}, ::Tuple{}, args...) = nothing

"""Preserve chronological shared effects; CPU feedback ends each causally closed batch."""
function receive_ordered_kernel!(payload, events, count)
    functions, arrays, contexts, groups, rows, weights = @inbounds payload[1]
    @inbounds for index in 1:count
        event = events[index]
        weight = event.edge == 0 ? 0.0 : weights[event.edge, 1]
        dispatch_event!(
            functions,
            arrays,
            contexts,
            groups[event.target],
            rows[event.target],
            event.time,
            event.edge,
            weight,
            event.flag,
        )
    end
    return nothing
end

function publish_external_spikes_kernel!(events, count, spikes, macro_spikes, errors)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    index > count && return nothing
    event = @inbounds events[index]
    index > 1 && events[index - 1].target == event.target && return nothing
    source = -event.target
    # One worker owns each source's sorted input sequence.
    while index <= count && events[index].target == event.target
        time = @inbounds events[index].time
        append_spike!(spikes, errors, source, time)
        append_spike!(macro_spikes, errors, source, time)
        index += 1
    end
    return nothing
end

function gather_event_requests_kernel!(values, outbox, count)
    index = Int64(blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    index > count && return nothing
    if index <= length(outbox.values)
        values[index] = outbox.values[index]
    else
        address = page_pointer(outbox.pages, 1, index - length(outbox.values), false, nothing)
        values[index] =
            UInt(address) == 0 ? ScheduledEvent(Inf, 0.0, 0, 0, 0) : unsafe_load(address)
    end
    return nothing
end

function detect_spikes_kernel!(detectors, above, spikes, macro_spikes, outbox, errors, time)
    detector = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    detector > length(above) && return nothing
    @inbounds begin
        value = detectors.banks[detectors.bank_ids[detector]][detectors.indices[detector]]
        current = value > detectors.thresholds[detector]
        if current && !above[detector]
            for index in detectors.offsets[detector]:(detectors.offsets[detector + 1] - 1)
                source = detectors.sources[index]
                append_spike!(spikes, errors, source, time)
                append_spike!(macro_spikes, errors, source, time)
                emit_event!(outbox, errors, ScheduledEvent(time, 0.0, 0, 0, -Int32(source)))
            end
        end
        above[detector] = current
    end
    return nothing
end

# Generated mechanism and ion kernels.

"""Accumulate FP64 sums with CUDA atomicAdd's device-scope relaxed ordering.

Numerical consumers wait at warp barriers or stream joins. These sums do not
publish unrelated data; event and allocation atomics retain their own ordering.
"""
@inline function atomic_sum!(address::Core.LLVMPtr{Float64, CUDA.AS.Global}, value::Float64)
    Base.llvmcall(
        """
        %previous = atomicrmw fadd ptr addrspace(1) %0, double %1 syncscope("device") monotonic
        ret void
        """,
        Cvoid,
        Tuple{Core.LLVMPtr{Float64, CUDA.AS.Global}, Float64},
        address,
        value,
    )
    return nothing
end

@inline function receive_ions!(data, row, context, ::Val{Publish}) where {Publish}
    node = context.nodes[row]
    node == 0 && return nothing
    for (field, column, slot, state) in context.ion_events
        ion_row = context.ion_rows[row, slot]
        if Publish
            state && (context.ions[ion_row, column] = data[row, field])
        else
            data[row, field] = context.ions[ion_row, column]
        end
    end
    return nothing
end

@inline function instance_indices(count, ::Val{false})
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    return index:min(index, count)
end

@inline function instance_indices(count, ::Val{true})
    first_thread = blockIdx().x == 1 && threadIdx().x == 1
    return 1:(first_thread ? count : 0)
end

@inline block_instances(count, ::Val{false}) = blockIdx().x:min(blockIdx().x, count)
@inline block_instances(count, ::Val{true}) = 1:(blockIdx().x == 1 ? count : 0)

function initialize_mechanism_kernel!(
    function_::F,
    data,
    context,
    ions,
    reads,
    writes,
    overrides,
    weights,
) where {F}
    for row in instance_indices(size(data, 1), context.serial)
        load_ions!(row, data, context.ion_rows, ions, reads)
        function_(data, row, context, weights === nothing ? 0.0 : weights[row], 0.0)
        if !isempty(overrides)
            for field in axes(data, 2)
                value = overrides[row, field]
                isnan(value) || (data[row, field] = value)
            end
        end
        store_ions!(row, data, context.ion_rows, ions, writes)
    end
    return nothing
end

@inline function load_ions!(row, data, rows, ions, ::Val{Bindings}) where {Bindings}
    @inbounds for (field, column, slot) in Bindings
        ion_row = rows[row, slot]
        ion_row != 0 && (data[row, field] = ions[ion_row, column])
    end
    return nothing
end

@inline function store_ions!(row, data, rows, ions, ::Val{Bindings}) where {Bindings}
    @inbounds for (field, column, slot) in Bindings
        ion_row = rows[row, slot]
        if ion_row != 0 && column != 4
            ions[ion_row, column] = data[row, field]
        end
    end
    return nothing
end

function mechanism_current_kernel!(
    function_::F,
    data,
    context,
    rhs,
    diagonal,
    current_fields,
    electrode_fields,
    ion_reads,
    ion_writes,
    ions,
    density::Val{Density},
) where {F, Density}
    for row in instance_indices(size(data, 1), context.serial)
        mechanism_current_instance!(
            function_,
            row,
            data,
            context,
            rhs,
            diagonal,
            current_fields,
            electrode_fields,
            ion_reads,
            ion_writes,
            ions,
            density,
            Val(false),
        )
    end
    return nothing
end

@inline function mechanism_current_instance!(
    function_::F,
    row,
    data,
    context,
    rhs,
    diagonal,
    ::Val{CurrentFields},
    ::Val{ElectrodeFields},
    ::Val{Reads},
    ::Val{Writes},
    ions,
    ::Val{Density},
    ::Val{Exclusive},
) where {F, CurrentFields, ElectrodeFields, Reads, Writes, Density, Exclusive}
    node = @inbounds context.nodes[row]
    node == 0 && return nothing
    @inbounds for (field, column, slot) in Reads
        data[row, field] = ions[context.ion_rows[row, slot], column]
    end
    voltage = @inbounds context.voltage[node]
    function_(data, row, context_at(context; voltage_probe = voltage + 0.001), 0.0, 0.0)
    shifted_ions =
        ntuple(index -> @inbounds(data[row, CurrentFields[index]]), Val(length(CurrentFields)))
    shifted = 0.0
    @inbounds for field in CurrentFields
        shifted += data[row, field]
    end
    @inbounds for field in ElectrodeFields
        shifted -= data[row, field]
    end
    function_(data, row, context_at(context; voltage_probe = voltage), 0.0, 0.0)
    value = 0.0
    @inbounds for field in CurrentFields
        value += data[row, field]
    end
    @inbounds for field in ElectrodeFields
        value -= data[row, field]
    end
    scale = Density ? (@inbounds context.area[node]) * 0.01 : 1.0
    if Exclusive
        @inbounds rhs[node] -= scale * value
        @inbounds diagonal[node] += scale * (shifted - value) / 0.001
    else
        atomic_sum!(pointer(rhs, node), -scale * value)
        atomic_sum!(pointer(diagonal, node), scale * (shifted - value) / 0.001)
    end
    @inbounds for (field, column, slot) in Writes
        ion_row = context.ion_rows[row, slot]
        if column == 4
            ion_scale = Density ? 1.0 : 100.0 / context.area[node]
            linear = ion_row + (column - 1) * size(ions, 1)
            value = ion_scale * data[row, field]
            if Exclusive
                ions[linear] += value
            else
                atomic_sum!(pointer(ions, linear), value)
            end
            for component in eachindex(CurrentFields)
                CurrentFields[component] == field || continue
                derivative = ion_scale * (shifted_ions[component] - data[row, field]) / 0.001
                offset = ion_row + column * size(ions, 1)
                if Exclusive
                    ions[offset] += derivative
                else
                    atomic_sum!(pointer(ions, offset), derivative)
                end
            end
        else
            ions[ion_row, column] = data[row, field]
        end
    end
    return nothing
end

@generated function density_current_kernel!(
    operations::T,
    rows,
    nodes,
    rhs,
    diagonal,
    ions,
    time,
) where {T <: Tuple}
    # CPU partitioning gives every row all operations in this membership group.
    body = Expr(:block)
    for index in eachindex(T.parameters)
        push!(
            body.args,
            quote
                row = rows[instance, $index]
                operation = operations[$index]
                mechanism_current_instance!(
                    operation.function_,
                    row,
                    operation.data,
                    context_at(operation.context; t = time),
                    rhs,
                    diagonal,
                    operation.current_fields,
                    operation.electrode_fields,
                    operation.ion_reads,
                    operation.ion_writes,
                    ions,
                    Val(true),
                    Val(true),
                )
            end,
        )
    end
    return quote
        instance = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        instance > length(nodes) && return nothing
        @inbounds $body
        return nothing
    end
end

@generated function density_advance_kernel!(
    operations::T,
    rows,
    nodes,
    ions,
    time,
) where {T <: Tuple}
    body = Expr(:block)
    for index in eachindex(T.parameters)
        push!(
            body.args,
            quote
                row = rows[instance, $index]
                operation = operations[$index]
                load_ions!(
                    row,
                    operation.data,
                    operation.context.ion_rows,
                    ions,
                    operation.ion_reads,
                )
                operation.function_(
                    operation.data,
                    row,
                    context_at(operation.context; t = time),
                    0.0,
                    0.0,
                )
                store_ions!(
                    row,
                    operation.data,
                    operation.context.ion_rows,
                    ions,
                    operation.ion_writes,
                )
            end,
        )
    end
    return quote
        instance = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        instance > length(nodes) && return nothing
        @inbounds $body
        return nothing
    end
end

function advance_mechanism_kernel!(function_::F, data, context, ions, reads, writes) where {F}
    for row in instance_indices(size(data, 1), context.serial)
        load_ions!(row, data, context.ion_rows, ions, reads)
        function_(data, row, context, 0.0, 0.0)
        store_ions!(row, data, context.ion_rows, ions, writes)
    end
    return nothing
end

"""Clear membrane sums and prepare contiguous ion instances before channel currents."""
function prepare_currents_kernel!(
    ions,
    valences,
    species_ids,
    styles,
    celsius,
    rhs,
    diagonal,
    ::Val{Initial} = Val(false),
) where {Initial}
    row = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    mask = Initial ? 0x20 : 0x40
    @inbounds begin
        if row <= length(rhs)
            rhs[row] = 0.0
            diagonal[row] = 0.0
        end
        if row <= size(ions, 1)
            ions[row, 4] = 0.0
            ions[row, 5] = 0.0
            if styles[row] & mask != 0x00
                species = species_ids[row]
                ions[row, 1] =
                    nernst_potential(ions[row, 2], ions[row, 3], valences[species], celsius)
            end
        end
    end
    return nothing
end

# Warp-scheduled DHS cable solve.

function assemble_cable_kernel!(
    diagonal,
    rhs,
    voltage,
    capacitance,
    parent,
    axial,
    axial_diagonal,
    dt,
)
    node = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    node > length(voltage) && return nothing
    @inbounds begin
        # Mechanisms already accumulated -I and dI/dV into the system.
        # Each diagonal has one writer; shared axial RHS additions stay atomic.
        diagonal[node] = capacitance[node] / dt + diagonal[node] + axial_diagonal[node]
        ancestor = parent[node]
        if ancestor != 0
            current = axial[node] * (voltage[ancestor] - voltage[node])
            atomic_sum!(pointer(rhs, node), current)
            atomic_sum!(pointer(rhs, ancestor), -current)
        end
    end
    return nothing
end

"""Apply the solved voltage increment and NEURON's optional ion-current correction."""
function update_voltage_kernel!(
    voltage,
    increments,
    ions,
    offsets,
    ::Val{SecondOrder},
) where {SecondOrder}
    node = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    node > length(voltage) && return nothing
    @inbounds begin
        delta = increments[node]
        if SecondOrder == 2
            for row in offsets[node]:(offsets[node + 1] - 1)
                ions[row, 4] += ions[row, 5] * delta
            end
        end
        voltage[node] += SecondOrder == 0 ? delta : 2.0 * delta
    end
    return nothing
end

"""Warp-local DHS elimination/back substitution over coalesced numerical storage.

Each child atomically eliminates into its parent, following NEURON-G. Active
round limits and numerical operands are cached in lane-local values, with four
rounds explicitly unrolled. Warp barriers preserve dependencies on NVIDIA GPUs
with independent thread scheduling, including lanes that finish early.
"""
function solve_dhs_kernel!(
    diagonal,
    rhs,
    parent,
    axial,
    warp_offsets,
    round_offsets,
    lane_rounds,
    errors,
)
    warp =
        (blockIdx().x - Int32(1)) * (blockDim().x ÷ Int32(32)) +
        (threadIdx().x - Int32(1)) ÷ Int32(32) +
        Int32(1)
    warp >= length(warp_offsets) && return nothing
    lane = CUDA.laneid() - Int32(1)
    @inbounds begin
        first_round = warp_offsets[warp]
        last_round = warp_offsets[warp + Int32(1)] - Int32(1)
        active_last =
            first_round + lane_rounds[(warp - Int32(1)) * Int32(32) + lane + Int32(1)] - Int32(1)
        round = first_round
        while round <= last_round
            Base.Cartesian.@nexprs 4 iteration -> begin
                if round <= last_round
                    if round <= active_last
                        node = round_offsets[round] + lane
                        ancestor = parent[node]
                        coupling, pivot, value = axial[node], diagonal[node], rhs[node]
                        if iszero(pivot) || !isfinite(pivot)
                            CUDA.atomic_max!(pointer(errors, 1), Int32(6))
                        end
                        if ancestor != 0
                            factor = coupling / pivot
                            atomic_sum!(pointer(diagonal, ancestor), -factor * coupling)
                            atomic_sum!(pointer(rhs, ancestor), factor * value)
                        end
                    end
                    CUDA.sync_warp()
                    round += Int32(1)
                end
            end
        end
        round = last_round
        while round >= first_round
            Base.Cartesian.@nexprs 4 iteration -> begin
                if round >= first_round
                    if round <= active_last
                        node = round_offsets[round] + lane
                        ancestor = parent[node]
                        coupling, pivot, value = axial[node], diagonal[node], rhs[node]
                        ancestor != 0 && (value += coupling * rhs[ancestor])
                        rhs[node] = value / pivot
                    end
                    CUDA.sync_warp()
                    round -= Int32(1)
                end
            end
        end
    end
    return nothing
end

# State initialization kernels.

function initialize_ions_kernel!(ions, styles, species_ids, initial_concentrations)
    row = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    row > size(ions, 1) && return nothing
    if styles[row] & 0x04 != 0x00
        species = species_ids[row]
        ions[row, 2] = initial_concentrations[1, species]
        ions[row, 3] = initial_concentrations[2, species]
    end
    return nothing
end

function initialize_concentration_reversals_kernel!(
    ions,
    rows,
    ::Val{Slots},
    species_ids,
    styles,
    valences,
    celsius,
) where {Slots}
    instance = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    instance > size(rows, 1) && return nothing
    for slot in Slots
        row = rows[instance, slot]
        row == 0 && continue
        if styles[row] & 0x20 != 0x00
            ions[row, 1] =
                nernst_potential(ions[row, 2], ions[row, 3], valences[species_ids[row]], celsius)
        end
    end
    return nothing
end

function finite_state_kernel!(values, errors)
    index = Int64(blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    index > length(values) && return nothing
    if !isfinite(values[index])
        # Preserve an earlier solver or queue failure as the primary cause.
        CUDA.atomic_cas!(pointer(errors, 1), Int32(0), Int32(8))
    end
    return nothing
end

function initialize_history_kernel!(history, exposures, provided)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    index > length(exposures) && return nothing
    owner = mod(index - 1, size(exposures, 1)) + 1
    column = fld(index - 1, size(exposures, 1)) + 1
    if !provided[owner, column]
        for slot in axes(history, 3)
            history[owner, column, slot] = exposures[owner, column]
        end
    else
        exposures[owner, column] = history[owner, column, 1]
    end
    return nothing
end

# Grow only beyond the coalesced initial allocation. Each row owns a linked
# sequence of 32, 64, 128, ... slots; published pages never move. A producer claims
# the link before allocation; its warp reconverges before waiting for publication.

@inline function free_device_page!(address::UInt64)
    Base.llvmcall((
        """
declare void @free(ptr)
define void @release(i64 %address) {
    %pointer = inttoptr i64 %address to ptr
    call void @free(ptr %pointer)
    ret void
}
""",
        "release",
    ), Cvoid, Tuple{UInt64}, address)
    return nothing
end

@inline function page_pointer(pages::DevicePageView{T}, row, index, create, errors) where {T}
    link = pointer(pages.heads, row)
    capacity = Int64(32)
    while true
        address = CUDA.atomic_add!(link, UInt64(0))
        if address == 0
            create || return reinterpret(Core.LLVMPtr{T, CUDA.AS.Global}, UInt64(0))
            # Claim the link before allocation: concurrent producers must not
            # allocate and discard duplicate large pages from the device heap.
            previous = CUDA.atomic_cas!(link, UInt64(0), UInt64(1))
            if previous == 0
                allocation = UInt64(CUDA.malloc(Csize_t(8 + capacity * sizeof(T))))
                if allocation != 0
                    next = reinterpret(Core.LLVMPtr{UInt64, CUDA.AS.Global}, allocation)
                    unsafe_store!(next, UInt64(0))
                    CUDA.threadfence()
                else
                    CUDA.atomic_max!(pointer(errors, 1), Int32(3))
                end
                CUDA.atomic_xchg!(link, allocation)
            end
        end
        # Publish before entering this wait, so lanes in the allocating warp
        # reconverge after its owner releases the link.
        address = CUDA.atomic_add!(link, UInt64(0))
        while address == 1
            address = CUDA.atomic_add!(link, UInt64(0))
        end
        address == 0 && return reinterpret(Core.LLVMPtr{T, CUDA.AS.Global}, UInt64(0))
        if index <= capacity
            return reinterpret(
                Core.LLVMPtr{T, CUDA.AS.Global},
                address + UInt64(8 + (index - 1) * sizeof(T)),
            )
        end
        index -= capacity
        capacity *= 2
        link = reinterpret(Core.LLVMPtr{UInt64, CUDA.AS.Global}, address)
    end
end

function free_pages_kernel!(heads)
    row = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    row > length(heads) && return nothing
    address = heads[row]
    while address != 0
        next = unsafe_load(reinterpret(Core.LLVMPtr{UInt64, CUDA.AS.Global}, address))
        free_device_page!(address)
        address = next
    end
    heads[row] = UInt64(0)
    return nothing
end

@inline function spike_time(spikes, source, slot)
    slot <= size(spikes.times, 2) && return spikes.times[source, slot]
    address = page_pointer(spikes.pages, source, slot - size(spikes.times, 2), false, nothing)
    return UInt(address) == 0 ? Inf : unsafe_load(address)
end

@inline function store_spike!(spikes, source, slot, time, errors)
    if slot <= size(spikes.times, 2)
        spikes.times[source, slot] = time
    else
        address = page_pointer(
            spikes.pages,
            source,
            slot - size(spikes.times, 2),
            errors !== nothing,
            errors,
        )
        UInt(address) != 0 && unsafe_store!(address, time)
    end
    return nothing
end
