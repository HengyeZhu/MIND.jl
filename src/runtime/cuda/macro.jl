# ROI execution and exposure transfers.

function read_exposures_kernel!(data, owners, exposures, ::Val{Bindings}) where {Bindings}
    row = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    row > size(data, 1) && return nothing
    owner = owners[row]
    @inbounds for (field, exposure) in Bindings
        data[row, field] = exposures[owner, exposure]
    end
    return nothing
end

function write_exposures_kernel!(data, owners, exposures, ::Val{Bindings}) where {Bindings}
    row = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    row > size(data, 1) && return nothing
    owner = owners[row]
    @inbounds for (field, exposure) in Bindings
        exposures[owner, exposure] = data[row, field]
    end
    return nothing
end

"""Fuse one ROI's exposure exchange and equation evaluation when rows are independent."""
function advance_region_kernel!(
    advance!::F,
    data,
    context,
    owners,
    exposures,
    ::Val{Reads},
    ::Val{Writes},
) where {F, Reads, Writes}
    for row in instance_indices(size(data, 1), context.serial)
        owner = owners[row]
        @inbounds for (field, exposure) in Reads
            data[row, field] = exposures[owner, exposure]
        end
        advance!(data, row, context, 0.0, 0.0)
        @inbounds for (field, exposure) in Writes
            exposures[owner, exposure] = data[row, field]
        end
    end
    return nothing
end

"""Snapshot the boundary and reset owned coupling fields in one device pass."""
function prepare_coupling_kernel!(
    exposures,
    inputs,
    snapshot,
    history,
    masks,
    step,
    macro_active,
    micro_active,
    copy_inputs,
)
    row = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    row > length(exposures) && return nothing
    @inbounds begin
        mask = masks[row]
        if macro_active
            isempty(snapshot) || (snapshot[row] = exposures[row])
            mask & 0x01 != 0x00 && (exposures[row] = 0.0)
        end
        if micro_active
            if mask & 0x02 != 0x00
                inputs[row] = 0.0
            elseif copy_inputs
                slot = mod(step, size(history, 3))
                inputs[row] = history[row + slot * length(exposures)]
            end
        end
    end
    return nothing
end

function macro_edge_kernel!(
    function_::F,
    data,
    context,
    sources,
    targets,
    weights,
    delays,
    history,
    snapshot,
    exposures,
    inputs,
    macro_targets,
    ::Val{Reads},
    ::Val{Writes},
    step::Int64,
    rows,
    ::Val{Deferred},
) where {F, Reads, Writes, Deferred}
    for index in instance_indices(rows === nothing ? size(data, 1) : length(rows), context.serial)
        row = rows === nothing ? index : rows[index]
        @inbounds for (field, exposure, direction) in Reads
            if direction == 1
                slot = mod(step - delays[row], size(history, 3)) + 1
                data[row, field] = history[sources[row], exposure, slot]
            else
                data[row, field] =
                    macro_targets[row] ? snapshot[targets[row], exposure] :
                    history[targets[row], exposure, mod(step, size(history, 3)) + 1]
            end
        end
        @inbounds for (field, _, _) in Writes
            data[row, field] = 0.0
        end
        function_(data, row, context, weights[row], 0.0)
        if !Deferred
            destination = macro_targets[row] ? exposures : inputs
            @inbounds for (field, exposure, direction) in Writes
                owner = direction == 1 ? sources[row] : targets[row]
                offset = owner + (exposure - 1) * size(destination, 1)
                CUDA.atomic_add!(pointer(destination, offset), data[row, field])
            end
        end
    end
    return nothing
end

"""Cooperatively reduce each projection destination in FP64, then publish once."""
function publish_projection_kernel!(
    exposures,
    inputs,
    data,
    publication,
    macro_active,
    micro_active,
)
    thread = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    row = thread ÷ Int32(32) + Int32(1)
    lane = CUDA.laneid()
    row > length(publication.destinations) && return nothing
    @inbounds begin
        destination = publication.destinations[row]
        is_macro = destination <= length(exposures)
        (is_macro ? macro_active : micro_active) || return nothing
        value = 0.0
        for index in (publication.offsets[row] + lane - 1):32:(publication.offsets[row + 1] - 1)
            value += data[publication.value_offsets[index]]
        end
        for shift in (16, 8, 4, 2, 1)
            value += CUDA.shfl_down_sync(CUDA.FULL_MASK, value, shift)
        end
        buffer = is_macro ? exposures : inputs
        offset = is_macro ? destination : destination - length(exposures)
        lane == 1 && CUDA.atomic_add!(pointer(buffer, offset), value)
    end
    return nothing
end

# Projection and cross-scale exchange kernels.

function output_transform_kernel!(
    receive!::F,
    advance!::G,
    data,
    context,
    offsets,
    sources,
    merge,
    spikes,
    start_time,
    stop_time,
) where {F, G}
    lane = threadIdx().x
    # Build a tournament once per window, then update only the consumed source's
    # path. Source selection costs O(S + E log S), without rescanning S sources
    # for each of E spikes. MOD integration remains ordered within each row.
    @inbounds for row in block_instances(size(data, 1), context.serial)
        first, last = offsets[row], offsets[row + 1] - 1
        count = last - first + 1
        base = 2 * (first - 1)

        # 1. Each leaf caches its source's next spike in this half-open window.
        for index in (first + lane - 1):blockDim().x:last
            source = sources[index]
            cursor = Int64(1)
            while cursor <= spikes.counts[source] && spike_time(spikes, source, cursor) < start_time
                cursor += 1
            end
            time = cursor <= spikes.counts[source] ? spike_time(spikes, source, cursor) : Inf
            merge.cursors[index] = cursor
            merge.times[index] = time < stop_time ? time : Inf
            merge.winners[base + count + index - first] = index
        end
        sync_threads()

        # 2. Build internal levels from leaves to root; ties keep logical source order.
        level = Int64(1)
        while level < count
            level <<= 1
        end
        level >>= 1
        while level > 0
            for node in (level + lane - 1):blockDim().x:min(2 * level - 1, count - 1)
                left, right = merge.winners[base + 2 * node], merge.winners[base + 2 * node + 1]
                a, b = merge.times[left], merge.times[right]
                merge.winners[base + node] = a < b || (a == b && left < right) ? left : right
            end
            sync_threads()
            level >>= 1
        end

        # 3. Advance to each selected spike, receive it, and update its ancestors.
        if lane == 1
            current_time = start_time
            while count > 0
                selected = merge.winners[base + 1]
                next_time = merge.times[selected]
                next_time == Inf && break
                advance!(
                    data,
                    row,
                    context_at(context; t = current_time, dt = next_time - current_time),
                    0.0,
                    0.0,
                )
                receive!(data, row, context_at(context; t = next_time, edge = Int64(0)), 1.0, 0.0)
                source = sources[selected]
                cursor = merge.cursors[selected] + 1
                while cursor <= spikes.counts[source] &&
                      spike_time(spikes, source, cursor) < next_time
                    cursor += 1
                end
                time = cursor <= spikes.counts[source] ? spike_time(spikes, source, cursor) : Inf
                merge.cursors[selected] = cursor
                merge.times[selected] = time < stop_time ? time : Inf
                node = (count + selected - first) ÷ 2
                while node > 0
                    left, right = merge.winners[base + 2 * node], merge.winners[base + 2 * node + 1]
                    a, b = merge.times[left], merge.times[right]
                    merge.winners[base + node] = a < b || (a == b && left < right) ? left : right
                    node ÷= 2
                end
                current_time = next_time
            end
            # 4. Integrate the remaining interval, including an empty source set.
            advance!(
                data,
                row,
                context_at(context; t = current_time, dt = stop_time - current_time),
                0.0,
                0.0,
            )
        end
        sync_threads()
    end
    return nothing
end

function compact_spikes_kernel!(spikes, cutoff)
    source = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    source > length(spikes.counts) && return nothing
    retained = 0
    count = spikes.counts[source]
    for index in 1:count
        time = spike_time(spikes, source, index)
        if time >= cutoff
            retained += 1
            store_spike!(spikes, source, retained, time, nothing)
        end
    end
    spikes.counts[source] = retained
    return nothing
end

function input_trigger_kernel!(
    receive!::F,
    data,
    context,
    owners,
    seed,
    stream_ids,
    step,
    exposures,
    ::Val{Reads},
) where {F, Reads}
    for row in instance_indices(size(data, 1), context.serial)
        stream_id = stream_ids[row]
        @inbounds for (field, exposure) in Reads
            data[row, field] = exposures[owners[row], exposure]
        end
        # Original MIND_Sim keys each input by ROI, input batch and source ID,
        # then resets the Random123 sequence at every macro sample.
        for column in axes(context.rng_counters, 2)
            offset = 3 * (column - 1)
            context.rng_keys[row, offset + 1] = UInt64((seed ⊻ stream_id) % UInt32)
            context.rng_keys[row, offset + 2] = context.logical_ids[row]
            context.rng_keys[row, offset + 3] =
                (seed >> 32) ⊻ (stream_id >> 32) ⊻ UInt64(owners[row] - 1) ⊻ UInt64(column - 1)
            set_random_sequence!(context, row, column, step)
        end
        receive!(data, row, context, 1.0, 0.0)
    end
    return nothing
end

"""Publish owned outputs in stable order and optionally commit the same boundary to history."""
function publish_outputs_kernel!(exposures, publication, history = nothing, step::Int64 = 0)
    row = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    row > length(exposures) && return nothing
    @inbounds begin
        value = exposures[row]
        if publication !== nothing
            first, stop = publication.offsets[row], publication.offsets[row + 1]
            if first < stop
                value = 0.0
                for index in first:(stop - 1)
                    bank = publication.bank_ids[index]
                    value += publication.banks[bank][publication.value_offsets[index]]
                end
                exposures[row] = value
            end
        end
        if history !== nothing
            slot = mod(step, size(history, 3))
            history[row + slot * length(exposures)] = value
        end
    end
    return nothing
end
