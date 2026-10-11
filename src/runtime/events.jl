# CPU event priority queue and batched mechanism execution.

"""Collect compact GPU feedback, then schedule self-events and source fanout on the CPU."""
function flush_spikes!(state::SimulationState; ordered::Bool = false)
    schedule = state.work[:event_schedule]
    # Read one aggregate count, never every target's queue or numerical state.
    download_event_feedback!(schedule.count, state.outbox.count, 1)
    received = Int(schedule.count[1])
    received == 0 && return nothing
    # Every internally generated spike also publishes an outbox request.
    # External sources are recorded and cleared directly in deliver_events!.
    if !isempty(state.spikes.counts)
        record_spike_batch!(state, state.spikes)
        clear_event_counts!(state.spikes.counts)
    end
    if received > length(schedule.requests)
        capacity = max(received, 2 * length(schedule.requests))
        schedule.requests = CUDA.pin(Vector{ScheduledEvent}(undef, capacity))
        schedule.collected = CuArray{ScheduledEvent}(undef, capacity)
    end
    if received <= length(state.outbox.values)
        download_event_feedback!(schedule.requests, state.outbox.values, received)
    else
        launch_kernel!(
            gather_event_requests_kernel!,
            received,
            schedule.collected,
            state.outbox,
            received,
        )
        download_event_feedback!(schedule.requests, schedule.collected, received)
    end
    clear_event_counts!(state.outbox.count)
    if !ordered && received > 1
        sort!(
            view(schedule.requests, 1:received);
            by = event -> (
                event.target > 0 ? schedule.target_ranks[event.target] :
                event.target < 0 ? schedule.source_ranks[-event.target] : 0,
                event.order,
            ),
            alg = MergeSort,
        )
    end
    @inbounds for index in 1:received
        event = schedule.requests[index]
        if event.target > 0
            enqueue_event!(schedule, event)
        elseif event.target < 0
            route_spike!(state, -Int(event.target), event.time)
        end
    end
    return nothing
end

"""Deliver due events with the same scalar priorities and compiled NET_RECEIVE semantics."""
function deliver_events!(state::SimulationState, cutoff::Float64)
    schedule = state.work[:event_schedule]
    pending = schedule.pending

    heap = schedule.heap
    while !isempty(heap) && first(heap).time <= cutoff
        empty!(pending)
        # 1. Explicit and pattern sources share the same queue as target events.
        # Fanout may put a receiver ahead of the next source, including at equal
        # times. Publish only the source events reached in this causal order.
        if first(heap).target < 0
            while !isempty(heap) && first(heap).time <= cutoff && first(heap).target < 0
                event = pop_event!(heap)
                push!(pending, event)
                route_spike!(state, -Int(event.target), event.time)
            end
            sort!(pending; by = event -> (event.target, event.time), alg = MergeSort)
            upload_events!(schedule, length(pending))
            launch_kernel!(
                publish_external_spikes_kernel!,
                length(pending),
                schedule.events,
                length(pending),
                state.spikes,
                state.macro_spikes,
                state.errors,
            )
            record_spike_batch!(state, state.spikes)
            clear_event_counts!(state.spikes.counts)
            continue
        end

        # 2. Batch ordinary receivers up to the next source or feedback event.
        # An effectful receiver runs alone, so its new events enter the queue
        # before the next delivery is selected, even through cycles or fan-in.
        feedback = false
        while !isempty(heap) && first(heap).time <= cutoff && first(heap).target > 0
            event = first(heap)
            needs_feedback = schedule.feedback[schedule.target_groups[event.target]]
            needs_feedback && !isempty(pending) && break
            push!(pending, pop_event!(heap))
            feedback = needs_feedback
            feedback && break
        end

        # 3. Stable grouping changes only independent target order.
        if schedule.ordered
            upload_events!(schedule, length(pending))
            launch_kernel!(
                receive_ordered_kernel!,
                1,
                state.work[:event_payload],
                schedule.events,
                length(pending),
            )
        else
            # The heap already established time/order, including equal signed
            # zeros. Stable grouping changes only independent target order.
            sort!(
                pending;
                by = event ->
                    (schedule.target_groups[event.target], schedule.target_rows[event.target]),
                alg = MergeSort,
            )
            empty!(schedule.offsets)
            empty!(schedule.rows)
            empty!(schedule.groups)
            previous_target, previous_group, first_row = Int32(0), 0, 1
            for (index, event) in enumerate(pending)
                event.target == previous_target && continue
                group = Int(schedule.target_groups[event.target])
                if group != previous_group
                    previous_group != 0 && push!(
                        schedule.groups,
                        (previous_group, first_row, length(schedule.rows) - first_row + 1),
                    )
                    previous_group, first_row = group, length(schedule.rows) + 1
                end
                push!(schedule.offsets, index)
                push!(schedule.rows, schedule.target_rows[event.target])
                previous_target = event.target
            end
            push!(schedule.offsets, length(pending) + 1)
            push!(
                schedule.groups,
                (previous_group, first_row, length(schedule.rows) - first_row + 1),
            )
            upload_events!(schedule, length(pending))
            if length(schedule.offsets) > length(schedule.device_offsets)
                capacity = max(length(schedule.offsets), 2 * length(schedule.device_offsets))
                schedule.device_offsets = CuArray{Int64}(undef, capacity)
                schedule.device_rows = CuArray{Int32}(undef, capacity)
            end
            copyto!(schedule.device_offsets, 1, schedule.offsets, 1, length(schedule.offsets))
            copyto!(schedule.device_rows, 1, schedule.rows, 1, length(schedule.rows))
            for (group_index, first, count) in schedule.groups
                group = state.work[:event_groups][group_index]
                receive_events!(group, state, first, count)
            end
        end

        # 4. Merge generated requests before choosing another event. For
        # ordinary synapses no output transfer or counter read is needed.
        if feedback
            flush_spikes!(state)
        end
    end
    return nothing
end
