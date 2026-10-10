# CPU event priority queues and batched mechanism execution.

"""Collect compact GPU feedback, then schedule self-events and source fanout on the CPU."""
function flush_spikes!(state::SimulationState)
    schedule = state.work[:event_schedule]
    count = length(state.spikes.counts)
    if count != 0
        record_spike_batch!(state, state.spikes)
        launch_kernel!(collect_spikes_kernel!, count, state.spikes, state.outbox, state.errors)
    end
    # Read one aggregate count, never every target's queue or numerical state.
    download_event_feedback!(schedule.count, state.outbox.count, 1)
    received = Int(schedule.count[1])
    received == 0 && return nothing
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

    # 1. Publish explicit/pattern source spikes and expand their outgoing edges once.
    empty!(pending)
    while !isempty(schedule.external) && first(schedule.external).time <= cutoff
        push!(pending, pop_event!(schedule.external))
    end
    plan = state.layout.micro
    if plan !== nothing
        cursor = schedule.pattern_cursor
        while cursor <= length(plan.pattern_times) && plan.pattern_times[cursor] <= cutoff
            target = plan.pattern_targets[cursor]
            for offset in
                schedule.target_source_offsets[target]:(schedule.target_source_offsets[target + 1] - 1)

                source = schedule.target_sources[offset]
                push!(pending, ScheduledEvent(plan.pattern_times[cursor], 0.0, 0, 0, -source))
            end
            cursor += 1
        end
        schedule.pattern_cursor = cursor
    end
    if !isempty(pending)
        sort!(
            pending;
            lt = (a, b) -> a.target < b.target || (a.target == b.target && a.time < b.time),
            alg = MergeSort,
        )
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
        for event in pending
            route_spike!(state, -Int(event.target), event.time)
        end
    end

    # 2. Follow acyclic target levels; shared effects/cycles use one ordered heap.
    for heap in schedule.heaps
        while !isempty(heap) && first(heap).time <= cutoff
            empty!(pending)
            empty!(schedule.postponed)
            feedback = false
            while !isempty(heap) && first(heap).time <= cutoff
                event = pop_event!(heap)
                group = schedule.target_groups[event.target]
                needs_feedback = schedule.feedback[group]
                if !schedule.ordered && needs_feedback && schedule.marked[event.target]
                    push!(schedule.postponed, event)
                    continue
                end
                push!(pending, event)
                schedule.marked[event.target] = true
                feedback |= needs_feedback
                # A new self-event/spike can precede the next queued event.
                schedule.ordered && needs_feedback && break
            end
            for event in pending
                schedule.marked[event.target] = false
            end
            for event in schedule.postponed
                push_event!(heap, event)
            end

            # 3. Only independent targets reorder physically. Each target keeps
            # chronological events; effectful receivers execute one event before feedback.
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
    end
    return nothing
end
