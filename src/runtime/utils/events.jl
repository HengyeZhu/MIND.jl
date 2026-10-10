# CPU priority ordering and transfer storage for the event workflow.

# Keep Int64 capacities while using CUDA's asynchronous integer memset. Clearing
# both 32-bit words avoids a generic fill kernel and its launch setup each step.
clear_event_counts!(counts) = fill!(reinterpret(UInt32, counts), UInt32(0))

@inline function event_before(a::ScheduledEvent, b::ScheduledEvent)
    return a.time < b.time ||
           (a.time == b.time && (a.order < b.order || (a.order == b.order && a.target < b.target)))
end

function push_event!(heap::Vector{ScheduledEvent}, event::ScheduledEvent)
    push!(heap, event)
    child = length(heap)
    @inbounds while child > 1
        parent = child >> 1
        event_before(event, heap[parent]) || break
        heap[child] = heap[parent]
        child = parent
    end
    @inbounds heap[child] = event
    return nothing
end

function pop_event!(heap::Vector{ScheduledEvent})
    event = first(heap)
    replacement = pop!(heap)
    count = length(heap)
    count == 0 && return event
    parent = 1
    @inbounds while 2 * parent <= count
        child = 2 * parent
        child < count && event_before(heap[child + 1], heap[child]) && (child += 1)
        event_before(heap[child], replacement) || break
        heap[parent] = heap[child]
        parent = child
    end
    @inbounds heap[parent] = replacement
    return event
end

function enqueue_event!(schedule::EventSchedule, event::ScheduledEvent)
    level = schedule.target_levels[event.target]
    level != 0 && push_event!(schedule.heaps[level], event)
    return nothing
end

function route_spike!(state, source::Int, time::Float64)
    plan = state.layout.micro
    schedule = state.work[:event_schedule]
    @inbounds for edge in plan.fanout_offsets[source]:(plan.fanout_offsets[source + 1] - 1)
        enqueue_event!(
            schedule,
            ScheduledEvent(time + plan.edge_delays[edge], 0.0, edge, edge, plan.edge_targets[edge]),
        )
    end
    return nothing
end

"""Read pinned feedback at a causal boundary, waiting on its CUDA stream directly."""
function download_event_feedback!(host::Vector{T}, device::CuArray{T}, count::Int) where {T}
    # The CPU needs these values before choosing the next events. As in
    # NEURON-G, enqueue the copy and wait directly, without handing each short
    # timestep wait to a Julia worker thread through Array copyto!.
    GC.@preserve host device begin
        unsafe_copyto!(pointer(host), pointer(device), count; async = true)
        CUDA.synchronize(CUDA.stream(); blocking = true)
    end
    return nothing
end

"""Reuse pinned transfer memory, growing geometrically only when a batch exceeds it."""
function upload_events!(schedule::EventSchedule, count::Int)
    # Only wait for the preceding H2D copy before overwriting pinned memory.
    # Device batches remain ordered on the primary stream without a host join.
    CUDA.synchronize(schedule.transferred; blocking = true)
    if count > length(schedule.staging)
        capacity = max(count, 2 * length(schedule.staging))
        schedule.staging = CUDA.pin(Vector{ScheduledEvent}(undef, capacity))
        schedule.events = CuArray{ScheduledEvent}(undef, capacity)
    end
    copyto!(schedule.staging, 1, schedule.pending, 1, count)
    copyto!(schedule.events, 1, schedule.staging, 1, count)
    CUDA.record(schedule.transferred)
    return nothing
end

function reset_events!(schedule::EventSchedule)
    foreach(empty!, schedule.heaps)
    empty!(schedule.external)
    empty!(schedule.pending)
    empty!(schedule.postponed)
    fill!(schedule.marked, false)
    schedule.pattern_cursor = 1
    return nothing
end

# Concrete generated-mechanism boundary for a selected receive batch.
function receive_events!(group::DeviceMechanism, state, first, count)
    schedule = state.work[:event_schedule]
    launch_kernel!(
        receive_batch_kernel!,
        count,
        group.mechanism.receive!,
        group.data,
        group.context,
        schedule.events,
        schedule.device_offsets,
        schedule.device_rows,
        first,
        count,
    )
    return nothing
end
