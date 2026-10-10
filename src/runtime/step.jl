# Run and dispatch simulation steps.

"""Run to an absolute stop time with CPU event scheduling and compiled GPU dynamics."""
function run!(
    state::SimulationState,
    stop_time::Real;
    records = RecordSpec[],
    macro_outputs = nothing,
    macro_rois = nothing,
    record_spikes = true,
    recording_buffer_samples::Integer = 4096,
    recording_buffer_spikes::Integer = 1048576,
)
    # 1. Resolve the absolute stop time against the fixed simulation timestep.
    state.initialized || throw(ArgumentError("simulation is not initialized"))
    macro_mode = state.layout.macro_model !== nothing
    dt = macro_mode ? state.configuration.dt_macro : state.configuration.dt_micro
    stop = Float64(stop_time)
    current_time = macro_mode ? macro_time(state) : micro_time(state)
    duration = stop - current_time
    isfinite(duration) || throw(ArgumentError("stop time must be finite"))
    count = round(Int, duration / dt)
    # Subtracting absolute times also carries their FP64 representation error.
    # The tolerance covers that subtraction, including an already-reached stop.
    tolerance = 64.0 * eps(max(abs(stop), abs(current_time), dt))
    count >= 0 && abs(duration - count * dt) <= tolerance ||
        throw(ArgumentError("stop time must be an integer number of steps after the current time"))
    macro_steps = macro_mode ? count : 0
    micro_steps =
        state.layout.micro === nothing ? 0 :
        macro_mode ? count * round(Int, dt / state.configuration.dt_micro) : count
    # 2. Allocate recording chunks; spike bursts can grow beyond their initial size.
    state.recording = allocate_recording(
        state,
        RecordSpec[records...],
        micro_steps,
        macro_steps;
        macro_outputs,
        macro_rois,
        record_spikes,
        recording_buffer_samples = Int(recording_buffer_samples),
        recording_buffer_spikes = Int(recording_buffer_spikes),
    )
    # 3. Record the starting boundary, advance to the stop time, then collect results.
    try
        own_phase_storage!(state, CUDA.stream())
        if state.step_micro == 0
            record_spike_batch!(state, state.macro_spikes)
        end
        if macro_mode
            record_macro!(state)
        else
            record_micro!(state)
        end
        for _ in 1:count
            if macro_mode
                advance_macro!(state)
            else
                advance_micro!(state)
            end
        end
        # State/ROI equations may emit on the final step. Preserve their
        # recording and future arrivals even though no following step collects them.
        state.layout.micro === nothing || flush_spikes!(state)
        validate_device_state!(state)
        return collect_result(state)
    catch
        state.initialized = false
        CUDA.synchronize()
        rethrow()
    finally
        state.recording = nothing
    end
end

"""Advance one macro step, or one micro step in a micro-only simulation."""
function step!(state::SimulationState)
    state.initialized || throw(ArgumentError("simulation is not initialized"))
    own_phase_storage!(state, CUDA.stream())
    if state.layout.macro_model === nothing
        advance_micro!(state)
    else
        advance_macro!(state)
    end
    return state
end

# Micro time step.

"""Advance events, membrane voltage, mechanism state and recordings by one micro step."""
function advance_micro!(state::SimulationState; record_sample::Bool = true)
    # 1. Detect from the completed boundary, collect generated requests, then
    # deliver through the midpoint before evaluating membrane currents.
    time = micro_time(state)
    dt = state.configuration.dt_micro
    detect_spikes!(state, time)
    flush_spikes!(state)
    deliver_events!(state, time + 0.5 * dt)
    micro_currents!(state, time + 0.5 * dt)
    # 2. Assemble and solve the cable system with warp DHS.
    secondorder = state.configuration.secondorder
    secondorder != 0 && copyto!(state.work[:cable_workspace].previous_voltage, state.voltage)
    launch_kernel!(
        assemble_cable_kernel!,
        length(state.voltage),
        state.diagonal,
        state.rhs,
        state.voltage,
        state.capacitance,
        state.current,
        state.conductance,
        state.work[:axial_diagonal],
        secondorder == 0 ? dt : 0.5 * dt,
    )
    if state.layout.micro !== nothing
        cable = state.work[:cable]
        warps = length(cable.warp_offsets) - 1
        if warps > 0
            CUDA.@cuda always_inline=true threads=128 blocks=cld(warps, 4) solve_dhs_kernel!(
                state.voltage,
                state.diagonal,
                state.rhs,
                state.parent,
                state.axial,
                cable.warp_offsets,
                cable.round_offsets,
                cable.lane_rounds,
                state.errors,
            )
        end
        secondorder != 0 && launch_kernel!(
            second_order_voltage_kernel!,
            length(state.voltage),
            state.voltage,
            state.work[:cable_workspace].previous_voltage,
            state.ions,
            state.work[:ion_offsets],
            Val(secondorder == 2),
        )
    end
    # 3. Advance channel states and scheduled post-voltage procedures.
    execute_phase!(state, state.work[:advance_schedule], Val(:advance), time + dt)
    # 4. Commit and record the completed state. Its next threshold check follows
    # any macro-boundary writes, at the beginning of the next micro step.
    state.step_micro += 1
    record_sample && record_micro!(state)
    state.layout.macro_model === nothing && clear_event_counts!(state.macro_spikes.counts)
    return nothing
end

# Cross-scale time step.

"""Advance coupling, micro dynamics, ROI equations and history by one macro step."""
function advance_macro!(state::SimulationState)
    start_time = macro_time(state)
    stop_time = macro_time(state, state.step_macro + 1)
    dt = state.configuration.dt_macro

    # 1. Read delayed coupling from the boundary snapshot. Initial macro-owned
    # coupling was already evaluated for the time-zero record.
    macro_coupling!(state; targets = state.step_macro == 0 ? :micro : :all)
    if state.layout.micro !== nothing
        # 2. Transform ROI inputs into device events for this exchange interval.
        for (index, group) in enumerate(state.input_groups)
            trigger_micro_input!(group, state, index, start_time, stop_time)
        end
        # 3. Solve each micro step. Defer the shared boundary sample until the
        # ROI equations and writes through aliases have also completed.
        count = round(Int, dt / state.configuration.dt_micro)
        for index in 1:count
            advance_micro!(state; record_sample = index != count)
        end

        # 4. Integrate output transforms in spike-time order, publish their sums,
        # and retain only spikes needed by the next exchange interval.
        for (index, group) in enumerate(state.output_groups)
            advance_micro_output!(group, state, index, start_time, stop_time)
        end
        publish_micro_outputs!(state)
        launch_kernel!(
            compact_spikes_kernel!,
            length(state.macro_spikes.counts),
            state.macro_spikes,
            stop_time,
        )
    end

    # 5. Advance ROI equations and publish the complete boundary state.
    execute_phase!(state, state.work[:region_schedule], Val(:region), start_time)
    state.step_macro += 1
    launch_kernel!(
        publish_outputs_kernel!,
        length(state.exposures),
        state.exposures,
        state.work[:output_publication],
        state.history,
        state.step_macro,
    )
    record_macro!(state)
    return nothing
end

# Micro currents, event delivery and stream execution.

function micro_currents!(state::SimulationState, time::Float64)
    if state.layout.micro !== nothing
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
        )
        execute_phase!(state, state.work[:current_schedule], Val(:current), time)
    end
    return nothing
end

function execute_phase!(state, schedule::PhaseSchedule, phase::Val, time)
    primary = CUDA.stream()
    if length(schedule.streams) == 1
        for job in schedule.jobs
            launch_phase_job!(state, job, phase, time, primary)
        end
        return nothing
    end
    # All jobs see preceding clears, ion updates and voltage solves.
    # Only conflicting jobs wait for each other. The solver stream joins the
    # tails before subsequent phases can observe the shared numerical arrays.
    CUDA.record(schedule.ready, primary)
    for stream in schedule.streams
        CUDA.wait(schedule.ready, stream)
    end
    try
        for (index, job) in enumerate(schedule.jobs)
            slot = schedule.stream_indices[index]
            stream = schedule.streams[slot]
            for predecessor in schedule.predecessors[index]
                schedule.stream_indices[predecessor] == slot && continue
                CUDA.wait(schedule.completed[predecessor], stream)
            end
            launch_phase_job!(state, job, phase, time, stream)
            completed = schedule.completed[index]
            completed === nothing || CUDA.record(completed, stream)
        end
    catch
        # A host compilation/launch failure must not leave streams using state
        # after the failed run releases its owners.
        foreach(CUDA.synchronize, schedule.streams)
        rethrow()
    end
    for tail in schedule.tails
        CUDA.wait(schedule.completed[tail], primary)
    end
    return nothing
end

# ROI, projection and transform execution.

function macro_coupling!(state::SimulationState; targets = :all)
    # Snapshot and reset each exposure at its own address. Separate macro and
    # micro-input mask bits preserve WRITE_SOURCE ownership without copy/clear races.
    isempty(state.edge_groups) && isempty(state.input_groups) && return nothing
    launch_kernel!(
        prepare_coupling_kernel!,
        length(state.exposures),
        state.exposures,
        state.input_exposures,
        state.exposure_snapshot,
        state.history,
        state.work[:coupling_reset_masks],
        state.step_macro,
        targets != :micro,
        targets != :macro,
        !isempty(state.input_groups),
    )
    isempty(state.edge_groups) && return nothing
    execute_phase!(
        state,
        state.work[:edge_schedule],
        Val(:edge),
        (; time = macro_time(state), targets),
    )
    return nothing
end

# Concrete group arguments preserve specialized generated CUDA functions.
function advance_region!(group::DeviceMechanism, state, time::Float64, stream)
    count = size(group.data, 1)
    launch_scheduled_kernel!(
        advance_region_kernel!,
        group.context.serial isa Val{true} ? min(1, count) : count,
        group.mechanism.advance!,
        group.data,
        context_at(group.phases.context; t = time),
        group.owners,
        state.exposures,
        group.exposure_reads,
        group.exposure_writes;
        stream,
    )
    return nothing
end

function advance_macro_edges!(group::DeviceMechanism, state, index, arguments, stream)
    # Literal NamedTuple keys keep row storage concrete. A computed Symbol
    # would widen this lookup to all unrelated work fields and lose inference.
    rows = if arguments.targets == :all
        nothing
    elseif arguments.targets == :macro
        state.work.edge_macro_rows[index]
    else
        state.work.edge_micro_rows[index]
    end
    count = rows === nothing ? size(group.data, 1) : length(rows)
    launch_scheduled_kernel!(
        macro_edge_kernel!,
        group.context.serial isa Val{true} ? min(1, count) : count,
        group.mechanism.advance!,
        group.data,
        context_at(group.phases.context; t = arguments.time),
        state.work[:edge_sources][index],
        state.work[:edge_targets][index],
        state.work[:edge_weights][index],
        state.work[:edge_delays][index],
        state.history,
        state.exposure_snapshot,
        state.exposures,
        state.input_exposures,
        state.work[:edge_macro_targets][index],
        group.exposure_reads,
        group.exposure_writes,
        state.step_macro,
        rows,
        Val(group.phases.deferred_projection);
        stream,
    )
    if group.phases.deferred_projection && count != 0
        publication = state.work[:edge_publication][index]
        launch_scheduled_kernel!(
            publish_projection_kernel!,
            32 * length(publication.destinations),
            state.exposures,
            state.input_exposures,
            group.data,
            publication,
            arguments.targets != :micro,
            arguments.targets != :macro;
            stream,
        )
    end
    return nothing
end

function trigger_micro_input!(group::DeviceMechanism, state, index, start_time, stop_time)
    count = size(group.data, 1)
    context = context_at(group.context; t = start_time, start_time, stop_time)
    launch_kernel!(
        input_trigger_kernel!,
        group.context.serial isa Val{true} ? min(1, count) : count,
        group.mechanism.receive!,
        group.data,
        context,
        group.owners,
        state.layout.seed,
        state.work[:input_stream_ids][index],
        state.step_macro,
        state.input_exposures,
        group.exposure_reads,
    )
    return nothing
end

function advance_micro_output!(group::DeviceMechanism, state, index, start_time, stop_time)
    count = size(group.data, 1)
    count == 0 && return nothing
    call = CUDA.KernelCall(
        output_transform_kernel!,
        group.mechanism.receive!,
        group.mechanism.advance!,
        group.data,
        group.context,
        state.work[:output_source_offsets][index],
        state.work[:output_sources][index],
        state.work[:output_merge][index],
        state.macro_spikes,
        start_time,
        stop_time,
    )
    kernel = CUDA.kernel_compile(call; always_inline = true)
    threads = min(128, kernel_configuration(kernel.fun).threads)
    blocks = group.context.serial isa Val{true} ? 1 : count
    CUDA.kernel_launch(kernel, call; threads, blocks)
    return nothing
end
