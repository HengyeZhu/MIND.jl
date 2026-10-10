# Bounded continuous recordings.

# Selection resolves once on the CPU; sampling kernels use only integer indices.
voltage_record(
    model::Union{MicroModel, MacroModel},
    cell::Cell;
    name = "v_$(cell.gid)",
    kwargs...,
) = RecordSpec(voltage_ref(model, cell; kwargs...), String(name))
mechanism_record(
    model::Union{MicroModel, MacroModel},
    point::PointProcess,
    field;
    name = string(field),
    kwargs...,
) = RecordSpec(mechanism_ref(model, point, field; kwargs...), String(name))
mechanism_record(
    model::Union{MicroModel, MacroModel},
    cell::Cell,
    mechanism,
    field;
    name = "$(cell.gid).$field",
    kwargs...,
) = RecordSpec(mechanism_ref(model, cell, mechanism, field; kwargs...), String(name))
mechanism_record(
    model::Union{MicroModel, MacroModel},
    region::ROI,
    field;
    name = string(field),
    kwargs...,
) = RecordSpec(mechanism_ref(model, region, field; kwargs...), String(name))
global_record(
    model::Union{MicroModel, MacroModel},
    mechanism,
    field;
    name = string(field),
    kwargs...,
) = RecordSpec(global_ref(model, mechanism, field; kwargs...), String(name))
ion_record(
    model::Union{MicroModel, MacroModel},
    cell::Cell,
    field;
    name = string(field),
    kwargs...,
) = RecordSpec(ion_ref(model, cell, field; kwargs...), String(name))
roi_record(model::Union{MicroModel, MacroModel}, region, exposure; name = string(exposure)) =
    RecordSpec(roi_ref(model, region, exposure), String(name))

voltage_record(
    model::Union{MicroModel, MacroModel},
    section::CellSection;
    name = "$(section.cell.gid).$(section.name).v",
    kwargs...,
) = RecordSpec(voltage_ref(model, section; kwargs...), String(name))
mechanism_record(
    model::Union{MicroModel, MacroModel},
    section::CellSection,
    mechanism,
    field;
    name = "$(section.cell.gid).$(section.name).$field",
    kwargs...,
) = RecordSpec(mechanism_ref(model, section, mechanism, field; kwargs...), String(name))
ion_record(
    model::Union{MicroModel, MacroModel},
    section::CellSection,
    field;
    name = string(field),
    kwargs...,
) = RecordSpec(ion_ref(model, section, field; kwargs...), String(name))
mechanism_record(
    model::Union{MicroModel, MacroModel},
    output::Micro2Macro,
    field;
    name = string(field),
    kwargs...,
) = RecordSpec(mechanism_ref(model, output, field; kwargs...), String(name))

struct SimulationResult
    time::Vector{Float64}
    traces::Dict{String, Vector{Float64}}
    macro_time::Vector{Float64}
    exposures::Array{Float64, 3}
    labels::Vector{String}
    exposure_names::Vector{Symbol}
    spike_times::Vector{Float64}
    spike_sources::Vector{Int64}
end

function recording_buffer(dimensions::NTuple{N, Int}, samples::Int) where {N}
    capacity = min(samples, dimensions[end])
    width = prod(dimensions[1:(end - 1)])
    shape = (dimensions[1:(end - 1)]..., capacity)
    return RecordingBuffer(CUDA.zeros(Float64, shape), zeros(Float64, dimensions), capacity, width)
end

function finish_recording_sample!(buffer::RecordingBuffer, sample::Int)
    buffer.width == 0 && return nothing
    chunk, offset = divrem(sample - 1, buffer.capacity)
    count = offset + 1
    count == buffer.capacity ||
        sample == size(buffer.values, ndims(buffer.values)) ||
        return nothing
    elements = buffer.width * count
    first_element = chunk * buffer.capacity * buffer.width + 1
    # Complete the host copy before the next chunk reuses this device buffer.
    copyto!(buffer.values, first_element, buffer.device, 1, elements)
    return nothing
end

function allocate_recording(
    state::SimulationState,
    selections::Vector{RecordSpec},
    micro_steps,
    macro_steps;
    macro_outputs = nothing,
    macro_rois = nothing,
    record_spikes = true,
    recording_buffer_samples::Int = 4096,
    recording_buffer_spikes::Int = 1048576,
)
    recording_buffer_samples > 0 ||
        throw(ArgumentError("recording buffer samples must be positive"))
    foreach(selection -> validate_reference(state.layout, selection.variable), selections)
    selections = [
        RecordSpec(resolved_reference(state, selection.variable), selection.name) for
        selection in selections
    ]
    names = getfield.(selections, :name)
    length(unique(names)) == length(names) || throw(ArgumentError("recording names must be unique"))
    # Resolve all numerical selections through the same variable-location API.
    # State owns these allocations; device descriptors gather every category in
    # one kernel instead of launching separately for each mechanism and GLOBAL.
    descriptor_type = typeof(CUDA.cudaconvert(state.voltage))
    banks = descriptor_type[]
    indices = IdDict{Any, Int32}()
    bank_ids, offsets = Int32[], Int64[]
    for spec in selections
        spec.kind == :exposure && continue
        data = reference_storage(state, spec.variable)
        bank = get!(indices, data) do
            push!(banks, CUDA.cudaconvert(vec(data)))
            Int32(length(banks))
        end
        push!(bank_ids, bank)
        push!(offsets, reference_offset(data, spec.variable))
    end
    macro_plan = state.layout.macro_model
    nr, ne = size(state.exposures)
    region_indices =
        macro_rois === nothing ? collect(1:nr) :
        [region_index(state.layout, region) for region in macro_rois]
    append!(
        region_indices,
        [selection.row for selection in selections if selection.kind == :exposure],
    )
    unique!(region_indices)
    exposure_indices = macro_outputs === nothing ? collect(1:ne) : Int[]
    if macro_outputs !== nothing
        for name in macro_outputs
            index =
                macro_plan === nothing ? nothing : findfirst(==(Symbol(name)), macro_plan.exposures)
            index === nothing && throw(ArgumentError("unknown recorded exposure $name"))
            push!(exposure_indices, index)
        end
    end
    append!(
        exposure_indices,
        [selection.field for selection in selections if selection.kind == :exposure],
    )
    unique!(exposure_indices)
    real_sources =
        record_spikes ?
        length(state.spikes.counts) -
        sum(input -> size(input.group.data, 1), state.layout.inputs; init = 0) : 0
    spikes = spike_recording_buffer(state, real_sources, recording_buffer_spikes)
    return DeviceRecording(
        selections,
        recording_buffer(
            (length(offsets), (state.layout.micro === nothing ? macro_steps : micro_steps) + 1),
            recording_buffer_samples,
        ),
        recording_buffer(
            (length(region_indices), length(exposure_indices), macro_steps + 1),
            recording_buffer_samples,
        ),
        spikes,
        real_sources,
        CuArray(Int32.(exposure_indices)),
        exposure_indices,
        CuArray(Int32.(region_indices)),
        region_indices,
        (; banks = CuArray(banks), bank_ids = CuArray(bank_ids), offsets = CuArray(offsets)),
        state.step_micro,
        state.step_macro,
        Vector{Float64}(undef, macro_steps + 1),
    )
end

function record_values_kernel!(output, gather, sample)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    index > length(gather.offsets) && return nothing
    @inbounds output[index, sample] = gather.banks[gather.bank_ids[index]][gather.offsets[index]]
    return nothing
end

function record_exposures_kernel!(output, exposures, owners, fields, sample)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    count = length(owners) * length(fields)
    index > count && return nothing
    owner = mod(index - 1, length(owners)) + 1
    column = fld(index - 1, length(owners)) + 1
    output[index + (sample - 1) * count] = exposures[owners[owner], fields[column]]
    return nothing
end

function record_micro!(state::SimulationState)
    recording = state.recording
    recording === nothing && return nothing
    sample =
        state.layout.micro === nothing ? state.step_macro - recording.start_macro_step + 1 :
        state.step_micro - recording.start_micro_step + 1
    output = recording.micro_records.device
    column = mod(sample - 1, recording.micro_records.capacity) + 1
    launch_kernel!(
        record_values_kernel!,
        length(recording.gather.offsets),
        output,
        recording.gather,
        column,
    )
    finish_recording_sample!(recording.micro_records, sample)
    return nothing
end

function record_macro!(state::SimulationState)
    recording = state.recording
    recording === nothing && return nothing
    record_micro!(state)
    sample = state.step_macro - recording.start_macro_step + 1
    recording.macro_times[sample] = macro_time(state)
    output = recording.macro_records.device
    column = mod(sample - 1, recording.macro_records.capacity) + 1
    launch_kernel!(
        record_exposures_kernel!,
        length(recording.exposure_owners) * length(recording.exposure_fields),
        output,
        state.exposures,
        recording.exposure_owners,
        recording.exposure_fields,
        column,
    )
    finish_recording_sample!(recording.macro_records, sample)
    return nothing
end

function collect_result(state::SimulationState)
    recording = state.recording
    micro_values = recording.micro_records.values
    macro_values = recording.macro_records.values
    start_micro, stop_micro = recording.start_micro_step, state.step_micro
    time =
        state.layout.micro === nothing ? copy(recording.macro_times) :
        [micro_time(state, step) for step in start_micro:stop_micro]
    macro_times = state.layout.macro_model === nothing ? Float64[] : recording.macro_times
    traces = Dict{String, Vector{Float64}}()
    micro_row = 0
    for spec in recording.selections
        if spec.kind == :exposure
            field = findfirst(==(spec.field), recording.exposure_indices)
            owner = findfirst(==(spec.row), recording.exposure_regions)
            traces[spec.name] = vec(macro_values[owner, field, :])
        else
            micro_row += 1
            traces[spec.name] = vec(micro_values[micro_row, :])
        end
    end
    spike_times, indices = collect_spikes!(recording.spikes)
    ids = state.layout.micro === nothing ? Int64[] : state.layout.micro.source_ids[indices]
    order = sortperm(
        eachindex(spike_times);
        by = index -> (spike_times[index], ids[index]),
        alg = MergeSort,
    )
    macro_plan = state.layout.macro_model
    return SimulationResult(
        time,
        traces,
        macro_times,
        macro_values,
        macro_plan === nothing ? String[] : macro_plan.labels[recording.exposure_regions],
        macro_plan === nothing ? Symbol[] : macro_plan.exposures[recording.exposure_indices],
        spike_times[order],
        ids[order],
    )
end

# Chunked spike recordings. The initial size is a hint, not a simulation limit.

function spike_recording_buffer(state, sources, chunk_capacity)
    chunk_capacity > 0 || throw(ArgumentError("spike recording buffer must be positive"))
    variables =
        state.layout.micro === nothing ? Union{Nothing, VariableRef}[] :
        state.layout.micro.source_variables
    targets = state.layout.micro === nothing ? Int32[] : state.layout.micro.source_targets
    # This estimate schedules ordinary host flushes. A larger event burst grows
    # device pages automatically, without polling a counter after every step.
    bound = sum(
        (
            variables[source] !== nothing && targets[source] == 0 ? 1 : size(state.spikes.times, 2) for source in 1:sources
        );
        init = 0,
    )
    capacity = sources == 0 ? 1 : chunk_capacity
    return SpikeRecordingBuffer(
        SpikeRecording(
            CuArray{Float64}(undef, capacity),
            CuArray{Int32}(undef, capacity),
            CUDA.zeros(Int64, 1),
            device_pages(RecordedSpike, 1),
        ),
        Float64[],
        Int32[],
        capacity,
        min(capacity, bound),
        0,
    )
end

function flush_spike_recording!(buffer::SpikeRecordingBuffer)
    buffer.reserved == 0 && return nothing
    count = Int(only(Array(buffer.device.count)))
    offset = length(buffer.times)
    resize!(buffer.times, offset + count)
    resize!(buffer.sources, offset + count)
    if count <= buffer.capacity
        copyto!(buffer.times, offset + 1, buffer.device.times, 1, count)
        copyto!(buffer.sources, offset + 1, buffer.device.sources, 1, count)
    else
        # Only overflow needs gathering: CUDA's device-heap pages cannot be
        # passed to a host memcpy. Gather them into normal CUDA-owned arrays.
        times, sources = CuArray{Float64}(undef, count), CuArray{Int32}(undef, count)
        launch_kernel!(gather_recorded_spikes_kernel!, count, times, sources, buffer.device)
        copyto!(buffer.times, offset + 1, times, 1, count)
        copyto!(buffer.sources, offset + 1, sources, 1, count)
    end
    clear_event_counts!(buffer.device.count)
    buffer.reserved = 0
    return nothing
end

function record_spike_batch!(state, spikes)
    recording = state.recording
    recording === nothing && return nothing
    sources = recording.spike_source_count
    sources == 0 && return nothing
    buffer = recording.spikes
    if buffer.reserved + buffer.flush_bound > buffer.capacity
        flush_spike_recording!(buffer)
    end
    buffer.reserved = min(buffer.capacity, buffer.reserved + max(1, buffer.flush_bound))
    launch_kernel!(record_spikes_kernel!, sources, buffer.device, spikes, state.errors, sources)
    return nothing
end

function collect_spikes!(buffer::SpikeRecordingBuffer)
    flush_spike_recording!(buffer)
    return buffer.times, buffer.sources
end

function record_spikes_kernel!(recording, spikes, errors, sources)
    source = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    source > sources && return nothing
    count = spikes.counts[source]
    count == 0 && return nothing
    offset = CUDA.atomic_add!(pointer(recording.count, 1), Int64(count))
    for index in 1:count
        slot = offset + index
        time = spike_time(spikes, source, index)
        if slot <= length(recording.times)
            recording.times[slot] = time
            recording.sources[slot] = source
        else
            address = page_pointer(recording.pages, 1, slot - length(recording.times), true, errors)
            UInt(address) != 0 && unsafe_store!(address, RecordedSpike(time, Int32(source)))
        end
    end
    return nothing
end

function gather_recorded_spikes_kernel!(times, sources, recording)
    slot = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    slot > length(times) && return nothing
    if slot <= length(recording.times)
        times[slot], sources[slot] = recording.times[slot], recording.sources[slot]
    else
        address = page_pointer(recording.pages, 1, slot - length(recording.times), false, nothing)
        spike = unsafe_load(address)
        times[slot], sources[slot] = spike.time, spike.source
    end
    return nothing
end
