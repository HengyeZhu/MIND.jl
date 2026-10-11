# Specialized launches and generated phase adapters.

# Shared launch policy and concrete-type boundaries for generated kernels.
# Keep forwarded kernel argument tuples specialized, including their arity.
function launch_kernel!(
    function_::F,
    count::Integer,
    arguments::Vararg{Any, N};
    stream::CUDA.CuStream = CUDA.stream(),
) where {F, N}
    count == 0 && return nothing
    call = CUDA.KernelCall(function_, arguments...)
    kernel = CUDA.kernel_compile(call; always_inline = true)
    launch_compiled_kernel!(kernel, call, count, stream)
    return nothing
end

function launch_compiled_kernel!(kernel, call, count, stream)
    # Four warps per block leave room for concurrent mechanism kernels.
    # Scalar ordered phases still launch exactly their requested worker count.
    threads = min(count, 128)
    CUDA.kernel_launch(kernel, call; threads, blocks = cld(count, threads), stream)
    return nothing
end

"""Launch on an explicitly ordered phase stream using retained device descriptors.

The phase DAG supplies CUDA events and the SimulationState owns all allocations.
Passing CuArray arguments directly would make CUDA.jl transfer array ownership
between streams and synchronize even disjoint rows or shared read-only inputs.
Normal launches keep CUDA.jl's automatic ownership management.
"""
function launch_scheduled_kernel!(
    function_::F,
    count::Integer,
    arguments::Vararg{Any, N};
    stream::CUDA.CuStream,
) where {F, N}
    count == 0 && return nothing
    GC.@preserve arguments begin
        device_arguments = CUDA.cudaconvert(arguments)
        launch_kernel!(function_, count, device_arguments...; stream)
    end
    return nothing
end

function mechanism_currents!(
    group::DeviceMechanism,
    state::SimulationState,
    time::Float64;
    stream = CUDA.stream(),
)
    ir = group.mechanism.ir
    count = size(group.data, 1)
    launch_scheduled_kernel!(
        mechanism_current_kernel!,
        group.context.serial isa Val{true} ? min(1, count) : count,
        group.mechanism.current!,
        group.data,
        context_at(group.phases.context; t = time),
        state.rhs,
        state.diagonal,
        group.phases.current_fields,
        group.phases.electrode_fields,
        group.ion_reads,
        group.ion_writes,
        state.ions,
        Val(ir.kind == :density);
        stream,
    )
    return nothing
end

function advance_mechanism!(
    group::DeviceMechanism,
    state::SimulationState,
    time::Float64;
    stream = CUDA.stream(),
)
    count = size(group.data, 1)
    launch_scheduled_kernel!(
        advance_mechanism_kernel!,
        group.context.serial isa Val{true} ? min(1, count) : count,
        group.mechanism.advance!,
        group.data,
        context_at(group.phases.context; t = time),
        state.ions,
        group.ion_reads,
        group.ion_writes;
        stream,
    )
    return nothing
end

function launch_phase_job!(state, job::PhaseJob, ::Val{:current}, time, stream)
    if job.density
        density = state.work[:density_current][job.index]
        launch_scheduled_kernel!(
            density_current_kernel!,
            length(density.nodes),
            density.operations,
            density.rows,
            density.nodes,
            state.rhs,
            state.diagonal,
            state.ions,
            time;
            stream,
        )
    else
        mechanism_currents!(state.work[:separate_current][job.index], state, time; stream)
    end
    return nothing
end

function launch_phase_job!(state, job::PhaseJob, ::Val{:advance}, time, stream)
    if job.density
        density = state.work[:density_advance][job.index]
        launch_scheduled_kernel!(
            density_advance_kernel!,
            length(density.nodes),
            density.operations,
            density.rows,
            density.nodes,
            state.ions,
            time;
            stream,
        )
    else
        advance_mechanism!(state.work[:separate_advance][job.index], state, time; stream)
    end
    return nothing
end

function launch_phase_job!(state, job::PhaseJob, ::Val{:region}, time, stream)
    return advance_region!(state.region_groups[job.index], state, time, stream)
end
function launch_phase_job!(state, job::PhaseJob, ::Val{:edge}, arguments, stream)
    index = job.index
    return advance_macro_edges!(state.edge_groups[index], state, index, arguments, stream)
end

function detect_spikes!(state, time)
    launch_kernel!(
        detect_spikes_kernel!,
        length(state.source_above),
        state.work[:detectors],
        state.source_above,
        state.spikes,
        state.macro_spikes,
        state.outbox,
        state.errors,
        time + 1.0e-10,
    )
    return nothing
end

# Publish cross-scale output fields.

function publish_micro_outputs!(state::SimulationState)
    publication = state.work[:output_publication]
    publication === nothing && return nothing
    launch_kernel!(publish_outputs_kernel!, length(state.exposures), state.exposures, publication)
    return nothing
end

# Explicit observation/update and numerical failure checks.
function validate_device_state!(state::SimulationState)
    for values in (state.voltage, state.ions, state.exposures)
        launch_kernel!(finite_state_kernel!, length(values), values, state.errors)
    end
    for values in state.work[:globals]
        launch_kernel!(finite_state_kernel!, length(values), values, state.errors)
    end
    for groups in (
        state.groups,
        state.region_groups,
        state.edge_groups,
        state.output_groups,
        state.input_groups,
    )
        for group in groups
            launch_kernel!(finite_state_kernel!, length(group.data), group.data, state.errors)
        end
    end
    CUDA.synchronize()
    code = only(Array(state.errors))
    state.work[:phase_owner][] = nothing
    code == 0 && return nothing
    messages = Dict(
        3 => "CUDA device allocator ran out of memory while growing an event or spike buffer",
        5 => "invalid self-event delay or target",
        6 => "singular or non-finite cable system",
        8 => "non-finite numerical state",
    )
    throw(ErrorException(get(messages, Int(code), "CUDA runtime error $code")))
end

# Stream ownership at execution boundaries.

function own_phase_storage!(state, primary::CUDA.CuStream)
    state.work[:phase_owner][] == primary && return nothing
    # Public pointer conversion moves each allocation's managed owner to the
    # primary stream. Its event joins then also protect deallocation. This is
    # needed only when a state moves between Julia tasks/streams.
    for array in state.work[:phase_storage]
        pointer(array)
    end
    state.work[:phase_owner][] = primary
    return nothing
end
# The ownership lookup ends here; the time loop specializes on concrete device storage.
run!(model::Union{MicroModel, MacroModel}, stop_time::Real; kwargs...) =
    run!(model.runtime.state, stop_time; kwargs...)
step!(model::Union{MicroModel, MacroModel}) = (step!(model.runtime.state); model)
