# Queue, source and exposure allocation.

function target_source_index(targets::Vector{Int32}, count::Int)
    offsets = ones(Int64, count + 1)
    for target in targets
        target != 0 && (offsets[target + 1] += 1)
    end
    for index in 2:length(offsets)
        offsets[index] += offsets[index - 1] - 1
    end
    sources = Vector{Int32}(undef, offsets[end] - 1)
    cursors = copy(offsets)
    for (source, target) in enumerate(targets)
        target == 0 && continue
        sources[cursors[target]] = source
        cursors[target] += 1
    end
    return offsets, sources
end

device_spikes(sources::Int, capacity::Int) = SpikeBuffer(
    CuArray{Float64}(undef, sources, capacity),
    CUDA.zeros(Int64, sources),
    device_pages(Float64, sources),
)

function device_pages(::Type{T}, rows::Int) where {T}
    heads = CUDA.zeros(UInt64, rows)
    pages = DevicePages{T, typeof(heads)}(heads)
    isempty(heads) && return pages
    context = CUDA.context()
    # Retain the underlying allocation independently of the CuArray finalizer.
    # Cleanup runs asynchronously, after the owner and array may both be dead.
    retained = copy(GPUArrays.storage(heads))
    device_heads = CUDA.cudaconvert(heads)
    finalizer(pages) do _
        errormonitor(@async begin
            CUDA.context!(context) do
                try
                    CUDA.synchronize()
                    launch_kernel!(free_pages_kernel!, rows, device_heads)
                    CUDA.synchronize()
                finally
                    GPUArrays.unsafe_free!(retained)
                end
            end
        end)
    end
    return pages
end

# Fuse only contiguous eligible groups. Splitting a run by node membership is
# safe because density groups have one row per node and no cross-row effects.
# A separate writer remains between the preceding and following fused batches.
function micro_phase_operations(groups, node_count, phase, fuse)
    batches, separate, jobs = NamedTuple[], DeviceMechanism[], PhaseJob[]
    pending = DeviceMechanism[]
    function flush_density!()
        for batch in density_operations(pending, node_count, phase)
            push!(batches, batch)
            push!(jobs, PhaseJob(true, length(batches)))
        end
        empty!(pending)
    end
    for group in groups
        getproperty(group.phases, phase) || continue
        ir = group.mechanism.ir
        effects = ir.blocks[:breakpoint].effects
        eligible =
            fuse &&
            ir.kind == :density &&
            !serial_instances(ir) &&
            !(effects.schedules_events || effects.emits_spikes || effects.discontinuous)
        if eligible
            push!(pending, group)
        else
            flush_density!()
            push!(separate, group)
            push!(jobs, PhaseJob(false, length(separate)))
        end
    end
    flush_density!()
    return batches, separate, jobs
end

# Projection exposure allocation.
function coupling_reset_masks(plan::MacroPlan)
    owners = length(plan.labels)
    masks = zeros(UInt8, owners * length(plan.exposures))
    for edge in plan.edges
        for row in eachindex(edge.targets)
            macro_target = plan.region_group[edge.targets[row]] != 0
            for index in axes(edge.group.exposure_writes, 2)
                exposure = edge.group.exposure_writes[2, index]
                direction = edge.group.exposure_writes[3, index]
                owner = direction == 1 ? edge.sources[row] : edge.targets[row]
                destination = owner + (exposure - 1) * owners
                masks[destination] |= macro_target ? 0x01 : 0x02
            end
        end
    end
    return masks
end

function projection_publication(edge::MacroEdgePlan, plan::MacroPlan)
    contributions = Dict{Int64, Vector{Int64}}()
    owners = length(plan.labels)
    stride = owners * length(plan.exposures)
    count = size(edge.group.data, 1)
    for row in 1:count, binding in axes(edge.group.exposure_writes, 2)
        field = edge.group.exposure_writes[1, binding]
        exposure = edge.group.exposure_writes[2, binding]
        direction = edge.group.exposure_writes[3, binding]
        owner = direction == 1 ? edge.sources[row] : edge.targets[row]
        destination = owner + (exposure - 1) * owners
        plan.region_group[edge.targets[row]] == 0 && (destination += stride)
        push!(get!(Vector{Int64}, contributions, destination), row + (field - 1) * count)
    end
    destinations = sort!(collect(keys(contributions)))
    values, offsets = Int64[], Int64[1]
    for destination in destinations
        append!(values, contributions[destination])
        push!(offsets, length(values) + 1)
    end
    return ProjectionPublication(CuArray(values), CuArray(destinations), CuArray(offsets))
end

# Storage collection and phase dependence.

function Adapt.adapt_storage(collector::PhaseStorageCollector, array::CuArray)
    if !haskey(collector.seen, array)
        collector.seen[array] = nothing
        push!(collector.arrays, array)
    end
    return array
end

function add_access!(access::ResourceAccess, key::ResourceKey, rows)
    isempty(rows) && return nothing
    union!(get!(BitSet, access, key), rows)
    return nothing
end

function add_reference_access!(access, reference::VariableRef)
    key = (reference.kind, reference.group, reference.field)
    add_access!(access, key, (reference.kind == :global ? 1 : reference.row,))
end

function overlapping_access(left::ResourceAccess, right::ResourceAccess)
    for (key, rows) in left
        other = get(right, key, nothing)
        other !== nothing && !isdisjoint(rows, other) && return true
    end
    return false
end

function phase_dependency(first::PhaseAccess, second::PhaseAccess)
    (first.barrier || second.barrier) && return true
    return overlapping_access(first.writes, second.reads) ||
           overlapping_access(first.writes, second.writes) ||
           overlapping_access(first.writes, second.adds) ||
           overlapping_access(first.reads, second.writes) ||
           overlapping_access(first.reads, second.adds) ||
           overlapping_access(first.adds, second.reads) ||
           overlapping_access(first.adds, second.writes)
end

"""Read the resolved field indices carried as constants into device phases."""
binding_fields(::Val{Fields}) where {Fields} = Fields
