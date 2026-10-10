# Construction validation.

function integer_ratio(numerator::Float64, denominator::Float64, name::AbstractString)
    ratio = numerator / denominator
    isfinite(ratio) && ratio > 0.0 ||
        throw(ArgumentError("$name must be positive and at least one base step"))
    rounded = round(Int, ratio)
    rounded >= 1 || throw(ArgumentError("$name must be at least one base step"))
    isapprox(ratio, rounded; rtol = 0.0, atol = 64.0 * eps(ratio)) ||
        throw(ArgumentError("$name must be an integer multiple of the base step"))
    return rounded
end

function validate_micro_region(region::ROI)
    region.model.micro !== nothing ||
        throw(ArgumentError("cross-scale binding requires a micro model"))
    haskey(region.model.micro_exposures, region.index) ||
        throw(ArgumentError("call use_micro! before cross-scale binding"))
    return nothing
end

# Contiguous mechanism rows and final column-major storage.

"""Temporary CPU row storage; final mechanism data are column-major matrices.

Micro instances use preallocated final column-major storage; physical_rows maps
logical evaluation order into that storage. Their identity arrays already follow
the physical order. Other groups append contiguous rows and are transposed once
when materialized. Only explicit INITIAL overrides need rows.
"""
mutable struct MechanismRows
    mechanism::CompiledMechanism
    ir::MechanismIR
    defaults::Vector{Float64}
    values::Vector{Float64}
    overrides::Dict{Int, Vector{Float64}}
    nodes::Vector{Int32}
    targets::Vector{Int32}
    owners::Vector{Int32}
    logical_ids::Vector{UInt64}
    physical_rows::Union{UnitRange{Int}, Vector{Int}}
end

function MechanismRows(mechanism; capacity::Int = 0)
    initial = defaults(mechanism.ir)
    return MechanismRows(
        mechanism,
        mechanism.ir,
        initial,
        sizehint!(Float64[], capacity * length(initial)),
        Dict{Int, Vector{Float64}}(),
        sizehint!(Int32[], capacity),
        sizehint!(Int32[], capacity),
        sizehint!(Int32[], capacity),
        sizehint!(UInt64[], capacity),
        Int[],
    )
end

function add_mechanism_row!(
    group::MechanismRows;
    node = 0,
    target = 0,
    owner = 0,
    logical_id = 0,
    params = NamedTuple(),
    initial_state = NamedTuple(),
    location = nothing,
)
    isempty(group.physical_rows) ||
        throw(ArgumentError("cannot append to a mechanism group with a finalized row layout"))
    # Append defaults first; parameter callbacks then run once for this logical row.
    row = length(group.nodes) + 1
    append!(group.values, group.defaults)
    apply_parameters!(mechanism_row(group, row), group.ir, params; location)
    if !isempty(initial_state)
        overrides = fill(NaN, length(group.defaults))
        apply_parameters!(overrides, group.ir, initial_state; location)
        group.overrides[row] = overrides
    end
    push!(group.nodes, Int32(node))
    push!(group.targets, Int32(target))
    push!(group.owners, Int32(owner))
    push!(group.logical_ids, UInt64(logical_id))
    return row
end

function finish_group(
    group::MechanismRows;
    ion_locations = Dict{Tuple{Int32, Symbol}, Int32}(),
    exposures = Symbol[],
)
    # 1. Materialize the final physical order in column-major field storage.
    width = length(group.defaults)
    count = length(group.nodes)
    data = if isempty(group.physical_rows)
        permutedims(reshape(group.values, width, count))
    else
        # Micro values already occupy their final physical rows and columns.
        reshape(group.values, count, width)
    end
    # Most density groups have no explicit INITIAL override. Retain the field
    # dimension for GLOBAL validation without materializing an unused matrix.
    overrides = fill(NaN, isempty(group.overrides) ? 0 : count, width)
    if !isempty(group.overrides)
        for (original, values) in group.overrides
            row = isempty(group.physical_rows) ? original : group.physical_rows[original]
            copyto!(view(overrides, row, :), values)
        end
    end
    # 2. Resolve ion transfers and exposure bindings into integer field indices.
    ion_reads, ion_writes = Vector{Int32}[], Vector{Int32}[]
    ion_rows = zeros(Int32, count, length(group.ir.ions))
    for (slot, ion) in enumerate(group.ir.ions)
        for (row, node) in enumerate(group.nodes)
            node != 0 && (ion_rows[row, slot] = ion_locations[(node, ion.name)])
        end
        # NMODL loads written concentrations even without an explicit READ.
        reads = unique(
            [
                ion.reads;
                filter(name -> name in (Symbol(ion.name, :i), Symbol(ion.name, :o)), ion.writes);
            ],
        )
        for (names, destination) in ((reads, ion_reads), (ion.writes, ion_writes))
            for name in names
                suffix =
                    name == Symbol(:e, ion.name) ? 1 :
                    name == Symbol(ion.name, :i) ? 2 :
                    name == Symbol(ion.name, :o) ? 3 :
                    name == Symbol(:i, ion.name) ? 4 :
                    throw(ArgumentError("unknown ion field $name"))
                push!(destination, Int32[fieldindex(group.ir, name), suffix, slot])
            end
        end
    end
    reads, writes = Vector{Int32}[], Vector{Int32}[]
    for binding in group.ir.bindings
        offset = findfirst(==(binding.exposure), exposures)
        offset === nothing && throw(ArgumentError("unknown exposure $(binding.exposure)"))
        direction = binding.direction in (:read_source, :write_source) ? 1 : 2
        destination = binding.direction in (:read_source, :read_target) ? reads : writes
        push!(destination, Int32[fieldindex(group.ir, binding.variable), offset, direction])
    end
    for name in group.ir.exposures
        offset = findfirst(==(name), exposures)
        offset === nothing && throw(ArgumentError("unknown exposure $name"))
        push!(writes, Int32[fieldindex(group.ir, name), offset, 2])
        push!(reads, Int32[fieldindex(group.ir, name), offset, 2])
    end
    pack(rows, width) = isempty(rows) ? zeros(Int32, width, 0) : hcat(rows...)
    return MechanismPlan(
        group.mechanism,
        data,
        overrides,
        group.nodes,
        group.targets,
        group.owners,
        group.logical_ids,
        ion_rows,
        pack(ion_reads, 3),
        pack(ion_writes, 3),
        pack(reads, 3),
        pack(writes, 3),
    )
end

function mechanism_row(group::MechanismRows, row::Integer)
    width = length(group.defaults)
    if !isempty(group.physical_rows)
        count = length(group.nodes)
        first = group.physical_rows[row]
        return view(group.values, first:count:(first + (width - 1) * count))
    end
    return view(group.values, ((row - 1) * width + 1):(row * width))
end
