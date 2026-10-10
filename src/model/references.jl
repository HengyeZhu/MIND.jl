# Resolve observation and variable identities.

"""A variable selection plus an output label; recording does not declare MOD storage."""
struct RecordSpec
    variable::VariableRef
    name::String
end

RecordSpec(
    variable::VariableRef;
    name = string(variable.kind, '.', variable.group, '.', variable.row, '.', variable.field),
) = RecordSpec(variable, String(name))

# Preserve the readable selector properties used by the recording backend.
@inline function Base.getproperty(selection::RecordSpec, name::Symbol)
    name in (:kind, :group, :row, :field) &&
        return getproperty(getfield(selection, :variable), name)
    return getfield(selection, name)
end

Base.propertynames(::RecordSpec) = (:variable, :name, :kind, :group, :row, :field)

function reference_groups(plan::ModelLayout, kind::Symbol)
    kind == :mechanism && return plan.micro === nothing ? MechanismPlan[] : plan.micro.groups
    kind == :region &&
        return plan.macro_model === nothing ? MechanismPlan[] : plan.macro_model.groups
    kind == :edge && return plan.macro_model === nothing ? MechanismPlan[] :
           getfield.(plan.macro_model.edges, :group)
    kind == :output && return getfield.(plan.outputs, :group)
    kind == :input && return getfield.(plan.inputs, :group)
    throw(ArgumentError("$kind does not select a mechanism group"))
end

function check_field_reference(group::MechanismPlan, selection::VariableRef)
    1 <= selection.row <= size(group.data, 1) && 1 <= selection.field <= size(group.data, 2) ||
        throw(ArgumentError("invalid model variable field or row"))
    info = fieldinfo(group.mechanism.ir, selection.field)
    info.scope == :global &&
        throw(ArgumentError("shared MOD fields must be selected with global_ref"))
    return selection
end

"""Resolve a named field in any compiled mechanism group, including MIND transforms."""
function mechanism_ref(
    plan::ModelLayout,
    kind::Symbol,
    group_index::Integer,
    row::Integer,
    field;
    index::Integer = 1,
)
    groups = reference_groups(plan, kind)
    1 <= group_index <= length(groups) || throw(ArgumentError("invalid mechanism group"))
    group = groups[group_index]
    1 <= row <= size(group.data, 1) || throw(ArgumentError("invalid mechanism row"))
    info = fieldinfo(group.mechanism.ir, Symbol(field))
    1 <= index <= fieldwidth(info) || throw(BoundsError(fieldcolumns(info), index))
    for (name, input) in group.mechanism.ir.inputs
        name == info.name || continue
        if input == VoltageInput
            node = group.nodes[row]
            node > 0 || throw(ArgumentError("this mechanism has no cable voltage location"))
            return VariableRef(plan.identity, :voltage, 0, node, 0)
        end
        throw(ArgumentError("$name is an execution input, not persistent mechanism storage"))
    end
    info.scope == :global && return global_ref(plan, group.mechanism, field; index)
    return VariableRef(plan.identity, kind, group_index, row, info.offset + index - 1)
end

"""Resolve a REGION field, whether or not the MOD exposes it for coupling."""
function mechanism_ref(plan::ModelLayout, region::ROI, field; index::Integer = 1)
    owner = region_index(plan, region)
    group, row = plan.macro_model.region_group[owner], plan.macro_model.region_row[owner]
    group > 0 || throw(ArgumentError("ROI has no REGION mechanism"))
    return mechanism_ref(plan, :region, group, row, field; index)
end

"""Resolve a field of an explicitly created MICRO2MACRO instance."""
function mechanism_ref(plan::ModelLayout, output::Micro2Macro, field; index::Integer = 1)
    macro_plan = plan.macro_model
    macro_plan !== nothing && macro_plan.model_id == UInt64(objectid(output.model)) ||
        throw(ArgumentError("transform belongs to another model"))
    row = output.index
    for (group_index, transform) in enumerate(plan.outputs)
        count = size(transform.group.data, 1)
        row <= count && return mechanism_ref(plan, :output, group_index, row, field; index)
        row -= count
    end
    throw(ArgumentError("transform was created after this plan was built"))
end

function voltage_ref(plan::ModelLayout, cell::Cell; section = :soma, x = 0.5)
    micro = plan.micro
    micro !== nothing && micro.model_id == cell.model.id ||
        throw(ArgumentError("cell belongs to another model"))
    node = plan_location(micro, cell.index, Symbol(section), Float64(x))
    return VariableRef(plan.identity, :voltage, 0, node, 0)
end

function mechanism_ref(plan::ModelLayout, point::PointProcess, field; index::Integer = 1)
    micro = plan.micro
    micro !== nothing && micro.model_id == point.model.id ||
        throw(ArgumentError("point process belongs to another model"))
    group, row = micro.point_group[point.index], micro.point_row[point.index]
    return mechanism_ref(plan, :mechanism, group, row, field; index)
end

function mechanism_ref(
    plan::ModelLayout,
    cell::Cell,
    mechanism,
    field;
    section = :soma,
    x = 0.5,
    index::Integer = 1,
)
    micro = plan.micro
    micro !== nothing && micro.model_id == cell.model.id ||
        throw(ArgumentError("cell belongs to another model"))
    node = range_node(micro.location_maps[cell.index], section, x)
    group_index, row = 0, 0
    for (candidate, group) in enumerate(micro.groups)
        ir = group.mechanism.ir
        ir.kind == :density && (
            mechanism isa CompiledMechanism ? ir.source_digest == mechanism.ir.source_digest :
            ir.name == Symbol(mechanism)
        ) || continue
        # Independent density rows follow sorted cable-node order. Stateful
        # serial groups retain logical order and require a linear location search.
        position = if serial_instances(ir)
            findfirst(==(node), group.nodes)
        else
            found = searchsortedfirst(group.nodes, node)
            found <= length(group.nodes) && group.nodes[found] == node ? found : nothing
        end
        position === nothing && continue
        group_index == 0 ||
            throw(ArgumentError("location must contain exactly one matching density mechanism"))
        group_index, row = candidate, position
    end
    group_index != 0 ||
        throw(ArgumentError("location must contain exactly one matching density mechanism"))
    return mechanism_ref(plan, :mechanism, group_index, row, field; index)
end

"""Select a shared MOD field, independent of any particular mechanism instance."""
function global_ref(plan::ModelLayout, mechanism, field; index::Integer = 1)
    group = global_index(plan, mechanism)
    info = fieldinfo(plan.globals[group].mechanism.ir, Symbol(field))
    info.scope == :global || throw(ArgumentError("$field is not a shared GLOBAL"))
    1 <= index <= fieldwidth(info) || throw(BoundsError(fieldcolumns(info), index))
    return VariableRef(plan.identity, :global, group, 0, info.offset + index - 1)
end

"""Select an ion concentration, reversal or current, such as `:cai`, `:eca` or `:ica`."""
function ion_ref(plan::ModelLayout, cell::Cell, field; section = :soma, x = 0.5)
    micro = plan.micro
    micro !== nothing && micro.model_id == cell.model.id ||
        throw(ArgumentError("cell belongs to another model"))
    column, species = 0, :none
    for ion in micro.ion_names
        for (component, candidate) in enumerate((
            Symbol(:e, ion),
            Symbol(ion, :i),
            Symbol(ion, :o),
            Symbol(:i, ion),
            Symbol(:di, ion, :_dv_),
        ))
            Symbol(field) == candidate && ((column, species) = (component, ion))
        end
    end
    column > 0 || throw(ArgumentError("unknown ion field $field"))
    node = range_node(micro.location_maps[cell.index], section, x)
    row = get(micro.ion_locations, (node, species), 0)
    row != 0 || throw(ArgumentError("ion $species is not present at this location"))
    return VariableRef(plan.identity, :ion, 0, row, column)
end

function region_index(plan::ModelLayout, region)
    macro_plan = plan.macro_model
    macro_plan === nothing && throw(ArgumentError("model has no ROIs"))
    region isa ROI &&
        UInt64(objectid(region.model)) != macro_plan.model_id &&
        throw(ArgumentError("ROI belongs to another model"))
    owner =
        region isa ROI ? region.index :
        region isa Integer ? Int(region) : findfirst(==(String(region)), macro_plan.labels)
    owner !== nothing && 1 <= owner <= length(macro_plan.labels) ||
        throw(ArgumentError("unknown ROI $region"))
    return owner
end

function roi_ref(plan::ModelLayout, region, exposure)
    owner = region_index(plan, region)
    macro_plan = plan.macro_model
    field = findfirst(==(Symbol(exposure)), macro_plan.exposures)
    field !== nothing || throw(ArgumentError("unknown exposure $exposure"))
    return VariableRef(plan.identity, :exposure, 0, owner, field)
end

function validate_reference(plan::ModelLayout, selection::VariableRef)
    selection.layout_id == plan.identity || throw(
        ArgumentError(
            "variable reference belongs to another model layout; resolve it again after build",
        ),
    )
    if selection.kind == :voltage
        plan.micro !== nothing &&
        1 <= selection.row <= length(plan.micro.voltage) &&
        selection.group == 0 &&
        selection.field == 0 || throw(ArgumentError("invalid voltage variable reference"))
    elseif selection.kind == :ion
        plan.micro !== nothing &&
        selection.group == 0 &&
        1 <= selection.row <= size(plan.micro.ion_data, 1) &&
        1 <= selection.field <= size(plan.micro.ion_data, 2) ||
            throw(ArgumentError("invalid ion variable reference"))
    elseif selection.kind == :mechanism
        plan.micro !== nothing && 1 <= selection.group <= length(plan.micro.groups) ||
            throw(ArgumentError("invalid mechanism variable group"))
        check_field_reference(plan.micro.groups[selection.group], selection)
    elseif selection.kind in (:region, :edge, :output, :input)
        groups = reference_groups(plan, selection.kind)
        1 <= selection.group <= length(groups) ||
            throw(ArgumentError("invalid model variable group"))
        group = groups[selection.group]
        check_field_reference(group, selection)
    elseif selection.kind == :global
        1 <= selection.group <= length(plan.globals) && selection.row == 0 ||
            throw(ArgumentError("invalid GLOBAL selection"))
        ir = plan.globals[selection.group].mechanism.ir
        1 <= selection.field <= fieldcount(ir) && fieldinfo(ir, selection.field).scope == :global ||
            throw(ArgumentError("invalid GLOBAL field"))
    elseif selection.kind == :exposure
        macro_plan = plan.macro_model
        macro_plan !== nothing &&
        selection.group == 0 &&
        1 <= selection.row <= length(macro_plan.labels) &&
        1 <= selection.field <= length(macro_plan.exposures) ||
            throw(ArgumentError("invalid ROI variable reference"))
    else
        throw(ArgumentError("unknown variable kind $(selection.kind)"))
    end
    return selection
end

function validate_writable(plan::ModelLayout, selection::VariableRef)
    validate_reference(plan, selection)
    ir = if selection.kind in (:mechanism, :region, :edge, :output, :input)
        reference_groups(plan, selection.kind)[selection.group].mechanism.ir
    elseif selection.kind == :global
        plan.globals[selection.group].mechanism.ir
    else
        return selection
    end
    fieldinfo(ir, selection.field).category == :constant &&
        throw(ArgumentError("CONSTANT fields cannot be changed"))
    return selection
end
"""Inspect the resolved ion style at a cell location in a built plan."""
function ion_style(plan::ModelLayout, cell::Cell, ion; section = :soma, x = 0.5)
    reference = ion_ref(plan, cell, Symbol(:e, ion); section, x)
    return decode_ion_style(plan.micro.ion_styles[reference.row])
end

# Link managed model-variable aliases.

const MECHANISM_REFERENCE_KINDS = (:mechanism, :region, :edge, :output, :input)

function reference_field(plan::ModelLayout, ref::VariableRef)
    ir =
        ref.kind == :global ? plan.globals[ref.group].mechanism.ir :
        ref.kind in MECHANISM_REFERENCE_KINDS ?
        reference_groups(plan, ref.kind)[ref.group].mechanism.ir : return nothing
    return fieldinfo(ir, ref.field)
end

function alias_target(ref::VariableRef, targets::Dict{VariableRef, VariableRef})
    visited = Set{VariableRef}()
    while haskey(targets, ref)
        ref in visited && throw(ArgumentError("cyclic POINTER binding"))
        push!(visited, ref)
        ref = targets[ref]
    end
    return ref
end

"""Link every declared POINTER. No GPU allocations or raw addresses enter this pass."""
function link_aliases(plan::ModelLayout, bindings)
    targets = Dict{VariableRef, VariableRef}()
    for (source, target) in bindings
        source isa VariableRef && target isa VariableRef ||
            throw(ArgumentError("POINTER bindings must pair two VariableRef values"))
        validate_reference(plan, source)
        validate_reference(plan, target)
        field = reference_field(plan, source)
        field !== nothing && field.scope == :pointer ||
            throw(ArgumentError("binding source is not a declared POINTER"))
        haskey(targets, source) && throw(ArgumentError("duplicate POINTER binding"))
        targets[source] = target
    end
    for kind in MECHANISM_REFERENCE_KINDS, (index, group) in enumerate(reference_groups(plan, kind))
        ir = group.mechanism.ir
        fields = filter(field -> field.scope == :pointer, ir.fields)
        isempty(fields) && continue
        for field in fields
            write = any(block -> field.name in block.effects.writes, values(ir.blocks))
            for column in fieldcolumns(field), row in axes(group.data, 1)
                source = VariableRef(plan.identity, kind, index, row, column)
                haskey(targets, source) || throw(
                    ModError(
                        field.location,
                        "unbound POINTER $(field.name), $kind group $index row $row",
                    ),
                )
                target = alias_target(source, targets)
                target_field = reference_field(plan, target)
                target_field !== nothing &&
                    target_field.scope == :pointer &&
                    throw(ArgumentError("POINTER chain ends at an unbound alias"))
                write && validate_writable(plan, target)
                targets[source] = target
            end
        end
    end
    return targets
end
