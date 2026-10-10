# Public selections resolve against the model's own construction layout.
voltage_ref(model::Union{MicroModel, MacroModel}, args...; kwargs...) =
    voltage_ref(model.runtime.layout, args...; kwargs...)
mechanism_ref(model::Union{MicroModel, MacroModel}, args...; kwargs...) =
    mechanism_ref(model.runtime.layout, args...; kwargs...)
ion_ref(model::Union{MicroModel, MacroModel}, args...; kwargs...) =
    ion_ref(model.runtime.layout, args...; kwargs...)
global_ref(model::Union{MicroModel, MacroModel}, args...; kwargs...) =
    global_ref(model.runtime.layout, args...; kwargs...)
roi_ref(model::Union{MicroModel, MacroModel}, args...; kwargs...) =
    roi_ref(model.runtime.layout, args...; kwargs...)

# Section objects use the same layout identities as whole-cell selections.
voltage_ref(plan::ModelLayout, section::CellSection; x = 0.5) =
    voltage_ref(plan, section.cell; section = section.name, x)
mechanism_ref(
    plan::ModelLayout,
    section::CellSection,
    mechanism,
    field;
    x = 0.5,
    index::Integer = 1,
) = mechanism_ref(plan, section.cell, mechanism, field; section = section.name, x, index)
ion_ref(plan::ModelLayout, section::CellSection, field; x = 0.5) =
    ion_ref(plan, section.cell, field; section = section.name, x)
ion_style(plan::ModelLayout, section::CellSection, ion; x = 0.5) =
    ion_style(plan, section.cell, ion; section = section.name, x)

# Resolved compartment and GLOBAL locations.

function plan_location(plan::MicroPlan, cell::Int, section::Symbol, x::Float64)
    return location_node(plan.location_maps[cell], plan.location_starts[cell], section, x)
end

function global_index(plan::ModelLayout, mechanism)
    compiled = mechanism isa CompiledMechanism ? mechanism : nothing
    matches = findall(plan.globals) do group
        compiled === nothing ? group.mechanism.ir.name == Symbol(mechanism) :
        group.mechanism.ir.source_digest == compiled.ir.source_digest
    end
    length(matches) == 1 || throw(ArgumentError("expected exactly one matching GLOBAL mechanism"))
    return only(matches)
end
