# Resolve the model's sole state before entering concrete runtime operations.
ion_style(model::Union{MicroModel, MacroModel}, args...; kwargs...) =
    ion_style(model.runtime.layout, args...; kwargs...)
simulation_time(model::Union{MicroModel, MacroModel}) = simulation_time(model.runtime.state)
runtime_configuration(model::Union{MicroModel, MacroModel}) =
    runtime_configuration(model.runtime.state)
read_value(model::Union{MicroModel, MacroModel}, selection) =
    read_value(model.runtime.state, selection)
connection_weights(model::Union{MicroModel, MacroModel}, connection::Integer) =
    connection_weights(model.runtime.state, connection)

function inject_event!(model::Union{MicroModel, MacroModel}, connection::Integer, time::Real)
    inject_event!(model.runtime.state, connection, time)
    return model
end

function inject_spike!(model::Union{MicroModel, MacroModel}, source::SpikeSource, time::Real)
    inject_spike!(model.runtime.state, source, time)
    return model
end

# Runtime clocks, scalar locations and configuration contexts.

micro_time(state::SimulationState, step::Integer = state.step_micro) =
    step * state.configuration.dt_micro

macro_time(state::SimulationState, step::Integer = state.step_macro) =
    step * state.configuration.dt_macro

function reference_storage(state::SimulationState, selection::VariableRef)
    selection.kind == :voltage && return state.voltage
    selection.kind == :ion && return state.ions
    selection.kind == :exposure && return state.exposures
    selection.kind == :global && return state.work[:globals][selection.group]
    groups =
        selection.kind == :mechanism ? state.groups :
        selection.kind == :region ? state.region_groups :
        selection.kind == :edge ? state.edge_groups :
        selection.kind == :output ? state.output_groups : state.input_groups
    return groups[selection.group].data
end

reference_offset(data, selection::VariableRef) = reference_offset(size(data, 1), selection)

function reference_offset(rows::Integer, selection::VariableRef)
    selection.kind == :global && return selection.field
    selection.kind == :voltage && return selection.row
    return selection.row + (selection.field - 1) * rows
end

function connection_edge(state::SimulationState, connection::Integer)
    plan = state.layout.micro
    plan === nothing && throw(ArgumentError("simulation has no micro connections"))
    1 <= connection <= length(plan.connection_edges) ||
        throw(ArgumentError("unknown connection index"))
    return plan.connection_edges[connection]
end
