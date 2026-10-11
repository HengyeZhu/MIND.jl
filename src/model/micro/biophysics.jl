# Ion species and styles.

"""Ion defaults use mM and mV; concentrations and charge are Float64."""
struct IonSpecies
    name::Symbol
    valence::Float64
    internal::Float64
    external::Float64
    reversal::Float64
    nernst::Bool
end

function IonSpecies(name; valence, internal, external, reversal = 0.0, nernst = false)
    valence isa Real && isfinite(valence) || throw(ArgumentError("ion valence must be finite"))
    all(isfinite, (internal, external, reversal)) ||
        throw(ArgumentError("non-finite ion parameter"))
    internal >= 0 && external >= 0 || throw(ArgumentError("ion concentrations must be nonnegative"))
    return IonSpecies(
        Symbol(name),
        Float64(valence),
        Float64(internal),
        Float64(external),
        Float64(reversal),
        nernst,
    )
end

"""Ion handling with NEURON's concentration/reversal styles and initialization flags.

`c_style` and `e_style` use 0 (unused), 1 (parameter), 2 (assigned) or 3 (state).
`cinit` resets concentrations from `IonSpecies`; `einit` and `eadvance` request
Nernst evaluation during initialization and current assembly, respectively.
"""
struct IonStyle
    c_style::UInt8
    e_style::UInt8
    einit::Bool
    eadvance::Bool
    cinit::Bool
    function IonStyle(c_style, e_style, einit, eadvance, cinit)
        all(value -> value isa Integer && 0 <= value <= 3, (c_style, e_style)) ||
            throw(ArgumentError("ion variable styles must be integers from 0 to 3"))
        all(value -> value isa Integer && value in (0, 1), (einit, eadvance, cinit)) ||
            throw(ArgumentError("ion initialization flags must be Bool or 0/1"))
        return new(UInt8(c_style), UInt8(e_style), Bool(einit), Bool(eadvance), Bool(cinit))
    end
end

IonStyle(; c_style = 0, e_style = 0, einit = false, eadvance = false, cinit = false) =
    IonStyle(c_style, e_style, einit, eadvance, cinit)

"""Nernst reversal in mV for concentrations in mM; continuous arithmetic is FP64."""
function nernst(internal::Real, external::Real, valence::Real; celsius = 6.3)
    validate_ion_arguments(internal, external, valence, celsius)
    return nernst_potential(
        Float64(internal),
        Float64(external),
        Float64(valence),
        Float64(celsius),
    )
end

"""Goldman–Hodgkin–Katz current factor, with NEURON's mV and mM conventions."""
function ghk(voltage::Real, internal::Real, external::Real, valence::Real; celsius = 6.3)
    validate_ion_arguments(internal, external, valence, celsius)
    isfinite(voltage) || throw(ArgumentError("voltage must be finite"))
    return ghk_current(
        Float64(voltage),
        Float64(internal),
        Float64(external),
        Float64(valence),
        Float64(celsius),
    )
end

# Mechanism parameter binding.

struct DensityBinding
    mechanism::CompiledMechanism
    sections::Vector{Symbol}
    parameters::Dict{Symbol, Any}
end

struct ParameterValue{T}
    name::Symbol
    offset::Int
    extent::Int
    value::T
end

"""Resolved CPU assignments; callable values remain unevaluated until each instance."""
struct ParameterBindings{T <: Tuple}
    values::T
    ions::Vector{Tuple{Symbol, Int, Bool}}
end

function apply_parameters!(row, ir::MechanismIR, parameters; location = nothing)
    # One-use assignments do not need a temporary binding program. Repeated
    # density assignments use the resolved, concretely typed method below.
    for (name, value) in pairs(parameters)
        field = checked_parameter_field(ir, name)
        actual = value isa Function ? value(location) : value
        assign_parameter!(row, field.offset, field.extent, name, actual)
    end
    return row
end

@inline function apply_parameters!(row, parameters::ParameterBindings; location = nothing)
    # Tuple map specializes on each field/callback type. Its all-Nothing result
    # is discarded; forwarding closures through tuple folds can box every row.
    map(parameters.values) do field
        name, value = field.name, field.value
        actual = value isa Function ? value(location) : value
        assign_parameter!(row, field.offset, field.extent, name, actual)
        return nothing
    end
    return row
end

@inline function collect_ion_parameters!(
    assignments,
    row,
    parameters::ParameterBindings,
    node::Int32,
)
    for (name, offset, concentration) in parameters.ions
        node > 0 || throw(ArgumentError("ion parameter $name requires a cable location"))
        value = row[offset]
        concentration && value < 0.0 && throw(ArgumentError("negative ion concentration $name"))
        assignments[(node, name)] = value
    end
    return nothing
end

"""Collect explicitly supplied USEION values for the compartment's shared storage.

Read the already evaluated row so spatial callbacks are executed exactly once.
Defaults in a MOD declaration do not override the ion species defaults.
"""
function collect_ion_parameters!(assignments, row, ir::MechanismIR, parameters, node::Int32)
    isempty(ir.ions) && return nothing
    for (name, _) in parameters
        for ion in ir.ions
            name in ion.reads || name in ion.writes || continue
            node > 0 || throw(ArgumentError("ion parameter $name requires a cable location"))
            value = row[fieldindex(ir, name)]
            concentration = name in (Symbol(ion.name, :i), Symbol(ion.name, :o))
            concentration && value < 0.0 && throw(ArgumentError("negative ion concentration $name"))
            assignments[(node, name)] = value
        end
    end
    return nothing
end

"""An owned scalar location resolved on the CPU, with no host or device address.

References are valid for the model layout that resolved them. Rebuilding the
model requires resolving its references again; resetting its state does not.
"""
struct VariableRef
    layout_id::UInt64
    kind::Symbol
    group::Int
    row::Int
    field::Int
end

struct SpikeSourceDefinition
    sid::Int64
    cell::Int
    section::Symbol
    x::Float64
    point::Int
    threshold::Float64
    mechanism::Union{Nothing, CompiledMechanism}
    field::Union{Nothing, Symbol}
    index::Int
end

struct Connection
    # Positive: registered source index. Negative: macro input index, rebased
    # after all micro sources have been created. Public SIDs stay on sources.
    source_index::Int
    target::Int
    weight::Float64
    weights::Union{Nothing, Vector{Float64}}
    arity::Int
    delay::Float64
end

Connection(source, target, weight::Float64, arity, delay) =
    Connection(source, target, weight, nothing, arity, delay)
Connection(source, target, weights::Vector{Float64}, arity, delay) =
    Connection(source, target, 0.0, weights, arity, delay)

"""Resolve a loaded MOD/builtin name or retain an already compiled mechanism."""
resolve_mechanism(mechanism::CompiledMechanism) = mechanism
function resolve_mechanism(name::Union{Symbol, AbstractString})
    key = Symbol(name)
    mechanism = lock(MOD_CACHE_LOCK) do
        get(LOADED_MODS, key, nothing)
    end
    return mechanism === nothing ? builtin(key) : mechanism
end
