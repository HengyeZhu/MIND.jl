# Ion style encoding and physical formulas.

ion_style_bits(style::IonStyle) =
    UInt8(style.c_style + 4style.cinit + 8style.e_style + 32style.einit + 64style.eadvance)

decode_ion_style(bits::UInt8) =
    IonStyle(bits & 0x03, (bits >> 3) & 0x03, bits & 0x20 != 0, bits & 0x40 != 0, bits & 0x04 != 0)

function promote_ion_style(style::UInt8, concentration::Integer, reversal::Integer)
    concentration = max(Int(style & 0x03), concentration)
    reversal = max(Int((style >> 3) & 0x03), reversal)
    concentration > 0 && reversal < 2 && (reversal = 2)
    return ion_style_bits(
        IonStyle(
            concentration,
            reversal,
            concentration > 0 && reversal == 2,
            concentration == 3 && reversal == 2,
            concentration == 3,
        ),
    )
end

ion_style_override(style::UInt8, override::IonStyle) = ion_style_bits(override)
function ion_style_override(style::UInt8, policy::Symbol)
    policy == :automatic && return style
    flags = policy == :fixed ? 0x00 : policy == :initial ? 0x20 : policy == :advance ? 0x40 : 0x60
    return (style & 0x1f) | flags
end

default_ions() = Dict(
    :na => IonSpecies(:na; valence = 1, internal = 10.0, external = 140.0, reversal = 50.0),
    :k => IonSpecies(:k; valence = 1, internal = 54.4, external = 2.5, reversal = -77.0),
    :ca => IonSpecies(
        :ca;
        valence = 2,
        internal = 5.0e-5,
        external = 2.0,
        reversal = 132.4579341637009,
    ),
)

@inline function nernst_potential(internal::Real, external::Real, valence::Real, celsius::Real)
    valence == 0.0 && return 0.0
    internal <= 0.0 && return 1.0e6
    external <= 0.0 && return -1.0e6
    gasconstant = 1.380649e-23 * 6.02214076e23
    faraday = 1.602176634e-19 * 6.02214076e23
    return (1000.0 * gasconstant * (celsius + 273.15) / faraday) / valence *
           log(external / internal)
end

@inline function ghk_current(
    voltage::Real,
    internal::Real,
    external::Real,
    valence::Real,
    celsius::Real,
)
    gasconstant = 1.380649e-23 * 6.02214076e23
    faraday = 1.602176634e-19 * 6.02214076e23
    thermal_voltage = 1000.0 * gasconstant * (celsius + 273.15) / faraday
    scaled_voltage = valence * voltage / thermal_voltage
    # Preserve NEURON's small-argument branch and units (mV, mM).
    quotient(x) = abs(x) < 1.0e-4 ? 1.0-x/2.0 : x/(exp(x)-1.0)
    return 0.001 *
           valence *
           faraday *
           (internal * quotient(-scaled_voltage) - external * quotient(scaled_voltage))
end

function validate_ion_arguments(internal, external, valence, celsius)
    all(isfinite, (internal, external, valence, celsius)) &&
    internal >= 0 &&
    external >= 0 &&
    celsius > -273.15 || throw(ArgumentError("invalid ion arguments"))
    return nothing
end

# Parameter access rules shared by immediate and reusable bindings.

@inline function assign_parameter!(row, offset, extent, name, value)
    if extent == 0
        converted = value isa Float64 ? value : Float64(value)::Float64
        isfinite(converted) || throw(ArgumentError("non-finite value for $name"))
        row[offset] = converted
    else
        length(value) == extent || throw(DimensionMismatch("array parameter $name"))
        for (index, element) in enumerate(value)
            converted = element isa Float64 ? element : Float64(element)::Float64
            isfinite(converted) || throw(ArgumentError("non-finite value for $name"))
            row[offset + index - 1] = converted
        end
    end
    return nothing
end

@inline function checked_parameter_field(ir::MechanismIR, name::Symbol)
    haskey(ir.fieldmap, name) || throw(ArgumentError("$(ir.name) has no field $name"))
    field = fieldinfo(ir, name)
    field.scope == :pointer &&
        throw(ArgumentError("POINTER $name needs a variable binding, not a numeric override"))
    field.category == :constant && throw(ArgumentError("CONSTANT $name cannot be overridden"))
    any(binding -> first(binding) == name, ir.inputs) && throw(
        ArgumentError(
            "$name is supplied by the simulation context; specify it when creating or building the model",
        ),
    )
    any(binding -> first(binding) == field.offset, ir.construction_inputs) && throw(
        ArgumentError(
            "$name is supplied by the connection builder; configure connectivity instead",
        ),
    )
    return field
end

function bind_parameters(ir::MechanismIR, parameters)
    # Resolve names and permissions once per binding, outside the instance loop.
    values = ParameterValue[]
    ions = Tuple{Symbol, Int, Bool}[]
    for (name, value) in pairs(parameters)
        field = checked_parameter_field(ir, name)
        push!(values, ParameterValue(name, field.offset, field.extent, value))
        for ion in ir.ions
            name in ion.reads || name in ion.writes || continue
            concentration = name in (Symbol(ion.name, :i), Symbol(ion.name, :o))
            push!(ions, (name, field.offset, concentration))
        end
    end
    # Preserve callback types without evaluating them during binding.
    return ParameterBindings(Tuple(values), ions)
end
