# Resolve lexical bindings before type, effect and equation analysis.

function array_extent!(parser::ModParser)
    accept!(parser, "[") || return 0
    expression = expression!(parser)
    expect!(parser, "]")
    value = constant_value(expression, parser.ast.definitions)
    isinteger(value) && value > 0 ||
        throw(ModError(expression.location, "array extent must be a positive integer constant"))
    return Int(value)
end

function equation_states(block::IRBlock)
    states = Symbol[]
    function collect_states!(body)
        for statement in body
            if statement isa Assignment && statement.derivative
                name = base_name(statement.target)
                name in states || push!(states, name)
            elseif statement isa Conditional
                collect_states!(statement.consequent)
                collect_states!(statement.alternative)
            elseif statement isa WhileStatement
                collect_states!(statement.body)
            end
        end
    end
    collect_states!(block.body)
    return states
end

function resolve_block_locals(block::ModBlock)
    bindings = LocalBinding[]
    scope_count = Ref(0)
    function expression(value::ModExpression, names)
        if value isa NameExpression
            return NameExpression(get(names, value.name, value.name), value.location)
        elseif value isa OperationExpression
            return OperationExpression(
                value.operation,
                ModExpression[expression(argument, names) for argument in value.arguments],
                value.location,
            )
        end
        return value
    end
    function statements(body, outer)
        scope_count[] += 1
        scope = scope_count[]
        names = copy(outer)
        # LOCAL belongs to its complete lexical block, including uses before
        # the declaration. Distinct scopes never share a generated binding.
        for statement in body
            statement isa LocalDeclaration || continue
            for (name, extent) in zip(statement.names, statement.extents)
                resolved = Symbol(name, "#local", scope)
                names[name] = resolved
                push!(bindings, LocalBinding(resolved, name, scope, extent, statement.location))
            end
        end
        result = ModStatement[]
        for statement in body
            location = statement.location
            if statement isa Assignment
                push!(
                    result,
                    Assignment(
                        expression(statement.target, names),
                        expression(statement.value, names),
                        statement.derivative,
                        location,
                    ),
                )
            elseif statement isa CallStatement
                push!(result, CallStatement(expression(statement.expression, names), location))
            elseif statement isa Conditional
                push!(
                    result,
                    Conditional(
                        expression(statement.condition, names),
                        statements(statement.consequent, names),
                        statements(statement.alternative, names),
                        location,
                    ),
                )
            elseif statement isa WhileStatement
                push!(
                    result,
                    WhileStatement(
                        expression(statement.condition, names),
                        statements(statement.body, names),
                        location,
                    ),
                )
            elseif statement isa EventInitialStatement
                push!(result, EventInitialStatement(statements(statement.body, names), location))
            elseif statement isa LocalDeclaration
                push!(
                    result,
                    LocalDeclaration(
                        [names[name] for name in statement.names],
                        statement.extents,
                        location,
                    ),
                )
            else
                push!(result, statement)
            end
        end
        return result
    end
    body = statements(block.body, Dict{Symbol, Symbol}())
    return ModBlock(
        block.kind,
        block.name,
        block.arguments,
        block.argument_units,
        block.units,
        body,
        block.location,
    ),
    bindings
end

# Recursive INCLUDE expansion; retain original token source locations.

function read_mod_tokens(filename::String, ancestors::Set{String}, sources)
    filename in ancestors && throw(ArgumentError("cyclic MOD INCLUDE: $filename"))
    source = get!(sources, filename) do
        read(filename, String)
    end
    next = union(ancestors, Set((filename,)))
    tokens = tokenize_mod(source; filename)
    expanded = Token[]
    index = 1
    while index <= length(tokens)
        token = tokens[index]
        if token.kind == :identifier && token.text == "INCLUDE"
            path = tokens[index + 1]
            path.kind == :string || throw(ModError(path.location, "INCLUDE requires a quoted path"))
            included = normpath(joinpath(dirname(filename), path.text))
            (haskey(sources, included) || isfile(included)) ||
                throw(ModError(path.location, "included file does not exist: $(path.text)"))
            append!(expanded, read_mod_tokens(included, next, sources)[1:(end - 1)])
            index += 2
        else
            push!(expanded, token)
            index += 1
        end
    end
    return expanded
end

# Dimensional analysis utilities. The basis follows nrnunits: length, mass, time,
# charge, light, currency, reserved, information, traffic and temperature. Mole
# is Avogadro's dimensionless count, as in NEURON's concentration convention.

const MOD_UNIT_BASES = Dict(
    "m" => 1,
    "kg" => 2,
    "sec" => 3,
    "coul" => 4,
    "candela" => 5,
    "dollar" => 6,
    "bit" => 8,
    "erlang" => 9,
    "K" => 10,
)
const MOD_UNIT_ALIASES = Dict(
    "meter" => "m",
    "metre" => "m",
    "second" => "sec",
    "s" => "sec",
    "coulomb" => "coul",
    "C" => "coul",
    "kelvin" => "K",
    "degK" => "K",
    "degC" => "K",
    "gram" => "0.001 kg",
    "gm" => "gram",
    "g" => "gram",
    "minute" => "60 sec",
    "min" => "minute",
    "hour" => "60 minute",
    "hr" => "hour",
    "day" => "24 hour",
    "week" => "7 day",
    "year" => "365.24219879 day",
    "liter" => "0.001 m3",
    "litre" => "liter",
    "L" => "liter",
    "micron" => "micrometer",
    "angstrom" => "1e-10 m",
    "ampere" => "coul/sec",
    "amp" => "ampere",
    "A" => "ampere",
    "newton" => "kg m/sec2",
    "N" => "newton",
    "joule" => "newton m",
    "J" => "joule",
    "watt" => "joule/sec",
    "W" => "watt",
    "volt" => "watt/amp",
    "V" => "volt",
    "ohm" => "volt/amp",
    "mho" => "/ohm",
    "siemens" => "mho",
    "S" => "siemens",
    "farad" => "coul/volt",
    "F" => "farad",
    "henry" => "sec2/farad",
    "H" => "henry",
    "weber" => "volt sec",
    "Wb" => "weber",
    "tesla" => "weber/m2",
    "T" => "tesla",
    "pascal" => "newton/m2",
    "Pa" => "pascal",
    "hertz" => "/sec",
    "Hz" => "hertz",
    "ms" => "millisecond",
    "us" => "microsecond",
    "cm" => "centimeter",
    "mm" => "millimeter",
    "nm" => "nanometer",
    "um" => "micrometer",
    "mV" => "millivolt",
    "uV" => "microvolt",
    "mA" => "milliamp",
    "uA" => "microamp",
    "nA" => "nanoamp",
    "pA" => "picoamp",
    "uS" => "microsiemens",
    "nS" => "nanosiemens",
    "mS" => "millisiemens",
    "uF" => "microfarad",
    "nF" => "nanofarad",
    "pF" => "picofarad",
    "Mohm" => "megaohm",
    "kohm" => "kiloohm",
    "megohm" => "megaohm",
    "molar" => "/liter",
    "M" => "molar",
    "mM" => "milli/liter",
    "uM" => "micro/liter",
    "mole" => "6.02214076e23",
    "mol" => "mole",
    "e" => "1.602176634e-19 coul",
    "faraday" => "e mole",
    "k" => "1.380649e-23 joule/K",
    "boltzmann" => "k",
    "gasconstant" => "k mole",
    "R" => "gasconstant",
    "planck" => "6.62607015e-34 joule sec",
    "hbar" => "planck/2 pi",
    "pi" => "3.141592653589793",
    "two" => "2",
    "fuzz" => "1",
    "radian" => "0.5/pi",
    "degree" => "1|180 pi radian",
    "rad" => "radian",
    "calorie" => "4.1868 joule",
    "cal" => "calorie",
    "erg" => "1e-7 joule",
    "electronvolt" => "e volt",
    "eV" => "electronvolt",
    "ev" => "electronvolt",
    "dyne" => "1e-5 newton",
    "bar" => "1e5 pascal",
    "atmosphere" => "101325 pascal",
    "torr" => "133.322 pascal",
    "cc" => "cm3",
    "byte" => "8 bit",
)
const MOD_UNIT_PREFIXES = (
    "yotta" => 1e24,
    "zetta" => 1e21,
    "exa" => 1e18,
    "peta" => 1e15,
    "tera" => 1e12,
    "giga" => 1e9,
    "mega" => 1e6,
    "meg" => 1e6,
    "kilo" => 1e3,
    "hecto" => 1e2,
    "hecta" => 1e2,
    "deka" => 1e1,
    "deci" => 1e-1,
    "centi" => 1e-2,
    "milli" => 1e-3,
    "micro" => 1e-6,
    "nano" => 1e-9,
    "pico" => 1e-12,
    "femto" => 1e-15,
    "atto" => 1e-18,
    "zepto" => 1e-21,
    "yocto" => 1e-24,
)

multiply_units(a::ModUnit, b::ModUnit) = ModUnit(
    a.dimensions .+ b.dimensions,
    a.scale * b.scale,
    a.literal && b.literal,
    a.wildcard || b.wildcard,
)
power_units(unit::ModUnit, power::Rational) =
    ModUnit(unit.dimensions .* power, unit.scale^Float64(power), unit.literal, unit.wildcard)
power_units(unit::ModUnit, power::Integer) = power_units(unit, power // 1)

function compatible_units(a::ModUnit, b::ModUnit, location; enabled = true)
    b.literal && return a
    a.literal && return b
    if enabled && !(a.wildcard || b.wildcard)
        a.dimensions == b.dimensions || throw(
            ModError(
                location,
                "inconsistent physical dimensions: $(a.dimensions) and $(b.dimensions)",
            ),
        )
        abs(b.scale / a.scale - 1.0) <= 1e-5 || throw(
            ModError(location, "missing explicit unit conversion factor $(b.scale / a.scale)"),
        )
    end
    return a
end

function mod_unit(ast::ModAST, text::AbstractString, location, active = Set{String}())
    isempty(strip(text)) && return ModUnit()
    tokens = [
        m.match for
        m in eachmatch(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]+[0-9]*|[^\s]", text)
    ]
    position = Ref(1)
    function named(name)
        name == "m9" && return ModUnit(wildcard = true)
        if haskey(MOD_UNIT_BASES, name)
            axis = MOD_UNIT_BASES[name]
            return ModUnit(ntuple(i -> (i == axis ? 1 : 0) // 1, 10), 1.0, false, false)
        end
        if haskey(ast.unit_definitions, name) || haskey(MOD_UNIT_ALIASES, name)
            name in active && throw(ModError(location, "cyclic unit definition $name"))
            definition = get(ast.unit_definitions, name, get(MOD_UNIT_ALIASES, name, ""))
            push!(active, name)
            value = mod_unit(ast, definition, location, active)
            delete!(active, name)
            return value
        end
        power = match(r"^([A-Za-z_]+)([0-9]+)$", name)
        power !== nothing && return power_units(named(String(power[1])), parse(Int, power[2]))
        for (prefix, scale) in MOD_UNIT_PREFIXES
            name == prefix && return ModUnit(; scale)
            startswith(name, prefix) || continue
            suffix = String(name[(length(prefix) + 1):end])
            return multiply_units(ModUnit(; scale), named(suffix))
        end
        endswith(name, "s") && length(name) > 1 && return named(name[1:(end - 1)])
        throw(ModError(location, "unknown unit $name"))
    end
    function product()
        value = ModUnit()
        denominator = false
        while position[] <= length(tokens)
            token = tokens[position[]]
            token == ")" && break
            position[] += 1
            if token == "/"
                denominator = true
                continue
            elseif token in ("*", "-")
                continue
            elseif token == "("
                factor = product()
                position[] <= length(tokens) && tokens[position[]] == ")" ||
                    throw(ModError(location, "unterminated unit expression"))
                position[] += 1
            else
                number = tryparse(Float64, token)
                factor = number === nothing ? named(String(token)) : ModUnit(scale = number)
                if number !== nothing && position[] <= length(tokens) && tokens[position[]] == "|"
                    position[] += 1
                    factor = ModUnit(scale = number / parse(Float64, tokens[position[]]))
                    position[] += 1
                end
            end
            if position[] <= length(tokens) && tokens[position[]] == "^"
                position[] += 1
                sign = tokens[position[]] == "-" ? -1 : 1
                tokens[position[]] in ("-", "+") && (position[] += 1)
                factor = power_units(factor, sign * parse(Int, tokens[position[]]))
                position[] += 1
            end
            value = multiply_units(value, denominator ? power_units(factor, -1) : factor)
        end
        return value
    end
    unit = product()
    position[] > length(tokens) || throw(ModError(location, "unexpected token in unit expression"))
    return unit
end

unit_scale(ast::ModAST, text::AbstractString, location::SourceLocation) =
    mod_unit(ast, text, location).scale

function recursive_call_graph(calls, functions)
    active, complete = Set{Symbol}(), Set{Symbol}()
    function visit(name)
        name in active && return true
        name in complete && return false
        push!(active, name)
        for target in calls[name]
            target in functions && visit(target) && return true
        end
        delete!(active, name)
        push!(complete, name)
        return false
    end
    return any(visit, functions)
end

function unit_constant_expression(expression, definitions)
    expression isa NumberExpression && return true
    expression isa NameExpression && return haskey(definitions, expression.name)
    return expression.operation in (
        :+,
        :-,
        :*,
        :/,
        :^,
        Symbol("unary+"),
        Symbol("unary-"),
        :exp,
        :log,
        :sqrt,
        :abs,
        :fabs,
    ) && all(argument -> unit_constant_expression(argument, definitions), expression.arguments)
end
# Keep BREAKPOINT's remaining control flow after SOLVE has moved to nrn_state.
function breakpoint_remainder(body::Vector{ModStatement})
    statements = ModStatement[]
    for statement in body
        statement isa SolveStatement && continue
        if statement isa Conditional
            statement = Conditional(
                statement.condition,
                breakpoint_remainder(statement.consequent),
                breakpoint_remainder(statement.alternative),
                statement.location,
            )
        elseif statement isa WhileStatement
            statement = WhileStatement(
                statement.condition,
                breakpoint_remainder(statement.body),
                statement.location,
            )
        end
        push!(statements, statement)
    end
    return statements
end

function contains_euler(body::Vector{ModStatement})
    for statement in body
        statement isa SolveStatement && statement.method == :euler && return true
        if statement isa Conditional
            (contains_euler(statement.consequent) || contains_euler(statement.alternative)) &&
                return true
        elseif statement isa WhileStatement || statement isa EventInitialStatement
            contains_euler(statement.body) && return true
        end
    end
    return false
end
