# Public MOD compilation workflow.

const MOD_CACHE_LOCK = ReentrantLock()
const MOD_CACHE = Dict{String, CompiledMechanism}()
const MOD_FILE_CACHE = Dict{String, Tuple{Dict{String, String}, CompiledMechanism}}()
const BUILTIN_CACHE = Dict{Symbol, CompiledMechanism}()
const LOADED_MODS = Dict{Symbol, CompiledMechanism}()

"""Load a MOD file or all .mod files in a directory, registering their declared names.

INCLUDE paths resolve relative to their source file. A file returns its compiled
mechanism; a directory returns nothing. Later loads replace the name registration;
existing model instances retain the compiled mechanisms they already selected.
"""
function load_mod(path::AbstractString; check_units::Bool = false)
    # 1. Load a directory in filename order through the same file compilation path.
    if isdir(path)
        for filename in readdir(path; join = true)
            isfile(filename) && endswith(filename, ".mod") || continue
            load_mod(filename; check_units)
        end
        return nothing
    end

    # 2. Reuse compiled code only when all transitive source contents still match.
    filename = abspath(path)
    cached = lock(MOD_CACHE_LOCK) do
        get(MOD_FILE_CACHE, filename, nothing)
    end
    # Check content, not timestamps: edits to any transitive INCLUDE must
    # invalidate the entry even when size and modification time are equal.
    unchanged =
        cached !== nothing && all(first(cached)) do (source_path, previous)
            isfile(source_path) && read(source_path, String) == previous
        end
    mechanism = if unchanged
        check_units && MIND.check_units(last(cached))
        last(cached)
    else
        sources = Dict{String, String}()
        tokens = read_mod_tokens(filename, Set{String}(), sources)
        compiled =
            compile_source(sources[filename], filename, tokens, sources, true; check_units)
        lock(MOD_CACHE_LOCK) do
            MOD_FILE_CACHE[filename] = (sources, compiled)
        end
        compiled
    end

    # 3. Publish the parsed MOD name for construction-time mechanism selection.
    lock(MOD_CACHE_LOCK) do
        LOADED_MODS[mechanism.ir.name] = mechanism
    end
    return mechanism
end

"""Select a packaged builtin explicitly, independently of loaded user MOD names."""
function builtin(name)
    key = Symbol(name)
    # Package mechanisms are fixed inputs. Read and compile each one on first use;
    # subsequent scalar insertions reuse the same compiled object.
    return lock(MOD_CACHE_LOCK) do
        get!(BUILTIN_CACHE, key) do
            filename = joinpath(pkgdir(@__MODULE__), "mod", "$key.mod")
            isfile(filename) ||
                throw(ArgumentError("unknown mechanism $key; load its MOD with load_mod"))
            sources = Dict{String, String}()
            tokens = read_mod_tokens(filename, Set{String}(), sources)
            compile_source(sources[filename], filename, tokens, sources, true)
        end
    end
end

"""Compile MOD source to typed IR and specialized Julia functions for CUDA execution."""
function compile_mod(
    source::AbstractString;
    filename::AbstractString = "<string>",
    cache::Bool = true,
    check_units::Bool = false,
)
    return compile_source(
        String(source),
        String(filename),
        nothing,
        Dict(String(filename) => String(source)),
        cache;
        check_units,
    )
end

function compile_source(source, filename, tokens, sources, cache; check_units = false)
    ast = check_units ? parse_mod(source; filename, tokens, sources) : nothing
    check_units && MIND.check_units(ast)
    digest = bytes2hex(sha256(string(filename, '\0', source_digest(sources, filename))))
    if cache
        cached = lock(MOD_CACHE_LOCK) do
            get(MOD_CACHE, digest, nothing)
        end
        cached !== nothing && return cached
    end
    # Resolve source syntax, language semantics and executable phases in order.
    ast === nothing && (ast = parse_mod(source; filename, tokens, sources))
    mechanism = ast |> analyze_mod |> compile_ir
    if cache
        lock(MOD_CACHE_LOCK) do
            MOD_CACHE[digest] = mechanism
        end
    end
    return mechanism
end

"""Drop compilation caches, retaining loaded name registrations and existing models."""
function clear_mod_cache!()
    lock(MOD_CACHE_LOCK) do
        empty!(MOD_CACHE)
        empty!(MOD_FILE_CACHE)
        empty!(BUILTIN_CACHE)
    end
    return nothing
end

# Tokenization.

"""Tokenize MOD without interpreting MIND blocks as embedded source strings."""
function tokenize_mod(source::AbstractString; filename::AbstractString = "<string>")
    text = String(source)
    tokens = Token[]
    index = firstindex(text)
    line = 1
    column = 1
    block_comment = false
    while index <= lastindex(text)
        character = text[index]
        location = SourceLocation(String(filename), line, column)
        if character == '\n'
            # Newlines carry locations, not statement boundaries, in NMODL.
            line += 1
            column = 1
            index = nextind(text, index)
            continue
        elseif isspace(character)
            column += 1
            index = nextind(text, index)
            continue
        elseif character in (':', '?') && !block_comment
            while index <= lastindex(text) && text[index] != '\n'
                index = nextind(text, index)
                column += 1
            end
            continue
        end
        if isletter(character) || character == '_'
            start = index
            while index <= lastindex(text) &&
                  (isletter(text[index]) || isdigit(text[index]) || text[index] == '_')
                index = nextind(text, index)
                column += 1
            end
            word = text[start:prevind(text, index)]
            if word == "TITLE" && !block_comment
                while index <= lastindex(text) && text[index] != '\n'
                    index = nextind(text, index)
                    column += 1
                end
            elseif word == "COMMENT"
                block_comment = true
            elseif word == "ENDCOMMENT"
                block_comment = false
            elseif !block_comment
                push!(tokens, Token(:identifier, word, location))
            end
            continue
        end
        if block_comment
            index = nextind(text, index)
            column += 1
            continue
        end
        if isdigit(character) ||
           (character == '.' && index < lastindex(text) && isdigit(text[nextind(text, index)]))
            matched = match(r"^(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?", SubString(text, index))
            word = matched.match
            push!(tokens, Token(:number, word, location))
            index = nextind(text, index, length(word))
            column += length(word)
            continue
        elseif character == '"'
            start = nextind(text, index)
            index = start
            while index <= lastindex(text) && text[index] != '"'
                text[index] == '\n' && throw(ModError(location, "unterminated string"))
                index = nextind(text, index)
            end
            index > lastindex(text) && throw(ModError(location, "unterminated string"))
            word = text[start:prevind(text, index)]
            push!(tokens, Token(:string, word, location))
            index = nextind(text, index)
            column += length(word) + 2
            continue
        end
        rest = SubString(text, index)
        operator = nothing
        for candidate in ("<=", ">=", "==", "!=", "&&", "||")
            if startswith(rest, candidate)
                operator = candidate
                break
            end
        end
        if operator === nothing
            character in (
                '{',
                '}',
                '(',
                ')',
                '[',
                ']',
                ',',
                ';',
                '+',
                '-',
                '*',
                '/',
                '^',
                '=',
                '<',
                '>',
                '!',
                '\'',
                '%',
            ) || throw(ModError(location, "unexpected character $(repr(character))"))
            operator = string(character)
        end
        push!(tokens, Token(:operator, operator, location))
        index = nextind(text, index, length(operator))
        column += length(operator)
    end
    block_comment &&
        throw(ModError(SourceLocation(String(filename), line, column), "unterminated COMMENT"))
    push!(tokens, Token(:eof, "", SourceLocation(String(filename), line, column)))
    return tokens
end

# MOD and MIND grammar.

mutable struct ModParser
    tokens::Vector{Token}
    position::Int
    ast::ModAST
    units_on::Bool
end

current(parser::ModParser) = parser.tokens[parser.position]
at(parser::ModParser, text::AbstractString) = current(parser).text == text

function take!(parser::ModParser)
    token = current(parser)
    parser.position += token.kind != :eof
    return token
end

function accept!(parser::ModParser, text::AbstractString)
    at(parser, text) || return false
    take!(parser)
    return true
end

function expect!(parser::ModParser, text::AbstractString)
    at(parser, text) || throw(
        ModError(
            current(parser).location,
            "expected $(repr(text)), found $(repr(current(parser).text))",
        ),
    )
    return take!(parser)
end

function identifier!(parser::ModParser)
    token = take!(parser)
    token.kind == :identifier || throw(ModError(token.location, "expected identifier"))
    return Symbol(token.text)
end

function separators!(parser::ModParser)
    while at(parser, ";")
        take!(parser)
    end
    return nothing
end

const MOD_PRECEDENCE = Dict(
    "||" => 1,
    "&&" => 2,
    "==" => 3,
    "!=" => 3,
    "<" => 4,
    ">" => 4,
    "<=" => 4,
    ">=" => 4,
    "+" => 5,
    "-" => 5,
    "*" => 6,
    "/" => 6,
    "%" => 6,
    "^" => 8,
)

function expression!(
    parser::ModParser,
    minimum::Int = 0;
    stop::Tuple = (),
    literal_units::Bool = true,
)
    token = take!(parser)
    if token.kind == :number
        value = parse(Float64, replace(token.text, 'D' => 'e', 'd' => 'e'))
        # NMODL allows dimensional literals such as 5(ms) and 0.062(/mV).
        # Units annotate the literal; they do not multiply it by a variable.
        units = literal_units && at(parser, "(") ? parentheses_text!(parser) : nothing
        left = NumberExpression(value, units, 1.0, token.location)
    elseif token.kind == :identifier
        left = NameExpression(Symbol(token.text), token.location)
        if accept!(parser, "(")
            arguments = ModExpression[]
            separators!(parser)
            if !at(parser, ")")
                while true
                    push!(arguments, expression!(parser))
                    separators!(parser)
                    accept!(parser, ",") || break
                    separators!(parser)
                end
            end
            expect!(parser, ")")
            left = OperationExpression(Symbol(token.text), arguments, token.location)
        end
        if accept!(parser, "[")
            index = expression!(parser)
            expect!(parser, "]")
            left = OperationExpression(Symbol("[]"), ModExpression[left, index], token.location)
        end
    elseif token.text == "("
        separators!(parser)
        left = expression!(parser)
        separators!(parser)
        expect!(parser, ")")
        # A parenthesized bare constant denotes an explicit conversion factor
        # to modlunit. Its numerical value is unchanged by unit analysis.
        if left isa NumberExpression && left.units === nothing
            left = NumberExpression(left.value, "", inv(left.value), token.location)
        elseif left isa NameExpression && haskey(parser.ast.definitions, left.name)
            value = parser.ast.definitions[left.name]
            left = NumberExpression(value, "", inv(value), token.location)
        end
    elseif token.text in ("+", "-", "!")
        left = OperationExpression(
            Symbol("unary", token.text),
            ModExpression[expression!(parser, 7; stop, literal_units)],
            token.location,
        )
    else
        throw(ModError(token.location, "expected expression, found $(repr(token.text))"))
    end
    while true
        operator = current(parser).text
        operator in stop && break
        precedence = get(MOD_PRECEDENCE, operator, -1)
        precedence < minimum && break
        take!(parser)
        separators!(parser)
        right = expression!(parser, precedence + (operator == "^" ? 0 : 1); stop, literal_units)
        left = OperationExpression(Symbol(operator), ModExpression[left, right], token.location)
    end
    return left
end

function constant_value(expression::ModExpression, definitions::Dict{Symbol, Float64})
    expression isa NumberExpression && return expression.value
    if expression isa NameExpression
        haskey(definitions, expression.name) || throw(
            ModError(expression.location, "$(expression.name) is not a compile-time constant"),
        )
        return definitions[expression.name]
    end
    values = [constant_value(argument, definitions) for argument in expression.arguments]
    operation = expression.operation
    operation == Symbol("unary-") && return -only(values)
    operation == Symbol("unary+") && return only(values)
    function_ =
        operation == :+ ? (+) :
        operation == :- ? (-) :
        operation == :* ? (*) :
        operation == :/ ? (/) :
        operation == :^ ? (^) :
        operation == :exp ? exp :
        operation == :log ? log :
        operation == :sqrt ? sqrt : operation in (:abs, :fabs) ? abs : nothing
    function_ !== nothing ||
        throw(ModError(expression.location, "invalid constant operation $operation"))
    return Float64(function_(values...))
end

function parentheses_text!(parser::ModParser)
    expect!(parser, "(")
    parts = String[]
    depth = 1
    while depth > 0
        token = take!(parser)
        token.kind == :eof && throw(ModError(token.location, "unterminated parentheses"))
        token.text == "(" && (depth += 1)
        token.text == ")" && (depth -= 1)
        depth > 0 && push!(parts, token.text)
    end
    return join(parts, " ")
end

function declarations!(parser::ModParser, storage::Symbol)
    expect!(parser, "{")
    separators!(parser)
    while !accept!(parser, "}")
        location = current(parser).location
        name = identifier!(parser)
        extent = array_extent!(parser)
        default =
            accept!(parser, "=") ? expression!(parser; stop = ("<",), literal_units = false) :
            nothing
        units = at(parser, "(") ? parentheses_text!(parser) : ""
        bounds = nothing
        if accept!(parser, "<")
            lower = constant_value(expression!(parser; stop = (">",)), parser.ast.definitions)
            expect!(parser, ",")
            upper = constant_value(expression!(parser; stop = (">",)), parser.ast.definitions)
            expect!(parser, ">")
            bounds = (lower, upper)
        elseif accept!(parser, "FROM")
            lower = constant_value(expression!(parser), parser.ast.definitions)
            expect!(parser, "TO")
            upper = constant_value(expression!(parser), parser.ast.definitions)
            bounds = (lower, upper)
        end
        declaration = Declaration(name, extent, storage, default, units, bounds, location)
        push!(parser.ast.declarations, declaration)
        parser.ast.unit_checks[declaration] = parser.units_on
        accept!(parser, ",")
        separators!(parser)
    end
end

const NEURON_KEYWORDS = Set((
    "SUFFIX",
    "POINT_PROCESS",
    "ARTIFICIAL_CELL",
    "RANGE",
    "GLOBAL",
    "USEION",
    "NONSPECIFIC_CURRENT",
    "ELECTRODE_CURRENT",
    "THREADSAFE",
    "RANDOM",
    "POINTER",
))

function names!(parser::ModParser; terminators = NEURON_KEYWORDS)
    names = Symbol[]
    while current(parser).kind == :identifier && current(parser).text ∉ terminators
        push!(names, identifier!(parser))
        accept!(parser, ",")
    end
    return names
end

function neuron_block!(parser::ModParser)
    expect!(parser, "{")
    separators!(parser)
    while !accept!(parser, "}")
        token = take!(parser)
        keyword = token.text
        if keyword in ("SUFFIX", "POINT_PROCESS", "ARTIFICIAL_CELL")
            parser.ast.name = identifier!(parser)
            parser.ast.kind =
                keyword == "SUFFIX" ? :density : keyword == "POINT_PROCESS" ? :point : :artificial
        elseif keyword in ("RANGE", "GLOBAL", "NONSPECIFIC_CURRENT", "ELECTRODE_CURRENT", "RANDOM")
            field =
                keyword == "RANGE" ? :ranges :
                keyword == "GLOBAL" ? :globals :
                keyword == "NONSPECIFIC_CURRENT" ? :currents :
                keyword == "ELECTRODE_CURRENT" ? :electrode_currents : :randoms
            append!(getfield(parser.ast, field), names!(parser))
        elseif keyword == "THREADSAFE"
            parser.ast.threadsafe = true
        elseif keyword == "USEION"
            name = identifier!(parser)
            reads = Symbol[]
            writes = Symbol[]
            valence = nothing
            stops = union(NEURON_KEYWORDS, Set(("READ", "WRITE", "VALENCE")))
            while current(parser).text in ("READ", "WRITE", "VALENCE")
                access = take!(parser).text
                if access == "VALENCE"
                    valence = constant_value(expression!(parser), parser.ast.definitions)
                else
                    append!(access == "READ" ? reads : writes, names!(parser; terminators = stops))
                end
            end
            push!(parser.ast.ions, IonDeclaration(name, reads, writes, valence, token.location))
        elseif keyword == "POINTER"
            append!(parser.ast.pointers, names!(parser))
        else
            throw(ModError(token.location, "unknown NEURON declaration $keyword"))
        end
        separators!(parser)
    end
end

function mind_block!(parser::ModParser)
    expect!(parser, "{")
    separators!(parser)
    keywords =
        Set(("ROLE", "EXPOSURE", "READ_SOURCE", "READ_TARGET", "WRITE_SOURCE", "WRITE_TARGET"))
    while !accept!(parser, "}")
        token = take!(parser)
        if token.text == "ROLE"
            role = lowercase(String(identifier!(parser)))
            role in ("region", "macro2macro", "macro2micro", "micro2macro") ||
                throw(ModError(token.location, "unknown MIND role $role"))
            parser.ast.role = Symbol(role)
        elseif token.text == "EXPOSURE"
            append!(parser.ast.exposures, names!(parser; terminators = keywords))
        elseif token.text in ("READ_SOURCE", "READ_TARGET", "WRITE_SOURCE", "WRITE_TARGET")
            while current(parser).kind == :identifier && current(parser).text ∉ keywords
                exposure = identifier!(parser)
                variable = accept!(parser, "AS") ? identifier!(parser) : exposure
                push!(
                    parser.ast.bindings,
                    ExposureBinding(
                        Symbol(lowercase(token.text)),
                        exposure,
                        variable,
                        token.location,
                    ),
                )
                accept!(parser, ",")
            end
        else
            throw(ModError(token.location, "unknown MIND declaration $(token.text)"))
        end
        separators!(parser)
    end
end

function statement_body!(parser::ModParser)
    expect!(parser, "{")
    body = ModStatement[]
    separators!(parser)
    while !accept!(parser, "}")
        current(parser).kind == :eof &&
            throw(ModError(current(parser).location, "unterminated block"))
        enabled = parser.units_on
        statement = statement!(parser)
        if statement !== nothing
            push!(body, statement)
            parser.ast.unit_checks[statement] = enabled
        end
        separators!(parser)
    end
    return body
end

function statement!(parser::ModParser)
    token = current(parser)
    if accept!(parser, "LOCAL")
        names = Symbol[identifier!(parser)]
        extents = Int[array_extent!(parser)]
        while accept!(parser, ",")
            push!(names, identifier!(parser))
            push!(extents, array_extent!(parser))
        end
        return LocalDeclaration(names, extents, token.location)
    elseif accept!(parser, "if")
        expect!(parser, "(")
        condition = expression!(parser)
        expect!(parser, ")")
        separators!(parser)
        consequent = statement_body!(parser)
        separators!(parser)
        alternative = ModStatement[]
        if accept!(parser, "else")
            separators!(parser)
            if at(parser, "if")
                enabled = parser.units_on
                statement = statement!(parser)
                parser.ast.unit_checks[statement] = enabled
                push!(alternative, statement)
            else
                alternative = statement_body!(parser)
            end
        end
        return Conditional(condition, consequent, alternative, token.location)
    elseif accept!(parser, "while")
        expect!(parser, "(")
        condition = expression!(parser)
        expect!(parser, ")")
        separators!(parser)
        return WhileStatement(condition, statement_body!(parser), token.location)
    elseif accept!(parser, "SOLVE")
        block = identifier!(parser)
        expect!(parser, "METHOD")
        method = identifier!(parser)
        return SolveStatement(block, method, token.location)
    elseif accept!(parser, "INITIAL")
        return EventInitialStatement(statement_body!(parser), token.location)
    elseif accept!(parser, "UNITSOFF") || accept!(parser, "UNITSON")
        parser.units_on = token.text == "UNITSON"
        return nothing
    end
    left = expression!(parser)
    derivative = accept!(parser, "'")
    if derivative && accept!(parser, "[")
        index = expression!(parser)
        expect!(parser, "]")
        left = OperationExpression(Symbol("[]"), ModExpression[left, index], token.location)
    end
    if accept!(parser, "=")
        right = expression!(parser)
        return Assignment(left, right, derivative, token.location)
    end
    derivative && throw(ModError(token.location, "derivative requires an assignment"))
    left isa OperationExpression ||
        throw(ModError(token.location, "expected assignment or function call"))
    return CallStatement(left, token.location)
end

function unit_block!(parser::ModParser)
    expect!(parser, "{")
    separators!(parser)
    while !accept!(parser, "}")
        if at(parser, "(")
            name = parentheses_text!(parser)
            expect!(parser, "=")
            parser.ast.unit_definitions[name] = parentheses_text!(parser)
        else
            token = current(parser)
            name = identifier!(parser)
            expect!(parser, "=")
            units = ""
            if at(parser, "(")
                source_units = parentheses_text!(parser)
                units = at(parser, "(") ? parentheses_text!(parser) : "1"
                factor =
                    unit_scale(parser.ast, source_units, token.location) /
                    unit_scale(parser.ast, units, token.location)
                value = NumberExpression(factor, token.location)
            else
                value = expression!(parser; literal_units = false)
                units = at(parser, "(") ? parentheses_text!(parser) : ""
            end
            push!(
                parser.ast.declarations,
                Declaration(name, 0, :constant, value, units, nothing, token.location),
            )
        end
        separators!(parser)
    end
end

function parse_mod(
    source::AbstractString;
    filename::AbstractString = "<string>",
    tokens = nothing,
    sources = Dict(String(filename) => String(source)),
)
    ast = ModAST(String(source), String(filename))
    ast.sources = sources
    parser = ModParser(tokens === nothing ? tokenize_mod(source; filename) : tokens, 1, ast, true)
    separators!(parser)
    while current(parser).kind != :eof
        token = take!(parser)
        kind = Symbol(token.text)
        if kind == :NEURON
            ast.unit_contracts = parser.units_on
            neuron_block!(parser)
        elseif kind == :MIND
            mind_block!(parser)
        elseif kind in (:PARAMETER, :ASSIGNED, :STATE)
            declarations!(parser, Symbol(lowercase(token.text)))
        elseif kind == :LOCAL
            parser.position -= 1
            declaration = statement!(parser)
            for (name, extent) in zip(declaration.names, declaration.extents)
                push!(
                    ast.declarations,
                    Declaration(name, extent, :local, nothing, "", nothing, declaration.location),
                )
            end
        elseif kind == :UNITS
            unit_block!(parser)
        elseif kind == :DEFINE
            name = identifier!(parser)
            ast.definitions[name] = constant_value(expression!(parser), ast.definitions)
        elseif kind in (:INITIAL, :BREAKPOINT, :DERIVATIVE, :PROCEDURE, :FUNCTION, :NET_RECEIVE)
            name =
                kind in (:INITIAL, :BREAKPOINT, :NET_RECEIVE) ? Symbol(lowercase(token.text)) :
                identifier!(parser)
            arguments = Symbol[]
            argument_units = String[]
            if accept!(parser, "(")
                while !accept!(parser, ")")
                    push!(arguments, identifier!(parser))
                    push!(argument_units, at(parser, "(") ? parentheses_text!(parser) : "")
                    accept!(parser, ",")
                end
            end
            units = at(parser, "(") ? parentheses_text!(parser) : ""
            separators!(parser)
            push!(
                ast.blocks,
                ModBlock(
                    Symbol(lowercase(token.text)),
                    name,
                    arguments,
                    argument_units,
                    units,
                    statement_body!(parser),
                    token.location,
                ),
            )
        elseif kind in (:UNITSOFF, :UNITSON)
            parser.units_on = kind == :UNITSON
        elseif kind == :INDEPENDENT
            expect!(parser, "{")
            while !accept!(parser, "}")
                current(parser).kind == :eof &&
                    throw(ModError(token.location, "unterminated INDEPENDENT"))
                take!(parser)
            end
        else
            throw(ModError(token.location, "unexpected top-level token $(token.text)"))
        end
        separators!(parser)
    end
    return ast
end

# Symbol resolution.

function execution_bindings(role::Symbol, kind::Symbol)
    common = (:t=>TimeInput, :dt=>StepInput, :celsius=>TemperatureInput, :flag=>EventFlagInput)
    role == :micro &&
        kind != :artificial &&
        return (
            common...,
            :v=>VoltageInput,
            :diam=>DiameterInput,
            :area=>AreaInput,
            :v_init=>InitialVoltageInput,
        )
    role == :macro2macro && return (common..., :weight=>EdgeWeightInput)
    role == :macro2micro &&
        return (common..., :start_time=>WindowStartInput, :stop_time=>WindowStopInput)
    return common
end

execution_variables(role::Symbol, kind::Symbol) = first.(execution_bindings(role, kind))

construction_bindings(role::Symbol) =
    role == :macro2macro ?
    (
        :weight=>ConnectionWeight,
        :delay=>ConnectionDelaySteps,
        :delay_steps=>ConnectionDelaySteps,
        :source_roi=>SourceROI,
        :target_roi=>TargetROI,
    ) : ()

const MOD_INTRINSICS = Set((
    :exp,
    :expm1,
    :log,
    :log1p,
    :log10,
    :sqrt,
    :sin,
    :cos,
    :tan,
    :sinh,
    :cosh,
    :tanh,
    :asin,
    :acos,
    :atan,
    :atan2,
    :abs,
    :fabs,
    :floor,
    :ceil,
    :min,
    :max,
    :pow,
    :fmod,
    :int,
    :erf,
    :erfc,
    :step,
    :heaviside,
    :scop_random,
    :random_uniform,
    :random_normal,
    :random_negexp,
    :random_setseq,
    :random_setids,
    :net_send,
    :net_event,
    :at_time,
    :normrand,
    :exprand,
    :nernst,
    :ghk,
    :nrn_ghk,
    :nrn_nernst,
))

const MOD_OPERATORS = Set(
    Symbol.((
        "+",
        "-",
        "*",
        "/",
        "^",
        "%",
        "<",
        ">",
        "<=",
        ">=",
        "==",
        "!=",
        "&&",
        "||",
        "unary+",
        "unary-",
        "unary!",
        "[]",
    )),
)

function declared_fields(ast::ModAST)
    fields = IRField[]
    fieldmap = Dict{Symbol, Int}()
    offset = 1
    definitions = copy(ast.definitions)
    declarations = copy(ast.declarations)
    language_inputs =
        (execution_variables(ast.role, ast.kind)..., first.(construction_bindings(ast.role))...)
    names = Set(declaration.name for declaration in declarations)
    event_initializer = findfirst(
        block ->
            block.kind == :net_receive &&
            any(statement -> statement isa EventInitialStatement, block.body),
        ast.blocks,
    )
    if event_initializer !== nothing
        # NET_RECEIVE INITIAL reuses the mechanism's private voltage copy from
        # INITIAL, including across multiple incoming connections.
        for (name, input) in execution_bindings(ast.role, ast.kind)
            input == VoltageInput && name ∉ names || continue
            push!(
                declarations,
                Declaration(
                    name,
                    0,
                    :assigned,
                    nothing,
                    "",
                    nothing,
                    ast.blocks[event_initializer].location,
                ),
            )
        end
    end
    # Context values are read from the execution environment; all other names
    # need an explicit declaration, including ion variables and MIND bindings.
    for declaration in declarations
        haskey(fieldmap, declaration.name) &&
            throw(ModError(declaration.location, "duplicate declaration of $(declaration.name)"))
        value =
            declaration.default === nothing ? 0.0 : constant_value(declaration.default, definitions)
        isfinite(value) || throw(ModError(declaration.location, "non-finite default value"))
        declaration.storage == :constant && (definitions[declaration.name] = value)
        name = declaration.name
        language_input = name in language_inputs
        if language_input
            declaration.extent == 0 ||
                throw(ModError(declaration.location, "language input $name is scalar"))
            declaration.storage in (:parameter, :assigned) || throw(
                ModError(
                    declaration.location,
                    "language input $name cannot be declared as $(declaration.storage)",
                ),
            )
        end
        is_ion = any(ion -> name in ion.reads || name in ion.writes, ast.ions)
        is_bound = name in ast.exposures || any(binding -> binding.variable == name, ast.bindings)
        instance_context =
            language_input ||
            is_ion ||
            is_bound ||
            name in ast.currents ||
            name in ast.electrode_currents
        if name in ast.pointers
            declaration.storage == :assigned &&
            !instance_context &&
            name ∉ ast.ranges &&
            name ∉ ast.globals || throw(
                ModError(
                    declaration.location,
                    "POINTER $name requires an unbound ASSIGNED declaration",
                ),
            )
        end
        name in ast.globals &&
            declaration.storage == :state &&
            throw(ModError(declaration.location, "STATE $name must have instance storage"))
        name in ast.globals &&
            (name in ast.ranges || instance_context) &&
            throw(ModError(declaration.location, "GLOBAL $name conflicts with instance storage"))
        scope =
            name in ast.pointers ? :pointer :
            declaration.storage == :local ? :thread :
            name in ast.globals ||
            (declaration.storage == :parameter && name ∉ ast.ranges && !instance_context) ?
            :global : :instance
        push!(
            fields,
            IRField(
                declaration.name,
                offset,
                declaration.extent,
                declaration.storage,
                scope,
                declaration.units,
                value,
                declaration.bounds,
                declaration.location,
            ),
        )
        fieldmap[declaration.name] = length(fields)
        offset += max(1, declaration.extent)
    end
    return fields, fieldmap, definitions
end

function local_symbols(body::Vector{ModStatement})
    names = Set{Symbol}()
    for statement in body
        if statement isa LocalDeclaration
            union!(names, statement.names)
        elseif statement isa Conditional
            union!(names, local_symbols(statement.consequent), local_symbols(statement.alternative))
        elseif statement isa Union{WhileStatement, EventInitialStatement}
            union!(names, local_symbols(statement.body))
        end
    end
    return names
end

function expression_names!(names::Set{Symbol}, expression::ModExpression)
    if expression isa NameExpression
        push!(names, expression.name)
    elseif expression isa OperationExpression
        for argument in expression.arguments
            expression_names!(names, argument)
        end
    end
    return names
end

function base_name(expression::ModExpression)
    expression isa NameExpression && return expression.name
    expression isa OperationExpression &&
        expression.operation == Symbol("[]") &&
        return base_name(first(expression.arguments))
    throw(ModError(expression.location, "assignment target must be a variable"))
end

function contains_solve(body::Vector{ModStatement})
    for statement in body
        statement isa SolveStatement && return true
        if statement isa Conditional
            (contains_solve(statement.consequent) || contains_solve(statement.alternative)) &&
                return true
        elseif statement isa WhileStatement
            contains_solve(statement.body) && return true
        end
    end
    return false
end

"""Fields read or changed by the statements executed in one generated phase.

This liveness pass leaves the source IR intact. In particular, a current probe
does not write the states and rate temporaries of a skipped SOLVE block.
"""
function executed_accesses(
    body::Vector{ModStatement},
    ir::MechanismIR,
    phase::Symbol;
    temporary = local_symbols(body),
    solve = phase != :current,
)
    reads, writes = Set{Symbol}(), Set{Symbol}()
    function expression!(expression)
        if expression isa NameExpression
            expression.name in temporary || push!(reads, expression.name)
            return
        end
        expression isa OperationExpression || return
        if haskey(ir.blocks, expression.operation)
            # Callee effects are already closed over its complete call graph,
            # including recursive strongly connected components.
            union!(reads, ir.blocks[expression.operation].effects.reads)
            union!(writes, ir.blocks[expression.operation].effects.writes)
        elseif expression.operation == :at_time
            union!(reads, (:t, :dt))
        elseif expression.operation in (:nernst, :ghk, :nrn_nernst, :nrn_ghk)
            push!(reads, :celsius)
        end
        foreach(expression!, expression.arguments)
    end
    function statements!(statements)
        for statement in statements
            if statement isa Assignment
                name = base_name(statement.target)
                name in temporary || push!(writes, name)
                statement.target isa OperationExpression &&
                    expression!(statement.target.arguments[2])
                expression!(statement.value)
            elseif statement isa CallStatement
                expression!(statement.expression)
            elseif statement isa Conditional
                expression!(statement.condition)
                statements!(statement.consequent)
                statements!(statement.alternative)
            elseif statement isa WhileStatement
                expression!(statement.condition)
                statements!(statement.body)
            elseif statement isa SolveStatement
                if solve
                    union!(reads, ir.blocks[statement.block].effects.reads)
                    union!(writes, ir.blocks[statement.block].effects.writes)
                end
            elseif statement isa EventInitialStatement && phase == :event_initialize
                statements!(statement.body)
            end
        end
    end
    statements!(body)
    return (; reads, writes)
end

function state_update_statements(body::Vector{ModStatement})
    # NMODL hoists BREAKPOINT's SOLVE statements into the state phase in source
    # order, independently of enclosing current-phase conditions and loops.
    # SOLVE inside a called procedure remains in that procedure's execution.
    statements = ModStatement[]
    for statement in body
        if statement isa SolveStatement
            push!(statements, statement)
        elseif statement isa Conditional
            append!(statements, state_update_statements(statement.consequent))
            append!(statements, state_update_statements(statement.alternative))
        elseif statement isa WhileStatement
            append!(statements, state_update_statements(statement.body))
        end
    end
    return statements
end

has_membrane_current(ir::MechanismIR) =
    !isempty(ir.currents) ||
    !isempty(ir.electrode_currents) ||
    any(ion -> Symbol(:i, ion.name) in ion.writes, ir.ions)

# Unit declarations.

"""Check retained MOD equations, including scale factors and callable signatures.

Like NEURON's separate modlunit tool, this analysis is explicit. `compile_mod` and
`load_mod` also accept `check_units=true`. No implicit numerical conversions are
inserted, and UNITSOFF/UNITSON retain their lexical source meaning.
"""
function check_units(ast::ModAST)
    # 1. Resolve declarations and language inputs, including ion/current contracts.
    for (name, definition) in ast.unit_definitions
        location = SourceLocation(ast.filename, 1, 1)
        mod_unit(ast, definition, location)
        if haskey(MOD_UNIT_BASES, name)
            compatible_units(
                mod_unit(ast, name, location),
                mod_unit(ast, definition, location),
                location,
            )
        end
    end
    environment = Dict(
        declaration.name => mod_unit(ast, declaration.units, declaration.location) for
        declaration in ast.declarations
    )
    location = SourceLocation(ast.filename, 1, 1)
    dimensionless = ModUnit()
    millisecond = mod_unit(ast, "ms", location)
    inputs = Dict(
        TimeInput => "ms",
        StepInput => "ms",
        TemperatureInput => "degC",
        VoltageInput => "mV",
        InitialVoltageInput => "mV",
        DiameterInput => "micron",
        AreaInput => "micron2",
        EventFlagInput => "",
        EdgeWeightInput => "",
        WindowStartInput => "ms",
        WindowStopInput => "ms",
    )
    for (name, input) in execution_bindings(ast.role, ast.kind)
        unit = mod_unit(ast, inputs[input], location)
        if haskey(environment, name)
            compatible_units(environment[name], unit, location; enabled = ast.unit_contracts)
        else
            environment[name] = unit
        end
    end
    if ast.role == :micro
        current = ast.kind == :density ? "mA/cm2" : "nA"
        contracts = Pair{Symbol, String}[
            name => current for name in union(ast.currents, ast.electrode_currents)
        ]
        for ion in ast.ions, name in union(ion.reads, ion.writes)
            unit =
                name == Symbol(:i, ion.name) ? current :
                name == Symbol(:e, ion.name) ? "mV" : "milli/liter"
            push!(contracts, name => unit)
        end
        for (name, units) in contracts
            haskey(environment, name) || throw(ModError(location, "undeclared variable $name"))
            compatible_units(
                environment[name],
                mod_unit(ast, units, location),
                location;
                enabled = ast.unit_contracts,
            )
        end
    end
    for name in keys(ast.definitions)
        environment[name] = ModUnit(literal = true)
    end
    for name in ast.randoms
        environment[name] = dimensionless
    end
    signatures = Dict(
        block.name => (
            [mod_unit(ast, units, block.location) for units in block.argument_units],
            mod_unit(ast, block.units, block.location),
        ) for block in ast.blocks if block.kind in (:function, :procedure)
    )

    # 2. Analyze expressions against contracts. Recursive calls check the same
    # declared signature, so the call graph never needs dimensional unrolling.
    function expression_unit(expression, variables, enabled)
        if expression isa NumberExpression
            expression.units === nothing && return ModUnit(literal = true)
            unit = mod_unit(ast, expression.units, expression.location)
            return ModUnit(
                unit.dimensions,
                unit.scale * expression.unit_factor,
                false,
                unit.wildcard,
            )
        elseif expression isa NameExpression
            haskey(variables, expression.name) ||
                throw(ModError(expression.location, "undeclared variable $(expression.name)"))
            return variables[expression.name]
        end
        op, arguments, loc = expression.operation, expression.arguments, expression.location
        values = [expression_unit(argument, variables, enabled) for argument in arguments]
        match_units(a, b) = compatible_units(a, b, loc; enabled)
        if haskey(signatures, op)
            inputs, output = signatures[op]
            length(inputs) == length(values) ||
                throw(ModError(loc, "argument count mismatch calling $op"))
            foreach(pair -> match_units(pair...), zip(inputs, values))
            return output
        elseif op == Symbol("[]")
            match_units(dimensionless, values[2])
            return values[1]
        elseif op in (Symbol("unary+"), Symbol("unary-"), :abs, :fabs, :floor, :ceil, :int)
            return only(values)
        elseif op in (Symbol("unary!"), Symbol("&&"), Symbol("||"), :step, :heaviside)
            return dimensionless
        elseif op in (:+, :-, :min, :max, :%, :fmod)
            return reduce(match_units, values)
        elseif op in (:<, :>, :<=, :>=, :(==), :(!=))
            match_units(values...)
            return dimensionless
        elseif op == :*
            return multiply_units(values...)
        elseif op == :/
            return multiply_units(values[1], power_units(values[2], -1))
        elseif op in (:^, :pow, :sqrt)
            if op == :sqrt
                return power_units(only(values), 1 // 2)
            end
            match_units(dimensionless, values[2])
            if unit_constant_expression(arguments[2], ast.definitions)
                exponent = constant_value(arguments[2], ast.definitions)
                return power_units(values[1], rationalize(exponent))
            end
            match_units(dimensionless, values[1])
            return dimensionless
        elseif op == :atan2
            match_units(values...)
            return dimensionless
        elseif op in (:net_send, :net_event, :at_time)
            match_units(millisecond, first(values))
            return dimensionless
        elseif op in (:nernst, :nrn_nernst, :ghk, :nrn_ghk)
            expected =
                op in (:nernst, :nrn_nernst) ? ("milli/liter", "milli/liter", "") :
                ("mV", "milli/liter", "milli/liter", "")
            length(expected) == length(values) ||
                throw(ModError(loc, "argument count mismatch calling $op"))
            for (units, value) in zip(expected, values)
                match_units(mod_unit(ast, units, loc), value)
            end
            return mod_unit(ast, op in (:nernst, :nrn_nernst) ? "mV" : "millicoulomb/cm3", loc)
        elseif op in (:random_uniform, :random_normal, :random_negexp, :normrand, :exprand)
            if startswith(String(op), "random_")
                match_units(dimensionless, first(values))
            end
            if op == :random_uniform && length(values) == 3 ||
               op in (:random_normal, :normrand) && length(values) >= 2
                return match_units(values[end - 1], values[end])
            elseif op == :random_negexp && length(values) == 2 || op == :exprand && !isempty(values)
                return last(values)
            end
            return dimensionless
        end
        op in MOD_INTRINSICS || throw(ModError(loc, "unknown function $op"))
        foreach(value -> match_units(dimensionless, value), values)
        return dimensionless
    end

    # 3. Follow lexical statements. As in modlunit, LOCAL acquires the unit of
    # each assignment; scoped locals do not overwrite the enclosing declaration.
    function statements(body, variables, local_names)
        shadowed = Dict{Symbol, Union{Nothing, ModUnit}}()
        local_names = copy(local_names)
        for statement in body
            statement isa LocalDeclaration || continue
            for name in statement.names
                shadowed[name] = get(variables, name, nothing)
                variables[name] = dimensionless
                push!(local_names, name)
            end
        end
        for statement in body
            enabled = get(ast.unit_checks, statement, true)
            if statement isa Assignment
                value = expression_unit(statement.value, variables, enabled)
                name = base_name(statement.target)
                if name in local_names && !statement.derivative
                    if statement.target isa OperationExpression
                        index_unit =
                            expression_unit(statement.target.arguments[2], variables, enabled)
                        compatible_units(dimensionless, index_unit, statement.location; enabled)
                    end
                    variables[name] = ModUnit(value.dimensions, value.scale, false, value.wildcard)
                else
                    target = expression_unit(statement.target, variables, enabled)
                    statement.derivative &&
                        (target = multiply_units(target, power_units(millisecond, -1)))
                    compatible_units(target, value, statement.location; enabled)
                end
            elseif statement isa CallStatement
                expression_unit(statement.expression, variables, enabled)
            elseif statement isa Conditional
                expression_unit(statement.condition, variables, enabled)
                statements(statement.consequent, variables, local_names)
                statements(statement.alternative, variables, local_names)
            elseif statement isa WhileStatement
                expression_unit(statement.condition, variables, enabled)
                statements(statement.body, variables, local_names)
            elseif statement isa EventInitialStatement
                statements(statement.body, variables, local_names)
            end
        end
        for (name, previous) in shadowed
            previous === nothing ? delete!(variables, name) : (variables[name] = previous)
        end
        return nothing
    end
    for declaration in ast.declarations
        declaration.default === nothing && continue
        enabled = get(ast.unit_checks, declaration, true)
        value = expression_unit(declaration.default, environment, enabled)
        compatible_units(environment[declaration.name], value, declaration.location; enabled)
    end
    for block in ast.blocks
        variables = copy(environment)
        for (name, units) in zip(block.arguments, block.argument_units)
            variables[name] = mod_unit(ast, units, block.location)
        end
        block.kind == :function && (variables[block.name] = signatures[block.name][2])
        statements(
            block.body,
            variables,
            Set(decl.name for decl in ast.declarations if decl.storage == :local),
        )
    end
    return true
end

check_units(source::AbstractString; filename::AbstractString = "<string>") =
    check_units(parse_mod(source; filename))

function check_units(mechanism::CompiledMechanism)
    ir = mechanism.ir
    sources = copy(ir.sources)
    tokens = read_mod_tokens(ir.filename, Set{String}(), sources)
    return check_units(parse_mod(ir.source; filename = ir.filename, tokens, sources))
end

# Types, scopes and transitive effects.

function type_expression!(types, expression::ModExpression, symbols, functions, arrays)
    if expression isa NumberExpression
        result = Real64
    elseif expression isa NameExpression
        expression.name in symbols ||
            throw(ModError(expression.location, "undeclared variable $(expression.name)"))
        result = expression.name in arrays ? RealArray : Real64
    else
        operation = expression.operation
        operation in MOD_OPERATORS ||
            operation in MOD_INTRINSICS ||
            operation in functions ||
            throw(ModError(expression.location, "unknown function $operation"))
        argument_types = [
            type_expression!(types, argument, symbols, functions, arrays) for
            argument in expression.arguments
        ]
        if operation == Symbol("[]")
            argument_types[1] == RealArray && argument_types[2] != RealArray || throw(
                ModError(
                    expression.location,
                    "array indexing requires an array and a scalar index",
                ),
            )
        elseif RealArray in argument_types
            throw(
                ModError(
                    expression.location,
                    "MOD arithmetic and calls require scalar array elements",
                ),
            )
        end
        result =
            operation in Symbol.(("<", ">", "<=", ">=", "==", "!=", "&&", "||", "unary!")) ?
            Boolean : Real64
    end
    types[expression] = result
    return result
end

function inspect_statements!(
    types,
    reads,
    writes,
    calls,
    body,
    symbols,
    functions,
    ast,
    block,
    arrays,
)
    for statement in body
        expressions = ModExpression[]
        if statement isa Assignment
            name = base_name(statement.target)
            name in symbols ||
                throw(ModError(statement.location, "undeclared assignment target $name"))
            if statement.derivative
                block.kind == :derivative ||
                    throw(ModError(statement.location, "derivative assignment outside DERIVATIVE"))
                any(
                    declaration -> declaration.name == name && declaration.storage == :state,
                    ast.declarations,
                ) || throw(ModError(statement.location, "derivative of non-STATE variable $name"))
            end
            push!(writes, name)
            type_expression!(types, statement.target, symbols, functions, arrays) != RealArray ||
                throw(ModError(statement.location, "assignment requires a scalar array element"))
            statement.target isa OperationExpression &&
                push!(expressions, statement.target.arguments[2])
            push!(expressions, statement.value)
        elseif statement isa CallStatement
            push!(expressions, statement.expression)
        elseif statement isa Conditional
            push!(expressions, statement.condition)
            inspect_statements!(
                types,
                reads,
                writes,
                calls,
                statement.consequent,
                symbols,
                functions,
                ast,
                block,
                arrays,
            )
            inspect_statements!(
                types,
                reads,
                writes,
                calls,
                statement.alternative,
                symbols,
                functions,
                ast,
                block,
                arrays,
            )
        elseif statement isa WhileStatement
            push!(expressions, statement.condition)
            inspect_statements!(
                types,
                reads,
                writes,
                calls,
                statement.body,
                symbols,
                functions,
                ast,
                block,
                arrays,
            )
        elseif statement isa EventInitialStatement
            inspect_statements!(
                types,
                reads,
                writes,
                calls,
                statement.body,
                symbols,
                functions,
                ast,
                block,
                arrays,
            )
        elseif statement isa SolveStatement
            push!(calls, statement.block)
            any(other -> other.name == statement.block, ast.blocks) ||
                throw(ModError(statement.location, "unknown SOLVE block $(statement.block)"))
            statement.method in (:euler, :cnexp, :after_cvode) || throw(
                ModError(statement.location, "unknown integration method $(statement.method)"),
            )
            if statement.method == :after_cvode
                equation = only(filter(other -> other.name == statement.block, ast.blocks))
                equation.kind == :procedure || throw(
                    ModError(
                        statement.location,
                        "after_cvode in fixed-step execution requires a PROCEDURE",
                    ),
                )
            end
        end
        for expression in expressions
            type_expression!(types, expression, symbols, functions, arrays) != RealArray ||
                throw(ModError(expression.location, "statement requires a scalar array element"))
            expression_names!(reads, expression)
            expression_calls!(calls, expression)
        end
    end
    return nothing
end

function expression_calls!(calls::Set{Symbol}, expression::ModExpression)
    if expression isa OperationExpression
        push!(calls, expression.operation)
        for argument in expression.arguments
            expression_calls!(calls, argument)
        end
    end
end

function analyze_mod(ast::ModAST)
    fields, fieldmap, definitions = declared_fields(ast)
    symbols = union(
        Set(keys(fieldmap)),
        Set(execution_variables(ast.role, ast.kind)),
        Set(keys(definitions)),
        Set(ast.randoms),
    )
    functions = Set(block.name for block in ast.blocks if block.kind in (:function, :procedure))
    types = IdDict{ModExpression, ValueType}()
    blocks = Dict{Symbol, IRBlock}()
    local_bindings = Dict{Symbol, Vector{LocalBinding}}()
    block_calls = Dict{Symbol, Set{Symbol}}()
    for source_block in ast.blocks
        block, bindings = resolve_block_locals(source_block)
        local_bindings[block.name] = bindings
        haskey(blocks, block.name) &&
            throw(ModError(block.location, "duplicate block $(block.name)"))
        temporary = union(Set(block.arguments), local_symbols(block.body))
        block.kind == :function && push!(temporary, block.name)
        locals = union(symbols, temporary)
        reads, writes, calls = Set{Symbol}(), Set{Symbol}(), Set{Symbol}()
        arrays = Set(field.name for field in fields if field.extent > 0 && field.name ∉ temporary)
        union!(arrays, (binding.name for binding in bindings if binding.extent > 0))
        inspect_statements!(
            types,
            reads,
            writes,
            calls,
            block.body,
            locals,
            functions,
            ast,
            block,
            arrays,
        )
        # Remove lexical bindings before adding hidden simulator inputs: a
        # function argument cannot shadow an intrinsic's time or temperature.
        setdiff!(reads, temporary)
        setdiff!(writes, temporary)
        # Intrinsics and integration read execution inputs even when their MOD
        # spelling does not mention them. Retain these dependencies in the IR.
        if :at_time in calls && :at_time ∉ functions
            union!(reads, (:t, :dt))
        end
        any(name -> name in calls && name ∉ functions, (:nernst, :ghk, :nrn_nernst, :nrn_ghk)) &&
            push!(reads, :celsius)
        block.kind == :derivative && push!(reads, :dt)
        # Resolve callee effects afterward so their global accesses survive
        # the caller's lexical shadowing as well.
        random = any(
            in(calls),
            (
                :random_uniform,
                :random_normal,
                :random_negexp,
                :random_setseq,
                :random_setids,
                :scop_random,
                :normrand,
                :exprand,
            ),
        )
        effects = BlockEffects(
            reads,
            writes,
            random,
            :net_send in calls,
            :net_event in calls,
            block.kind == :net_receive,
        )
        blocks[block.name] = IRBlock(
            block.kind,
            block.name,
            block.arguments,
            block.argument_units,
            block.units,
            block.body,
            effects,
            block.location,
        )
        block_calls[block.name] = calls
    end
    resolve_effects!(blocks, block_calls)
    writes = Set{Symbol}()
    foreach(block -> union!(writes, block.effects.writes), values(blocks))
    for (index, field) in enumerate(fields)
        field.category == :constant &&
            field.name in writes &&
            throw(ModError(field.location, "CONSTANT $(field.name) cannot be assigned"))
        if field.scope == :global && ast.threadsafe && field.name in writes
            # NMODL's writable thread scratch starts at zero, even when the
            # original PARAMETER declaration has a nonzero global default.
            fields[index] = IRField(
                field.name,
                field.offset,
                field.extent,
                field.category,
                :thread,
                field.units,
                0.0,
                field.bounds,
                field.location,
            )
        end
    end
    for name in union(
        ast.ranges,
        ast.globals,
        ast.pointers,
        ast.currents,
        ast.electrode_currents,
        ast.exposures,
    )
        name in symbols ||
            throw(ModError(SourceLocation(ast.filename, 1, 1), "undeclared public variable $name"))
    end
    for binding in ast.bindings
        binding.variable in symbols ||
            throw(ModError(binding.location, "undeclared bound variable $(binding.variable)"))
    end
    for ion in ast.ions
        for name in union(ion.reads, ion.writes)
            name in symbols || throw(ModError(ion.location, "undeclared ion variable $name"))
        end
    end
    return MechanismIR(
        ast.name,
        ast.kind,
        ast.role,
        ast.threadsafe,
        collect(execution_bindings(ast.role, ast.kind)),
        Pair{Int, ConstructionInput}[
            fields[fieldmap[name]].offset => input for
            (name, input) in construction_bindings(ast.role) if haskey(fieldmap, name)
        ],
        fields,
        fieldmap,
        blocks,
        recursive_call_graph(block_calls, functions),
        local_bindings,
        types,
        ast.currents,
        ast.electrode_currents,
        ast.ions,
        ast.randoms,
        ast.exposures,
        ast.bindings,
        definitions,
        source_digest(ast.sources, ast.filename),
        ast.source,
        ast.filename,
        ast.sources,
    )
end

function resolve_effects!(blocks, calls)
    # Monotone closure handles recursive and mutually recursive call graphs.
    # Each iteration only adds storage/effects; the finite IR bounds convergence.
    changed = true
    while changed
        changed = false
        for name in keys(blocks)
            block = blocks[name]
            own = block.effects
            reads, writes = copy(own.reads), copy(own.writes)
            random, schedules, emits, discontinuous =
                own.random, own.schedules_events, own.emits_spikes, own.discontinuous
            for call in calls[name]
                haskey(blocks, call) || continue
                nested = blocks[call].effects
                union!(reads, nested.reads)
                union!(writes, nested.writes)
                random |= nested.random
                schedules |= nested.schedules_events
                emits |= nested.emits_spikes
                discontinuous |= nested.discontinuous
            end
            different =
                reads != own.reads ||
                writes != own.writes ||
                random != own.random ||
                schedules != own.schedules_events ||
                emits != own.emits_spikes ||
                discontinuous != own.discontinuous
            different || continue
            effects = BlockEffects(reads, writes, random, schedules, emits, discontinuous)
            blocks[name] = IRBlock(
                block.kind,
                block.name,
                block.arguments,
                block.argument_units,
                block.units,
                block.body,
                effects,
                block.location,
            )
            changed = true
        end
    end
    return nothing
end
