# Compile resolved IR into specialized execution phases.

mutable struct CodegenContext
    ir::MechanismIR
    variables::Dict{Symbol, Any}
    derivatives::Dict{Symbol, Any}
    phase::Symbol
    globals::Dict{Symbol, Any}
    solver::Symbol
    calls::Dict{Symbol, Int}
end

function compile_ir(ir::MechanismIR)
    # A concrete tuple carries the call graph, including self and mutual calls.
    # It needs no process-global method definitions or manually expanded recursion.
    names = sort!([name for (name, block) in ir.blocks if block.kind in (:function, :procedure)])
    indices = Dict{Symbol, Int}(name => index for (index, name) in enumerate(names))
    calls = Tuple(generated_callable(ir, name, indices) for name in names)
    return CompiledMechanism(
        ir,
        GeneratedPhase(generated_phase(ir, :initialize, indices), calls),
        GeneratedPhase(generated_phase(ir, :current, indices), calls),
        GeneratedPhase(generated_phase(ir, :advance, indices), calls),
        GeneratedPhase(generated_phase(ir, :receive, indices), calls),
        GeneratedPhase(generated_phase(ir, :event_initialize, indices), calls),
    )
end

# Expression and statement generation.

function julia_expression(expression::ModExpression, context::CodegenContext)
    if expression isa NumberExpression
        return expression.value
    elseif expression isa NameExpression
        haskey(context.variables, expression.name) && return context.variables[expression.name]
        haskey(context.ir.definitions, expression.name) &&
            return context.ir.definitions[expression.name]
        expression.name in context.ir.randoms &&
            return findfirst(==(expression.name), context.ir.randoms)
        throw(ModError(expression.location, "unresolved variable $(expression.name)"))
    end
    operation = expression.operation
    arguments = expression.arguments
    if haskey(context.ir.blocks, operation) &&
       context.ir.blocks[operation].kind in (:function, :procedure)
        return mod_call(operation, arguments, context)
    end
    values = [julia_expression(argument, context) for argument in arguments]
    if operation == Symbol("[]")
        # NMODL indices are zero based; Julia's storage views are one based.
        return :($(values[1])[Int(trunc($(values[2]))) + 1])
    elseif operation == Symbol("unary-")
        return Expr(:call, :-, only(values))
    elseif operation == Symbol("unary+")
        return only(values)
    elseif operation == Symbol("unary!")
        return :(ifelse(iszero($(only(values))), 1.0, 0.0))
    elseif operation in (Symbol("&&"), Symbol("||"))
        condition = Expr(operation, :(!iszero($(values[1]))), :(!iszero($(values[2]))))
        return :(ifelse($condition, 1.0, 0.0))
    elseif operation in (:<, :>, :<=, :>=, :(==), :(!=))
        # MOD logical results also enter scalar equations and function arguments.
        # Preserve short-circuit evaluation above, with numerical FP64 results.
        return :(ifelse($(Expr(:call, operation, values...)), 1.0, 0.0))
    elseif operation == :net_send
        return :(schedule_self!(_context, _instance, $(values[1]), $(values[2])))
    elseif operation == :net_event
        return :(emit_spike!(_context, _instance, $(values[1])))
    elseif operation in (:random_uniform, :scop_random)
        stream = isempty(values) ? 1 : values[1]
        uniform = :(uniform_random!(_context, _instance, $stream))
        if length(values) == 3
            lower, upper = gensym(:lower), gensym(:upper)
            # Arguments may call MOD functions with effects. Evaluate each once,
            # before drawing, as for an ordinary function call.
            return quote
                $lower = $(values[2])
                $upper = $(values[3])
                $lower + $uniform * ($upper - $lower)
            end
        end
        return uniform
    elseif operation in (:random_negexp, :exprand)
        stream = operation == :exprand ? 1 : first(values)
        scale =
            operation == :exprand && !isempty(values) ? values[1] :
            length(values) == 2 ? values[2] : 1.0
        value = gensym(:scale)
        return quote
            $value = $scale
            -log(uniform_random!(_context, _instance, $stream)) * $value
        end
    elseif operation in (:random_normal, :normrand)
        stream = operation == :normrand ? 1 : first(values)
        average = length(values) >= 2 ? values[end - 1] : 0.0
        deviation = length(values) >= 2 ? values[end] : 1.0
        return :($average + $deviation * normal_random!(_context, _instance, $stream))
    elseif operation == :random_setseq
        return :(set_random_sequence!(_context, _instance, $(values[1]), $(values[2])))
    elseif operation == :random_setids
        return :(set_random_ids!(_context, _instance, $(values...)))
    elseif operation == :at_time
        # Native intrinsics read simulator inputs, independently of MOD arguments
        # or locals named t, dt or celsius.
        time = :(_context.t)
        dt = :(_context.dt)
        event = gensym(:event_time)
        return quote
            $event = $(only(values)) - 1.0e-11
            ifelse($event <= $time && $event > $time - $dt, 1.0, 0.0)
        end
    elseif operation in (:nernst, :ghk, :nrn_nernst, :nrn_ghk)
        function_ = operation in (:nernst, :nrn_nernst) ? :nernst_potential : :ghk_current
        temperature = :(_context.celsius)
        return Expr(:call, function_, values..., temperature)
    elseif operation in (:step, :heaviside)
        return :(ifelse($(values[1]) >= 0.0, 1.0, 0.0))
    elseif operation == :int
        return :(trunc($(values[1])))
    end
    operation =
        operation == :fabs ? :abs :
        operation == :pow ? :^ : operation == :atan2 ? :atan : operation == :fmod ? :rem : operation
    # Preserve integer powers in the generated expression; values remain FP64.
    if operation == :^ && length(arguments) == 2 && arguments[2] isa NumberExpression
        power = arguments[2].value
        isinteger(power) && abs(power) <= 64 && (values[2] = Int(power))
    end
    return Expr(:call, operation, values...)
end

function local_initializers!(body::Vector{Any}, variables, declarations)
    for declaration in declarations
        name = declaration.name
        symbol = gensym(name)
        variables[name] = symbol
        value =
            declaration.extent == 0 ? 0.0 :
            :(MVector{$(declaration.extent), Float64}(ntuple(_ -> 0.0, Val($(declaration.extent)))))
        push!(body, :($symbol = $value))
    end
end

"""Call a MOD function with concrete live fields and return its scalar updates.

Array views and POINTER aliases retain identity. Ordinary scalar fields travel
in a concrete tuple, which Julia can keep in registers after inlining. Recursive
calls use the same interface and the native device call stack.
"""
function mod_call(name::Symbol, arguments::Vector{ModExpression}, context::CodegenContext)
    block = context.ir.blocks[name]
    length(arguments) == length(block.arguments) ||
        throw(ModError(block.location, "argument count mismatch calling $name"))
    body, values = Any[], Any[]
    # Evaluate argument effects before taking the callee's live field snapshot.
    for argument in arguments
        value = gensym(:argument)
        push!(body, :($value = $(julia_expression(argument, context))))
        push!(values, value)
    end
    fields = callable_fields(context.ir, block)
    frame = Expr(
        :tuple,
        (
            name in fields.aliases ? callable_value(context.globals[name]) : context.globals[name] for name in fields.names
        )...,
    )
    result, updated = gensym(:result), gensym(:fields)
    execution =
        isempty(fields.context) ? nothing :
        Expr(
            :tuple,
            Expr(
                :parameters,
                (
                    Expr(:kw, name, Expr(:., :_context, QuoteNode(name))) for name in fields.context
                )...,
            ),
        )
    call = Expr(
        :call,
        Expr(:ref, :_calls, context.calls[name]),
        :_calls,
        frame,
        :_instance,
        execution,
        values...,
    )
    # Acyclic call graphs can address the generated body for inlining. Keep
    # recursive graphs on the callable interface that preserves native frames.
    if !context.ir.recursive
        pushfirst!(call.args, GlobalRef(RuntimeGeneratedFunctions, :generated_callfunc))
    end
    push!(body, :(($result, $updated) = @inline $call))
    for (index, name) in enumerate(fields.updates)
        variable = context.globals[name]
        # A caller using immediate storage may call a function with no alias
        # access. Commit that function's scalar results before its next statement.
        push!(body, :($variable = $updated[$index]))
    end
    push!(body, result)
    return Expr(:block, body...)
end

function callable_fields(ir::MechanismIR, block::IRBlock)
    # Effects already include transitive calls. Preserve declaration order and
    # written fields' entry values for conditional writes and recursive calls.
    effects = block.effects
    live(name) = name in effects.reads || name in effects.writes
    direct = any(field -> field.scope == :pointer && live(field.name), ir.fields)
    names, updates = Symbol[], Symbol[]
    aliases = Set{Symbol}()
    for name in unique([getfield.(ir.fields, :name); first.(ir.inputs)])
        live(name) || continue
        push!(names, name)
        field = haskey(ir.fieldmap, name) ? fieldinfo(ir, name) : nothing
        field !== nothing && field.extent > 0 && continue
        if direct &&
           field !== nothing &&
           all(
               binding ->
                   first(binding) != name || last(binding) ∉ (EventFlagInput, EdgeWeightInput),
               ir.inputs,
           )
            push!(aliases, name)
        elseif name in effects.writes
            push!(updates, name)
        end
    end
    # Ordinary fields and aliases already travel in the live frame. Only
    # intrinsics need execution context; avoid copying the device queues and
    # storage descriptors through every numerical or recursive call.
    context = Symbol[]
    for (name, input) in ir.inputs
        name in effects.reads || continue
        input == TimeInput && push!(context, :t)
        input == StepInput && push!(context, :dt)
        input == TemperatureInput && push!(context, :celsius)
    end
    effects.random && append!(context, (:rng_counters, :rng_keys, :rng_words, :seed))
    if ir.role == :micro &&
       :dt in effects.reads &&
       any(block -> contains_euler(block.body), values(ir.blocks))
        push!(context, :secondorder)
    end
    effects.schedules_events && append!(context, (:targets, :outbox, :errors, :t, :edge))
    effects.emits_spikes && append!(
        context,
        (
            :targets,
            :target_offsets,
            :target_sources,
            :spikes,
            :macro_spikes,
            :outbox,
            :errors,
            :edge,
        ),
    )
    unique!(context)
    return (; names, aliases, updates, context)
end

callable_value(variable) = variable isa Expr && variable.head == :ref ? variable.args[1] : variable

function generated_callable(ir::MechanismIR, name::Symbol, calls)
    block = ir.blocks[name]
    fields = callable_fields(ir, block)
    variables, derivatives = Dict{Symbol, Any}(), Dict{Symbol, Any}()
    # The annotation must be in the generated body itself. Otherwise the GPU
    # may retain calls that copy the whole live-field tuple through local memory.
    # Julia still keeps genuine recursive calls on the native device stack.
    body = Any[Expr(:meta, :inline)]
    for (index, name) in enumerate(fields.names)
        symbol = gensym(name)
        push!(body, :($symbol = _values[$index]))
        variables[name] = name in fields.aliases ? :($symbol[1]) : symbol
    end
    for field in ir.fields
        field.category == :state && field.name in block.effects.writes || continue
        derivative = gensym(Symbol(:derivative_, field.name))
        derivatives[field.name] = derivative
        value =
            field.extent == 0 ? 0.0 :
            :(MVector{$(field.extent), Float64}(ntuple(_ -> 0.0, Val($(field.extent)))))
        push!(body, :($derivative = $value))
    end
    globals = copy(variables)
    parameters = Symbol[]
    for argument in block.arguments
        variable = gensym(argument)
        push!(parameters, variable)
        variables[argument] = variable
    end
    local_initializers!(body, variables, ir.locals[block.name])
    result = block.kind == :function ? gensym(block.name) : 1.0
    if block.kind == :function
        variables[block.name] = result
        push!(body, :($result = 0.0))
    end
    context = CodegenContext(ir, variables, derivatives, :call, globals, :none, calls)
    append!(body, julia_statements(block.body, context))
    updated = Expr(:tuple, (globals[name] for name in fields.updates)...)
    # A concrete return contract also terminates recursive GPU type inference.
    result_type = :(Tuple{Float64, NTuple{$(length(fields.updates)), Float64}})
    push!(body, :(return (Float64($result), $updated)::$result_type))
    signature = Expr(
        :tuple,
        :_calls,
        :_values,
        :_instance,
        :_context,
        (Expr(:(::), parameter, :Float64) for parameter in parameters)...,
    )
    expression = Expr(:function, signature, Expr(:block, body...))
    return RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, expression) |> drop_expr
end

function julia_statements(statements::Vector{ModStatement}, context::CodegenContext)
    body = Any[]
    for statement in statements
        if statement isa Assignment
            if statement.derivative && context.solver == :cnexp
                append!(body, cnexp_statement(statement, context))
                continue
            end
            target =
                statement.derivative ? context.derivatives[base_name(statement.target)] :
                julia_expression(statement.target, context)
            if statement.derivative && statement.target isa OperationExpression
                index = julia_expression(statement.target.arguments[2], context)
                target = :($target[Int(trunc($index)) + 1])
            end
            push!(body, Expr(:(=), target, julia_expression(statement.value, context)))
        elseif statement isa CallStatement
            push!(body, julia_expression(statement.expression, context))
        elseif statement isa Conditional
            condition = julia_expression(statement.condition, context)
            consequent = Expr(:block, julia_statements(statement.consequent, context)...)
            alternative = Expr(:block, julia_statements(statement.alternative, context)...)
            push!(body, Expr(:if, :(!iszero($condition)), consequent, alternative))
        elseif statement isa WhileStatement
            condition = julia_expression(statement.condition, context)
            nested = Expr(:block, julia_statements(statement.body, context)...)
            push!(body, Expr(:while, :(!iszero($condition)), nested))
        elseif statement isa SolveStatement
            context.phase != :current && append!(body, emit_solve(statement, context))
        elseif statement isa EventInitialStatement
            context.phase == :event_initialize &&
                append!(body, julia_statements(statement.body, context))
        elseif statement isa LocalDeclaration
            nothing
        else
            throw(
                ModError(statement.location, "statement has not been lowered for code generation"),
            )
        end
    end
    return body
end

function generated_phase(ir::MechanismIR, phase::Symbol, calls)
    name =
        phase == :initialize ? :initial :
        phase in (:receive, :event_initialize) ? :net_receive : :breakpoint
    block = get(ir.blocks, name, nothing)
    statements =
        block === nothing ? ModStatement[] :
        phase == :event_initialize ?
        ModStatement[statement for statement in block.body if statement isa EventInitialStatement] :
        block.body
    if phase == :advance && ir.role == :micro
        equations = state_update_statements(statements)
        # Native nrn_state executes hoisted SOLVEs first. A mechanism without
        # membrane currents then executes the rest of BREAKPOINT as well.
        statements =
            has_membrane_current(ir) ? equations : [equations; breakpoint_remainder(statements)]
    end
    temporary =
        block === nothing ? Set{Symbol}() : union(Set(block.arguments), local_symbols(block.body))
    accesses = executed_accesses(statements, ir, phase; temporary)
    writes = accesses.writes
    direct = any(
        field ->
            field.scope == :pointer && (field.name in accesses.reads || field.name in writes),
        ir.fields,
    )
    variables = Dict{Symbol, Any}()
    derivatives = Dict{Symbol, Any}()
    body = Any[Expr(:meta, :inline)]
    if phase == :advance &&
       ir.role == :micro &&
       block !== nothing &&
       contains_solve(block.body) &&
       any(block -> contains_euler(block.body), values(ir.blocks))
        push!(body, :(
            if _context.secondorder != 0
                _context = context_at(_context; dt = 0.5 * _context.dt)
            end
        ))
    end
    voltage_copy = has_event_initializer(ir) && phase in (:initialize, :event_initialize)
    phase == :receive && push!(body, :(receive_ions!(_data, _instance, _context, Val(false))))
    for field in ir.fields
        symbol = gensym(field.name)
        variables[field.name] = symbol
        if direct || field.extent > 0 || field.scope == :pointer
            storage =
                field.scope == :pointer ?
                :(AliasField(_context.references, _instance, $(field.offset))) :
                field.scope == :global ? :(DirectField(_context.shared, 0, $(field.offset))) :
                :(DirectField(_data, _instance, $(field.offset)))
            push!(body, :($symbol = $storage))
            variables[field.name] = field.extent == 0 ? :($symbol[1]) : symbol
        else
            value =
                field.scope == :global ? :(_context.shared[$(field.offset)]) :
                :(_data[_instance, $(field.offset)])
            push!(body, :($symbol = @inbounds $value))
        end
        if field.category == :state
            derivative = gensym(Symbol(:derivative_, field.name))
            derivatives[field.name] = derivative
            initial =
                field.extent == 0 ? 0.0 :
                :(MVector{$(field.extent), Float64}(ntuple(_ -> 0.0, Val($(field.extent)))))
            push!(body, :($derivative = $initial))
        end
    end
    for (name, input) in ir.inputs
        if input == EventFlagInput
            variables[name] = :_flag
        elseif input == EdgeWeightInput
            variables[name] = :_weight
        else
            voltage_copy && phase == :event_initialize && input == VoltageInput && continue
            symbol = get!(variables, name, gensym(name))
            push!(body, :($symbol = context_value(_context, Val($input), _instance)))
        end
    end
    globals = copy(variables)
    if block !== nothing
        local_initializers!(body, variables, ir.locals[block.name])
        arguments = block.arguments
        for (index, argument) in enumerate(arguments)
            variable = gensym(argument)
            variables[argument] = variable
            push!(body, :($variable = event_weight(_context, _weight, $index)))
        end
        context = CodegenContext(ir, variables, derivatives, phase, globals, :none, calls)
        append!(body, julia_statements(statements, context))
        for (index, argument) in enumerate(arguments)
            push!(body, :(set_event_weight!(_context, $index, $(variables[argument]))))
        end
    end
    # NEURON commits its voltage copy at the end of NET_RECEIVE. INITIAL,
    # current probes keep their local voltage assignment.
    if phase == :receive
        for (name, input) in ir.inputs
            input == VoltageInput && name in writes || continue
            push!(body, :(@inbounds _context.voltage[_context.nodes[_instance]] = $(globals[name])))
        end
    end
    for field in ir.fields
        (direct || field.extent > 0 || field.scope == :pointer) && continue
        saved_voltage = voltage_copy && (field.name => VoltageInput) in ir.inputs
        field.name in writes || saved_voltage || continue
        any(binding -> first(binding) == field.name, ir.inputs) && !saved_voltage && continue
        symbol = globals[field.name]
        destination =
            field.scope == :global ? :(_context.shared[$(field.offset)]) :
            :(_data[_instance, $(field.offset)])
        push!(body, :(@inbounds $destination = $symbol))
    end
    phase == :receive && push!(body, :(receive_ions!(_data, _instance, _context, Val(true))))
    push!(body, :(return nothing))
    expression = Expr(
        :function,
        Expr(:tuple, :_calls, :_data, :_instance, :_context, :_weight, :_flag),
        Expr(:block, body...),
    )
    # Keep the inspectable equation IR on the host; only the cached function ID
    # crosses the CUDA boundary. An Expr contains references and is not isbits.
    return RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, expression) |> drop_expr
end

"""Stable exponential Euler factor, including the zero-rate limit."""
@inline function exponential_factor(value::Float64)
    if abs(value) < 1.0e-7
        return 1.0 + value * (0.5 + value * (1.0 / 6.0 + value / 24.0))
    end
    return expm1(value) / value
end

# Native scalar cnexp coefficients.

"""One RHS value and its coefficient in the equation's own state.

Names assigned before this equation and function returns are frozen inputs,
matching CoreNEURON's default NeuronSolveVisitor/DiffeqDriver. This local pair
is eliminated by scalar CUDA compilation; it is not an AD state or buffer.
"""
struct AffineRate
    value::Float64
    coefficient::Float64
end

AffineRate(value::Real) = AffineRate(value, 0.0)
Base.:+(a::AffineRate, b::AffineRate) = AffineRate(a.value + b.value, a.coefficient + b.coefficient)
Base.:-(a::AffineRate, b::AffineRate) = AffineRate(a.value - b.value, a.coefficient - b.coefficient)
Base.:+(a::AffineRate) = a
Base.:-(a::AffineRate) = AffineRate(-a.value, -a.coefficient)
Base.:*(a::AffineRate, b::AffineRate) =
    AffineRate(a.value * b.value, a.coefficient * b.value + a.value * b.coefficient)
Base.:/(a::AffineRate, b::AffineRate) =
    AffineRate(a.value / b.value, (a.coefficient - a.value / b.value * b.coefficient) / b.value)

function cnexp_degree(expression::ModExpression, state::ModExpression)
    expression isa NumberExpression && return 0
    expression isa NameExpression && return Int(expression.name == base_name(state))
    if expression.operation == Symbol("[]")
        base_name(expression) == base_name(state) || return 0
        first_index, second_index = expression.arguments[2], state.arguments[2]
        # Constant distinct array elements are independent. Dynamic indices may
        # name the same element and are resolved when evaluating the coefficient.
        return first_index isa NumberExpression && second_index isa NumberExpression ?
               Int(first_index.value == second_index.value) : 1
    end
    name, arguments = expression.operation, expression.arguments
    degrees = map(argument -> cnexp_degree(argument, state), arguments)
    # The native solver freezes calls, but cannot solve a call whose explicit
    # arguments contain this equation's state. Precomputed names remain inputs.
    name in (:+, :-, :*, :/, Symbol("unary+"), Symbol("unary-")) ||
        return any(!iszero, degrees) ? 2 : 0
    name in (:+, :-, Symbol("unary+"), Symbol("unary-")) && return maximum(degrees; init = 0)
    name == :* && return min(2, sum(degrees))
    return degrees[2] == 0 ? degrees[1] : 2
end

function cnexp_rate(expression::ModExpression, state::ModExpression, context::CodegenContext)
    if expression isa NameExpression
        value = julia_expression(expression, context)
        return :(AffineRate($value, $(Float64(expression.name == base_name(state)))))
    elseif expression isa OperationExpression && expression.operation == Symbol("[]")
        value = julia_expression(expression, context)
        base_name(expression) == base_name(state) || return :(AffineRate($value))
        index = julia_expression(expression.arguments[2], context)
        target = julia_expression(state.arguments[2], context)
        array = julia_expression(expression.arguments[1], context)
        element = gensym(:element)
        return quote
            $element = Int(trunc($index))
            AffineRate($array[$element + 1], Float64($element == $target))
        end
    elseif expression isa OperationExpression &&
           expression.operation in (:+, :-, :*, :/, Symbol("unary+"), Symbol("unary-"))
        name = expression.operation
        operator = name == Symbol("unary+") ? :+ : name == Symbol("unary-") ? :- : name
        return Expr(
            :call,
            operator,
            (cnexp_rate(argument, state, context) for argument in expression.arguments)...,
        )
    end
    # Calls run exactly once in ordinary FP64 scope, in expression order.
    return :(AffineRate($(julia_expression(expression, context))))
end

function cnexp_statement(statement::Assignment, context::CodegenContext)
    name = base_name(statement.target)
    cnexp_degree(statement.value, statement.target) <= 1 ||
        throw(ModError(statement.location, "cnexp requires an equation affine in its own state"))
    # Evaluate a dynamic target index once, before the equation can change any
    # of its inputs. RHS indices likewise execute once in cnexp_rate.
    prefix = Any[]
    target = statement.target
    if target isa OperationExpression
        index = gensym(:element)
        index_name = gensym(:array_index)
        value = julia_expression(target.arguments[2], context)
        push!(prefix, :($index = Int(trunc($value))))
        variables = copy(context.variables)
        variables[index_name] = index
        context = CodegenContext(
            context.ir,
            variables,
            context.derivatives,
            context.phase,
            context.globals,
            context.solver,
            context.calls,
        )
        target = OperationExpression(
            Symbol("[]"),
            ModExpression[target.arguments[1], NameExpression(index_name, target.location)],
            target.location,
        )
    end
    state = julia_expression(target, context)
    dt = context.variables[:dt]
    # Lower relaxation and pure decay directly, as in native cnexp. Simple
    # scalar operands have no call/index effects to duplicate or reorder.
    # Other affine equations retain their frozen inputs and zero-rate limit.
    expression = statement.value
    if expression isa OperationExpression && expression.operation == :/
        numerator, tau = expression.arguments
        if target isa NameExpression &&
           numerator isa OperationExpression &&
           (tau isa NumberExpression || tau isa NameExpression && tau.name != name)
            equilibrium = nothing
            if numerator.operation == :- &&
               numerator.arguments[2] isa NameExpression &&
               numerator.arguments[2].name == name
                value = numerator.arguments[1]
                if value isa NumberExpression || value isa NameExpression && value.name != name
                    equilibrium = julia_expression(value, context)
                end
            elseif numerator.operation == Symbol("unary-") &&
                   only(numerator.arguments) isa NameExpression &&
                   only(numerator.arguments).name == name
                equilibrium = 0.0
            end
            if equilibrium !== nothing
                scale = julia_expression(tau, context)
                return Any[:(
                    $state = $state + (1.0 - exp($dt * (-1.0 / $scale))) * ($equilibrium - $state)
                )]
            end
        end
    end
    original, rate = gensym(:state_value), gensym(:affine_rate)
    append!(
        prefix,
        Any[
            :($original = $state),
            :($rate = $(cnexp_rate(statement.value, target, context))),
            :($state = $original + $dt * exponential_factor($rate.coefficient * $dt) * $rate.value),
        ],
    )
    return prefix
end

# Euler and exponential integration.

function emit_solve(solve::SolveStatement, context::CodegenContext)
    block = context.ir.blocks[solve.block]
    block.kind == :procedure && return Any[mod_call(block.name, ModExpression[], context)]
    block.kind == :derivative || throw(ModError(solve.location, "SOLVE requires an equation block"))
    # A SOLVE gets its own lexical locals, while referring to the caller's live
    # mechanism fields. POINTER reads/writes retain their immediate identity.
    nested = CodegenContext(
        context.ir,
        copy(context.globals),
        context.derivatives,
        context.phase,
        context.globals,
        solve.method,
        context.calls,
    )
    body = Any[]
    local_initializers!(body, nested.variables, context.ir.locals[block.name])
    states = equation_states(block)
    if solve.method == :euler
        for name in states
            rate = nested.derivatives[name]
            if fieldinfo(context.ir, name).extent == 0
                push!(body, :($rate = 0.0))
            else
                push!(body, :(fill!($rate, 0.0)))
            end
        end
    end
    append!(body, julia_statements(block.body, nested))
    if solve.method == :euler
        # Native Euler evaluates all rates before integrating their targets.
        # Keep the two rounded additions used by secondorder, with half-step
        # dt visible throughout nrn_state (including the remaining BREAKPOINT).
        updates = Any[]
        for name in states
            state, rate = nested.variables[name], nested.derivatives[name]
            dt = nested.variables[:dt]
            extent = fieldinfo(context.ir, name).extent
            if extent == 0
                push!(updates, :($state = $state + $dt * $rate))
            else
                push!(updates, :(
                    for index in 1:($extent)
                        $state[index] = $state[index] + $dt * $rate[index]
                    end
                ))
            end
        end
        append!(body, updates)
        if context.ir.role == :micro
            push!(body, Expr(:if, :(_context.secondorder != 0), Expr(:block, updates...)))
        end
    end
    return body
end
