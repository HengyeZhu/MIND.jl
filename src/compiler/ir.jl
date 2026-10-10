# Source syntax and locations.

"""Location in a MOD source, retained through lowering for diagnostics."""
struct SourceLocation
    file::String
    line::Int
    column::Int
end

struct ModError <: Exception
    location::SourceLocation
    message::String
end

function Base.showerror(io::IO, error::ModError)
    loc = error.location
    print(io, loc.file, ':', loc.line, ':', loc.column, ": ", error.message)
end

struct Token
    kind::Symbol
    text::String
    location::SourceLocation
end

"""Physical dimensions and scale used only by the compiler's unit analysis."""
struct ModUnit
    dimensions::NTuple{10, Rational{Int}}
    scale::Float64
    literal::Bool
    wildcard::Bool
end

ModUnit(; scale = 1.0, literal = false, wildcard = false) =
    ModUnit(ntuple(_ -> 0 // 1, 10), Float64(scale), literal, wildcard)

abstract type ModExpression end

struct NumberExpression <: ModExpression
    value::Float64
    units::Union{Nothing, String}
    unit_factor::Float64
    location::SourceLocation
end

NumberExpression(value::Float64, location::SourceLocation) =
    NumberExpression(value, nothing, 1.0, location)

struct NameExpression <: ModExpression
    name::Symbol
    location::SourceLocation
end

struct OperationExpression <: ModExpression
    operation::Symbol
    arguments::Vector{ModExpression}
    location::SourceLocation
end

abstract type ModStatement end

struct Assignment <: ModStatement
    target::ModExpression
    value::ModExpression
    derivative::Bool
    location::SourceLocation
end

struct CallStatement <: ModStatement
    expression::ModExpression
    location::SourceLocation
end

struct Conditional <: ModStatement
    condition::ModExpression
    consequent::Vector{ModStatement}
    alternative::Vector{ModStatement}
    location::SourceLocation
end

struct WhileStatement <: ModStatement
    condition::ModExpression
    body::Vector{ModStatement}
    location::SourceLocation
end

struct LocalDeclaration <: ModStatement
    names::Vector{Symbol}
    extents::Vector{Int}
    location::SourceLocation
end

struct SolveStatement <: ModStatement
    block::Symbol
    method::Symbol
    location::SourceLocation
end

struct EventInitialStatement <: ModStatement
    body::Vector{ModStatement}
    location::SourceLocation
end

struct Declaration
    name::Symbol
    extent::Int
    storage::Symbol
    default::Union{Nothing, ModExpression}
    units::String
    bounds::Union{Nothing, Tuple{Float64, Float64}}
    location::SourceLocation
end

struct IonDeclaration
    name::Symbol
    reads::Vector{Symbol}
    writes::Vector{Symbol}
    valence::Union{Nothing, Float64}
    location::SourceLocation
end

struct ExposureBinding
    direction::Symbol
    exposure::Symbol
    variable::Symbol
    location::SourceLocation
end

struct ModBlock
    kind::Symbol
    name::Symbol
    arguments::Vector{Symbol}
    argument_units::Vector{String}
    units::String
    body::Vector{ModStatement}
    location::SourceLocation
end

mutable struct ModAST
    name::Symbol
    kind::Symbol
    role::Symbol
    declarations::Vector{Declaration}
    ranges::Vector{Symbol}
    globals::Vector{Symbol}
    pointers::Vector{Symbol}
    currents::Vector{Symbol}
    electrode_currents::Vector{Symbol}
    randoms::Vector{Symbol}
    ions::Vector{IonDeclaration}
    exposures::Vector{Symbol}
    bindings::Vector{ExposureBinding}
    blocks::Vector{ModBlock}
    definitions::Dict{Symbol, Float64}
    unit_definitions::Dict{String, String}
    unit_checks::IdDict{Union{ModStatement, Declaration}, Bool}
    unit_contracts::Bool
    threadsafe::Bool
    source::String
    filename::String
    sources::Dict{String, String}
end

function ModAST(source::String, filename::String)
    return ModAST(
        :anonymous,
        :density,
        :micro,
        Declaration[],
        Symbol[],
        Symbol[],
        Symbol[],
        Symbol[],
        Symbol[],
        Symbol[],
        IonDeclaration[],
        Symbol[],
        ExposureBinding[],
        ModBlock[],
        Dict{Symbol, Float64}(),
        Dict{String, String}(),
        IdDict{Union{ModStatement, Declaration}, Bool}(),
        true,
        false,
        source,
        filename,
        Dict(filename => source),
    )
end

# Resolved fields, equations and execution effects.

@enum ValueType::UInt8 Real64 Boolean RealArray

"""Language-defined inputs, resolved before GPU code generation.

User field spellings have no runtime meaning. These tags bind only the reserved
inputs for a mechanism's language role; REGION fields named v or weight remain
ordinary declared fields.
"""
@enum ExecutionInput::UInt8 begin
    TimeInput
    StepInput
    TemperatureInput
    VoltageInput
    DiameterInput
    AreaInput
    InitialVoltageInput
    EventFlagInput
    EdgeWeightInput
    WindowStartInput
    WindowStopInput
end

"""MIND inputs supplied once by the CPU connection builder."""
@enum ConstructionInput::UInt8 ConnectionWeight ConnectionDelaySteps SourceROI TargetROI

"""Resolved storage field; extent zero denotes a scalar, otherwise a MOD array."""
struct IRField
    name::Symbol
    offset::Int
    extent::Int
    category::Symbol
    scope::Symbol
    units::String
    default::Float64
    bounds::Union{Nothing, Tuple{Float64, Float64}}
    location::SourceLocation
end

"""Effects are explicit so future differentiation can distinguish dynamics from events."""
struct BlockEffects
    reads::Set{Symbol}
    writes::Set{Symbol}
    random::Bool
    schedules_events::Bool
    emits_spikes::Bool
    discontinuous::Bool
end

struct IRBlock
    kind::Symbol
    name::Symbol
    arguments::Vector{Symbol}
    argument_units::Vector{String}
    units::String
    body::Vector{ModStatement}
    effects::BlockEffects
    location::SourceLocation
end

"""A lexical LOCAL binding, with its source spelling retained for inspection and AD."""
struct LocalBinding
    name::Symbol
    source_name::Symbol
    scope::Int
    extent::Int
    location::SourceLocation
end

"""Typed and name-resolved MOD representation, independent of the CUDA storage layout."""
# Shared descriptor identity avoids repeated boxing in heterogeneous model collections.
# Fields retain their fixed bindings; AST/IR payloads remain inspectable.
mutable struct MechanismIR
    const name::Symbol
    const kind::Symbol
    const role::Symbol
    const threadsafe::Bool
    const inputs::Vector{Pair{Symbol, ExecutionInput}}
    const construction_inputs::Vector{Pair{Int, ConstructionInput}}
    const fields::Vector{IRField}
    const fieldmap::Dict{Symbol, Int}
    const blocks::Dict{Symbol, IRBlock}
    const recursive::Bool
    const locals::Dict{Symbol, Vector{LocalBinding}}
    const expression_types::IdDict{ModExpression, ValueType}
    const currents::Vector{Symbol}
    const electrode_currents::Vector{Symbol}
    const ions::Vector{IonDeclaration}
    const randoms::Vector{Symbol}
    const exposures::Vector{Symbol}
    const bindings::Vector{ExposureBinding}
    const definitions::Dict{Symbol, Float64}
    const source_digest::String
    const source::String
    const filename::String
    const sources::Dict{String, String}
end

function has_shared_writes(ir::MechanismIR)
    for field in ir.fields
        field.scope == :global || continue
        any(block -> field.name in block.effects.writes, values(ir.blocks)) && return true
    end
    return false
end

has_pointers(ir::MechanismIR) = any(field -> field.scope == :pointer, ir.fields)

function has_event_initializer(ir::MechanismIR)
    block = get(ir.blocks, :net_receive, nothing)
    return block !== nothing && any(statement -> statement isa EventInitialStatement, block.body)
end

function has_voltage_reset(ir::MechanismIR)
    block = get(ir.blocks, :net_receive, nothing)
    block === nothing && return false
    return any(
        binding -> last(binding) == VoltageInput && first(binding) in block.effects.writes,
        ir.inputs,
    )
end
serial_instances(ir::MechanismIR) =
    has_shared_writes(ir) || has_pointers(ir) || has_voltage_reset(ir)

function source_digest(sources::Dict{String, String}, filename::AbstractString)
    length(sources) == 1 && return bytes2hex(sha256(only(values(sources))))
    # INCLUDE identity follows the source tree, not its installation directory.
    # This digest also seeds mechanism RNG streams and must survive relocation.
    paths = sort!(collect(keys(sources)); by = path -> relpath(path, dirname(filename)))
    content =
        join(string(relpath(path, dirname(filename)), '\0', sources[path], '\0') for path in paths)
    return bytes2hex(sha256(content))
end

fieldwidth(field::IRField) = max(1, field.extent)
fieldcolumns(field::IRField) = field.offset:(field.offset + fieldwidth(field) - 1)
fieldcount(ir::MechanismIR) = sum(fieldwidth, ir.fields; init = 0)
fieldinfo(ir::MechanismIR, name::Symbol) = ir.fields[ir.fieldmap[name]]
function fieldinfo(ir::MechanismIR, offset::Integer)
    for field in ir.fields
        offset in fieldcolumns(field) && return field
    end
    throw(BoundsError(ir.fields, offset))
end
fieldindex(ir::MechanismIR, name::Symbol) = fieldinfo(ir, name).offset

defaults(ir::MechanismIR) = [field.default for field in ir.fields for _ in fieldcolumns(field)]

mutable struct CompiledMechanism{I, C, S, E, N}
    const ir::MechanismIR
    const initialize!::I
    const current!::C
    const advance!::S
    const receive!::E
    const event_initialize!::N
end

"""A generated execution phase and its concrete MOD function call graph."""
struct GeneratedPhase{F, C}
    function_::F
    calls::C
end

@inline (phase::GeneratedPhase)(data, row, context, weight, flag) =
    @inline phase.function_(phase.calls, data, row, context, weight, flag)

"""Only forward execution is supplied; equation IR and event effects remain inspectable."""
mechanism_ir(mechanism::CompiledMechanism) = mechanism.ir
