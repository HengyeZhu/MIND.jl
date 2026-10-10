# ROI exposure ownership and binding validation.

Base.length(model::MacroModel) = length(model.regions)
Base.eltype(::Type{MacroModel}) = ROI
Base.iterate(model::MacroModel, index::Int = 1) =
    index > length(model) ? nothing : (ROI(model, index), index + 1)

Base.show(io::IO, model::MacroModel) = print(
    io,
    "MacroModel(",
    length(model.regions),
    " ROIs, ",
    length(model.projections),
    " projections, micro=",
    model.micro !== nothing,
    ")",
)
Base.show(io::IO, region::ROI) =
    print(io, "ROI(", repr(region.model.connectivity.labels[region.index]), ")")
Base.show(io::IO, output::Micro2Macro) = print(
    io,
    "Micro2Macro(index=",
    output.index,
    ", sources=",
    length(output.model.outputs[output.index].sources),
    ")",
)

# Each exposure's samples run from oldest to newest; capture them at ROI ownership creation.
function capture_history!(region::ROI, history)
    for (name, samples) in pairs(history)
        values = collect(Float64, samples)
        !isempty(values) && all(isfinite, values) ||
            throw(ArgumentError("initial history must be nonempty and finite"))
        region.model.initial_histories[(region.index, Symbol(name))] = values
    end
    return nothing
end

function roi_exposures(model::MacroModel)
    available = [Set{Symbol}() for _ in model.regions]
    names = Symbol[]
    for (index, region) in enumerate(model.regions)
        region === nothing && continue
        for name in region.mechanism.ir.exposures
            push!(available[index], name)
            name in names || push!(names, name)
        end
    end
    for index in sort!(collect(keys(model.micro_exposures)))
        for name in model.micro_exposures[index]
            push!(available[index], name)
            name in names || push!(names, name)
        end
    end
    return available, names
end

function validate_exposure_bindings(mechanism, available, source, target)
    for binding in mechanism.ir.bindings
        owner = binding.direction in (:read_source, :write_source) ? source : target
        binding.exposure in available[owner] || throw(
            ArgumentError(
                "ROI $owner does not expose $(binding.exposure) required by $(mechanism.ir.name)",
            ),
        )
    end
    return nothing
end

# Connectivity input parsing.

function csv_fields(line::AbstractString)
    # Matrix rows normally need only delimiter scanning; quoted labels retain
    # the same escaping rules without making every numerical field use IOBuffer.
    if !occursin('"', line)
        return String[strip(field) for field in split(line, ','; keepempty = true)]
    end
    result = String[]
    buffer = IOBuffer()
    quoted = false
    characters = collect(line)
    index = 1
    while index <= length(characters)
        character = characters[index]
        if character == '"'
            if quoted && index < length(characters) && characters[index + 1] == '"'
                print(buffer, '"')
                index += 1
            else
                quoted = !quoted
            end
        elseif character == ',' && !quoted
            push!(result, strip(String(Base.take!(buffer))))
        else
            print(buffer, character)
        end
        index += 1
    end
    quoted && throw(ArgumentError("unterminated quoted CSV field"))
    push!(result, strip(String(Base.take!(buffer))))
    return result
end
