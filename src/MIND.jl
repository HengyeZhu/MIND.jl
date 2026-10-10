module MIND

using Adapt
using CUDA
import GPUArrays
using LinearAlgebra
using Random
using RuntimeGeneratedFunctions
using SHA
using StaticArrays: MVector
import SpecialFunctions: erf, erfc
import Base: insert!

RuntimeGeneratedFunctions.init(@__MODULE__)

include("compiler/ir.jl")
include("compiler/frontend.jl")
include("compiler/codegen.jl")
include("compiler/utils/frontend.jl")
include("model/micro/morphology.jl")
include("model/micro/utils/morphology.jl")
include("model/micro/biophysics.jl")
include("model/micro/utils/biophysics.jl")
include("model/micro/model.jl")
include("model/macro/model.jl")
include("model/build.jl")
include("model/references.jl")
include("model/micro/utils/model.jl")
include("model/macro/utils/model.jl")
include("model/utils/build.jl")
include("model/utils/references.jl")
include("runtime/state.jl")
include("runtime/initialize.jl")
include("runtime/events.jl")
include("runtime/step.jl")
include("runtime/recording.jl")
include("runtime/utils/state.jl")
include("runtime/utils/initialize.jl")
include("runtime/utils/events.jl")
include("runtime/utils/step.jl")
include("runtime/cuda/micro.jl")
include("runtime/cuda/macro.jl")

export runtime_configuration, simulation_time

export compile_mod, load_mod, check_units, mechanism_ir, clear_mod_cache!, builtin
export Point3D,
    Section, MicroModel, Population, Cell, CellSection, SectionGroup, PointProcess, SpikeSource
export load_swc,
    load_asc,
    insert!,
    add_population!,
    add_pointprocess!,
    register_spike_source!,
    spike_source,
    connect!
export MorphologyLayout, path_distance, segment_diameter
export section, section_group
export delete_sections!
export IonSpecies, IonStyle, ion_style, Connectivity, load_connectivity, MacroModel, ROI, roi
export nernst, ghk
export use_macro!, use_micro!, Micro2Macro, micro2macro!, macro2micro!
export build!
export initialize!, step!, run!, RecordSpec, SimulationResult
export voltage_record, mechanism_record, ion_record, roi_record, global_record, read_value
export VariableRef, voltage_ref, mechanism_ref, ion_ref, roi_ref, global_ref
export connection_weights, inject_event!, inject_spike!

end
