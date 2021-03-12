module ITensorInfiniteMPS

using ITensors
# For optional ITensorsVisualization dependency.
using Requires
# For using ∞ as lengths, ranges, etc.
using Infinities
# For functions like `isdiag`
using LinearAlgebra
# For indexing starting from something other than 0.
using OffsetArrays
using IterTools
using KrylovKit

import Base:
  getindex,
  length,
  setindex!

import ITensors:
  AbstractMPS

include("ITensors.jl")
include("ITensorNetworks.jl")
include("itensormap.jl")
include("abstractinfinitemps.jl")
include("infinitemps.jl")
include("infinitempo.jl")
include("orthogonalize.jl")
include("vumps.jl")
include("contraction_sequence_optimization/contraction_sequence_optimization.jl")

export
  InfiniteMPS,
  ITensorMap,
  ITensorNetwork,
  input_inds,
  nsites,
  output_inds,
  vumps,
  ⊕,
  ⊗,
  ×

function __init__()
  # This is used for debugging using visualizations
  @require ITensorsVisualization="f2aed53d-2f32-47c3-a7b9-1ee253853786" @eval using ITensorsVisualization
end

end
