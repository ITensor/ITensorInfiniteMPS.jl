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

using ITensors.NDTensors: eachdiagblock
using KrylovKit: eigsolve, linsolve

import Base:
  getindex,
  length,
  setindex!

import ITensors:
  AbstractMPS

include("ITensors.jl")
include("ITensorNetworks.jl")
include("itensormap.jl")
include("celledvectors.jl")
include("abstractinfinitemps.jl")
include("infinitemps.jl")
include("infinitempo.jl")
include("infinitecanonicalmps.jl")
include("models.jl")
include("orthogonalize.jl")
include("infinitemps_approx.jl")
include("nullspace.jl")
include("subspace_expansion.jl")
include("vumps_localham.jl")

export
  Cell,
  CelledVector,
  InfiniteMPS,
  InfiniteCanonicalMPS,
  InfMPS,
  InfiniteITensorSum,
  InfiniteMPO,
  InfiniteSumLocalOps,
  ITensorMap,
  ITensorNetwork,
  Model,
  @Model_str,
  input_inds,
  infinitemps_approx,
  infsiteinds,
  nsites,
  output_inds,
  subspace_expansion,
  translatecell,
  vumps,
  ⊕,
  ⊗,
  ×

function __init__()
  # This is used for debugging using visualizations
  @require ITensorsVisualization="f2aed53d-2f32-47c3-a7b9-1ee253853786" @eval using ITensorsVisualization
end

end
