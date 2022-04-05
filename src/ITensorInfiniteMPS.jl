module ITensorInfiniteMPS

# For `only`, which was introduced in Julia v1.4
using Compat
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
# For HDF5 support
using HDF5
# For integration support when computing exact reference results
using QuadGK

using ITensors.NDTensors: eachdiagblock
using KrylovKit: eigsolve, linsolve, exponentiate

import Base: getindex, length, setindex!, +, -, *

import ITensors: AbstractMPS

include("ITensors.jl")
include("ITensorNetworks.jl")
include("itensormap.jl")
include("celledvectors.jl")
include("abstractinfinitemps.jl")
include("infinitemps.jl")
include("infinitempo.jl")
include("infinitecanonicalmps.jl")
include("infinitempomatrix.jl")
include("transfermatrix.jl")
include("models/models.jl")
include("models/ising.jl")
include("models/ising_extended.jl")
include("models/heisenberg.jl")
include("models/hubbard.jl")
include("models/xx.jl")
include("orthogonalize.jl")
include("infinitemps_approx.jl")
include("nullspace.jl")
include("subspace_expansion.jl")
include("vumps_localham.jl")
include("vumps_nonlocalham.jl")
include("vumps_mpo.jl")

export Cell,
  CelledVector,
  InfiniteMPS,
  InfiniteCanonicalMPS,
  InfMPS,
  InfiniteSum,
  InfiniteMPO,
  InfiniteMPOMatrix,
  InfiniteSumLocalOps,
  ITensorMap,
  ITensorNetwork,
  TransferMatrix,
  @Model_str,
  Model,
  @Observable_str,
  Observable,
  input_inds,
  infinitemps_approx,
  infsiteinds,
  nsites,
  output_inds,
  reference,
  subspace_expansion,
  translatecell,
  tdvp,
  vumps,
  ⊕,
  ⊗,
  ×

function __init__()
  # This is used for debugging using visualizations
  @require ITensorsVisualization = "f2aed53d-2f32-47c3-a7b9-1ee253853786" @eval using ITensorsVisualization
end

end
