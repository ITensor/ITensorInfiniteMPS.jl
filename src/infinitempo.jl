
#
# InfiniteMPO
#

mutable struct InfiniteMPO <: AbstractInfiniteMPS
  data::CelledVector{ITensor}
  llim::Int #RealInfinity
  rlim::Int #RealInfinity
  reverse::Bool
end

## struct InfiniteSumLocalOps <: AbstractInfiniteMPS
##   data::CelledVector{ITensor}
## end
## InfiniteSumLocalOps(N::Int) = InfiniteSumLocalOps(Vector{ITensor}(undef, N))
## InfiniteSumLocalOps(data::Vector{ITensor}) = InfiniteSumLocalOps(CelledVector(data))
## getindex(l::InfiniteSumLocalOps, n::Integer) = ITensors.data(l)[n]

# TODO? Instead of having a big quasi empty ITensor, store only the non zero blocks

translator(mpo::InfiniteMPO) = mpo.data.translator
