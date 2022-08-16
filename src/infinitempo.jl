
#
# InfiniteMPO
#

mutable struct InfiniteMPO <: AbstractInfiniteMPS
  data::CelledVector{ITensor}
  llim::Int #RealInfinity
  rlim::Int #RealInfinity
  reverse::Bool
end

translator(mpo::InfiniteMPO) = mpo.data.translator
