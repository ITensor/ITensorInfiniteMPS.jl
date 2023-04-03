
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

InfiniteMPO(data::CelledVector{ITensor}) = InfiniteMPO(data, 0, size(data, 1), false)
