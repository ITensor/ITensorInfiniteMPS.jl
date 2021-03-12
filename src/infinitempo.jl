
#
# InfiniteMPO
#

mutable struct InfiniteMPO <: AbstractInfiniteMPS
  data::Vector{ITensor}
  llim::Int #RealInfinity
  rlim::Int #RealInfinity
  reverse::Bool
end

