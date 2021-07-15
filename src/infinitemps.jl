
#
# InfiniteMPS
#

# TODO: store the cell 1 as an MPS
# Implement `getcell(::InfiniteMPS, n::Integer) -> MPS`
mutable struct InfiniteMPS <: AbstractInfiniteMPS
  data::Vector{ITensor}
  llim::Int #RealInfinity
  rlim::Int #RealInfinity
  reverse::Bool
end

#
# InfiniteCanonicalMPS
#

# L and R are the orthogonalized MPS tensors.
# C are the center bond matrices (singular values of C
# are the singular values of the MPS at the specified
# cut)
struct InfiniteCanonicalMPS <: AbstractInfiniteMPS
  AL::InfiniteMPS
  C::InfiniteMPS
  AR::InfiniteMPS
end

# TODO: check if `isempty(ψ.AL)`, if so use `ψ.AR`
nsites(ψ::InfiniteCanonicalMPS) = nsites(ψ.AL)
isreversed(ψ::InfiniteCanonicalMPS) = isreversed(ψ.AL)
#ITensors.data(ψ::InfiniteCanonicalMPS) = data(ψ.AL)
ITensors.data(ψ::InfiniteCanonicalMPS) = ψ.AL.data

