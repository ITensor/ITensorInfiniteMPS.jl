
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

function fmap(f, ψ::InfiniteCanonicalMPS)
  return InfiniteCanonicalMPS(f(ψ.AL), f(ψ.C), f(ψ.AR))
end

function ITensors.prime(ψ::InfiniteCanonicalMPS, args...; kwargs...)
  return fmap(x -> prime(x, args...; kwargs...), ψ)
end

function ITensors.prime(f::typeof(linkinds), ψ::InfiniteCanonicalMPS, args...; kwargs...)
  return fmap(x -> prime(f, x, args...; kwargs...), ψ)
end

function ITensors.dag(ψ::InfiniteCanonicalMPS, args...; kwargs...)
  return fmap(x -> dag(x, args...; kwargs...), ψ)
end

