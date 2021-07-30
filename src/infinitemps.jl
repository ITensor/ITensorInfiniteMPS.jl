
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

Base.copy(ψ::InfiniteCanonicalMPS) = InfiniteCanonicalMPS(copy(ψ.AL), copy(ψ.C), copy(ψ.AR))

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

ITensors.siteinds(f::typeof(only), ψ::InfiniteCanonicalMPS) = siteinds(f, ψ.AL)

# For now, only represents nearest neighbor interactions
# on a linear chain
struct InfiniteITensorSum
  data::CelledVector{ITensor}
end
InfiniteITensorSum(N::Int) = InfiniteITensorSum(Vector{ITensor}(undef, N))
InfiniteITensorSum(data::Vector{ITensor}) = InfiniteITensorSum(CelledVector(data))
function Base.getindex(l::InfiniteITensorSum, n1n2::Tuple{Int,Int})
  n1, n2 = n1n2
  @assert n2 == n1 + 1
  return l.data[n1]
end
nsites(h::InfiniteITensorSum) = length(l.data)
