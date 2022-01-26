
#
# InfiniteMPS
#

# TODO: store the cell 1 as an MPS
# Implement `getcell(::InfiniteMPS, n::Integer) -> MPS`
mutable struct InfiniteMPS <: AbstractInfiniteMPS
  data::CelledVector{ITensor}
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
  # Doesn't prime ψ.C
  #return fmap(x -> prime(f, x, args...; kwargs...), ψ)
  AL = prime(linkinds, ψ.AL, args...; kwargs...)
  C = prime(ψ.C, args...; kwargs...)
  AR = prime(linkinds, ψ.AR, args...; kwargs...)
  return InfiniteCanonicalMPS(AL, C, AR)
end

function ITensors.dag(ψ::InfiniteCanonicalMPS, args...; kwargs...)
  return fmap(x -> dag(x, args...; kwargs...), ψ)
end

ITensors.siteinds(f::typeof(only), ψ::InfiniteCanonicalMPS) = siteinds(f, ψ.AL)

ITensorInfiniteMPS.getcell(i::Index) = ITensorInfiniteMPS.getcell(tags(i))
getsite(i::Index) = getsite(tags(i))
function ITensors.findfirstsiteind(ψ::InfiniteMPS, i::Index)
  c = ITensorInfiniteMPS.getcell(i)
  i1 = translatecell(i, -(c - 1))
  n1 = findfirst(hascommoninds(i1), ψ[Cell(1)])
  return (c - 1) * nsites(ψ) + n1
end
function ITensors.findsites(ψ::InfiniteMPS, T::ITensor)
  s = filterinds(T; plev=0)
  return sort([ITensors.findfirstsiteind(ψ, i) for i in s])
end
ITensors.findsites(ψ::InfiniteCanonicalMPS, T::ITensor) = findsites(ψ.AL, T)

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
nsites(h::InfiniteITensorSum) = length(h.data)
#Gives the range of the Hamiltonian. Useful for better optimized contraction in VUMPS
nsites_support(h::InfiniteITensorSum) = order.(h.data) ÷ 2
nsites_support(h::InfiniteITensorSum, n::Int64) = order(h.data[n]) ÷ 2

nrange(h::InfiniteITensorSum) = nrange.(h.data, ncell=nsites(h))
nrange(h::InfiniteITensorSum, n::Int64) = nrange(h.data[n]; ncell=nsites(h))
function nrange(h::ITensor; ncell=0)
  ns = findsites(h; ncell=ncell)
  return ns[end] - ns[1] + 1
end

function ITensors.findfirstsiteind(h::ITensor, i::Index, ncell::Int64)
  c = ITensorInfiniteMPS.getcell(i)
  n1 = getsite(i)
  return (c - 1) * ncell + n1
end
function ITensors.findsites(h::ITensor; ncell::Int64=1)
  s = filterinds(h; plev=0)
  return sort([ITensors.findfirstsiteind(h, i, ncell) for i in s])
end
ITensors.findsites(h::InfiniteITensorSum) = [findsites(h, n) for n in 1:nsites(h)]
ITensors.findsites(h::InfiniteITensorSum, n::Int64) = findsites(h.data[n]; ncell=nsites(h))

## HDF5 support for the InfiniteCanonicalMPS type

function HDF5.write(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ψ::InfiniteCanonicalMPS
)
  g = create_group(parent, name)
  attributes(g)["type"] = "InfiniteCanonicalMPS"
  attributes(g)["version"] = 1
  N = nsites(ψ)
  write(g, "length", N)
  for n in 1:N
    write(g, "AL[$(n)]", ψ.AL[n])
    write(g, "AR[$(n)]", ψ.AR[n])
    write(g, "C[$(n)]", ψ.C[n])
  end
end

function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{InfiniteCanonicalMPS}
)
  g = open_group(parent, name)
  if read(attributes(g)["type"]) != "InfiniteCanonicalMPS"
    error("HDF5 group or file does not contain InfiniteCanonicalMPS data")
  end
  N = read(g, "length")
  AL = InfiniteMPS([read(g, "AL[$(i)]", ITensor) for i in 1:N])
  AR = InfiniteMPS([read(g, "AR[$(i)]", ITensor) for i in 1:N])
  C = InfiniteMPS([read(g, "C[$(i)]", ITensor) for i in 1:N])
  return InfiniteCanonicalMPS(AL, C, AR)
end
