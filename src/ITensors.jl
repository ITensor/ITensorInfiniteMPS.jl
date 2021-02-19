
############################################################################
# imports.jl
#

import Base: *

############################################################################
# NDTensors
#

# Take the square root of a diagonal order 2 tensor
# TODO: make a AbstractDiagTensor supertype?
# TODO: make this work for non-diagonal tensors.
# TODO: define SecondOrderTensor?
function Base.sqrt(T::NDTensors.Tensor{<: Number, 2})
  @assert isdiag(T)
  return sqrt.(T)
end

############################################################################
# tagset.jl
#

# Used for findfirst
Base.keys(ts::TagSet) = Base.OneTo(length(ts))

import ITensors: Tag

macro tag_str(s)
  Tag(s)
end

maxlength(tag::Tag) = length(tag.data)

function Base.length(tag::Tag)
  n = 1
  while n <= maxlength(tag) && tag[n] != zero(eltype(tag))
    n += 1
  end
  return n-1
end

Base.lastindex(tag::Tag) = length(tag)

Base.getindex(tag::Tag, r::UnitRange) = Tag([tag[n] for n in r])

# TODO: make this work directly on a Tag, without converting
# to String
function Base.parse(::Type{T}, tag::Tag) where {T <: Integer}
  return parse(T, string(tag))
end

function Base.startswith(tag::Tag, subtag::Tag)
  for n in 1:length(subtag)
    tag[n] ≠ subtag[n] && return false
  end
  return true
end

function tag_starting_with(ts::TagSet, prefix)
  return ts[findfirst(t -> startswith(t, Tag(prefix)), ts)]
end

ITensors.replacetags(ts::TagSet, p::Pair) = replacetags(ts, first(p), last(p))

############################################################################
# qn.jl
#

#
# New QN interface.
# Allows constructors like:
# Index((("Sz", 0) × ("SzParity", 0, 2) => 1) ⊕
#       (("Sz", 1) × ("SzParity", 1, 2) => 2))
#

LinearAlgebra.cross(qn1::Tuple{String, Int, Vararg{Int}},
                    qn2::Tuple{String, Int, Vararg{Int}}) = QN(qn1, qn2)
LinearAlgebra.cross(qn1::QN, qn2::Tuple{String, Int, Vararg{Int}}) =
  ITensors.addqnval(qn1, ITensors.QNVal(qn2...))

⊕(qn1::Pair{QN, Int}, qn2::Pair{QN, Int}) = [qn1, qn2]
⊕(qn1::Pair{QN, Int64}, qn2::Pair{<:Tuple{String, Int, Vararg{Int}}, Int}) =
  [qn1, QN(first(qn2)) => last(qn2)]
⊕(qn1::Pair{<:Tuple{String, Int, Vararg{Int}}, Int}, qn2::Pair{QN, Int}) =
  [QN(first(qn1)) => last(qn1), qn2]
⊕(qn1::Pair{<:Tuple{String, Int, Vararg{Int}}, Int}, qn2::Pair{<:Tuple{String, Int, Vararg{Int}}, Int}) =
  [QN(first(qn1)) => last(qn1), QN(first(qn2)) => last(qn2)]
⊕(qns::Vector{Pair{QN, Int}}, qn::Pair{QN, Int}) = push!(copy(qns), qn)

############################################################################
# indexset.jl
#

ITensors.IndexSet(is::IndexSet...) = unioninds(is)
ITensors.IndexSet(is::Tuple{Vararg{<:IndexSet}}) = unioninds(is...)

Base.copy(is::IndexSet) = IndexSet(copy.(ITensors.data(is)))

ITensors.noncommoninds(is::IndexSet) = is


############################################################################
# itensor.jl
#

function LinearAlgebra.ishermitian(T::ITensor, pairs = 0 => 1; kwargs...)
  Tᴴ = swapprime(dag(T), pairs)
  return isapprox(Tᴴ, T; kwargs...)
end

# Helpful for making sure the ITensor doesn't contract
ITensors.sim(A::ITensor) = ITensors.setinds(A, sim(inds(A)))
using ITensors.NDTensors

LinearAlgebra.isdiag(T::ITensor) = isdiag(tensor(T))

function eigendecomp(T::ITensor; ishermitian = true, kwargs...)
  @assert ishermitian
  D, U = eigen(T; ishermitian = ishermitian, kwargs...)
  return U', D, dag(U)
end

*(A::ITensor) = A

ITensors.noncommoninds(A::ITensor) = inds(A)

# TODO: implement something like this?
#function sqrt(::Order{2}, T::ITensor)
#end

# Take the square root of T assuming it is Hermitian
# TODO: add more general index structures
function Base.sqrt(T::ITensor; ishermitian = true)
  @assert ishermitian
  if isdiag(T) && order(T) == 2
    return itensor(sqrt(tensor(T)))
  end
  U′, D, Uᴴ = eigendecomp(T; ishermitian = ishermitian)
  # XXX: if T is order 2 and diagonal, D may just be a view
  # of T so this would also modify T
  D .= sqrt.(D)
  return U′ * D * Uᴴ
end


############################################################################
# mps.jl
#

# TODO: make this definition AbstractMPS
# Handle orthogonality center correctly
Base.getindex(ψ::MPS, r::UnitRange{Int}) =
  MPS([ψ[n] for n in r])


