
#
# General helper functionality
#

# 
# Contracting index sets and getting costs
#

function _dim(is::Vector{<: Index})
  isempty(is) && return 1
  return mapreduce(dim, *, is)
end

# `is` could be Vector{Int} for BitSet
function _dim(is::IndexSetT, ind_dims::Vector) where {IndexSetT <: Union{Vector{Int}, BitSet}}
  isempty(is) && return one(eltype(inds_dims))
  dim = one(eltype(ind_dims))
  for i in is
    dim *= ind_dims[i]
  end
  return dim
end

function _dim(is::Unsigned, ind_dims::Vector)
  _isemptyset(is) && return one(eltype(inds_dims))
  dim = one(eltype(ind_dims))
  i = 1
  @inbounds while !iszero(is)
    if isodd(is)
      # TODO: determine if overflow really needs to be checked
      #dim = dim * ind_dims[i]
      dim = Base.Checked.checked_mul(dim, ind_dims[i])
    end
    is = is >> 1
    i += 1
  end
  return dim
end

# XXX: is this being used?
function contraction_cost(T::Vector{Vector{IndexT}}, a::Vector{Int},
                          b::Vector{Int}) where {IndexT <: Index}
  # The set of tensor indices `a` and `b`
  Tᵃ = T[a]
  Tᵇ = T[b]

  # XXX TODO: this should use a cache to store the results
  # of the contraction
  indsTᵃ = symdiff(Tᵃ...)
  indsTᵇ = symdiff(Tᵇ...)
  indsTᵃTᵇ = symdiff(indsTᵃ, indsTᵇ)
  cost = Int(sqrt(prod(_dim(indsTᵃ) * prod(_dim(indsTᵇ) * _dim(indsTᵃTᵇ)))))
  return cost
end

# XXX: is this being used?
function contraction_cost(T::Vector{BitSet}, dims::Vector{Int}, a::BitSet,
                          b::BitSet) where {IndexT <: Index}
  # The set of tensor indices `a` and `b`
  Tᵃ = [T[aₙ] for aₙ in a]
  Tᵇ = [T[bₙ] for bₙ in b]

  # XXX TODO: this should use a cache to store the results
  # of the contraction
  indsTᵃ = symdiff(Tᵃ...)
  indsTᵇ = symdiff(Tᵇ...)
  indsTᵃTᵇ = symdiff(indsTᵃ, indsTᵇ)
  dim_a = _dim(indsTᵃ, dims)
  dim_b = _dim(indsTᵇ, dims)
  dim_ab = _dim(indsTᵃTᵇ, dims)
  cost = Int(sqrt(Base.Checked.checked_mul(dim_a, dim_b, dim_ab)))
  return cost
end

# XXX: is this being used?
function remove_common_pair!(isR, N1, n1)
  if n1 > N1
    return
  end
  N = length(isR)
  is1 = isR[n1]
  is2 = @view isR[N1+1:N]
  n2 = findfirst(==(is1), is2)
  if isnothing(n2)
    n1 += 1
    remove_common_pair!(isR, N1, n1)
  else
    deleteat!(isR, (n1, N1+n2))
    N1 -= 1
    remove_common_pair!(isR, N1, n1)
  end
end

# XXX: is this being used?
function contract_inds(is1, is2)
  IndexT = eltype(is1)
  N1 = length(is1)
  N2 = length(is2)
  isR = IndexT[]
  for n1 in 1:N1
    i1 = is1[n1]
    n2 = findfirst(==(i1), is2)
    if isnothing(n2)
      push!(isR, i1)
    end
  end
  for n2 in 1:N2
    i2 = is2[n2]
    n1 = findfirst(==(i2), is1)
    if isnothing(n1)
      push!(isR, dag(i2))
    end
  end
  return isR
end

# Return the noncommon indices and the cost of contraction
# Recursively removes pairs of indices that are common
# between the IndexSets (TODO: use this for symdiff in ITensors.jl)
function contract_inds_cost(is1is2::Tuple{Vector{Int}, Vector{Int}}, ind_dims::Vector{Int})
  is1, is2 = is1is2
  N1 = length(is1)
  isR = vcat(is1, is2)
  remove_common_pair!(isR, N1, 1)
  cost = Int(sqrt(_dim(is1, ind_dims) * _dim(is2, ind_dims) * _dim(isR, ind_dims)))
  return isR, cost
end

function contract_inds_cost(is1::Vector{IndexT}, is2::Vector{IndexT}) where {IndexT <: Index}
  N1 = length(is1)

  # This is pretty slow
  #isR = vcat(is1, is2)
  #remove_common_pair!(isR, N1, 1)

  isR = contract_inds(is1, is2)

  cost = Int(sqrt(_dim(is1) * _dim(is2) * _dim(isR)))
  return isR, cost
end

function contraction_cost(indsTᵃ::BitSet, indsTᵇ::BitSet, dims::Vector)
  indsTᵃTᵇ = _symdiff(indsTᵃ, indsTᵇ)
  dim_a = _dim(indsTᵃ, dims)
  dim_b = _dim(indsTᵇ, dims)
  dim_ab = _dim(indsTᵃTᵇ, dims)
  # Perform the sqrt first to avoid overflow.
  # Alternatively, use a larger integer type.
  cost = round(Int, sqrt(dim_a) * sqrt(dim_b) * sqrt(dim_ab))
  return cost, indsTᵃTᵇ
end

function contraction_cost(indsTᵃ::IndexSetT, indsTᵇ::IndexSetT, dims::Vector) where {IndexSetT <: Unsigned}
  unionTᵃTᵇ = _union(indsTᵃ, indsTᵇ)
  cost = _dim(unionTᵃTᵇ, dims)
  indsTᵃTᵇ = _setdiff(unionTᵃTᵇ, _intersect(indsTᵃ, indsTᵇ))
  return cost, indsTᵃTᵇ
end

#
# Converting between represtentations of indices
#

# Converts the indices to integer labels
# and returns a Vector that takes those labels
# and returns the original integer dimensions
function inds_to_ints(::Type{DimT}, T::Vector{Vector{IndexT}}) where {IndexT <: Index, DimT}
  N = length(T)
  ind_to_int = Dict{IndexT, Int}()
  ints = Vector{Int}[Vector{Int}(undef, length(T[n])) for n in 1:N]
  ind_dims = Vector{DimT}(undef, sum(length, T))
  x = 0
  @inbounds for n in 1:N
    T_n = T[n]
    ints_n = ints[n]
    for j in 1:length(ints_n)
      i = T_n[j]
      i_label = get!(ind_to_int, i) do
        x += 1
        ind_dims[x] = dim(i)
        x
      end
      ints_n[j] = i_label
    end
  end
  resize!(ind_dims, x)
  return ints, ind_dims
end

# Converts the indices to integer labels stored in BitSets
# and returns a Vector that takes those labels
# and returns the original integer dimensions
# TODO: directly make an unsigned integer instead of a BitSet.
function inds_to_bitsets(::Type{BitSet}, ::Type{DimT}, T::Vector{Vector{IndexT}}) where {IndexT <: Index, DimT}
  N = length(T)
  ind_to_int = Dict{IndexT, Int}()
  ints = map(_ -> BitSet(), 1:N)
  ind_dims = Vector{DimT}(undef, sum(length, T))
  x = 0
  @inbounds for n in 1:N
    T_n = T[n]
    @inbounds for j in 1:length(T_n)
      i = T_n[j]
      i_label = get!(ind_to_int, i) do
        x += 1
        ind_dims[x] = dim(i)
        x
      end
      push!(ints[n], i_label)
    end
  end
  resize!(ind_dims, x)
  return ints, ind_dims
end

function inds_to_bitsets(::Type{T}, ::Type{DimT}, Tinds::Vector{Vector{IndexT}}) where {T <: Unsigned, IndexT <: Index, DimT}
  ints, ind_dims = inds_to_bitsets(BitSet, DimT, Tinds)
  uints = T[bitset(T, int) for int in ints]
  return uints, ind_dims
end

# Convert a contraction sequence in pair form to tree format
function pair_sequence_to_tree(pairs::Vector{Pair{Int, Int}}, N::Int)
  trees = Any[1:N...]
  for p in pairs
    push!(trees, Any[trees[p[1]], trees[p[2]]])
  end
  return trees[end]
end

#
# BitSet utilities
#

function _cmp(A::BitSet, B::BitSet)
  for (a, b) in zip(A, B)
    if !isequal(a, b)
      return isless(a, b) ? -1 : 1
    end
  end
  return cmp(length(A), length(B))
end

# Returns true when `A` is less than `B` in lexicographic order.
_isless(A::BitSet, B::BitSet) = _cmp(A, B) < 0

bitset(::Type{BitSet}, ints) = BitSet(ints)

function bitset(::Type{T}, ints) where {T <: Unsigned}
  set = zero(T)
  u = one(T)
  for i in ints
    set |= (u<<(i-1))
  end
  return set
end

# Return the position of the first nonzero bit
function findfirst_nonzero_bit(i::Unsigned)
  n = 0
  @inbounds while !iszero(i)
    if isodd(i)
      return n+1
    end
    i = i >> 1
    n += 1
  end
  return n
end

_isless(s1::T, s2::T) where {T <: Unsigned} = s1 < s2
_intersect(s1::BitSet, s2::BitSet) = intersect(s1, s2)
_intersect(s1::T, s2::T) where {T<:Unsigned} = s1 & s2
_union(s1::BitSet, s2::BitSet) = union(s1, s2)
_union(s1::T, s2::T) where {T<:Unsigned} = s1 | s2
_setdiff(s1::BitSet, s2::BitSet) = setdiff(s1, s2)
_setdiff(s1::T, s2::T) where {T<:Unsigned} = s1 & (~s2)
_symdiff(s1::BitSet, s2::BitSet) = symdiff(s1, s2)
_symdiff(s1::T, s2::T) where {T<:Unsigned} = xor(s1, s2)
_isemptyset(s::BitSet) = isempty(s)
_isemptyset(s::Unsigned) = iszero(s)

# TODO: use _first instead, optimize to avoid using _set
_only(s::BitSet) = only(s)
_only(s::Unsigned) = findfirst_nonzero_bit(s)

