
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
  _isemptyset(is) && return one(eltype(ind_dims))
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
# Convert indices into unique integer labels
#

function contraction_labels!(labels1, labels2, is1, is2)
  nextlabel = 1
  nextlabel = common_contraction_labels!(labels1, labels2, is1, is2, nextlabel)
  nextlabel = uncommon_contraction_labels!(labels1, is1, nextlabel)
  nextlabel = uncommon_contraction_labels!(labels2, is2, nextlabel)
  return labels1, labels2
end

# Compute the common contraction labels and return the next label
function common_contraction_labels!(labels1, labels2, is1, is2, label)
  N1 = length(is1)
  N2 = length(is2)
  @inbounds for n1 in 1:N1, n2 in 1:N2
    i1 = is1[n1]
    i2 = is2[n2]
    if i1 == i2
      labels1[n1] = labels2[n2] = label
      label += 1
    end
  end
  return label
end

function uncommon_contraction_labels!(labels, is, label)
  N = length(labels)
  @inbounds for n in 1:N
    if iszero(labels[n])
      labels[n] = label
      label += 1
    end
  end
  return label
end

function contraction_labels!(labels, is)
  ntensors = length(is)
  nextlabel = 1
  # Loop through each tensor pair searching for
  # common indices
  @inbounds for n1 in 1:ntensors-1, n2 in n1+1:ntensors
    nextlabel = common_contraction_labels!(labels[n1], labels[n2], is[n1], is[n2], nextlabel)
  end
  @inbounds for n in 1:ntensors
    nextlabel = uncommon_contraction_labels!(labels[n], is[n], nextlabel)
  end
  return nextlabel - 1
end

function empty_labels(is::NTuple{N}) where {N}
  return ntuple(n -> fill(0, length(is[n])), Val(N))
end

function empty_labels(is::Vector)
  ntensors = length(is)
  labels = Vector{Vector{Int}}(undef, ntensors)
  @inbounds for n in 1:ntensors
    labels[n] = fill(0, length(is[n]))
  end
  return labels
end

function contraction_labels(is)
  labels = empty_labels(is)
  contraction_labels!(labels, is)
  return labels
end

contraction_labels(is...) = contraction_labels(is)

#
# Use a Dict as a cache to map the indices to the integer label
# This only helps with many nodes/tensors (nnodes > 30)
# TODO: determine the crossover when this is useful and use
# it in `depth_first_constructive`/`breadth_first_constructive`
#

contraction_labels_caching(is) =
  contraction_labels_caching(eltype(eltype(is)), is)

function contraction_labels_caching(::Type{IndexT}, is) where {IndexT}
  labels = empty_labels(is)
  contraction_labels_caching!(labels, IndexT, is)
end

function contraction_labels_caching!(labels, ::Type{IndexT}, is) where {IndexT}
  N = length(is)
  ind_to_label = Dict{IndexT, Int}()
  label = 0
  @inbounds for n in 1:N
    isₙ = is[n]
    labelsₙ = labels[n]
    @inbounds for j in 1:length(labelsₙ)
      i = isₙ[j]
      i_label = get!(ind_to_label, i) do
        label += 1
      end
      labelsₙ[j] = i_label
    end
  end
  return label
end

#
# Compute the labels and also return a data structure storing the dims.
#

function label_dims(::Type{DimT}, is) where {DimT <: Integer}
  labels = empty_labels(is)
  nlabels = contraction_labels!(labels, is)
  dims = fill(zero(DimT), nlabels)
  @inbounds for i in 1:length(is)
    labelsᵢ = labels[i]
    isᵢ = is[i]
    @inbounds for n in 1:length(labelsᵢ)
      lₙ = labelsᵢ[n]
      if iszero(dims[lₙ])
        dims[lₙ] = dim(isᵢ[n])
      end
    end
  end
  return labels, dims
end

label_dims(is...) = label_dims(is)

# Convert a contraction sequence in pair form to tree format.
# This is used in `depth_first_constructive` to convert the output.
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

##
## To delete, not used anymore
##

## # XXX: is this being used?
## function remove_common_pair!(isR, N1, n1)
##   if n1 > N1
##     return
##   end
##   N = length(isR)
##   is1 = isR[n1]
##   is2 = @view isR[N1+1:N]
##   n2 = findfirst(==(is1), is2)
##   if isnothing(n2)
##     n1 += 1
##     remove_common_pair!(isR, N1, n1)
##   else
##     deleteat!(isR, (n1, N1+n2))
##     N1 -= 1
##     remove_common_pair!(isR, N1, n1)
##   end
## end
## 
## # XXX: is this being used?
## function contract_inds(is1, is2)
##   IndexT = eltype(is1)
##   N1 = length(is1)
##   N2 = length(is2)
##   isR = IndexT[]
##   for n1 in 1:N1
##     i1 = is1[n1]
##     n2 = findfirst(==(i1), is2)
##     if isnothing(n2)
##       push!(isR, i1)
##     end
##   end
##   for n2 in 1:N2
##     i2 = is2[n2]
##     n1 = findfirst(==(i2), is1)
##     if isnothing(n1)
##       push!(isR, dag(i2))
##     end
##   end
##   return isR
## end
## 
## # Return the noncommon indices and the cost of contraction
## # Recursively removes pairs of indices that are common
## # between the IndexSets (TODO: use this for symdiff in ITensors.jl)
## function contract_inds_cost(is1is2::Tuple{Vector{Int}, Vector{Int}}, ind_dims::Vector{Int})
##   is1, is2 = is1is2
##   N1 = length(is1)
##   isR = vcat(is1, is2)
##   remove_common_pair!(isR, N1, 1)
##   cost = Int(sqrt(_dim(is1, ind_dims) * _dim(is2, ind_dims) * _dim(isR, ind_dims)))
##   return isR, cost
## end
## 
## function contract_inds_cost(is1::Vector{IndexT}, is2::Vector{IndexT}) where {IndexT <: Index}
##   N1 = length(is1)
## 
##   # This is pretty slow
##   #isR = vcat(is1, is2)
##   #remove_common_pair!(isR, N1, 1)
## 
##   isR = contract_inds(is1, is2)
## 
##   cost = Int(sqrt(_dim(is1) * _dim(is2) * _dim(isR)))
##   return isR, cost
## end
## 
## # XXX: is this being used?
## function contraction_cost(T::Vector{Vector{IndexT}}, a::Vector{Int},
##                           b::Vector{Int}) where {IndexT <: Index}
##   # The set of tensor indices `a` and `b`
##   Tᵃ = T[a]
##   Tᵇ = T[b]
## 
##   # XXX TODO: this should use a cache to store the results
##   # of the contraction
##   indsTᵃ = symdiff(Tᵃ...)
##   indsTᵇ = symdiff(Tᵇ...)
##   indsTᵃTᵇ = symdiff(indsTᵃ, indsTᵇ)
##   cost = Int(sqrt(prod(_dim(indsTᵃ) * prod(_dim(indsTᵇ) * _dim(indsTᵃTᵇ)))))
##   return cost
## end
## 
## # XXX: is this being used?
## function contraction_cost(T::Vector{BitSet}, dims::Vector{Int}, a::BitSet,
##                           b::BitSet) where {IndexT <: Index}
##   # The set of tensor indices `a` and `b`
##   Tᵃ = [T[aₙ] for aₙ in a]
##   Tᵇ = [T[bₙ] for bₙ in b]
## 
##   # XXX TODO: this should use a cache to store the results
##   # of the contraction
##   indsTᵃ = symdiff(Tᵃ...)
##   indsTᵇ = symdiff(Tᵇ...)
##   indsTᵃTᵇ = symdiff(indsTᵃ, indsTᵇ)
##   dim_a = _dim(indsTᵃ, dims)
##   dim_b = _dim(indsTᵇ, dims)
##   dim_ab = _dim(indsTᵃTᵇ, dims)
##   cost = Int(sqrt(Base.Checked.checked_mul(dim_a, dim_b, dim_ab)))
##   return cost
## end

