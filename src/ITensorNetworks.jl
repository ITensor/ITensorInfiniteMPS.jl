
export eachneighborindex, nnodes

const IndexSetNetwork = Vector{Dict{Int,IndexSet}}

IndexSetNetwork(n::Integer) = [eltype(IndexSetNetwork)() for _ in 1:n]

struct ITensorNetwork
  itensors::Vector{ITensor}
  indsnetwork::IndexSetNetwork
end

Base.length(tn::ITensorNetwork) = length(itensors(tn))
Base.copy(tn::ITensorNetwork) = ITensorNetwork(copy(itensors(tn)), copy.(inds(tn)))

# Call this with tn[1:end]
itensors(tn::ITensorNetwork) = tn.itensors

ITensors.inds(tn::ITensorNetwork) = tn.indsnetwork

# TODO: should this return an ITensorNetwork with 1 node?
Base.getindex(tn::ITensorNetwork, i::Integer) = itensors(tn)[i]

# TODO: should this update reverse(ij)?
function Base.setindex!(isn::IndexSetNetwork, is::IndexSet, ij::Pair{<:Integer,<:Integer})
  i, j = ij
  isn[i][j] = is
  #isn[j][i] = dag(is)
  return isn
end

# TODO: also update indsnetwork with the new indices
Base.setindex!(tn::ITensorNetwork, t::ITensor, i::Integer) = (itensors(tn)[i] = t)

function setinds!(tn::ITensorNetwork, is::IndexSet, ij::Pair{<:Integer,<:Integer})
  return inds(tn)[ij] = is
end

# Iteration iterates through the ITensors of the network
Base.iterate(tn::ITensorNetwork, args...) = iterate(itensors(tn), args...)

# Iterate over the nodes that are neighbors of the specified node
# For example, for an ITensorNetwork `tn = [A₁, A₂, A₃, A₄]`:
#
# --A₁--A₂--A₃--A₄--
#
# `collect(eachneighborindex(tn, 2)) == [1, 3]`
eachneighborindex(isn::IndexSetNetwork, i::Integer) = Iterators.filter(==(i), keys(isn[i]))

# Iterator over each IndexSet connected to the node `i`
eachlinkinds(isn::IndexSetNetwork, i::Integer) = values(isn[i])

eachneighborindex(tn::ITensorNetwork, i::Integer) = eachneighborindex(inds(tn))

function ITensorNetwork(tensors::Vector{<:ITensor})
  N = length(tensors)
  indsnetwork = [eltype(IndexSetNetwork)() for n in 1:N]
  # Get the "link" indices
  # Determine the neighborhood of each node/tensor
  # through common indices
  for i in 1:N
    for j in (i+1):N
      linkindsᵢⱼ = commoninds(tensors[i], tensors[j])
      if !isempty(linkindsᵢⱼ)
        indsnetwork[i][j] = linkindsᵢⱼ
        indsnetwork[j][i] = dag(indsnetwork[i][j])
      end
    end
  end
  # Get the "site" indices
  for i in 1:N
    indsnetwork[i][i] = uniqueinds(tensors[i], eachlinkinds(indsnetwork, i)...)
  end
  return ITensorNetwork(tensors, indsnetwork)
end

function Base.getindex(tn::IndexSetNetwork, ij::Pair{<:Integer,<:Integer})
  i, j = ij
  @assert j ∈ 1:nnodes(tn)
  return get(tn[i], j, IndexSet())
end

ITensors.inds(tn::ITensorNetwork, ij::Pair{<:Integer,<:Integer}) = inds(tn)[ij]

function ITensors.linkinds(tn::ITensorNetwork, ij::Pair{<:Integer,<:Integer})
  i, j = ij
  @assert i ≠ j
  return inds(tn, ij)
end

# Get the self edges/site indices of node `i`
ITensors.siteinds(tn::ITensorNetwork, i::Integer) = inds(tn, i => i)

# Get all of the self edges/site indices of the network, returning `s` such 
# that `s[i] == siteinds(tn, i)`
ITensors.siteinds(tn::ITensorNetwork) = IndexSet[siteinds(tn, i) for i in 1:nnodes(tn)]

# Number of nodes/vertices/tensors of the graph/network
nnodes(isn::IndexSetNetwork) = length(isn)
function nnodes(tn::ITensorNetwork)
  n = nnodes(inds(tn))
  @assert n == length(itensors(tn))
  return n
end

⊗(Ts::ITensor...) = ITensorNetwork([Ts...])

# TODO: build this by appending T to the network instead
# of rebuilding the entire network
⊗(Ts::ITensorNetwork, T::ITensor) = ITensorNetwork([Ts..., T])

⊗(TN1::ITensorNetwork, TN2::ITensorNetwork) = ITensorNetwork([TN1..., TN2...])

#
# ITensors.jl extensions
#

ITensors.unioninds(is::Vector{<:IndexSet}) = unioninds(is...)

Base.:*(tn::ITensorNetwork) = *(tn...)
Base.:*(t::ITensor, tn::ITensorNetwork) = *(t, tn...)
Base.:*(tn::ITensorNetwork, t::ITensor) = *(tn..., t)
Base.:*(tn1::ITensorNetwork, tn2::ITensorNetwork) = *(tn1..., tn2...)

function ITensors.dag(tn::ITensorNetwork)
  tn = copy(tn)
  N = nnodes(tn)
  for i in 1:N
    tn[i] = dag(tn[i])
    for j in keys(inds(tn)[i])
      setinds!(tn, dag(inds(tn, i => j)), i => j)
    end
  end
  return tn
end

#
# Contraction sequence optimization
#

function trivial_contraction_sequence(tn::ITensorNetwork)
  N = nnodes(tn)
  # N-1 pairwise contractions
  t = 1
  for n in 2:N
    t = [n, t]
  end
  return t
end

function contract_pair!(T⃗, ab)
  a, b = ab
  Tᵃ, Tᵇ = T⃗[a], T⃗[b]
  # Contract a and b
  Tᵈ = Any[Tᵃ, Tᵇ]
  # Remove tensors Tᵃ and Tᵇ from the list of tensors
  deleteat!(T⃗, [a, b])
  # Append Tᵈ to the list of tensors
  push!(T⃗, Tᵈ)
  return T⃗
end

# Get all tensor pairs for a given number of tensors
function tensor_pairs(n::Integer)
  pairs = Pair{Int,Int}[]
  for i in 1:(n-1)
    for j in (i+1):n
      push!(pairs, i => j)
    end
  end
  return pairs
end

# Depth-first constructive approach
# TODO: this isn't pruning repeats
function each_contraction_sequence(n::Integer)
  Ts = Any[]
  each_each_ab = Iterators.product(
    (ITensorInfiniteMPS.tensor_pairs(n) for n in reverse(2:n))...
  )
  for each_ab in each_each_ab
    T = Any[1:n...]
    for ab in each_ab
      contract_pair!(T, ab)
    end
    push!(Ts, T[])
  end
  return Ts
end

each_contraction_sequence(tn::ITensorNetwork) = each_contraction_sequence(nnodes(tn))
function each_contraction_sequence(isn::Vector{<:Vector{<:Index}})
  return each_contraction_sequence(length(isn))
end
each_contraction_sequence(isn::IndexSetNetwork) = each_contraction_sequence(length(isn))

#
# Contraction cost using Vector{<: Vector{<: IndexSet}}
#

contraction_cost(T1::ITensor, T2::ITensor) = contraction_cost(inds(T1), inds(T2))

#
# TODO: use the following for symdiff
# Index[setdiff(inds(TN[1]), inds(TN[2]))..., setdiff(inds(TN[2]), inds(TN[1]))...]
#

function _intersect(A::Vector{IndexT}, B::Vector{IndexT}) where {IndexT<:Index}
  R = IndexT[]
  for a in A
    a ∈ B && push!(R, a)
  end
  return R
end

function _setdiff(A::Vector{IndexT}, B::Vector{IndexT}) where {IndexT<:Index}
  R = IndexT[]
  for a in A
    a ∉ B && push!(R, a)
  end
  return R
end

# Faster symdiff
function _symdiff(is1::Vector{IndexT}, is2::Vector{IndexT}) where {IndexT<:Index}
  setdiff12 = _setdiff(is1, is2)
  setdiff21 = _setdiff(is2, is1)
  display(setdiff12)
  display(setdiff21)
  return IndexT[_setdiff(is1, is2)..., _setdiff(is2, is1)...]
end

# Return the noncommon indices and the cost of contraction
function _contract_inds(is1::Vector{IndexT}, is2::Vector{IndexT}) where {IndexT<:Index}
  N1 = length(is1)
  N2 = length(is2)
  Ncommon = 0
  dim_common = 1
  is1_contracted = fill(false, N1)
  is2_contracted = fill(false, N2)
  # Determine the contracted indices
  for (n1, i1) in pairs(is1)
    n2 = findfirst(==(i1), is2)
    if !isnothing(n2)
      Ncommon += 1
      dim_common *= dim(i1)
      is1_contracted[n1] = true
      is2_contracted[n2] = true
    end
  end
  cost = _dim(is1) * _dim(is2) ÷ dim_common
  Nnoncommon = N1 + N2 - 2 * Ncommon
  isR = Vector{IndexT}(undef, Nnoncommon)
  n = 1
  for n1 in 1:N1
    if !is1_contracted[n1]
      isR[n] = is1[n1]
      n += 1
    end
  end
  for n2 in 1:N2
    if !is2_contracted[n2]
      isR[n] = is2[n2]
      n += 1
    end
  end
  return isR, cost
end

function _dim(is::Vector{<:Index})
  isempty(is) && return 1
  return mapreduce(dim, *, is)
end

function contraction_cost(is1::Vector{IndexT}, is2::Vector{IndexT}) where {IndexT<:Index}
  # 1.169417 s for N = 7 network
  #common_is = _intersect(is1, is2)
  #noncommon_is = _symdiff(is1, is2)
  #return noncommon_is, _dim(common_is) * _dim(noncommon_is)

  # 0.635962 s for N = 7 network
  return _contract_inds(is1, is2)
end

function contraction_cost!(
  cost::Ref{Int}, is1::Vector{IndexT}, is2::Vector{IndexT}
) where {IndexT<:Index}
  contracted_is, current_cost = contraction_cost(is1, is2)
  cost[] += current_cost
  return contracted_is
end

contraction_cost!(cost::Ref{Int}, ::Vector{Union{}}, ::Vector{Union{}}) = (cost[] += 1)

function contraction_cost!(cost::Ref{Int}, is::Vector{<:Vector{<:Index}}, n::Integer)
  return is[n]
end

function contraction_cost!(cost::Ref{Int}, is::Vector{<:Vector{<:Index}}, sequence)
  return contraction_cost!(
    cost, contraction_cost!(cost, is, sequence[1]), contraction_cost!(cost, is, sequence[2])
  )
end

function contraction_cost(isn::Vector{<:Vector{<:Index}}, sequence)
  # TODO: use the network itself
  cost = Ref(0)
  contraction_cost!(cost, isn, sequence)
  return cost[]
end

function contraction_cost(tn::ITensorNetwork, sequence)
  return contraction_cost(collect.(inds.(tn)), sequence)
end

# Return the optimal contraction sequence
function optimal_contraction_sequence(isn::Vector{<:Vector{<:Index}})
  sequences = each_contraction_sequence(isn)
  contraction_costs = map(sequence -> contraction_cost(isn, sequence), sequences)
  mincost, minsequenceindex = findmin(contraction_costs)
  return sequences[minsequenceindex], mincost
end

# Return the optimal contraction sequence
function optimal_contraction_sequence(tn::ITensorNetwork)
  if length(tn) == 1
    return Any[1], 0
  end
  #return optimal_contraction_sequence(inds(tn))
  #return optimal_contraction_sequence(inds.(itensors(tn)))
  return optimal_contraction_sequence(collect.(inds.(itensors(tn))))
end
