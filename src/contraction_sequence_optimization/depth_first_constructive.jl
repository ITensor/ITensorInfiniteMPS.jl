
function depth_first_constructive(T::Vector{<: ITensor})
  indsT = [inds(Tₙ) for Tₙ in T]
  return depth_first_constructive(indsT)
end

function depth_first_constructive(T::Vector{IndexSetT}) where {IndexSetT <: IndexSet}
  N = length(T)
  IndexT = eltype(IndexSetT)
  Tinds = Vector{IndexT}[Vector{IndexT}(undef, length(T[n])) for n in 1:N]
  for n in 1:N
    T_n = T[n]
    Tinds_n = Tinds[n]
    for j in 1:length(Tinds_n)
      Tinds_n[j] = T_n[j]
    end
  end
  return depth_first_constructive(Tinds)
end

# TODO: convert to unsigned integer/BitSet for labels
function depth_first_constructive(T::Vector{Vector{IndexT}}) where {IndexT <: Index}
  optimal_cost = Ref(typemax(Int))
  optimal_sequence = Vector{Pair{Int, Int}}(undef, length(T)-1)
  _depth_first_constructive!(optimal_sequence, optimal_cost, Pair{Int, Int}[], T, collect(1:length(T)), 0)
  return pair_sequence_to_tree(optimal_sequence, length(T)), optimal_cost[]
end

function _depth_first_constructive!(optimal_sequence, optimal_cost, sequence, T, remaining, cost)
  if length(remaining) == 1
    # Only should get here if the contraction was the best
    # Otherwise it would have hit the `continue` below
    @assert cost ≤ optimal_cost[]
    optimal_cost[] = cost
    optimal_sequence .= sequence
  end
  for aᵢ in 1:length(remaining)-1, bᵢ in aᵢ+1:length(remaining)
    a = remaining[aᵢ]
    b = remaining[bᵢ]
    Tᵈ, current_cost = contract_inds_cost(T[a], T[b])
    new_cost = cost + current_cost
    if new_cost ≥ optimal_cost[]
      continue
    end
    new_sequence = push!(copy(sequence), a => b)
    new_T = push!(copy(T), Tᵈ)
    new_remaining = deleteat!(copy(remaining), (aᵢ, bᵢ))
    push!(new_remaining, length(new_T))
    _depth_first_constructive!(optimal_sequence, optimal_cost, new_sequence, new_T, new_remaining, new_cost)
  end
end

