
module ContractionSequenceOptimization

  using ITensors
  using IterTools

  export depth_first_constructive

  function _dim(is::Vector{<: Index})
    isempty(is) && return 1
    return mapreduce(dim, *, is)
  end

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

  # Return the noncommon indices and the cost of contraction
  # Recursively removes pairs of indices that are common
  # between the IndexSets (TODO: use this for symdiff in ITensors.jl)
  function contract_inds_cost(is1::Vector{IndexT},
                              is2::Vector{IndexT}) where {IndexT <: Index}
    N1 = length(is1)
    isR = vcat(is1, is2)
    remove_common_pair!(isR, N1, 1)
    cost = Int(sqrt(_dim(is1) * _dim(is2) * _dim(isR)))
    return isR, cost
  end

  function depth_first_constructive(T::Vector{<: ITensor})
    return depth_first_constructive(collect.(inds.(T)))
  end

  # TODO: use the initial sequence as a guess sequence, which
  # can be used to prune the tree
  function depth_first_constructive(T::Vector{Vector{IndexT}}) where {IndexT <: Index}
    best_cost = Ref(typemax(Int))
    best_sequence = Vector{Pair{Int, Int}}(undef, length(T)-1)
    function _depth_first_constructive(sequence, T, remaining, cost)
      if length(remaining) == 1
        # Only should get here if the contraction was the best
        # Otherwise it would have hit the `continue` below
        @assert cost ≤ best_cost[]
        best_cost[] = cost
        best_sequence .= sequence
      end
      for aᵢ in 1:length(remaining)-1, bᵢ in aᵢ+1:length(remaining)
        a = remaining[aᵢ]
        b = remaining[bᵢ]
        Tᵈ, current_cost = contract_inds_cost(T[a], T[b])
        new_cost = cost + current_cost
        if new_cost ≥ best_cost[]
          continue
        end
        new_sequence = push!(copy(sequence), a => b)
        new_T = push!(copy(T), Tᵈ)
        new_remaining = deleteat!(copy(remaining), (aᵢ, bᵢ))
        push!(new_remaining, length(new_T))
        _depth_first_constructive(new_sequence, new_T, new_remaining, new_cost)
      end
    end
    _depth_first_constructive(Pair{Int, Int}[], T, collect(1:length(T)), 0)
    return best_sequence, best_cost[]
  end

end # module ContractionSequenceOptimization

