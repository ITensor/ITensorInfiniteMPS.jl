
module ContractionSequenceOptimization

  using ITensors

  export depth_first_constructive

  function _dim(is::Vector{<: Index})
    isempty(is) && return 1
    return mapreduce(dim, *, is)
  end

  function _dim(is::Vector{Int}, ind_dims::Vector{Int})
    isempty(is) && return 1
    dim = 1
    for n in 1:length(is)
      dim *= ind_dims[is[n]]
    end
    return dim
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
    isR = vcat(is1, is2)
    remove_common_pair!(isR, N1, 1)
    cost = Int(sqrt(_dim(is1) * _dim(is2) * _dim(isR)))
    return isR, cost
  end

  function optimize_three_tensor_contraction_sequence(T::Vector{<: ITensor})
    @assert length(T) == 3
    indsT = [inds(t) for t in T]
    return optimize_three_tensor_contraction_sequence(indsT)
  end

  function optimize_three_tensor_contraction_sequence(T::Vector{<: IndexSet})
    N = length(T)
    IndexT = eltype(T[1])
    Tinds = Vector{IndexT}[Vector{IndexT}(undef, length(T[n])) for n in 1:N]
    for n in 1:N
      T_n = T[n]
      Tinds_n = Tinds[n]
      for j in 1:length(Tinds_n)
        Tinds_n[j] = T_n[j]
      end
    end
    return optimize_three_tensor_contraction_sequence(Tinds)
  end

  function compute_cost(external_dims::Tuple{Int, Int, Int},
                        internal_dims::Tuple{Int, Int, Int})
    dim11, dim22, dim33 = external_dims
    dim12, dim23, dim31 = internal_dims
    cost12 = dim11 * dim22 * dim12 * dim23 * dim31
    return cost12 + dim11 * dim22 * dim33 * dim31 * dim23
  end

  function three_tensor_contraction_sequence(which_sequence::Int)::Vector{Any}
    @assert 1 ≤ which_sequence ≤ 3
    return if which_sequence == 1
      Any[3, Any[1, 2]]
    elseif which_sequence == 2
      Any[1, Any[2, 3]]
    else
      Any[2, Any[3, 1]]
    end
  end

  function optimize_three_tensor_contraction_sequence(is::Vector{Vector{IndexT}}) where {IndexT <: Index}
    @assert length(is) == 3
    is1 = is[1]
    is2 = is[2]
    is3 = is[3]
    dim2 = _dim(is2)
    dim3 = _dim(is3)
    dim11 = 1
    dim12 = 1
    dim31 = 1
    @inbounds for n1 in 1:length(is1)
      i1 = is1[n1]
      n2 = findfirst(==(i1), is2)
      if isnothing(n2)
        n3 = findfirst(==(i1), is3)
        if isnothing(n3)
          dim11 *= dim(i1)
          continue
        end
        dim31 *= dim(i1)
        deleteat!(is3, n3)
        continue
      end
      dim12 *= dim(i1)
      deleteat!(is2, n2)
    end
    dim23 = 1
    @inbounds for n2 in 1:length(is2)
      i2 = is2[n2]
      n3 = findfirst(==(i2), is3)
      if !isnothing(n3)
        dim23 *= dim(i2)
        deleteat!(is3, n3)
      end
    end
    dim22 = dim2 ÷ (dim12 * dim23)
    dim33 = dim3 ÷ (dim23 * dim31)
    external_dims1 = (dim11, dim22, dim33)
    internal_dims1 = (dim12, dim23, dim31)
    external_dims2 = (dim22, dim33, dim11)
    internal_dims2 = (dim23, dim31, dim12)
    external_dims3 = (dim33, dim11, dim22)
    internal_dims3 = (dim31, dim12, dim23)
    cost1 = compute_cost(external_dims1, internal_dims1)
    cost2 = compute_cost(external_dims2, internal_dims2)
    cost3 = compute_cost(external_dims3, internal_dims3)
    mincost, which_sequence = findmin((cost1, cost2, cost3))
    sequence = three_tensor_contraction_sequence(which_sequence)
    return sequence, mincost
  end

  # Converts the indices to integer labels
  # and returns a Vector that takes those labels
  # and returns the original integer dimensions
  function inds_to_ints(T::Vector{Vector{IndexT}}) where {IndexT <: Index}
    N = length(T)
    ind_to_int = Dict{IndexT, Int}()
    ints = Vector{Int}[Vector{Int}(undef, length(T[n])) for n in 1:N]
    ind_dims = Vector{Int}(undef, sum(length, T))
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

  # Convert a contraction sequence in pair form to tree format
  function pair_sequence_to_tree(pairs::Vector{Pair{Int, Int}}, N::Int)
    trees = Any[1:N...]
    for p in pairs
      push!(trees, Any[trees[p[1]], trees[p[2]]])
    end
    return trees[end]
  end

  function _depth_first_constructive!(optimal_sequence, optimal_cost, cost_cache, sequence, T, ind_dims, remaining, cost)
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
      Tᵈ, current_cost = get!(cost_cache, (T[a], T[b])) do
        contract_inds_cost((T[a], T[b]), ind_dims)
      end
      new_cost = cost + current_cost
      if new_cost ≥ optimal_cost[]
        continue
      end
      new_sequence = push!(copy(sequence), a => b)
      new_T = push!(copy(T), Tᵈ)
      new_remaining = deleteat!(copy(remaining), (aᵢ, bᵢ))
      push!(new_remaining, length(new_T))
      _depth_first_constructive!(optimal_sequence, optimal_cost, cost_cache, new_sequence, new_T, ind_dims, new_remaining, new_cost)
    end
  end

  # No caching
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

  function depth_first_constructive(T::Vector{<: ITensor}; enable_caching::Bool = false)
    if length(T) == 1
      return Any[1], 0
    elseif length(T) == 2
      return Any[1 2], 0
    elseif length(T) == 3
      return optimize_three_tensor_contraction_sequence(T)
    end
    return depth_first_constructive(inds.(T); enable_caching = enable_caching)
  end

  function depth_first_constructive(T::Vector{IndexSetT}; enable_caching::Bool = false) where {IndexSetT <: IndexSet}
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
    return depth_first_constructive(Tinds; enable_caching = enable_caching)
  end

  function depth_first_constructive(T::Vector{Vector{IndexT}}; enable_caching = enable_caching) where {IndexT <: Index}
    return if enable_caching
      T′, ind_dims = inds_to_ints(T)
      depth_first_constructive_caching(T′, ind_dims)
    else
      depth_first_constructive_no_caching(T)
    end
  end

  # TODO: use the initial sequence as a guess sequence, which
  # can be used to prune the tree
  function depth_first_constructive_caching(T::Vector{Vector{Int}}, ind_dims::Vector{Int})
    optimal_cost = Ref(typemax(Int))
    optimal_sequence = Vector{Pair{Int, Int}}(undef, length(T)-1)
    # Memoize index contractions and costs that have already been seen
    cost_cache = Dict{Tuple{Vector{Int}, Vector{Int}}, Tuple{Vector{Int}, Int}}()
    _depth_first_constructive!(optimal_sequence, optimal_cost, cost_cache, Pair{Int, Int}[], T, ind_dims, collect(1:length(T)), 0)
    return pair_sequence_to_tree(optimal_sequence, length(T)), optimal_cost[]
  end

  function depth_first_constructive_no_caching(T::Vector{Vector{IndexT}}) where {IndexT <: Index}
    optimal_cost = Ref(typemax(Int))
    optimal_sequence = Vector{Pair{Int, Int}}(undef, length(T)-1)
    _depth_first_constructive!(optimal_sequence, optimal_cost, Pair{Int, Int}[], T, collect(1:length(T)), 0)
    return pair_sequence_to_tree(optimal_sequence, length(T)), optimal_cost[]
  end

end # module ContractionSequenceOptimization

