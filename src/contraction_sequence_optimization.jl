
module ContractionSequenceOptimization

  using ITensors
  using IterTools

  export depth_first_constructive

  function _dim(is::Vector{<: Index})
    isempty(is) && return 1
    return mapreduce(dim, *, is)
  end

  # Return the noncommon indices and the cost of contraction
  function contract_inds_cost(is1::Vector{IndexT},
                              is2::Vector{IndexT}) where {IndexT <: Index}
    N1 = length(is1)
    N2 = length(is2)
    Ncommon = 0
    dim_common = 1
    is1_contracted = fill(false, N1)
    is2_contracted = fill(false, N2)
    # Determine the contracted indices
    for (n1, i1) ∈ pairs(is1)
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

  function depth_first_constructive(T::Vector{<: ITensor})
    return depth_first_constructive(collect.(inds.(T)))
  end

  # TODO: use the initial sequence as a guess sequence, which
  # can be used to prune the tree
  function depth_first_constructive(T::Vector{Vector{IndexT}}) where {IndexT <: Index}
    best_cost = Ref(typemax(Int))
    best_sequence = Vector{Pair{Int, Int}}(undef, length(T)-1) #Any[]
    function _depth_first_constructive(sequence, T, remaining, cost)
      # Only should get here if the contraction was the best
      if length(remaining) == 1
        @assert cost ≤ best_cost[]
        best_cost[] = cost
        best_sequence .= sequence
      end

      for (a, b) in IterTools.subsets(remaining, Val(2))
        Tᵈ, current_cost = contract_inds_cost(T[a], T[b])
        new_cost = cost + current_cost
        if new_cost ≥ best_cost[]
          continue
        end
        new_sequence = push!(copy(sequence), a => b)
        new_T = push!(copy(T), Tᵈ)
        new_remaining = setdiff(remaining, [a, b])
        push!(new_remaining, length(new_T))
        _depth_first_constructive(new_sequence, new_T, new_remaining, new_cost)
      end
    end
    _depth_first_constructive(Pair{Int, Int}[], T, collect(1:length(T)), 0)
    return best_sequence, best_cost[]
  end

  #function _depth_first_constructive(T::Vector{<: Vector{IndexT}},
  #                                   S::Vector{Any},
  #                                   a, b, c, d, cost) where {IndexT <: Index}
  #  n = length(T)

  #  @show T
  #  @show S
  #  @show c
  #  @show n

  #  if n == 2
  #    _, current_cost = contract_inds_cost(T[1], T[2])
  #    cost += current_cost
  #    println()
  #    println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
  #    println("n = 2, finished, before loop")
  #    @show S
  #    @show cost
  #    @show a, b
  #    println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
  #    println()
  #    c = 0
  #    return T, S, a, b, c, d, cost
  #  end

  #  for a in 1:n-1, b in max(a, c) + 1:n
  #    n = length(T)
  #    if n == 2
  #      #_, current_cost = contract_inds_cost(T[1], T[2])
  #      #cost += current_cost
  #      println()
  #      println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
  #      println("n = 2, finished, inside loop")
  #      @show S
  #      @show cost
  #      @show a, b
  #      println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
  #      println()
  #      c = 0
  #      return T, S, a, b, c, d, cost
  #    end

  #    d += 1
  #    println()
  #    println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
  #    println("n = $n")
  #    @show S
  #    @show cost
  #    @show a, b
  #    println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
  #    println()

  #    # Choose b > a and b > c
  #    @assert b > a && b > c

  #    # TODO: add this back
  #    # First check if for any a, b, Tᵃ has the
  #    # same indices as Tᵇ
  #    #for α in 1:n-1, β in α+1:n
  #    #  if issetequal(T[α], T[β])
  #    #    @show α, β
  #    #    a, b = α, β
  #    #    break
  #    #  end
  #    #end

  #    Sᵈ = Any[S[a], S[b]]
  #    S = copy(S)
  #    deleteat!(S, a => b)
  #    push!(S, Sᵈ)

  #    Tᵈ, current_cost = contract_inds_cost(T[a], T[b])
  #    T = copy(T)
  #    deleteat!(T, a => b)
  #    push!(T, Tᵈ)

  #    cost += current_cost
  #    c = a
  #    n = length(T)

  #    T, S, a, b, c, d, cost = _depth_first_constructive(T, S, a, b, c, d, cost)
  #  end
  #end

  #function depth_first_constructive(Tᵢ::Vector{<: Vector{IndexT}}) where {IndexT <: Index}
  #  n = length(Tᵢ)
  #  Sᵢ = Any[i for i in 1:n]
  #  minS = Any[]
  #  mincost = typemax(Int) # Minimal cost found so far
  #  c = 0
  #  for a in 1:n-1, b in max(a, c)+1:n
  #    println()
  #    println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
  #    println("n = $n, start")
  #    @show a, b
  #    println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
  #    println()

  #    cost = 0 # Current contraction cost
  #    d = n # Counter for total number of tensors

  #    # Restart the sequence
  #    T = copy(Tᵢ)
  #    S = copy(Sᵢ)

  #    Sᵈ = Any[S[a], S[b]]
  #    deleteat!(S, a => b)
  #    push!(S, Sᵈ)

  #    Tᵈ, current_cost = contract_inds_cost(T[a], T[b])
  #    deleteat!(T, a => b)
  #    push!(T, Tᵈ)

  #    cost += current_cost
  #    c = a

  #    #T, S, a, b, c, d, cost = _depth_first_constructive(T, S, a, b, c, d, cost)
  #    @show _depth_first_constructive(T, S, a, b, c, d, cost)
  #    println("FINISHED")
  #    if cost < mincost
  #      minS = S
  #      mincost = cost
  #    end
  #  end
  #  return minS, mincost
  #end
end # module ContractionSequenceOptimization

