
#
# Breadth-first constructive approach
#

function breadth_first_constructive(T::Vector{<: ITensor}; fscale::Function = maximum)
  indsT = [inds(Tₙ) for Tₙ in T]
  ntensors = length(indsT)
  if ntensors ≤ 16
    return breadth_first_constructive(UInt16, DimT, indsT; fscale = fscale)
  elseif ntensors ≤ 32
    return breadth_first_constructive(UInt32, DimT, indsT; fscale = fscale)
  elseif ntensors ≤ 64
    return breadth_first_constructive(UInt64, DimT, indsT; fscale = fscale)
  elseif ntensors ≤ 128
    return breadth_first_constructive(UInt128, DimT, indsT; fscale = fscale)
  else
    return breadth_first_constructive(BitSet, DimT, indsT; fscale = fscale)
  end
end

function breadth_first_constructive(::Type{TensorSetT}, ::Type{DimT},
                                    T::Vector{IndexSetT}; fscale = maximum) where {IndexSetT <: IndexSet, TensorSetT, DimT}
  labels, dims = label_dims(DimT, T)
  nlabels = length(dims)
  if nlabels ≤ 16
    return breadth_first_constructive(TensorSetT, UInt16, labels, dims; fscale = fscale)
  elseif nlabels ≤ 32
    return breadth_first_constructive(TensorSetT, UInt32, labels, dims; fscale = fscale)
  elseif nlabels ≤ 64
    return breadth_first_constructive(TensorSetT, UInt64, labels, dims; fscale = fscale)
  elseif nlabels ≤ 128
    return breadth_first_constructive(TensorSetT, UInt128, labels, dims; fscale = fscale)
  else
    return breadth_first_constructive(TensorSetT, BitSet, labels, dims; fscale = fscale)
  end
end

function breadth_first_constructive(::Type{TensorSetT}, ::Type{LabelSetT}, labels::Vector, dims::Vector; fscale = maximum) where {TensorSetT, LabelSetT}
  return breadth_first_constructive(TensorSetT, map(label -> bitset(LabelSetT, label), labels), dims; fscale = fscale)
end

function breadth_first_constructive(::Type{TensorSetT}, ::Type{LabelSetT}, ::Type{DimT},
                                    T::Vector{<: ITensor}; fscale::Function = maximum) where {TensorSetT, LabelSetT, DimT}
  indsT = [inds(Tₙ) for Tₙ in T]
  return breadth_first_constructive(TensorSetT, LabelSetT, DimT, indsT; fscale = fscale)
end

function breadth_first_constructive(::Type{TensorSetT}, ::Type{LabelSetT}, ::Type{DimT},
                                    T::Vector{IndexSetT}; fscale = maximum) where {IndexSetT <: IndexSet, TensorSetT, LabelSetT, DimT}
  labels, dims = label_dims(DimT, T)
  return breadth_first_constructive(TensorSetT, map(label -> bitset(LabelSetT, label), labels), dims; fscale = fscale)
end

# A type storing information about subnetworks
const SubNetwork{LabelSetT} = NamedTuple{(:inds, :cost, :sequence, :isnew), Tuple{LabelSetT, Int64, Vector{Any}, Bool}}

function breadth_first_constructive(::Type{TensorSetT},
                                    T::Vector{LabelSetT},
                                    dims::Vector;
                                    fscale::Function = maximum) where {TensorSetT, LabelSetT}
  n = length(T)

  # `cache[c]` is the set of all objects made up by
  # contracting `c` unique tensors from the original tensors `1:n`.
  cache = Vector{Dict{TensorSetT, SubNetwork{LabelSetT}}}(undef, n)
  for c in 1:n
    # Initialized to empty
    cache[c] = eltype(cache)()
  end
  # Fill the first cache with trivial data
  for i in 1:n
    cache[1][bitset(TensorSetT, [i])] = (inds = T[i], cost = 0, sequence = Any[], isnew = false)
  end

  μᶜᵃᵖ = 1
  μᵒˡᵈ = 0
  # Scale the cost lower bound by this amount
  ξᶠᵃᶜᵗ = fscale(dims)
  # For now, don't support dimension 1 indices
  @assert ξᶠᵃᶜᵗ > 1

  while isempty(cache[n])
    μⁿᵉˣᵗ = typemax(Int)

    # c is the total number of tensors being contracted
    # in the current sequence
    for c in 2:n
      # For each pair of sets Sᵈ, Sᶜ⁻ᵈ, 1 ≤ d ≤ ⌊c/2⌋
      for d in 1:c÷2
        for a in keys(cache[d]), b in keys(cache[c-d])

          if d == c-d && _isless(b, a)
            # When d == c-d (the subset sizes are equal), check that
            # b > a so that that case (a,b) and (b,a) are not repeated
            continue
          end
          if !_isemptyset(_intersect(a, b))
            # Check that each element of S¹ appears
            # at most once in (TᵃTᵇ).
            continue
          end
          # Determine the cost μ of contracting Tᵃ, Tᵇ
          # These dictionary calls and `contraction_cost` take
          # up most of the time.
          cache_a = cache[d][a]
          cache_b = cache[c-d][b]
          μ, inds_ab = contraction_cost(cache_a.inds, cache_b.inds, dims)
          if d > 1
            μ += cache_a.cost
          end
          if c-d > 1
            μ += cache_b.cost
          end
          if cache_a.isnew || cache_a.isnew
            μ⁰ = 0
          else
            μ⁰ = μᵒˡᵈ
          end
          if μ > μᶜᵃᵖ && μ < μⁿᵉˣᵗ
            μⁿᵉˣᵗ = μ
          end

          if μ⁰ < μ ≤ μᶜᵃᵖ
            ab = _union(a, b)
            # TODO: perform get and set `ab` in a single call
            cache_c = @inbounds cache[c]
            cache_ab = get(cache_c, ab, nothing)
            old_cost = isnothing(cache_ab) ? typemax(Int) : cache_ab.cost
            if μ < old_cost
              cost_ab = μ
              inds_ab = inds_ab
              if d == 1
                sequence_a = _only(a)
              else
                sequence_a = cache_a.sequence
              end
              if c-d == 1
                sequence_b = _only(b)
              else
                sequence_b = cache_b.sequence
              end
              sequence_ab = Any[sequence_a, sequence_b]
              isnew_ab = true
              # XXX: this call is pretty slow (maybe takes 1/3 of total time in large n limit)
              cache_c[ab] = (inds = inds_ab, cost = cost_ab, sequence = sequence_ab, isnew = isnew_ab)
            end
          end # if μ⁰ < μ ≤ μᶜᵃᵖ
        end # for a in S[d], b in S[c-d]
      end # for d in 1:c÷2
    end # for c in 2:n
    μᵒˡᵈ = μᶜᵃᵖ
    μᶜᵃᵖ = max(μⁿᵉˣᵗ, ξᶠᵃᶜᵗ * μᶜᵃᵖ)
    # Reset all tensors to old
    for i in 1:n
      for a in eachindex(cache[i])
        cache_a = cache[i][a]
        cache[i][a] = (inds = cache_a.inds, cost = cache_a.cost, sequence = cache_a.sequence, isnew = false)
      end
    end
  end # while isempty(S[n])
  Sⁿ = bitset(TensorSetT, 1:n)
  return cache[n][Sⁿ].sequence, cache[n][Sⁿ].cost
end

