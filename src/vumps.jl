
# Also input C bond matrices to help compute the right fixed points
# of ψ (R ≈ C * dag(C))
function left_environment(H::InfiniteMPO, ψ::InfiniteMPS)
  N = nsites(H)
  @assert N == nsites(ψ)

  # Do the 1-site case first
  @assert N == 1

  # MPO bond dimensions
  d⃗ₕ = OffsetArray([dim(commoninds(H[n], H[n+1])) for n in 0:N], -1)
  @show d⃗ₕ

  l⃗ = OffsetArray([commoninds(H[n], H[n+1]) for n in 0:N], -1)
  @show l⃗

  # Collect the on-site MPO operators obtained by projecting
  # the link indices onto certain elements
  H⃡ = Vector{Matrix{ITensor}}(undef, N)
  for n in 1:N
    @show n
    H⃡[n] = Matrix{ITensor}(undef, d⃗ₕ[n-1], d⃗ₕ[n])
    for i in 1:d⃗ₕ[n-1], j in 1:d⃗ₕ[n]
      H⃡[n][i, j] = H[n] * setelt((l⃗[n-1] .=> i)...) * setelt((l⃗[n] .=> j)...)
      @show i, j
      @show norm(H⃡[n][i, j])
      # Assume lower triangular
      if j > i
        @assert norm(H⃡[n][i, j]) ≈ 0
      end
      # Also assume nonzero diagonal entries
      # are proportional to zero
    end
  end

  # Starting point of identity
  n = 1 # Assuming we have only one site
  # This is the set of left quasi fixed points
  L⃗ = Vector{ITensor}(undef, d⃗ₕ[n])
  # The last one is defined to be identity
  L⃗[d⃗ₕ[n]] = δ(l⃗[n]'..., dag(l⃗[n])...)
  for a in reverse(1:d⃗ₕ[n]-1)
    @show a
    YLᵃ = emptyITensor(Any)
    for b in a+1:d⃗ₕ[n] 
      YLᵃ += L⃗[b] * ψ[n] * H⃡[n][b, a] * dag(ψ[n]')
    end
    error("Now calculate L⃗[a] using geometric sums")
    if H⃡[n][a, a] # proportional to identity
      # Solve using C21
    elseif H⃡[n][a, a] # equal to identity
      # Solve using C25a
    else H⃡[n][a, a] # is zero
      L⃗[a] = YLᵃ
    end
  end
end

vumps(H::InfiniteMPO, ψ::InfiniteMPS; kwargs...) = vumps(H, orthogonalize(ψ, :); kwargs...)

function vumps(H::InfiniteMPO, ψ::InfiniteCanonicalMPS)
  L = left_environment(H, ψ.AL)
end

