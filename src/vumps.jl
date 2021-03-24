
# Make an ITensorMap representing the transfer matrix T|v> = |Tv>
function transfer_matrix(ψ::InfiniteMPS)
  N = nsites(ψ)
  ψᴴ = prime(linkinds, dag(ψ))

  # The unit cell range
  # Turn into function `eachcellindex(ψ::InfiniteMPS, cell::Integer = 1)`
  cell₁ = 1:N
  # A transfer matrix made from the 1st unit cell of
  # the infinite MPS
  # TODO: make a `Cell` Integer type and call as `ψ[Cell(1)]`
  # TODO: make a TransferMatrix wrapper that automatically
  # primes and daggers, so it can be called with:
  # T = TransferMatrix(ψ[Cell(1)])
  ψ₁ = ψ[cell₁]
  ψ₁ᴴ = ψᴴ[cell₁]
  T₀₁ = ITensorMap(ψ₁, ψ₁ᴴ;
                   input_inds = unioninds(commoninds(ψ[N], ψ[N+1]), commoninds(ψᴴ[N], ψᴴ[N+1])),
                   output_inds = unioninds(commoninds(ψ[1], ψ[0]), commoninds(ψᴴ[1], ψᴴ[0])))
  return T₀₁
end

# Combine Left and Right, rename ProjectedTransferMatrix
struct ProjectedTransferMatrix
  T::ITensorMap
  R::ITensor
end
function (T::ProjectedTransferMatrix)(v::ITensor)
  return v - T.T(v) + (v * T.R) * δ(inds(v))
end

function operator_matrix(H::InfiniteMPO)
  N = nsites(H)

  # MPO bond dimensions
  d⃗ₕ = OffsetArray([dim(commoninds(H[n], H[n+1])) for n in 0:N], -1)

  # MPO link indices
  l⃗ₕ = OffsetArray([commoninds(H[n], H[n+1]) for n in 0:N], -1)

  # Collect the on-site MPO operators obtained by projecting
  # the link indices onto certain elements
  H⃡ = Vector{Matrix{ITensor}}(undef, N)
  for n in 1:N
    H⃡[n] = Matrix{ITensor}(undef, d⃗ₕ[n-1], d⃗ₕ[n])
    for i in 1:d⃗ₕ[n-1], j in 1:d⃗ₕ[n]
      H⃡[n][i, j] = H[n] * setelt((l⃗ₕ[n-1] .=> i)...) * setelt((l⃗ₕ[n] .=> j)...)
      # Assume lower triangular
      if j > i
        @assert norm(H⃡[n][i, j]) ≈ 0
      end
      # Also assume nonzero diagonal entries
      # are proportional to zero
    end

    println("norm.(H⃡[n])")
    display(norm.(H⃡[n]))

  end
  return H⃡
end

# Also input C bond matrices to help compute the right fixed points
# of ψ (R ≈ C * dag(C))
function left_environment(H::InfiniteMPO, ψ::InfiniteMPS, C::InfiniteMPS)
  N = nsites(H)
  @assert N == nsites(ψ)

  # Do the 1-site case first
  @assert N == 1

  # Transfer matrix between cells
  # T(v) performs a left multiplication
  T = transpose(transfer_matrix(ψ))
  R = C[N] * prime(dag(C[N]), linkinds(ψ, (N, N+1)))
  R = translatecell(R, -1)

  # Operator valued matrix representation of MPO
  H⃡ = operator_matrix(H)

  # MPO bond dimensions
  d⃗ₕ = OffsetArray([dim(commoninds(H[n], H[n+1])) for n in 0:N], -1)

  # MPO link indices
  l⃗ₕ = OffsetArray([commoninds(H[n], H[n+1]) for n in 0:N], -1)

  # MPS link indices
  l⃗ = OffsetArray([commoninds(ψ[n], ψ[n+1]) for n in 0:N], -1)

  # Starting point of identity
  n = 1 # Assuming we have only one site
  # This is the set of left quasi fixed points
  L⃗ = Vector{ITensor}(undef, d⃗ₕ[n])
  # The last one is defined to be identity
  L⃗[d⃗ₕ[n]] = δ(l⃗[n-1]'..., dag(l⃗[n-1])...)

  for a in reverse(1:d⃗ₕ[n]-1)
    YLᵃ = emptyITensor()
    for b in a+1:d⃗ₕ[n] 
      YLᵃ += L⃗[b] * ψ[n] * H⃡[n][b, a] * dag(ψ[n]')
    end
    YLᵃ = translatecell(YLᵃ, -1)
    Hnᵃᵃ = H⃡[n][a, a]
    λᵃ = Hnᵃᵃ[1, 1]
    δᵃ = δ(inds(Hnᵃᵃ))

    if abs(λᵃ) > 1
      error("Diagonal operator of the MPO must be proportional to identity with proprtionality constant less than or equal to 1.")
    end

    if Hnᵃᵃ ≠ λᵃ * δᵃ
      error("Diagonal operator of the MPO must be proportional to identity.")
    end

    if λᵃ < 1
      error("Diagonal elements proportional to identity not currently supported.")
      # Solve using C21
    elseif isone(λᵃ) # equal to identity
      println("Diagonal elements equal to identity.")
      # Solve using C25a
      # Solve:
      # (Lᵃ|[𝟏 - Tᴸ + |R)(𝟏|] = (YLᵃ| - (YLᵃ|R)(𝟏|
      b = YLᵃ - (YLᵃ * R) * δ(inds(YLᵃ))
      L⃗[a], _ = linsolve(ProjectedTransferMatrix(T, R), b)

      # Get error
      err_lhs = L⃗[a] - translatecell(L⃗[a] * ψ[1] * dag(prime(ψ[1], "Link")), -1) + L⃗[a] * R * δ(inds(L⃗[a]))
      err_rhs = YLᵃ - YLᵃ * R * δ(inds(YLᵃ))
      @show norm(err_lhs - err_rhs)
    elseif iszero(λᵃ) # is zero
      println("Diagonal element is zero")
      L⃗[a] = YLᵃ
    else
      error("The diagonal operator of the MPO must be zero, identity, or proportional to identity.")
    end
  end
  # TODO: make this a vector of length N
  L = emptyITensor()
  n = 1
  for a in 1:d⃗ₕ[n]
    L += translatecell(L⃗[a], 1) * setelt(l⃗ₕ[n]... => a)
  end
  return InfiniteMPS([L])
end

function right_environment(H::InfiniteMPO, ψ::InfiniteMPS, C::InfiniteMPS)
  N = nsites(H)
  @assert N == nsites(ψ)

  # Do the 1-site case first
  @assert N == 1

  # Transfer matrix between cells
  # T(v) performs a right multiplication
  T = transfer_matrix(ψ)
  L = C[N] * prime(dag(C[N]), linkinds(ψ, (N, N+1)))

  # Operator valued matrix representation of MPO
  H⃡ = operator_matrix(H)

  # MPO bond dimensions
  d⃗ₕ = OffsetArray([dim(commoninds(H[n], H[n+1])) for n in 0:N], -1)

  # MPO link indices
  l⃗ₕ = OffsetArray([commoninds(H[n], H[n+1]) for n in 0:N], -1)

  # MPS link indices
  l⃗ = OffsetArray([commoninds(ψ[n], ψ[n+1]) for n in 0:N], -1)

  # Starting point of identity
  n = 1 # Assuming we have only one site
  # This is the set of left quasi fixed points
  R⃗ = Vector{ITensor}(undef, d⃗ₕ[n])
  # The first one is defined to be identity
  R⃗[1] = δ(l⃗[n]'..., dag(l⃗[n])...)

  for a in 2:d⃗ₕ[n]
    YRᵃ = emptyITensor()
    for b in 1:a-1
      YRᵃ += R⃗[b] * ψ[n] * H⃡[n][a, b] * dag(ψ[n]')
    end
    YRᵃ = translatecell(YRᵃ, 1)
    Hnᵃᵃ = H⃡[n][a, a]
    λᵃ = Hnᵃᵃ[1, 1]
    δᵃ = δ(inds(Hnᵃᵃ))

    if abs(λᵃ) > 1
      error("Diagonal operator of the MPO must be proportional to identity with proprtionality constant less than or equal to 1.")
    end

    if Hnᵃᵃ ≠ λᵃ * δᵃ
      error("Diagonal operator of the MPO must be proportional to identity.")
    end

    if λᵃ < 1
      error("Diagonal elements proportional to identity not currently supported.")
      # Solve using C22
    elseif isone(λᵃ) # equal to identity
      println("Diagonal elements equal to identity.")
      # Solve using C25b
      # Solve:
      # [𝟏 - Tᴿ + |𝟏)(L|]|Rᵃ) = |YRᵃ) - |𝟏)(L|YRᵃ)
      b = YRᵃ - (L * YRᵃ) * δ(inds(YRᵃ))
      R⃗[a], _ = linsolve(ProjectedTransferMatrix(T, L), b)

      # Get error
      err_lhs = R⃗[a] - translatecell(ψ[1] * dag(prime(ψ[1], "Link")) * R⃗[a], 1) + L * R⃗[a] * δ(inds(R⃗[a]))
      err_rhs = YRᵃ - L * YRᵃ * δ(inds(YRᵃ))
      @show norm(err_lhs - err_rhs)
    elseif iszero(λᵃ) # is zero
      println("Diagonal element is zero")
      L⃗[a] = YLᵃ
    else
      error("The diagonal operator of the MPO must be zero, identity, or proportional to identity.")
    end
  end
  # TODO: make this a vector of length N
  R = emptyITensor()
  n = 1
  for a in 1:d⃗ₕ[n]
    R += R⃗[a] * setelt(l⃗ₕ[n]... => a)
  end
  return InfiniteMPS([R])
end

vumps(H::InfiniteMPO, ψ::InfiniteMPS; kwargs...) = vumps(H, orthogonalize(ψ, :); kwargs...)

# Find the best orthogonal approximation given
# the center tensors AC and C
function ortho(AC::ITensor, C::ITensor)
  E = AC * dag(C)
  U, P = polar(E, uniqueinds(AC, C))
  l = commoninds(U, P)
  return noprime(U, l)
end

#function right_ortho(C::ITensor, AC::ITensor)
#  E = dag(C) * AC
#  U, P = polar(E, uniqueinds(C, AC))
#  l = commoninds(U, P)
#  return noprime(U, l)
#end

function vumps(H::InfiniteMPO, ψ::InfiniteCanonicalMPS; nsweeps = 10)
  for sweep in 1:nsweeps
    L = left_environment(H, ψ.AL, ψ.C)
    R = right_environment(H, ψ.AR, ψ.C)

    n = 1

    # 0-site effective Hamiltonian
    H⁰ = ITensorMap([L[n], R[n]])
    vals0, vecs0, info0 = eigsolve(H⁰, ψ.C[n])
    E0 = vals0[1]
    Cⁿ = vecs0[1]
    C = InfiniteMPS([Cⁿ])

    @show E0
    @show inds(Cⁿ)

    # 1-site effective Hamiltonian
    H¹ = ITensorMap([L[n-1], H[n], R[n]])
    vals1, vecs1, info1 = eigsolve(H¹, ψ.AL[n] * ψ.C[n]; ishermition = true)
    E1 = vals1[1]
    ACⁿ = vecs1[1]
    AC = InfiniteMPS([ACⁿ])

    @show E1
    @show inds(ACⁿ)

    ALⁿ = ortho(ACⁿ, Cⁿ)
    ψL = InfiniteMPS([ALⁿ])

    @show norm(ALⁿ * Cⁿ - ACⁿ)

    @show inds(ALⁿ)
    #ARⁿ = ortho(ACⁿ, translatecell(Cⁿ, -1))
    ARⁿ = replacetags(ALⁿ, "Left" => "Right")
 
    @show norm(translatecell(Cⁿ, -1) * ARⁿ - ACⁿ)
    ψR = InfiniteMPS([ARⁿ])

    ψ = InfiniteCanonicalMPS(ψL, C, ψR)
  end

  return ψ
end

