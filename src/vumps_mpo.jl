#
# The fixed point equations for the environments of an `InfiniteBlockMPO` follow
# appendix C 2 of Zauner-Stijl et al., arXiv:1701.07035. With a lower triangular
# MPO `W` and diagonal blocks `W[a, a] = λₐ 1`, the unit cell transfer matrix of
# channel `a` is `T^{aa} = λₐ T`, and the fixed point equations (C19)/(C20) are
# solved recursively (their table VI) by dispatching on `λₐ`:
#
#   T^{aa} = 0        -> (Lₐ| = (Y_Lₐ|                                  (nothing to solve)
#   |λₐ| < 1          -> (Lₐ|[1 - λₐ T] = (Y_Lₐ|                        (C21)/(C22)
#   λₐ = 1            -> (Lₐ|[1 - T + |R)(1|] = (Y_Lₐ| - (Y_Lₐ|R)(1|    (C25a)/(C25b)
#
# Only the last case has a zero mode to project out, and only there is the energy
# density (C27) subtracted. Applying that regularization to a `|λₐ| < 1` channel
# shifts `(Lₐ|` by a spurious `(Y_Lₐ|R)/(1 - λₐ) (1|`, which then contaminates the
# energy through (C17).

# Struct for use in linear system solver.
# `projector = true` gives the operator of (C25a), `false` the one of (C21).
struct AOᴸ
  ψ::InfiniteCanonicalMPS
  H::InfiniteBlockMPO
  n::Int
  projector::Bool
end

AOᴸ(ψ::InfiniteCanonicalMPS, H::InfiniteBlockMPO, n::Int) = AOᴸ(ψ, H, n, true)

function (A::AOᴸ)(x)
  ψ = A.ψ
  H = A.H
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  n = A.n
  N = nsites(ψ)
  #@assert n == N

  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)
  δˡ(n) = δ(l[n], l′[n])
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  xT = translatecell(translator(ψ), x, -1)
  for j in (2 - N):1
    xT = xT * H[j][n, n] * ψ.AL[j] * ψ′.AL[j]
  end
  A.projector || return xT
  xR = x * ψ.C[1] * (ψ′.C[1] * δʳ(1)) * denseblocks(δˡ(1))
  return xT - xR
end

"""
    local_diagonal_scalar(W::ITensor, s::Index; rtol=1e-12)

The scalar `λ` such that the diagonal MPO block `W` equals `λ` times the identity on
the physical index `s`, and the identity (a pass through) on the MPO link indices.

Errors if `W` is not of that form, which is the assumption under which the fixed point
equations (C19)/(C20) of arXiv:1701.07035 decouple channel by channel.
"""
function local_diagonal_scalar(W::ITensor, s::Index; rtol=1e-12)
  δˢ = δ(dag(s), prime(s))
  links = uniqueinds(W, δˢ)
  identity_block = if length(links) == 0
    denseblocks(δˢ)
  elseif length(links) == 2
    il, ir = links
    if dim(il) != dim(ir)
      error(
        "Diagonal MPO block has left link $(il) and right link $(ir) of different dimension, so it cannot be proportional to the identity.",
      )
    end
    # densify before the outer product, NDTensors has no `outer!` for two Diag tensors
    denseblocks(δˢ) * denseblocks(δ(il, ir))
  else
    error("Diagonal MPO block of order $(order(W)) is not supported.")
  end
  λ = (W * dag(identity_block))[] / (identity_block * dag(identity_block))[]
  if norm(W - λ * identity_block) > rtol * max(norm(W), one(real(eltype(W))))
    error(
      "Only diagonal MPO blocks proportional to the identity are supported, the block on site index $(s) is not.",
    )
  end
  return λ
end

"""
    mpo_diagonal_scalar(H::InfiniteBlockMPO, s, a::Int; rtol=1e-12)

The scalar `λ` of the unit cell transfer matrix `T^{aa} = λ T` of channel `a`, i.e. the
product of the per site scalars of the diagonal blocks `H[n][a, a]`. Returns `nothing`
when `T^{aa} = 0`, i.e. when the diagonal block vanishes on at least one site of the
unit cell.

This is the quantity that table VI of arXiv:1701.07035 dispatches on when solving the
environment fixed point equations.
"""
function mpo_diagonal_scalar(H::InfiniteBlockMPO, s, a::Int; rtol=1e-12)
  λ = nothing
  for n in 1:nsites(H)
    W = H[n][a, a]
    isempty(W) && return nothing
    λₙ = local_diagonal_scalar(W, s[n]; rtol=rtol)
    iszero(λₙ) && return nothing
    λ = isnothing(λ) ? λₙ : λ * λₙ
  end
  return λ
end

function initialize_left_environment(
  H::InfiniteBlockMPO, ψ::InfiniteCanonicalMPS, n::Int64; init_last=false
)
  dₕ = size(H[n + 1], 1)
  sit = inds(H[n + 1][1, 1])
  link = commonind(ψ.AL[n], ψ.AL[n + 1])
  Ls = Vector{ITensor}(undef, dₕ)
  Ls[1] = ITensor(Float64, link, dag(prime(link)))
  if init_last
    Ls[end] = denseblocks(δ(link, dag(prime(link))))
  else
    Ls[end] = ITensor(Float64, link, dag(prime(link)))
  end
  for j in 2:(dₕ - 1)
    mpo_link = only(uniqueinds(H[n + 1][j, 1], sit))
    Ls[j] = ITensor(Float64, dag(mpo_link), link, dag(prime(link)))
  end
  return Ls
end

function apply_local_left_transfer_matrix(
  Lstart::Vector{ITensor}, H::InfiniteBlockMPO, ψ::InfiniteCanonicalMPS, n_1::Int64
)
  dₕ = length(Lstart)
  ψ′ = dag(ψ)'

  Ltarget = initialize_left_environment(H, ψ, n_1; init_last=false)
  for j in 1:dₕ
    for k in reverse(j:dₕ)
      if !isempty(H[n_1][k, j]) && !isempty(Lstart[k])
        Ltarget[j] .+= Lstart[k] * ψ.AL[n_1] * H[n_1][k, j] * ψ′.AL[n_1]
      end
    end
  end
  return Ltarget
end

# apply the left transfer matrix at position n1 to the vector Lstart considering it at position m, adding to Ltarget
function apply_local_left_transfer_matrix(
  Lstart::ITensor, m::Int64, H::InfiniteBlockMPO, ψ::InfiniteCanonicalMPS, n_1::Int64;
)
  Ltarget = initialize_left_environment(H, ψ, n_1; init_last=false)
  for j in 1:m
    Ltarget[j] = Lstart * ψ.AL[n_1] * H[n_1][m, j] * dag(prime(ψ.AL[n_1]))
  end
  return Ltarget
end

#apply the left transfer matrix n1:n1+nsites(ψ)-1
function apply_left_transfer_matrix(
  Lstart::ITensor, m::Int64, H::InfiniteBlockMPO, ψ::InfiniteCanonicalMPS, n_1::Int64
)
  Ltarget = apply_local_left_transfer_matrix(Lstart, m, H, ψ, n_1)
  for j in 1:(nsites(ψ) - 1)
    Ltarget = apply_local_left_transfer_matrix(Ltarget, H, ψ, n_1 + j)
  end
  return Ltarget
end

# Also input C bond matrices to help compute the right fixed points
# of ψ (R ≈ C * dag(C))
function left_environment(
  H::InfiniteBlockMPO,
  ψ::InfiniteCanonicalMPS;
  tol=1e-10,
  geometric_tol=1e-14,
  identity_tol=1e-12,
)
  N = nsites(H)
  @assert N == nsites(ψ)

  # Do the 1-site case first
  ψ′ = dag(ψ)'

  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)
  s = siteinds(only, ψ)
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˡ(n) = δ(l[n], l′[n])
  # op("Id", s[1]) is permuted w.r.t. below line
  δˢ(n) = δ(dag(s[n]), prime(s[n]))

  EType = ITensorMPS.promote_itensor_eltype(ψ)

  eₗ = zeros(EType, 1)
  dₕ = size(H[1])[1]
  #Ls = [Vector{ITensor}(undef, dₕ) for j in 1:N]
  Ls = [initialize_left_environment(H, ψ, j; init_last=true) for j in 1:N]
  #Building the L vector for n_1 = 1
  # TM is 2 3 ... N 1
  localR = ψ.C[1] * δʳ(1) * ψ′.C[1] #to revise
  for a in reverse(1:(dₕ - 1))
    temp_Ls = apply_left_transfer_matrix(
      translatecell(translator(ψ), Ls[1][a + 1], -1), a + 1, H, ψ, 2 - N
    )
    for j in 1:a
      if isassigned(temp_Ls, j)
        if isassigned(Ls[1], j)
          Ls[1][j] += temp_Ls[j]
        else
          Ls[1][j] = temp_Ls[j]
        end
      end
    end
    # Dispatch on the scalar of the unit cell transfer matrix T^{aa} = λ T of this
    # channel, following table VI of arXiv:1701.07035.
    λ = mpo_diagonal_scalar(H, s, a)
    if isnothing(λ)
      # T^{aa} = 0, the fixed point equation reduces to (Lₐ| = (Y_Lₐ|.
      continue
    elseif abs(λ - one(λ)) <= identity_tol
      # Eq. (C25a). This is the only channel with a zero mode, and the only one whose
      # diagonal correction (C27) is the energy density. Per the paper it can only be
      # the first one, whose diagonal block is the terminating identity.
      if a != 1
        error(
          "The identity is only allowed on the first and last entry of the diagonal of the MPO, found it on entry $a of $dₕ.",
        )
      end
      eₗ[1] = (Ls[1][a] * localR)[]
      Ls[1][a] -= eₗ[1] * denseblocks(δˡ(1))
      Ls[1][a], info = linsolve(AOᴸ(ψ, H, a), Ls[1][a], 1, -1; tol=tol)
    elseif abs(λ) < one(abs(λ))
      # Eq. (C21). 1 - λ T is invertible, so there is neither a zero mode to project out
      # nor an energy to subtract. The paper solves this to machine precision.
      Ls[1][a], info = linsolve(
        AOᴸ(ψ, H, a, false), Ls[1][a], 1, -1; tol=min(tol, geometric_tol)
      )
    else
      error(
        "Diagonal MPO entry $a of $dₕ is $λ times the identity, |λ| ≥ 1 gives a diverging geometric series.",
      )
    end
  end
  for a in 2:N
    Ls[a] = apply_local_left_transfer_matrix(Ls[a - 1], H, ψ, a)
  end
  return CelledVector(Ls), eₗ[1]
end

# Struct for use in linear system solver.
# `projector = true` gives the operator of (C25b), `false` the one of (C22).
struct AOᴿ
  ψ::InfiniteCanonicalMPS
  H::InfiniteBlockMPO
  n::Int
  projector::Bool
end

AOᴿ(ψ::InfiniteCanonicalMPS, H::InfiniteBlockMPO, n::Int) = AOᴿ(ψ, H, n, true)

function (A::AOᴿ)(x)
  ψ = A.ψ
  H = A.H
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  n = A.n
  N = nsites(ψ)
  #@assert n == N

  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)
  δˡ(n) = δ(l[n], l′[n])
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  xT = translatecell(translator(ψ), x, 1)
  for j in reverse(1:N)
    xT = xT * ψ.AR[j] * H[j][n, n] * ψ′.AR[j]
  end
  A.projector || return xT
  xR = x * ψ.C[0] * (ψ′.C[0] * δˡ(0)) * denseblocks(δʳ(0))
  return xT - xR
end

function initialize_right_environment(
  H::InfiniteBlockMPO, ψ::InfiniteCanonicalMPS, n::Int64; init_first=false
)
  dₕ = size(H[n - 1], 2)
  sit = inds(H[n - 1][1, 1])
  link = commonind(ψ.AR[n], ψ.AR[n - 1])
  Rs = Vector{ITensor}(undef, dₕ)
  Rs[end] = ITensor(Float64, link, dag(prime(link)))
  if init_first
    Rs[1] = denseblocks(δ(link, dag(prime(link))))
  else
    Rs[1] = ITensor(Float64, link, dag(prime(link)))
  end
  for j in 2:(dₕ - 1)
    mpo_link = only(uniqueinds(H[n - 1][dₕ, j], sit))
    Rs[j] = ITensor(Float64, dag(mpo_link), link, dag(prime(link)))
  end
  return Rs
end

function apply_local_right_transfer_matrix(
  Lstart::Vector{ITensor}, H::InfiniteBlockMPO, ψ::InfiniteCanonicalMPS, n_1::Int64
)
  dₕ = length(Lstart)
  ψ′ = dag(ψ)'
  Ltarget = initialize_right_environment(H, ψ, n_1)
  for j in reverse(1:dₕ)
    for k in reverse(1:j)
      if !isempty(H[n_1][j, k]) && isassigned(Lstart, k) && !isempty(Lstart[k])
        Ltarget[j] += Lstart[k] * ψ.AR[n_1] * H[n_1][j, k] * ψ′.AR[n_1]
      end
    end
  end
  return Ltarget
end

# apply the left transfer matrix at position n1 to the vector Lstart considering it at position m, adding to Ltarget
function apply_local_right_transfer_matrix(
  Lstart::ITensor,
  m::Int64,
  H::InfiniteBlockMPO,
  ψ::InfiniteCanonicalMPS,
  n_1::Int64;
  reset=true,
)
  dₕ = size(H[n_1])[1]
  ψ′ = dag(prime(ψ.AR[n_1]))
  Ltarget = initialize_right_environment(H, ψ, n_1)
  for j in m:dₕ
    if !isempty(H[n_1][j, m])
      Ltarget[j] = Lstart * ψ.AR[n_1] * H[n_1][j, m] * ψ′
    end
  end
  return Ltarget
end

#apply the right transfer matrix n1:n1+nsites(ψ)-1
function apply_right_transfer_matrix(
  Lstart::ITensor, m::Int64, H::InfiniteBlockMPO, ψ::InfiniteCanonicalMPS, n_1::Int64
)
  Ltarget = apply_local_right_transfer_matrix(Lstart, m, H, ψ, n_1)
  for j in 1:(nsites(ψ) - 1)
    Ltarget = apply_local_right_transfer_matrix(Ltarget, H, ψ, n_1 - j)
  end
  return Ltarget
end

function right_environment(
  H::InfiniteBlockMPO,
  ψ::InfiniteCanonicalMPS;
  tol=1e-10,
  geometric_tol=1e-14,
  identity_tol=1e-12,
)
  N = nsites(H)
  @assert N == nsites(ψ)

  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  s = siteinds(only, ψ)
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˡ(n) = δ(l[n], dag(prime(l[n])))
  δˢ(n) = δ(dag(s[n]), prime(s[n]))

  EType = ITensorMPS.promote_itensor_eltype(ψ)

  eᵣ = zeros(EType, 1)
  dₕ = size(H[1])[1]
  Rs = [initialize_right_environment(H, ψ, j; init_first=true) for j in 1:N]
  #Building the R vector for n_1 = 1
  # TM is 2-N 3-N ... 0
  localL = ψ.C[0] * δˡ(0) * dag(prime(ψ.C[0]))
  for a in 2:dₕ
    temp_Rs = apply_right_transfer_matrix(
      translatecell(translator(ψ), Rs[1][a - 1], 1), a - 1, H, ψ, N
    )
    for j in a:dₕ
      if isassigned(temp_Rs, j)
        if isassigned(Rs[1], j)
          Rs[1][j] += temp_Rs[j]
        else
          Rs[1][j] = temp_Rs[j]
        end
      end
    end
    # Dispatch on the scalar of the unit cell transfer matrix T^{aa} = λ T of this
    # channel, following table VI of arXiv:1701.07035.
    λ = mpo_diagonal_scalar(H, s, a)
    if isnothing(λ)
      # T^{aa} = 0, the fixed point equation reduces to |Rₐ) = |Y_Rₐ).
      continue
    elseif abs(λ - one(λ)) <= identity_tol
      # Eq. (C25b). This is the only channel with a zero mode, and the only one whose
      # diagonal correction (C27) is the energy density. Per the paper it can only be
      # the last one, whose diagonal block is the terminating identity.
      if a != dₕ
        error(
          "The identity is only allowed on the first and last entry of the diagonal of the MPO, found it on entry $a of $dₕ.",
        )
      end
      eᵣ[1] = (localL * Rs[1][a])[]
      Rs[1][a] -= eᵣ[1] * denseblocks(δʳ(0))
      Rs[1][a], info = linsolve(AOᴿ(ψ, H, a), Rs[1][a], 1, -1; tol=tol)
    elseif abs(λ) < one(abs(λ))
      # Eq. (C22). 1 - λ T is invertible, so there is neither a zero mode to project out
      # nor an energy to subtract. The paper solves this to machine precision.
      Rs[1][a], info = linsolve(
        AOᴿ(ψ, H, a, false), Rs[1][a], 1, -1; tol=min(tol, geometric_tol)
      )
    else
      error(
        "Diagonal MPO entry $a of $dₕ is $λ times the identity, |λ| ≥ 1 gives a diverging geometric series.",
      )
    end
  end
  if N > 1
    Rs[N] = apply_local_right_transfer_matrix(
      translatecell(translator(ψ), Rs[1], 1), H, ψ, N
    )
    for n in reverse(2:(N - 1))
      Rs[n] = apply_local_right_transfer_matrix(Rs[n + 1], H, ψ, n)
    end
  end
  return CelledVector(Rs), eᵣ[1]
end

function vumps(H::InfiniteBlockMPO, ψ::InfiniteMPS; kwargs...)
  return vumps(H, orthogonalize(ψ, :); kwargs...)
end

struct H⁰
  L::Vector{ITensor}
  R::Vector{ITensor}
end

function (H::H⁰)(x)
  L = H.L
  R = H.R
  dₕ = length(L)
  result = L[1] * x * R[1]
  for j in 2:dₕ
    result += L[j] * x * R[j]
  end
  return noprime(result)
end

struct H¹
  L::Vector{ITensor}
  R::Vector{ITensor}
  T::Matrix{ITensor}
end

function (H::H¹)(x)
  L = H.L
  R = H.R
  T = H.T
  dₕ = length(L)
  result = ITensor(prime(inds(x)))
  for i in 1:dₕ
    for j in 1:dₕ
      if !isempty(T[i, j])
        result += L[i] * x * T[i, j] * R[j]
      end
    end
  end
  return noprime(result)
end

function tdvp_iteration_sequential(
  solver::Function,
  H::InfiniteBlockMPO,
  ψ::InfiniteCanonicalMPS;
  (ϵᴸ!)=fill(1e-15, nsites(ψ)),
  (ϵᴿ!)=fill(1e-15, nsites(ψ)),
  time_step,
  solver_tol=(x -> x / 100),
  eager=true,
)
  ψ = copy(ψ)
  ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
  _solver_tol = solver_tol(ϵᵖʳᵉˢ)
  N = nsites(ψ)

  C̃ = InfiniteMPS(Vector{ITensor}(undef, N))
  Ãᶜ = InfiniteMPS(Vector{ITensor}(undef, N))
  Ãᴸ = InfiniteMPS(Vector{ITensor}(undef, N))
  Ãᴿ = InfiniteMPS(Vector{ITensor}(undef, N))

  EType_ψ = ITensorMPS.promote_itensor_eltype(ψ)

  EType_t = typeof(time_step)

  EType = typeof(one(EType_ψ) * one(EType_t))

  eL = zeros(EType, N)
  eR = zeros(EType, N)
  for n in 1:N
    L, eL[n] = left_environment(H, ψ; tol=_solver_tol) #TODO currently computing two many of them
    R, eR[n] = right_environment(H, ψ; tol=_solver_tol) #TODO currently computing two many of them
    if N == 1
      # 0-site effective Hamiltonian
      E0, C̃[n], info0 = solver(H⁰(L[1], R[2]), time_step, ψ.C[1], _solver_tol, eager)
      # 1-site effective Hamiltonian
      E1, Ãᶜ[n], info1 = solver(
        H¹(L[0], R[2], H[1]), time_step, ψ.AL[1] * ψ.C[1], _solver_tol, eager
      )
      Ãᴸ[1] = ortho_polar(Ãᶜ[1], C̃[1])
      Ãᴿ[1] = ortho_polar(Ãᶜ[1], C̃[0])
      ψ.AL[1] = Ãᴸ[1]
      ψ.AR[1] = Ãᴿ[1]
      ψ.C[1] = C̃[1]
    else
      # 0-site effective Hamiltonian
      E0, C̃[n], info0 = solver(H⁰(L[n], R[n + 1]), time_step, ψ.C[n], _solver_tol, eager)
      E0′, C̃[n - 1], info0′ = solver(
        H⁰(L[n - 1], R[n]), time_step, ψ.C[n - 1], _solver_tol, eager
      )
      # 1-site effective Hamiltonian
      E1, Ãᶜ[n], info1 = solver(
        H¹(L[n - 1], R[n + 1], H[n]), time_step, ψ.AL[n] * ψ.C[n], _solver_tol, eager
      )
      Ãᴸ[n] = ortho_polar(Ãᶜ[n], C̃[n])
      Ãᴿ[n] = ortho_polar(Ãᶜ[n], C̃[n - 1])
      ψ.AL[n] = Ãᴸ[n]
      ψ.AR[n] = Ãᴿ[n]
      ψ.C[n] = C̃[n]
      ψ.C[n - 1] = C̃[n - 1]
    end
  end
  for n in 1:N
    ϵᴸ![n] = norm(Ãᶜ[n] - Ãᴸ[n] * C̃[n])
    ϵᴿ![n] = norm(Ãᶜ[n] - C̃[n - 1] * Ãᴿ[n])
  end
  return ψ, (eL / N, eR / N)
end

function tdvp_iteration_parallel(
  solver::Function,
  H::InfiniteBlockMPO,
  ψ::InfiniteCanonicalMPS;
  (ϵᴸ!)=fill(1e-15, nsites(ψ)),
  (ϵᴿ!)=fill(1e-15, nsites(ψ)),
  time_step,
  solver_tol=(x -> x / 100),
  eager=true,
)
  ψ = copy(ψ)
  ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
  _solver_tol = solver_tol(ϵᵖʳᵉˢ)
  N = nsites(ψ)

  C̃ = InfiniteMPS(Vector{ITensor}(undef, N))
  Ãᶜ = InfiniteMPS(Vector{ITensor}(undef, N))
  Ãᴸ = InfiniteMPS(Vector{ITensor}(undef, N))
  Ãᴿ = InfiniteMPS(Vector{ITensor}(undef, N))

  EType_ψ = ITensorMPS.promote_itensor_eltype(ψ)
  EType_t = typeof(time_step)
  EType = typeof(one(EType_ψ) * one(EType_t))

  eL = zeros(EType, 1)
  eR = zeros(EType, 1)

  L, eL[1] = left_environment(H, ψ; tol=_solver_tol) #TODO currently computing two many of them
  R, eR[1] = right_environment(H, ψ; tol=_solver_tol) #TODO currently computing two many of them
  for n in 1:N
    if N == 1
      # 0-site effective Hamiltonian
      E0, C̃[n], info0 = solver(H⁰(L[1], R[2]), time_step, ψ.C[1], _solver_tol, eager)
      # 1-site effective Hamiltonian
      E1, Ãᶜ[n], info1 = solver(
        H¹(L[0], R[2], H[1]), time_step, ψ.AL[1] * ψ.C[1], _solver_tol, eager
      )
      Ãᴸ[1] = ortho_polar(Ãᶜ[1], C̃[1])
      Ãᴿ[1] = ortho_polar(Ãᶜ[1], C̃[0])
      ψ.AL[1] = Ãᴸ[1]
      ψ.AR[1] = Ãᴿ[1]
      ψ.C[1] = C̃[1]
    else
      # 0-site effective Hamiltonian
      for n in 1:N
        E0, C̃[n], info0 = solver(H⁰(L[n], R[n + 1]), time_step, ψ.C[n], _solver_tol, eager)
        E1, Ãᶜ[n], info1 = solver(
          H¹(L[n - 1], R[n + 1], H[n]), time_step, ψ.AL[n] * ψ.C[n], _solver_tol, eager
        )
      end
      # 1-site effective Hamiltonian
      for n in 1:N
        Ãᴸ[n] = ortho_polar(Ãᶜ[n], C̃[n])
        Ãᴿ[n] = ortho_polar(Ãᶜ[n], C̃[n - 1])
        ψ.AL[n] = Ãᴸ[n]
        ψ.AR[n] = Ãᴿ[n]
        ψ.C[n] = C̃[n]
      end
    end
  end
  for n in 1:N
    ϵᴸ![n] = norm(Ãᶜ[n] - Ãᴸ[n] * C̃[n])
    ϵᴿ![n] = norm(Ãᶜ[n] - C̃[n - 1] * Ãᴿ[n])
  end
  return ψ, (eL / N, eR / N)
end
