function ITensors.NDTensors.contraction_output(
  A::NDTensors.EmptyTensor, B::NDTensors.DiagBlockSparseTensor, label
)
  return ITensor(eltype(B), label)
end

# Struct for use in linear system solver
struct AOᴸ
  ψ::InfiniteCanonicalMPS
  H::InfiniteMPOMatrix
  n::Int
end

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
  xR = x * ψ.C[1] * (ψ′.C[1] * δʳ(1)) * denseblocks(δˡ(1))
  return xT - xR
end

function initialize_left_environment(
  H::InfiniteMPOMatrix, ψ::InfiniteCanonicalMPS, n::Int64; init_last=false
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
  Lstart::Vector{ITensor}, H::InfiniteMPOMatrix, ψ::InfiniteCanonicalMPS, n_1::Int64
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
  Lstart::ITensor, m::Int64, H::InfiniteMPOMatrix, ψ::InfiniteCanonicalMPS, n_1::Int64;
)
  Ltarget = initialize_left_environment(H, ψ, n_1; init_last=false)
  for j in 1:m
    Ltarget[j] = Lstart * ψ.AL[n_1] * H[n_1][m, j] * dag(prime(ψ.AL[n_1]))
  end
  return Ltarget
end

#apply the left transfer matrix n1:n1+nsites(ψ)-1
function apply_left_transfer_matrix(
  Lstart::ITensor, m::Int64, H::InfiniteMPOMatrix, ψ::InfiniteCanonicalMPS, n_1::Int64
)
  Ltarget = apply_local_left_transfer_matrix(Lstart, m, H, ψ, n_1)
  for j in 1:(nsites(ψ) - 1)
    Ltarget = apply_local_left_transfer_matrix(Ltarget, H, ψ, n_1 + j)
  end
  return Ltarget
end

# Also input C bond matrices to help compute the right fixed points
# of ψ (R ≈ C * dag(C))
function left_environment(H::InfiniteMPOMatrix, ψ::InfiniteCanonicalMPS; tol=1e-10)
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
  δˢ(n) = δ(dag(s[n]), prime(s[n]))

  eₗ = [0.0]
  dₕ = size(H[1])[1]
  #Ls = [Vector{ITensor}(undef, dₕ) for j in 1:N]
  Ls = [initialize_left_environment(H, ψ, j; init_last=true) for j in 1:N]
  #Building the L vector for n_1 = 1
  # TM is 2 3 ... N 1
  localR = ψ.C[1] * δʳ(1) * ψ′.C[1] #to revise
  for n in reverse(1:(dₕ - 1))
    temp_Ls = apply_left_transfer_matrix(
      translatecell(translator(ψ), Ls[1][n + 1], -1), n + 1, H, ψ, 2 - N
    )
    for j in 1:n
      if isassigned(temp_Ls, j)
        if isassigned(Ls[1], j)
          Ls[1][j] += temp_Ls[j]
        else
          Ls[1][j] = temp_Ls[j]
        end
      end
    end
    if !isempty(H[1][n, n])
      λ = H[1][n, n][1, 1]
      δiag = δˢ(1)
      #@assert norm(H[1][n, n] - λ * δiag) == 0 "Non identity diagonal not implemented in MPO"
      @assert abs(λ) <= 1 "Diverging term"
      eₗ[1] = (Ls[1][n] * localR)[]
      Ls[1][n] += -(eₗ[1] * denseblocks(δˡ(1)))
      if λ == 1
        A = AOᴸ(ψ, H, n)
        Ls[1][n], info = linsolve(A, Ls[1][n], 1, -1; tol=tol)
      else
        println("Not implemented")
        flush(stdout)
        flush(stderr)
      end
    end
  end
  for n in 2:N
    Ls[n] = apply_local_left_transfer_matrix(Ls[n - 1], H, ψ, n)
  end
  return CelledVector(Ls), eₗ[1]
end

# Struct for use in linear system solver
struct AOᴿ
  ψ::InfiniteCanonicalMPS
  H::InfiniteMPOMatrix
  n::Int
end

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
  xR = x * ψ.C[0] * (ψ′.C[0] * δˡ(0)) * denseblocks(δʳ(0))
  return xT - xR
end

function initialize_right_environment(
  H::InfiniteMPOMatrix, ψ::InfiniteCanonicalMPS, n::Int64; init_first=false
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
    mpo_link = only(uniqueinds(H[n - 1][1, j], sit))
    Rs[j] = ITensor(Float64, dag(mpo_link), link, dag(prime(link)))
  end
  return Rs
end

function apply_local_right_transfer_matrix(
  Lstart::Vector{ITensor}, H::InfiniteMPOMatrix, ψ::InfiniteCanonicalMPS, n_1::Int64
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
  H::InfiniteMPOMatrix,
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
  Lstart::ITensor, m::Int64, H::InfiniteMPOMatrix, ψ::InfiniteCanonicalMPS, n_1::Int64
)
  Ltarget = apply_local_right_transfer_matrix(Lstart, m, H, ψ, n_1)
  for j in 1:(nsites(ψ) - 1)
    Ltarget = apply_local_right_transfer_matrix(Ltarget, H, ψ, n_1 - j)
  end
  return Ltarget
end

function right_environment(H::InfiniteMPOMatrix, ψ::InfiniteCanonicalMPS; tol=1e-10)
  N = nsites(H)
  @assert N == nsites(ψ)

  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  s = siteinds(only, ψ)
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˡ(n) = δ(l[n], dag(prime(l[n])))
  δˢ(n) = δ(dag(s[n]), prime(s[n]))

  eᵣ = [0.0]
  dₕ = size(H[1])[1]
  Rs = [initialize_right_environment(H, ψ, j; init_first=true) for j in 1:N]
  #Building the R vector for n_1 = 1
  # TM is 2-N 3-N ... 0
  localL = ψ.C[0] * δˡ(0) * dag(prime(ψ.C[0]))
  for n in 2:dₕ
    temp_Rs = apply_right_transfer_matrix(
      translatecell(translator(ψ), Rs[1][n - 1], 1), n - 1, H, ψ, N
    )
    for j in n:dₕ
      if isassigned(temp_Rs, j)
        if isassigned(Rs[1], j)
          Rs[1][j] += temp_Rs[j]
        else
          Rs[1][j] = temp_Rs[j]
        end
      end
    end
    if !isempty(H[1][n, n])
      λ = H[1][n, n][1, 1]
      δiag = δˢ(1)
      #@assert norm(H[1][n, n] - λ * δiag) == 0 "Non identity diagonal not implemented in MPO"
      @assert abs(λ) <= 1 "Diverging term"
      eᵣ[1] = (localL * Rs[1][n])[]
      Rs[1][n] += -(eᵣ[1] * denseblocks(δʳ(0)))
      if λ == 1
        A = AOᴿ(ψ, H, n)
        Rs[1][n], info = linsolve(A, Rs[1][n], 1, -1; tol=tol)
      else
        println("Not yet implemented")
        flush(stdout)
        flush(stderr)
      end
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

function vumps(H::InfiniteMPOMatrix, ψ::InfiniteMPS; kwargs...)
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
  H::InfiniteMPOMatrix,
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

  eL = zeros(N)
  eR = zeros(N)
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
  H::InfiniteMPOMatrix,
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

  eL = zeros(1)
  eR = zeros(1)
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
