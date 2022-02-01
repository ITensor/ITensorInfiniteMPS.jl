
#
# VUMPS code
#

struct Hᶜ
  ∑h::InfiniteITensorSum
  Hᴸ::InfiniteMPS
  Hᴿ::InfiniteMPS
  ψ::InfiniteCanonicalMPS
  n::Int
end

struct Hᴬᶜ
  ∑h::InfiniteITensorSum
  Hᴸ::InfiniteMPS
  Hᴿ::InfiniteMPS
  ψ::InfiniteCanonicalMPS
  n::Int
end

function Base.:*(H::Hᶜ, v::ITensor)
  ∑h = H.∑h
  Hᴸ = H.Hᴸ
  Hᴿ = H.Hᴿ
  ψ = H.ψ
  ψ′ = dag(ψ)'
  n = H.n
  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)
  δˡ = δ(l[n], l′[n])
  δˡ⁻¹ = δ(l[n - 1], l′[n - 1])
  δʳ = δ(r[n], r′[n])
  δʳ⁺¹ = δ(r[n + 1], r′[n + 1])
  Hᶜᴸv = v * Hᴸ[n] * dag(δʳ)
  Hᶜᴿv = v * δˡ * Hᴿ[n]
  Hᶜʰv =
    v * ψ.AL[n] * δˡ⁻¹ * ψ′.AL[n] * ∑h[(n, n + 1)] * ψ.AR[n + 1] * dag(δʳ⁺¹) * ψ′.AR[n + 1]
  Hᶜv = Hᶜᴸv + Hᶜʰv + Hᶜᴿv
  return Hᶜv * dag(δˡ) * δʳ
end

function Base.:*(H::Hᴬᶜ, v::ITensor)
  ∑h = H.∑h
  Hᴸ = H.Hᴸ
  Hᴿ = H.Hᴿ
  ψ = H.ψ
  ψ′ = dag(ψ)'
  n = H.n
  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)
  s = siteinds(only, ψ)
  s′ = siteinds(only, ψ′)

  δˢ(n) = δ(s[n], s′[n])
  δˡ(n) = δ(l[n], l′[n])
  δʳ(n) = δ(r[n], r′[n])

  Hᴬᶜᴸv = v * Hᴸ[n - 1] * dag(δˢ(n)) * dag(δʳ(n))
  Hᴬᶜᴿv = v * δˡ(n - 1) * dag(δˢ(n)) * Hᴿ[n]
  Hᴬᶜʰ¹v = v * ψ.AL[n - 1] * δˡ(n - 2) * ψ′.AL[n - 1] * ∑h[(n - 1, n)] * dag(δʳ(n))
  Hᴬᶜʰ²v = v * δˡ(n - 1) * ψ.AR[n + 1] * dag(δʳ(n + 1)) * ψ′.AR[n + 1] * ∑h[(n, n + 1)]
  Hᴬᶜv = Hᴬᶜᴸv + Hᴬᶜʰ¹v + Hᴬᶜʰ²v + Hᴬᶜᴿv
  return Hᴬᶜv * dag(δˡ(n - 1)) * δˢ(n) * δʳ(n)
end

function (H::Hᶜ)(v)
  Hᶜv = H * v
  ## return Hᶜv * δˡ * δʳ
  return noprime(Hᶜv)
end

function (H::Hᴬᶜ)(v)
  Hᴬᶜv = H * v
  ## return Hᶜv * δˡ⁻¹ * δˢ * δʳ
  return noprime(Hᴬᶜv)
end

# Struct for use in linear system solver
struct Aᴸ
  ψ::InfiniteCanonicalMPS
  n::Int
end

function (A::Aᴸ)(x)
  ψ = A.ψ
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  ψ̃ = prime(linkinds, ψᴴ)
  n = A.n

  N = length(ψ)
  #@assert n == N

  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)

  xT = translatecell(x, -1)
  for k in (n - N + 1):n
    xT = xT * ψ.AL[k] * ψ̃.AL[k]
  end
  δˡ = δ(l[n], l′[n])
  δʳ = δ(r[n], r′[n])
  xR = x * ψ.C[n] * ψ′.C[n] * dag(δʳ) * denseblocks(δˡ)
  return xT - xR
end

function left_environment(hᴸ, 𝕙ᴸ, ψ; tol=1e-15)
  ψ̃ = prime(linkinds, dag(ψ))
  N = nsites(ψ)

  Aᴺ = Aᴸ(ψ, N)
  Hᴸᴺ¹, info = linsolve(Aᴺ, 𝕙ᴸ[N], 1, -1; tol=tol)
  # Get the rest of the environments in the unit cell
  Hᴸ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴸ[N] = Hᴸᴺ¹
  Hᴸᴺ¹ = translatecell(Hᴸᴺ¹, -1)
  for n in 1:(N - 1)
    Hᴸ[n] = Hᴸ[n - 1] * ψ.AL[n] * ψ̃.AL[n] + hᴸ[n]
  end
  # Compute more accurate environments
  # Not currently working
  #for n in 1:(N - 1)
  #  Aⁿ = Aᴸ(ψ, n)
  #  Hᴸ[n], info = linsolve(Aⁿ, 𝕙ᴸ[n], Hᴸ[n], 1, -1; tol=tol)
  #end
  return Hᴸ
end

# Struct for use in linear system solver
struct Aᴿ
  hᴿ::InfiniteMPS
  ψ::InfiniteCanonicalMPS
  n::Int
end

function (A::Aᴿ)(x)
  hᴿ = A.hᴿ
  ψ = A.ψ
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  ψ̃ = prime(linkinds, ψᴴ)
  n = A.n

  N = length(ψ)
  @assert n == N

  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)

  xT = x
  for k in reverse(1:N)
    xT = xT * ψ.AR[k] * ψ̃.AR[k]
  end
  xT = translatecell(xT, 1)
  δˡ = δ(l[n], l′[n])
  δʳ = δ(r[n], r′[n])
  xR = x * ψ.C[n] * ψ′.C[n] * δˡ * denseblocks(dag(δʳ))
  return xT - xR
end

function right_environment(hᴿ, ψ; tol=1e-15)
  ψ̃ = prime(linkinds, dag(ψ))
  # XXX: replace with `nsites`
  #N = nsites(ψ)
  N = length(ψ)

  A = Aᴿ(hᴿ, ψ, N)
  Hᴿᴺ¹, info = linsolve(A, hᴿ[N], 1, -1; tol=tol)
  # Get the rest of the environments in the unit cell
  Hᴿ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴿ[N] = Hᴿᴺ¹
  for n in reverse(1:(N - 1))
    Hᴿ[n] = Hᴿ[n + 1] * ψ.AR[n + 1] * ψ̃.AR[n + 1] + hᴿ[n]
  end
  return Hᴿ
end

function tdvp_iteration(args...; multisite_update_alg="sequential", kwargs...)
  if multisite_update_alg == "sequential"
    return tdvp_iteration_sequential(args...; kwargs...)
  elseif multisite_update_alg == "parallel"
    return tdvp_iteration_parallel(args...; kwargs...)
  else
    error(
      "Multisite update algorithm multisite_update_alg = $multisite_update_alg not supported, use \"parallel\" or \"sequential\"",
    )
  end
end

function tdvp_iteration_sequential(
  solver::Function,
  ∑h::InfiniteITensorSum,
  ψ::InfiniteCanonicalMPS;
  (ϵᴸ!)=fill(1e-15, nsites(ψ)),
  (ϵᴿ!)=fill(1e-15, nsites(ψ)),
  time_step,
  solver_tol=(x -> x / 100),
)
  Nsites = nsites(ψ)
  ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
  _solver_tol = solver_tol(ϵᵖʳᵉˢ)
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  # XXX: make this prime the center sites
  ψ̃ = prime(linkinds, ψᴴ)

  # TODO: replace with linkinds(ψ)
  l = CelledVector([commoninds(ψ.AL[n], ψ.AL[n + 1]) for n in 1:Nsites])
  l′ = CelledVector([commoninds(ψ′.AL[n], ψ′.AL[n + 1]) for n in 1:Nsites])
  r = CelledVector([commoninds(ψ.AR[n], ψ.AR[n + 1]) for n in 1:Nsites])
  r′ = CelledVector([commoninds(ψ′.AR[n], ψ′.AR[n + 1]) for n in 1:Nsites])

  ψ = copy(ψ)
  C̃ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  Ãᶜ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  Ãᴸ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  Ãᴿ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  eᴸ = Vector{Float64}(undef, Nsites)
  eᴿ = Vector{Float64}(undef, Nsites)
  for n in 1:Nsites
    hᴸ = InfiniteMPS([
      δ(only(l[k - 2]), only(l′[k - 2])) *
      ψ.AL[k - 1] *
      ψ.AL[k] *
      ∑h[(k - 1, k)] *
      ψ′.AL[k - 1] *
      ψ′.AL[k] for k in 1:Nsites
    ])
    hᴿ = InfiniteMPS([
      δ(only(dag(r[k + 2])), only(dag(r′[k + 2]))) *
      ψ.AR[k + 2] *
      ψ.AR[k + 1] *
      ∑h[(k + 1, k + 2)] *
      ψ′.AR[k + 2] *
      ψ′.AR[k + 1] for k in 1:Nsites
    ])
    eᴸ = [
      (hᴸ[k] * ψ.C[k] * δ(only(dag(r[k])), only(dag(r′[k]))) * ψ′.C[k])[] for k in 1:Nsites
    ]
    eᴿ = [(hᴿ[k] * ψ.C[k] * δ(only(l[k]), only(l′[k])) * ψ′.C[k])[] for k in 1:Nsites]
    for k in 1:Nsites
      # TODO: remove `denseblocks` once BlockSparse + DiagBlockSparse is supported
      hᴸ[k] -= eᴸ[k] * denseblocks(δ(inds(hᴸ[k])))
      hᴿ[k] -= eᴿ[k] * denseblocks(δ(inds(hᴿ[k])))
    end

    function left_environment_cell(ψ, ψ̃, hᴸ, n)
      Nsites = nsites(ψ)
      𝕙ᴸ = copy(hᴸ)
      for k in reverse((n - Nsites + 2):n)
        𝕙ᴸ[k] = 𝕙ᴸ[k - 1] * ψ.AL[k] * ψ̃.AL[k] + 𝕙ᴸ[k]
      end
      return 𝕙ᴸ[n]
    end

    #for k in 2:Nsites
    #  hᴸ[k] = hᴸ[k - 1] * ψ.AL[k] * ψ̃.AL[k] + hᴸ[k]
    #end
    𝕙ᴸ = copy(hᴸ)
    for k in 1:Nsites
      𝕙ᴸ[k] = left_environment_cell(ψ, ψ̃, hᴸ, k)
    end
    Hᴸ = left_environment(hᴸ, 𝕙ᴸ, ψ; tol=_solver_tol)
    for k in 2:Nsites
      hᴿ[k] = hᴿ[k + 1] * ψ.AR[k + 1] * ψ̃.AR[k + 1] + hᴿ[k]
    end
    Hᴿ = right_environment(hᴿ, ψ; tol=_solver_tol)

    Cvalsₙ₋₁, Cvecsₙ₋₁, Cinfoₙ₋₁ = solver(
      Hᶜ(∑h, Hᴸ, Hᴿ, ψ, n - 1), time_step, ψ.C[n - 1], _solver_tol
    )
    Cvalsₙ, Cvecsₙ, Cinfoₙ = solver(Hᶜ(∑h, Hᴸ, Hᴿ, ψ, n), time_step, ψ.C[n], _solver_tol)
    Avalsₙ, Avecsₙ, Ainfoₙ = solver(
      Hᴬᶜ(∑h, Hᴸ, Hᴿ, ψ, n), time_step, ψ.AL[n] * ψ.C[n], _solver_tol
    )

    C̃[n - 1] = Cvecsₙ₋₁
    C̃[n] = Cvecsₙ
    Ãᶜ[n] = Avecsₙ

    function ortho_overlap(AC, C)
      AL, _ = polar(AC * dag(C), uniqueinds(AC, C))
      return noprime(AL)
    end

    function ortho_polar(AC, C)
      UAC, _ = polar(AC, uniqueinds(AC, C))
      UC, _ = polar(C, commoninds(C, AC))
      return noprime(UAC) * noprime(dag(UC))
    end

    Ãᴸ[n] = ortho_polar(Ãᶜ[n], C̃[n])
    Ãᴿ[n] = ortho_polar(Ãᶜ[n], C̃[n - 1])

    # Update state for next iteration
    #ψ = InfiniteCanonicalMPS(Ãᴸ, C̃, Ãᴿ)
    ψ.AL[n] = Ãᴸ[n]
    ψ.AR[n] = Ãᴿ[n]
    ψ.C[n - 1] = C̃[n - 1]
    ψ.C[n] = C̃[n]
    ψᴴ = dag(ψ)
    ψ′ = ψᴴ'
    # XXX: make this prime the center sites
    ψ̃ = prime(linkinds, ψᴴ)

    # TODO: replace with linkinds(ψ)
    l = CelledVector([commoninds(ψ.AL[n], ψ.AL[n + 1]) for n in 1:Nsites])
    l′ = CelledVector([commoninds(ψ′.AL[n], ψ′.AL[n + 1]) for n in 1:Nsites])
    r = CelledVector([commoninds(ψ.AR[n], ψ.AR[n + 1]) for n in 1:Nsites])
    r′ = CelledVector([commoninds(ψ′.AR[n], ψ′.AR[n + 1]) for n in 1:Nsites])
  end
  for n in 1:Nsites
    ϵᴸ![n] = norm(Ãᶜ[n] - Ãᴸ[n] * C̃[n])
    ϵᴿ![n] = norm(Ãᶜ[n] - C̃[n - 1] * Ãᴿ[n])
  end
  return ψ, (eᴸ, eᴿ)
end

function tdvp_iteration_parallel(
  solver::Function,
  ∑h::InfiniteITensorSum,
  ψ::InfiniteCanonicalMPS;
  (ϵᴸ!)=fill(1e-15, nsites(ψ)),
  (ϵᴿ!)=fill(1e-15, nsites(ψ)),
  solver_tol=(x -> x / 100),
)
  Nsites = nsites(ψ)
  ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
  _solver_tol = solver_tol(ϵᵖʳᵉˢ)
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  # XXX: make this prime the center sites
  ψ̃ = prime(linkinds, ψᴴ)

  # TODO: replace with linkinds(ψ)
  l = CelledVector([commoninds(ψ.AL[n], ψ.AL[n + 1]) for n in 1:Nsites])
  l′ = CelledVector([commoninds(ψ′.AL[n], ψ′.AL[n + 1]) for n in 1:Nsites])
  r = CelledVector([commoninds(ψ.AR[n], ψ.AR[n + 1]) for n in 1:Nsites])
  r′ = CelledVector([commoninds(ψ′.AR[n], ψ′.AR[n + 1]) for n in 1:Nsites])

  hᴸ = InfiniteMPS([
    δ(only(l[n - 2]), only(l′[n - 2])) *
    ψ.AL[n - 1] *
    ψ.AL[n] *
    ∑h[(n - 1, n)] *
    ψ′.AL[n - 1] *
    ψ′.AL[n] for n in 1:Nsites
  ])

  hᴿ = InfiniteMPS([
    δ(only(dag(r[n + 2])), only(dag(r′[n + 2]))) *
    ψ.AR[n + 2] *
    ψ.AR[n + 1] *
    ∑h[(n + 1, n + 2)] *
    ψ′.AR[n + 2] *
    ψ′.AR[n + 1] for n in 1:Nsites
  ])

  eᴸ = [
    (hᴸ[n] * ψ.C[n] * δ(only(dag(r[n])), only(dag(r′[n]))) * ψ′.C[n])[] for n in 1:Nsites
  ]
  eᴿ = [(hᴿ[n] * ψ.C[n] * δ(only(l[n]), only(l′[n])) * ψ′.C[n])[] for n in 1:Nsites]

  for n in 1:Nsites
    # TODO: use these instead, for now can't subtract
    # BlockSparse and DiagBlockSparse tensors
    #hᴸ[n] -= eᴸ[n] * δ(inds(hᴸ[n]))
    #hᴿ[n] -= eᴿ[n] * δ(inds(hᴿ[n]))
    hᴸ[n] -= eᴸ[n] * denseblocks(δ(inds(hᴸ[n])))
    hᴿ[n] -= eᴿ[n] * denseblocks(δ(inds(hᴿ[n])))
  end

  # Sum the Hamiltonian terms in the unit cell
  for n in 2:Nsites
    hᴸ[n] = hᴸ[n - 1] * ψ.AL[n] * ψ̃.AL[n] + hᴸ[n]
  end
  Hᴸ = left_environment(hᴸ, ψ; tol=_solver_tol)

  for n in 2:Nsites
    hᴿ[n] = hᴿ[n + 1] * ψ.AR[n + 1] * ψ̃.AR[n + 1] + hᴿ[n]
  end
  Hᴿ = right_environment(hᴿ, ψ; tol=_solver_tol)

  C̃ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  Ãᶜ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  for n in 1:Nsites
    Cvalsₙ, Cvecsₙ, Cinfoₙ = solver(Hᶜ(∑h, Hᴸ, Hᴿ, ψ, n), time_step, ψ.C[n], _solver_tol)
    Avalsₙ, Avecsₙ, Ainfoₙ = solver(
      Hᴬᶜ(∑h, Hᴸ, Hᴿ, ψ, n), time_step, ψ.AL[n] * ψ.C[n], _solver_tol
    )

    C̃[n] = Cvecsₙ[1]
    Ãᶜ[n] = Avecsₙ[1]
  end

  # TODO: based on minimum singular values of C̃, use more accurate
  # method for finding Ãᴸ, Ãᴿ
  Ãᴸ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  Ãᴿ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  for n in 1:Nsites
    Ãᴸⁿ, X = polar(Ãᶜ[n] * dag(C̃[n]), uniqueinds(Ãᶜ[n], C̃[n]))
    Ãᴿⁿ, _ = polar(Ãᶜ[n] * dag(C̃[n - 1]), uniqueinds(Ãᶜ[n], C̃[n - 1]))
    Ãᴸⁿ = noprime(Ãᴸⁿ)
    Ãᴿⁿ = noprime(Ãᴿⁿ)
    Ãᴸ[n] = Ãᴸⁿ
    Ãᴿ[n] = Ãᴿⁿ
  end

  for n in 1:Nsites
    ϵᴸ![n] = norm(Ãᶜ[n] - Ãᴸ[n] * C̃[n])
    ϵᴿ![n] = norm(Ãᶜ[n] - C̃[n - 1] * Ãᴿ[n])
  end
  return InfiniteCanonicalMPS(Ãᴸ, C̃, Ãᴿ), (eᴸ, eᴿ)
end

function tdvp(
  solver::Function,
  ∑h,
  ψ;
  maxiter=10,
  tol=1e-8,
  outputlevel=1,
  multisite_update_alg="sequential",
  solver_tol=(x -> x / 100),
  time_step,
)
  N = nsites(ψ)
  (ϵᴸ!) = fill(tol, nsites(ψ))
  (ϵᴿ!) = fill(tol, nsites(ψ))
  outputlevel > 0 &&
    println("Running VUMPS with multisite_update_alg = $multisite_update_alg")
  for iter in 1:maxiter
    ψ, (eᴸ, eᴿ) = tdvp_iteration(
      solver,
      ∑h,
      ψ;
      (ϵᴸ!)=(ϵᴸ!),
      (ϵᴿ!)=(ϵᴿ!),
      multisite_update_alg=multisite_update_alg,
      solver_tol=solver_tol,
      time_step=time_step,
    )
    ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
    maxdimψ = maxlinkdim(ψ[0:(N + 1)])
    outputlevel > 0 && println(
      "VUMPS iteration $iter (out of maximum $maxiter). Bond dimension = $maxdimψ, energy = $((eᴸ, eᴿ)), ϵᵖʳᵉˢ = $ϵᵖʳᵉˢ, tol = $tol",
    )
    if ϵᵖʳᵉˢ < tol
      println(
        "Precision error $ϵᵖʳᵉˢ reached tolerance $tol, stopping VUMPS after $iter iterations (of a maximum $maxiter).",
      )
      break
    end
  end
  return ψ
end

function vumps_solver(M, time_step, v₀, solver_tol)
  λ⃗, v⃗, info = eigsolve(M, v₀, 1, :SR; ishermitian=true, tol=solver_tol)
  return λ⃗[1], v⃗[1], info
end

return function tdvp_solver(M, time_step, v₀, solver_tol)
  v, info = exponentiate(M, time_step, v₀; ishermitian=true, tol=solver_tol)
  return nothing, v, info
end

function vumps(
  args...; time_step=-Inf, eigsolve_tol=(x -> x / 100), solver_tol=eigsolve_tol, kwargs...
)
  @assert isinf(time_step) && time_step < 0
  println("Using VUMPS solver with time step $time_step")
  return tdvp(vumps_solver, args...; time_step=time_step, solver_tol=solver_tol, kwargs...)
end

function tdvp(args...; time_step, solver_tol=(x -> x / 100), kwargs...)
  solver = if !isinf(time_step)
    println("Using TDVP solver with time step $time_step")
    tdvp_solver
  elseif time_step < 0
    # Call VUMPS instead
    println("Using VUMPS solver with time step $time_step")
    vumps_solver
  else
    error("Time step $time_step not supported.")
  end
  return tdvp(solver, args...; time_step=time_step, solver_tol=solver_tol, kwargs...)
end

##################################################################
# Old functionality, only used for testing

## function left_environment_recursive(hᴸ, ψ; niter=10)
##   ψ̃ = prime(linkinds, dag(ψ))
##   # XXX: replace with `nsites`
##   #N = nsites(ψ)
##   N = length(ψ)
##   Hᴸᴺ¹ = hᴸ[N]
##   for _ in 1:niter
##     Hᴸᴺ¹ = translatecell(Hᴸᴺ¹, -1)
##     for n in 1:N
##       Hᴸᴺ¹ = Hᴸᴺ¹ * ψ.AL[n] * ψ̃.AL[n]
##     end
##     # Loop over the Hamiltonian terms in the unit cell
##     for n in 1:N
##       hᴸⁿ = hᴸ[n]
##       for k in (n + 1):N
##         hᴸⁿ = hᴸⁿ * ψ.AL[k] * ψ̃.AL[k]
##       end
##       Hᴸᴺ¹ += hᴸⁿ
##     end
##   end
##   # Get the rest of the environments in the unit cell
##   Hᴸ = InfiniteMPS(Vector{ITensor}(undef, N))
##   Hᴸ[N] = Hᴸᴺ¹
##   Hᴸᴺ¹ = translatecell(Hᴸᴺ¹, -1)
##   for n in 1:(N - 1)
##     Hᴸ[n] = Hᴸ[n - 1] * ψ.AL[n] * ψ̃.AL[n] + hᴸ[n]
##   end
##   return Hᴸ
## end
## 
## function right_environment_recursive(hᴿ, ψ; niter=10)
##   ψ̃ = prime(linkinds, dag(ψ))
##   # XXX: replace with `nsites`
##   #N = nsites(ψ)
##   N = length(ψ)
##   Hᴿᴺ¹ = hᴿ[0]
##   for _ in 1:niter
##     Hᴿᴺ¹ = translatecell(Hᴿᴺ¹, 1)
##     for n in reverse(1:N)
##       Hᴿᴺ¹ = Hᴿᴺ¹ * ψ.AR[n] * ψ̃.AR[n]
##     end
##     # Loop over the Hamiltonian terms in the unit cell
##     for n in reverse(0:(N - 1))
##       hᴿⁿ = hᴿ[n]
##       for k in reverse(1:n)
##         hᴿⁿ = hᴿⁿ * ψ.AR[k] * ψ̃.AR[k]
##       end
##       Hᴿᴺ¹ += hᴿⁿ
##     end
##   end
##   Hᴿᴺ¹ = translatecell(Hᴿᴺ¹, 1)
##   # Get the rest of the environments in the unit cell
##   Hᴿ = InfiniteMPS(Vector{ITensor}(undef, N))
##   Hᴿ[N] = Hᴿᴺ¹
##   for n in reverse(1:(N - 1))
##     Hᴿ[n] = Hᴿ[n + 1] * ψ.AR[n + 1] * ψ̃.AR[n + 1] + hᴿ[n]
##   end
##   return Hᴿ
## end
