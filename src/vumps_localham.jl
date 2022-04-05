function Base.:*(H::Hᶜ{ITensor}, v::ITensor)
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

function Base.:*(H::Hᴬᶜ{ITensor}, v::ITensor)
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

function tdvp_iteration_sequential(
  solver::Function,
  ∑h::InfiniteSum{ITensor},
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
  ∑h::InfiniteSum{ITensor},
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

    C̃[n] = Cvecsₙ
    Ãᶜ[n] = Avecsₙ
  end

  function ortho_overlap(AC, C)
    AL, _ = polar(AC * dag(C), uniqueinds(AC, C))
    return noprime(AL)
  end

  function ortho_polar(AC, C)
    UAC, _ = polar(AC, uniqueinds(AC, C))
    UC, _ = polar(C, commoninds(C, AC))
    return noprime(UAC) * noprime(dag(UC))
  end

  Ãᴸ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  Ãᴿ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  for n in 1:Nsites
    Ãᴸ[n] = ortho_polar(Ãᶜ[n], C̃[n])
    Ãᴿ[n] = ortho_polar(Ãᶜ[n], C̃[n - 1])
  end

  for n in 1:Nsites
    ϵᴸ![n] = norm(Ãᶜ[n] - Ãᴸ[n] * C̃[n])
    ϵᴿ![n] = norm(Ãᶜ[n] - C̃[n - 1] * Ãᴿ[n])
  end
  return InfiniteCanonicalMPS(Ãᴸ, C̃, Ãᴿ), (eᴸ, eᴿ)
end
