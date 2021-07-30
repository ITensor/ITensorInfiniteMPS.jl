
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

function left_environment_recursive(hᴸ, ψ; niter=10)
  ψ̃ = prime(linkinds, dag(ψ))
  # XXX: replace with `nsites`
  #N = nsites(ψ)
  N = length(ψ)
  Hᴸᴺ¹ = hᴸ[N]
  for _ in 1:niter
    Hᴸᴺ¹ = translatecell(Hᴸᴺ¹, -1)
    for n in 1:N
      Hᴸᴺ¹ = Hᴸᴺ¹ * ψ.AL[n] * ψ̃.AL[n]
    end
    # Loop over the Hamiltonian terms in the unit cell
    for n in 1:N
      hᴸⁿ = hᴸ[n]
      for k in (n + 1):N
        hᴸⁿ = hᴸⁿ * ψ.AL[k] * ψ̃.AL[k]
      end
      Hᴸᴺ¹ += hᴸⁿ
    end
  end
  # Get the rest of the environments in the unit cell
  Hᴸ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴸ[N] = Hᴸᴺ¹
  Hᴸᴺ¹ = translatecell(Hᴸᴺ¹, -1)
  for n in 1:(N - 1)
    Hᴸ[n] = Hᴸ[n - 1] * ψ.AL[n] * ψ̃.AL[n] + hᴸ[n]
  end
  return Hᴸ
end

function right_environment_recursive(hᴿ, ψ; niter=10)
  ψ̃ = prime(linkinds, dag(ψ))
  # XXX: replace with `nsites`
  #N = nsites(ψ)
  N = length(ψ)
  Hᴿᴺ¹ = hᴿ[0]
  for _ in 1:niter
    Hᴿᴺ¹ = translatecell(Hᴿᴺ¹, 1)
    for n in reverse(1:N)
      Hᴿᴺ¹ = Hᴿᴺ¹ * ψ.AR[n] * ψ̃.AR[n]
    end
    # Loop over the Hamiltonian terms in the unit cell
    for n in reverse(0:(N - 1))
      hᴿⁿ = hᴿ[n]
      for k in reverse(1:n)
        hᴿⁿ = hᴿⁿ * ψ.AR[k] * ψ̃.AR[k]
      end
      Hᴿᴺ¹ += hᴿⁿ
    end
  end
  Hᴿᴺ¹ = translatecell(Hᴿᴺ¹, 1)
  # Get the rest of the environments in the unit cell
  Hᴿ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴿ[N] = Hᴿᴺ¹
  for n in reverse(1:(N - 1))
    Hᴿ[n] = Hᴿ[n + 1] * ψ.AR[n + 1] * ψ̃.AR[n + 1] + hᴿ[n]
  end
  return Hᴿ
end

function vumps_iteration(
  ∑h::InfiniteITensorSum, ψ::InfiniteCanonicalMPS; environment_iterations=10
)
  Nsites = nsites(ψ)
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  # XXX: make this prime the center sites
  ψ̃ = prime(linkinds, ψᴴ)

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

  Hᴸ = left_environment_recursive(hᴸ, ψ; niter=environment_iterations)
  Hᴿ = right_environment_recursive(hᴿ, ψ; niter=environment_iterations)

  C̃ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  for n in 1:Nsites
    valsₙ, vecsₙ, infoₙ = eigsolve(Hᶜ(∑h, Hᴸ, Hᴿ, ψ, n), ψ.C[n], 1, :SR; ishermitian=true)
    C̃[n] = vecsₙ[1]
  end

  Ãᶜ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  for n in 1:Nsites
    valsₙ, vecsₙ, infoₙ = eigsolve(
      Hᴬᶜ(∑h, Hᴸ, Hᴿ, ψ, n), ψ.AL[n] * ψ.C[n], 1, :SR; ishermitian=true
    )
    Ãᶜ[n] = vecsₙ[1]
  end

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
  return InfiniteCanonicalMPS(Ãᴸ, C̃, Ãᴿ), (eᴸ, eᴿ)
end

function vumps(∑h, ψ; niter=10, environment_iterations=10, outputlevel=1)
  N = nsites(ψ)
  for iter in 1:niter
    ψ, (eᴸ, eᴿ) = vumps_iteration(∑h, ψ; environment_iterations=environment_iterations)
    maxdimψ = maxlinkdim(ψ[0:(N + 1)])
    if outputlevel > 0
      println("VUMPS iteration $iter of $niter. Bond dimension = $maxdimψ, energy = ", eᴸ)
    end
  end
  return ψ
end
