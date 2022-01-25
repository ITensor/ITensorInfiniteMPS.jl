
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


#Assume all local Hamiltonians have the same nrange
function Base.:*(H::Hᶜ, v::ITensor)
  ∑h = H.∑h
  Hᴸ = H.Hᴸ
  Hᴿ = H.Hᴿ
  ψ = H.ψ
  ψ′ = dag(ψ)'
  Nsites = nsites(ψ)
  range_∑h = nrange(∑h, 1)
  n = H.n
  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)
  s = siteinds(only, ψ)
  s′ = siteinds(only, ψ′)
  δˢ(n) = δ(s[n], s′[n])

  δˡ = δ(l[n], l′[n])
  δʳ = δ(r[n], r′[n])
  Hᶜᴸv = v * Hᴸ[n] * dag(δʳ)
  Hᶜᴿv = v * δˡ * Hᴿ[n]
  # TODO do we prefer a formalism like the commented one? if yes, need to define contraction of ITensor and MPS
  #Hᶜʰv = v * ψ.AL[n] * δ(l[n - 1], l′[n - 1]) * ψ′.AL[n] * ∑h[(n, n + 1)] *
  #  ψ.AR[n+1:n + range_∑h - 1]... * ψ′.AR[n+1:n + range_∑h - 1]... * dag(δ(r[n+range_∑h - 1], r′[n+range_∑h - 1]))
  Hᶜʰv = v * ψ.AL[n] * δ(l[n - 1], l′[n - 1]) * ψ′.AL[n] * ∑h[(n, n + 1)]
  for k in 1:range_∑h-2
      Hᶜʰv =  Hᶜʰv * ψ.AR[n + k] * ψ′.AR[n + k]
  end
  Hᶜʰv =  Hᶜʰv * ψ.AR[n + range_∑h - 1] * dag(δ(r[n+range_∑h - 1], r′[n+range_∑h - 1])) * ψ′.AR[n + range_∑h - 1]
  for j in 1:range_∑h-2
    temp_Hᶜʰv = ψ.AL[n-j] * δ(l[n - 1 - j], l′[n - 1 - j]) * ψ′.AL[n - j] * ∑h[(n - j, n + 1 - j)];
    for k in 1:j
        temp_Hᶜʰv  =  temp_Hᶜʰv  * ψ.AL[n - j + k] * ψ′.AL[n - j + k]
    end
    temp_Hᶜʰv = temp_Hᶜʰv  * v
    for k in j+1:range_∑h-2
        temp_Hᶜʰv =  temp_Hᶜʰv * ψ.AR[n - j + k] * ψ′.AR[n - j + k]
    end
    temp_Hᶜʰv =  temp_Hᶜʰv * ψ.AR[n - j + range_∑h - 1] * dag(δ(r[n - j +  range_∑h - 1], r′[n - j +  range_∑h - 1])) * ψ′.AR[n - j + range_∑h - 1]
    Hᶜʰv = Hᶜʰv + temp_Hᶜʰv
  end

  Hᶜv = Hᶜᴸv + Hᶜʰv + Hᶜᴿv
  return Hᶜv * dag(δˡ) * δʳ
end


function Base.:*(H::Hᴬᶜ, v::ITensor)
  ∑h = H.∑h
  Hᴸ = H.Hᴸ
  Hᴿ = H.Hᴿ
  ψ = H.ψ
  ψ′ = dag(ψ)'
  Nsites = nsites(ψ)
  range_∑h = nrange(∑h, 1)
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

  #
  Hᴬᶜʰv = v * δˡ(n - 1) * ∑h[(n, n + 1)]
  for k in 1:range_∑h-2
      Hᴬᶜʰv =  Hᴬᶜʰv * ψ.AR[n + k] * ψ′.AR[n + k]
  end
  Hᴬᶜʰv =  Hᴬᶜʰv * ψ.AR[n + range_∑h - 1] * ψ′.AR[n + range_∑h - 1] * dag(δʳ(n + range_∑h - 1))

  for j in 1:range_∑h-1
    temp_Hᴬᶜʰv = ψ.AL[n-j] * δˡ(n -j - 1) * ψ′.AL[n-j] * ∑h[(n - j, n - j + 1)]
    for k in 1:j-1
        temp_Hᴬᶜʰv  =  temp_Hᴬᶜʰv  * ψ.AL[n - j + k] * ψ′.AL[n - j + k]
    end
    temp_Hᴬᶜʰv = temp_Hᴬᶜʰv  * v
    for k in j+1:range_∑h-1
        temp_Hᴬᶜʰv =  temp_Hᴬᶜʰv * ψ.AR[n + k - j] * ψ′.AR[n + k - j]
    end
    temp_Hᴬᶜʰv =   temp_Hᴬᶜʰv * dag(δʳ(n - j + range_∑h - 1))

    Hᴬᶜʰv = Hᴬᶜʰv + temp_Hᴬᶜʰv
  end
  Hᴬᶜv = Hᴬᶜᴸv + Hᴬᶜʰv + Hᴬᶜᴿv
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

# TODO Generate all environments, why? Only one is needed in the sequential version
function left_environment(hᴸ, 𝕙ᴸ, ψ; tol=1e-15)
  ψ̃ = prime(linkinds, dag(ψ))
  N = nsites(ψ)

  Aᴺ = Aᴸ(ψ, N)
  Hᴸᴺ¹, info = linsolve(Aᴺ, 𝕙ᴸ[N], 1, -1; tol=tol)
  # Get the rest of the environments in the unit cell
  Hᴸ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴸ[N] = Hᴸᴺ¹
  #Hᴸᴺ¹ = translatecell(Hᴸᴺ¹, -1) #Found it was useless?
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

# TODO Generate all environments, why? Only one is needed in the sequential version
function right_environment(hᴿ, 𝕙ᴿ, ψ; tol=1e-15)
  ψ̃ = prime(linkinds, dag(ψ))
  N = nsites(ψ)

  A = Aᴿ(hᴿ, ψ, N)
  Hᴿᴺ¹, info = linsolve(A, 𝕙ᴿ[N], 1, -1; tol=tol)
  # Get the rest of the environments in the unit cell
  Hᴿ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴿ[N] = Hᴿᴺ¹
  for n in reverse(1:(N - 1))
    Hᴿ[n] = Hᴿ[n + 1] * ψ.AR[n + 1] * ψ̃.AR[n + 1] + hᴿ[n]
  end
  return Hᴿ
end

function vumps_iteration(args...; multisite_update_alg="sequential", kwargs...)
  if multisite_update_alg == "sequential"
    return vumps_iteration_sequential(args...; kwargs...)
  elseif multisite_update_alg == "parallel"
    return vumps_iteration_parallel(args...; kwargs...)
  else
    error(
      "Multisite update algorithm multisite_update_alg = $multisite_update_alg not supported, use \"parallel\" or \"sequential\"",
    )
  end
end


function vumps_iteration_sequential(
  ∑h::InfiniteITensorSum,
  ψ::InfiniteCanonicalMPS;
  (ϵᴸ!)=fill(1e-15, nsites(ψ)),
  (ϵᴿ!)=fill(1e-15, nsites(ψ)),
  eigsolve_tol=(x -> x / 100),
)
  Nsites = nsites(ψ)
  range_∑h = nrange(∑h, 1)
  ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
  krylov_tol = eigsolve_tol(ϵᵖʳᵉˢ)
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
    # TODO improve the multisite contraction such that we contract with identities
    hᴸ = Vector{ITensor}(undef, Nsites)
    for k in 1:Nsites
    hᴸ[k] = δ(only(l[k - range_∑h]), only(l′[k - range_∑h])) *
      ψ.AL[k - range_∑h + 1] *
      ∑h[(k - range_∑h + 1, k - range_∑h + 2)] * ψ′.AL[k - range_∑h + 1]
      for j in 2:range_∑h
          hᴸ[k] = hᴸ[k] * ψ.AL[k - range_∑h + j] * ψ′.AL[k - range_∑h + j]
      end
    end
    hᴸ = InfiniteMPS(hᴸ)

    hᴿ = Vector{ITensor}(undef, Nsites)
    for k in 1:Nsites
      hᴿ[k] = ψ.AR[k + range_∑h] * ∑h[(k + 1, k + 2)] * ψ′.AR[k + range_∑h]* δ(only(dag(r[k + range_∑h])), only(dag(r′[k + range_∑h])))
      for j in range_∑h-1:-1:1
          hᴿ[k] = hᴿ[k] * ψ.AR[k + j] * ψ′.AR[k + j]
      end
    end
    hᴿ = InfiniteMPS(hᴿ)
    eᴸ = [(hᴸ[k] * ψ.C[k] * δ(only(dag(r[k])), only(dag(r′[k]))) * ψ′.C[k])[] for k in 1:Nsites]
    eᴿ = [(hᴿ[k] * ψ.C[k] * δ(only(l[k]), only(l′[k])) * ψ′.C[k])[] for k in 1:Nsites]
    for k in 1:Nsites
      # TODO: remove `denseblocks` once BlockSparse + DiagBlockSparse is supported
      hᴸ[k] -= eᴸ[k] * denseblocks(δ(inds(hᴸ[k])))
      hᴿ[k] -= eᴿ[k] * denseblocks(δ(inds(hᴿ[k])))
    end

    # TODO Promote full function?
    function left_environment_cell(ψ, ψ̃, hᴸ)
      Nsites = nsites(ψ)
      𝕙ᴸ = copy(hᴸ)
      # TODO restrict to the useful ones only?
      for n = 1:Nsites
        for k in  1:Nsites - 1
          temp = copy(hᴸ[n-k])
          for kp in reverse(0:k-1)
            temp = temp * ψ.AL[n - kp] * ψ̃.AL[n - kp]
          end
          𝕙ᴸ[n] =temp + 𝕙ᴸ[n]
        end
      end
      return 𝕙ᴸ
    end


    𝕙ᴸ = left_environment_cell(ψ, ψ̃, hᴸ)
    Hᴸ = left_environment(hᴸ, 𝕙ᴸ, ψ; tol=krylov_tol)


    # TODO Promote full function
    function right_environment_cell(ψ, ψ̃, hᴿ)
      Nsites = nsites(ψ)
      𝕙ᴿ = copy(hᴿ)
      # TODO restrict to the useful ones only
      for n = 1:Nsites
        for k in  1:Nsites - 1
          temp = copy(hᴿ[n + k])
          for kp in reverse(1:k)
            temp = temp * ψ.AR[n + kp] * ψ̃.AR[n + kp]
          end
          𝕙ᴿ[n] =temp + 𝕙ᴿ[n]
        end
      end
      return 𝕙ᴿ
    end

    𝕙ᴿ = right_environment_cell(ψ, ψ̃, hᴿ)
    Hᴿ = right_environment(hᴿ, 𝕙ᴿ, ψ; tol=krylov_tol)

    # TODO check indices coherences
    Cvalsₙ₋₁, Cvecsₙ₋₁, Cinfoₙ₋₁ = eigsolve(
      Hᶜ(∑h, Hᴸ, Hᴿ, ψ, n - 1), ψ.C[n - 1], 1, :SR; ishermitian=true, tol=krylov_tol
    )
    Cvalsₙ, Cvecsₙ, Cinfoₙ = eigsolve(
      Hᶜ(∑h, Hᴸ, Hᴿ, ψ, n), ψ.C[n], 1, :SR; ishermitian=true, tol=krylov_tol
    )

    Avalsₙ, Avecsₙ, Ainfoₙ = eigsolve(
      Hᴬᶜ(∑h, Hᴸ, Hᴿ, ψ, n), ψ.AL[n] * ψ.C[n], 1, :SR; ishermitian=true, tol=krylov_tol
    )

    C̃[n - 1] = Cvecsₙ₋₁[1]
    C̃[n] = Cvecsₙ[1]
    Ãᶜ[n] = Avecsₙ[1]

    function ortho_overlap(AC, C)
      AL, _ = polar(AC * dag(C), uniqueinds(AC, C))
      return noprime(AL)
    end

    function ortho_polar(AC, C)
      UAC, _ = polar(AC, uniqueinds(AC, C))
      UC, _ = polar(C, commoninds(C, AC))
      return noprime(UAC) * noprime(dag(UC))
    end

    function ortho_polar_right(AC, C)
      UAC, _ = polar(dag(AC), uniqueinds(AC, C))
      UC, _ = polar(dag(C), commoninds(C, AC))
      return noprime(dag(UAC)) * noprime(UC)
    end

    Ãᴸ[n] = ortho_polar(Ãᶜ[n], C̃[n])
    Ãᴿ[n] = ortho_polar_right(Ãᶜ[n], C̃[n - 1])
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

function vumps_iteration_parallel(
  ∑h::InfiniteITensorSum,
  ψ::InfiniteCanonicalMPS;
  (ϵᴸ!)=fill(1e-15, nsites(ψ)),
  (ϵᴿ!)=fill(1e-15, nsites(ψ)),
  eigsolve_tol=(x -> x / 100),
)
  Nsites = nsites(ψ)
  ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
  krylov_tol = ϵᵖʳᵉˢ / 100
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
  Hᴸ = left_environment(hᴸ, ψ; tol=krylov_tol)

  for n in 2:Nsites
    hᴿ[n] = hᴿ[n + 1] * ψ.AR[n + 1] * ψ̃.AR[n + 1] + hᴿ[n]
  end
  Hᴿ = right_environment(hᴿ, ψ; tol=krylov_tol)

  C̃ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  Ãᶜ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  for n in 1:Nsites
    Cvalsₙ, Cvecsₙ, Cinfoₙ = eigsolve(
      Hᶜ(∑h, Hᴸ, Hᴿ, ψ, n), ψ.C[n], 1, :SR; ishermitian=true, tol=krylov_tol
    )
    Avalsₙ, Avecsₙ, Ainfoₙ = eigsolve(
      Hᴬᶜ(∑h, Hᴸ, Hᴿ, ψ, n), ψ.AL[n] * ψ.C[n], 1, :SR; ishermitian=true, tol=krylov_tol
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

function vumps(
  ∑h,
  ψ;
  maxiter=10,
  tol=1e-8,
  outputlevel=1,
  multisite_update_alg="sequential",
  eigsolve_tol=(x -> x / 100),
)
  N = nsites(ψ)
  (ϵᴸ!) = fill(tol, nsites(ψ))
  (ϵᴿ!) = fill(tol, nsites(ψ))
  outputlevel > 0 &&
    println("Running VUMPS with multisite_update_alg = $multisite_update_alg")
  for iter in 1:maxiter
    ψ, (eᴸ, eᴿ) = vumps_iteration(
      ∑h,
      ψ;
      (ϵᴸ!)=(ϵᴸ!),
      (ϵᴿ!)=(ϵᴿ!),
      multisite_update_alg=multisite_update_alg,
      eigsolve_tol=eigsolve_tol,
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
