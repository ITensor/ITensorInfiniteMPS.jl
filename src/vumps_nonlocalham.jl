#Assume all local Hamiltonians have the same nrange
function Base.:*(H::Hᶜ{MPO}, v::ITensor)
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

  # Remove temporarily in favor of `replaceinds`,
  # which is more efficient.
  # δˢ(n) = δ(dag(s[n]), prime(s[n]))
  # δʳ(n) = δ(dag(r[n]), prime(r[n]))
  # δˡ(n) = δ(l[n], l′[n])

  #Build the contribution of the left environment
  # Hᶜᴸv = v * Hᴸ[n] * δʳ(n)
  Hᶜᴸv = replaceinds(v * Hᴸ[n], r[n] => r[n]')
  #Build the contribution of the right environment
  # Hᶜᴿv = v * δˡ(n) * Hᴿ[n]
  Hᶜᴿv = replaceinds(v, l[n] => l[n]') * Hᴿ[n]
  #We now start building terms where C overlap with the local Hamiltonian
  # We start with the tensor AL[n] - v - AR[n+1] ... AR[n + range_∑h - 1]
  # Hᶜʰv =
  #   δʳ(n + range_∑h - 1) * ψ.AR[n + range_∑h - 1] * ∑h[n][end] * ψ′.AR[n + range_∑h - 1]
  Hᶜʰv =
    replaceinds(ψ.AR[n + range_∑h - 1], r[n + range_∑h - 1] => r[n + range_∑h - 1]') *
    ∑h[n][end] *
    ψ′.AR[n + range_∑h - 1]
  common_sites = findsites(ψ, ∑h[n])
  idx = length(∑h[n]) - 1 #list the sites Σh, we start at 2 because n is already taken into account
  for k in reverse(1:(range_∑h - 2))
    if n + k == common_sites[idx]
      Hᶜʰv = Hᶜʰv * ψ.AR[n + k] * ∑h[n][idx] * ψ′.AR[n + k]
      idx -= 1
    else
      # Hᶜʰv = Hᶜʰv * ψ.AR[n + k] * δˢ(n + k) * ψ′.AR[n + k]
      Hᶜʰv = replaceinds(Hᶜʰv * ψ.AR[n + k], s[n + k] => s[n + k]') * ψ′.AR[n + k]
    end
  end
  # Hᶜʰv = v * ψ.AL[n] * δˡ(n - 1) * ∑h[n][1] * ψ′.AL[n] * Hᶜʰv #left extremity
  Hᶜʰv = replaceinds(v * ψ.AL[n], l[n - 1] => l[n - 1]') * ∑h[n][1] * ψ′.AL[n] * Hᶜʰv #left extremity
  #Now we are building contributions of the form AL[n - j] ... AL[n] - v - AR[n + 1] ... AR[n + range_∑h - 1 - j]
  for j in 1:(range_∑h - 2)
    # temp_Hᶜʰv = ψ.AL[n - j] * δˡ(n - 1 - j) * ∑h[n - j][1] * ψ′.AL[n - j]
    temp_Hᶜʰv =
      replaceinds(ψ.AL[n - j], l[n - 1 - j] => l[n - 1 - j]') * ∑h[n - j][1] * ψ′.AL[n - j]
    common_sites = findsites(ψ, ∑h[n - j])
    idx = 2
    for k in 1:j
      if n - j + k == common_sites[idx]
        temp_Hᶜʰv = temp_Hᶜʰv * ψ.AL[n - j + k] * ∑h[n - j][idx] * ψ′.AL[n - j + k]
        idx += 1
      else
        # temp_Hᶜʰv = temp_Hᶜʰv * ψ.AL[n - j + k] * δˢ(n - j + k) * ψ′.AL[n - j + k]
        temp_Hᶜʰv =
          replaceinds(temp_Hᶜʰv * ψ.AL[n - j + k], s[n - j + k] => s[n - j + k]') *
          ψ′.AL[n - j + k]
      end
    end
    # Finished the AL part
    # temp_Hᶜʰv_r =
    #   ψ.AR[n - j + range_∑h - 1] *
    #   δʳ(n - j + range_∑h - 1) *
    #   ∑h[n - j][end] *
    #   ψ′.AR[n - j + range_∑h - 1]
    temp_Hᶜʰv_r =
      replaceinds(
        ψ.AR[n - j + range_∑h - 1], r[n - j + range_∑h - 1] => r[n - j + range_∑h - 1]'
      ) *
      ∑h[n - j][end] *
      ψ′.AR[n - j + range_∑h - 1]
    idx = length(∑h[n]) - 1
    for k in reverse((j + 1):(range_∑h - 2))
      if n - j + k == common_sites[idx]
        temp_Hᶜʰv_r = temp_Hᶜʰv_r * ψ.AR[n - j + k] * ∑h[n - j][idx] * ψ′.AR[n - j + k]
        idx -= 1
      else
        # temp_Hᶜʰv_r = temp_Hᶜʰv_r * ψ.AR[n - j + k] * δˢ(n - j + k) * ψ′.AR[n - j + k]
        temp_Hᶜʰv_r =
          replaceinds(temp_Hᶜʰv_r * ψ.AR[n - j + k], s[n - j + k] => s[n - j + k]') *
          ψ′.AR[n - j + k]
      end
    end
    temp_Hᶜʰv = temp_Hᶜʰv * v * temp_Hᶜʰv_r
    Hᶜʰv = Hᶜʰv + temp_Hᶜʰv
  end
  Hᶜv = Hᶜᴸv + Hᶜʰv + Hᶜᴿv
  # return Hᶜv * dag(δˡ(n)) * dag(δʳ(n))
  return replaceinds(Hᶜv, (l[n], r[n]) => (l[n]', r[n]'))
end

function Base.:*(H::Hᴬᶜ{MPO}, v::ITensor)
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

  # Remove temporarily in favor of `replaceinds`,
  # which is more efficient.
  # δˢ(n) = δ(dag(s[n]), prime(s[n]))
  # δʳ(n) = δ(dag(r[n]), prime(r[n]))
  # δˡ(n) = δ(l[n], l′[n])

  #Contribution of the left environment
  # Hᴬᶜᴸv = v * Hᴸ[n - 1] * δˢ(n) * δʳ(n)
  Hᴬᶜᴸv = replaceinds(v * Hᴸ[n - 1], (s[n], r[n]) => (s[n]', r[n]'))
  #Contribution of the right environment
  # Hᴬᶜᴿv = v * δˡ(n - 1) * δˢ(n) * Hᴿ[n]
  Hᴬᶜᴿv = replaceinds(v, (l[n - 1], s[n]) => (l[n - 1]', s[n]')) * Hᴿ[n]
  #We now start building terms where AC overlap with the local Hamiltonian
  # We start with the tensor v - AR[n+1] ... AR[n + range_∑h - 1]
  # Hᴬᶜʰv =
  #   δʳ(n + range_∑h - 1) * ψ.AR[n + range_∑h - 1] * ∑h[n][end] * ψ′.AR[n + range_∑h - 1] #rightmost extremity
  Hᴬᶜʰv =
    replaceinds(ψ.AR[n + range_∑h - 1], r[n + range_∑h - 1] => r[n + range_∑h - 1]') *
    ∑h[n][end] *
    ψ′.AR[n + range_∑h - 1] #rightmost extremity
  common_sites = findsites(ψ, ∑h[n])
  idx = length(∑h[n]) - 1  #list the sites Σh, we start at 2 because n is already taken into account
  for k in reverse(1:(range_∑h - 2))
    if n + k == common_sites[idx]
      Hᴬᶜʰv = Hᴬᶜʰv * ψ.AR[n + k] * ∑h[n][idx] * ψ′.AR[n + k]
      idx -= 1
    else
      # Hᴬᶜʰv = Hᴬᶜʰv * ψ.AR[n + k] * δˢ(n + k) * ψ′.AR[n + k]
      Hᴬᶜʰv = replaceinds(Hᴬᶜʰv * ψ.AR[n + k], s[n + k] => s[n + k]') * ψ′.AR[n + k]
    end
  end
  # Hᴬᶜʰv = v * Hᴬᶜʰv * ∑h[n][1] * δˡ(n - 1)
  Hᴬᶜʰv = replaceinds(v * Hᴬᶜʰv * ∑h[n][1], l[n - 1] => l[n - 1]')
  #Now we are building contributions of the form AL[n - j] ... AL[n-1] - v - AR[n + 1] ... AR[n + range_∑h - 1 - j]
  for j in 1:(range_∑h - 1)
    # temp_Hᴬᶜʰv = ψ.AL[n - j] * δˡ(n - j - 1) * ∑h[n - j][1] * ψ′.AL[n - j]
    temp_Hᴬᶜʰv =
      replaceinds(ψ.AL[n - j], l[n - j - 1] => l[n - j - 1]') * ∑h[n - j][1] * ψ′.AL[n - j]
    common_sites = findsites(ψ, ∑h[n - j])
    idx = 2
    for k in 1:(j - 1)
      if n - j + k == common_sites[idx]
        temp_Hᴬᶜʰv = temp_Hᴬᶜʰv * ψ.AL[n - j + k] * ∑h[n - j][idx] * ψ′.AL[n - j + k]
        idx += 1
      else
        # temp_Hᴬᶜʰv = temp_Hᴬᶜʰv * ψ.AL[n - j + k] * δˢ(n - j + k) * ψ′.AL[n - j + k]
        temp_Hᴬᶜʰv =
          replaceinds(temp_Hᴬᶜʰv * ψ.AL[n - j + k], s[n - j + k] => s[n - j + k]') *
          ψ′.AL[n - j + k]
      end
    end
    #Finished with AL, treating the center AC = v
    if j == range_∑h - 1
      # temp_Hᴬᶜʰv = temp_Hᴬᶜʰv * v * ∑h[n - j][end] * δʳ(n - j + range_∑h - 1)
      temp_Hᴬᶜʰv = replaceinds(
        temp_Hᴬᶜʰv * v * ∑h[n - j][end], r[n - j + range_∑h - 1] => r[n - j + range_∑h - 1]'
      )
    else
      # temp_Hᴬᶜʰv_r =
      #   ψ.AR[n + range_∑h - 1 - j] *
      #   δʳ(n - j + range_∑h - 1) *
      #   ∑h[n - j][end] *
      #   ψ′.AR[n + range_∑h - 1 - j]
      temp_Hᴬᶜʰv_r =
        replaceinds(
          ψ.AR[n + range_∑h - 1 - j], r[n - j + range_∑h - 1] => r[n - j + range_∑h - 1]'
        ) *
        ∑h[n - j][end] *
        ψ′.AR[n + range_∑h - 1 - j]

      idx = length(∑h[n]) - 1
      for k in reverse((j + 1):(range_∑h - 2))
        if n + k - j == common_sites[idx]
          temp_Hᴬᶜʰv_r = temp_Hᴬᶜʰv_r * ψ.AR[n + k - j] * ∑h[n - j][idx] * ψ′.AR[n + k - j]
          idx -= 1
        else
          # temp_Hᴬᶜʰv_r = temp_Hᴬᶜʰv_r * ψ.AR[n + k - j] * δˢ(n + k - j) * ψ′.AR[n + k - j]
          temp_Hᴬᶜʰv_r =
            replaceinds(temp_Hᴬᶜʰv_r * ψ.AR[n + k - j], s[n + k - j] => s[n + k - j]') *
            ψ′.AR[n + k - j]
        end
      end

      if n == common_sites[idx] #need to check whether we need to branch v
        temp_Hᴬᶜʰv = temp_Hᴬᶜʰv * v * ∑h[n - j][idx] * temp_Hᴬᶜʰv_r
      else
        # temp_Hᴬᶜʰv = temp_Hᴬᶜʰv * v * δˢ(n) * temp_Hᴬᶜʰv_r
        temp_Hᴬᶜʰv = replaceinds(temp_Hᴬᶜʰv * v * temp_Hᴬᶜʰv_r, s[n] => s[n]')
      end
    end
    Hᴬᶜʰv = Hᴬᶜʰv + temp_Hᴬᶜʰv
  end
  Hᴬᶜv = Hᴬᶜᴸv + Hᴬᶜʰv + Hᴬᶜᴿv
  # return Hᴬᶜv * dag(δˡ(n - 1)) * dag(δˢ(n)) * dag(δʳ(n))
  return replaceinds(Hᴬᶜv, (l[n - 1], s[n], r[n]) => (l[n - 1]', s[n]', r[n]'))
end

function left_environment(∑h::InfiniteSum{MPO}, ψ::InfiniteCanonicalMPS; tol=1e-15)
  Nsites = nsites(ψ)
  range_∑h = nrange(∑h, 1)
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  ψ̃ = prime(linkinds, ψᴴ)

  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)
  s = siteinds(only, ψ)
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˡ(n) = δ(l[n], l′[n])
  δˢ(n) = δ(dag(s[n]), prime(s[n]))
  hᴸ = Vector{ITensor}(undef, Nsites)
  for k in 1:Nsites
    hᴸ[k] =
      δˡ(k - range_∑h) *
      ψ.AL[k - range_∑h + 1] *
      ∑h[(k - range_∑h + 1, k - range_∑h + 2)][1] *
      ψ′.AL[k - range_∑h + 1]
    common_sites = findsites(ψ, ∑h[(k - range_∑h + 1, k - range_∑h + 2)])
    idx = 2
    for j in 2:range_∑h
      if k - range_∑h + j == common_sites[idx]
        hᴸ[k] =
          hᴸ[k] *
          ψ.AL[k - range_∑h + j] *
          ∑h[(k - range_∑h + 1, k - range_∑h + 2)][idx] *
          ψ′.AL[k - range_∑h + j]
        idx += 1
      else
        hᴸ[k] =
          hᴸ[k] * ψ.AL[k - range_∑h + j] * δˢ(k - range_∑h + j) * ψ′.AL[k - range_∑h + j]
      end
    end
  end
  hᴸ = InfiniteMPS(hᴸ)
  eᴸ = [(hᴸ[k] * ψ.C[k] * δʳ(k) * ψ′.C[k])[] for k in 1:Nsites]
  for k in 1:Nsites
    # TODO: remove `denseblocks` once BlockSparse + DiagBlockSparse is supported
    hᴸ[k] -= eᴸ[k] * denseblocks(δ(inds(hᴸ[k])))
  end

  𝕙ᴸ = copy(hᴸ)
  # TODO restrict to the useful ones only?
  for n in 1:Nsites
    for k in 1:(Nsites - 1)
      temp = copy(hᴸ[n - k])
      for kp in reverse(0:(k - 1))
        temp = temp * ψ.AL[n - kp] * ψ̃.AL[n - kp]
      end
      𝕙ᴸ[n] = temp + 𝕙ᴸ[n]
    end
  end
  Hᴸ = left_environment(hᴸ, 𝕙ᴸ, ψ; tol=tol)

  return Hᴸ, eᴸ
end

function right_environment(∑h::InfiniteSum{MPO}, ψ::InfiniteCanonicalMPS; tol=1e-15)
  Nsites = nsites(ψ)
  range_∑h = nrange(∑h, 1)
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  ψ̃ = prime(linkinds, ψᴴ)

  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)
  s = siteinds(only, ψ)
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˡ(n) = δ(l[n], l′[n])
  δˢ(n) = δ(dag(s[n]), prime(s[n]))

  hᴿ = Vector{ITensor}(undef, Nsites)
  for k in 1:Nsites
    hᴿ[k] = ψ.AR[k + range_∑h] * δʳ(k + range_∑h) * ∑h[k + 1][end] * ψ′.AR[k + range_∑h]
    common_sites = findsites(ψ, ∑h[k + 1])
    idx = length(common_sites) - 1
    for j in (range_∑h - 1):-1:1
      if k + j == common_sites[idx]
        hᴿ[k] = hᴿ[k] * ψ.AR[k + j] * ∑h[k + 1][idx] * ψ′.AR[k + j]
        idx -= 1
      else
        hᴿ[k] = hᴿ[k] * ψ.AR[k + j] * δˢ(k + j) * ψ′.AR[k + j]
      end
    end
  end
  hᴿ = InfiniteMPS(hᴿ)
  eᴿ = [(hᴿ[k] * ψ.C[k] * δˡ(k) * ψ′.C[k])[] for k in 1:Nsites]
  for k in 1:Nsites
    hᴿ[k] -= eᴿ[k] * denseblocks(δ(inds(hᴿ[k])))
  end

  𝕙ᴿ = copy(hᴿ)
  #TODO restrict to the useful ones only
  for n in 1:Nsites
    for k in 1:(Nsites - 1)
      temp = copy(hᴿ[n + k])
      for kp in reverse(1:k)
        temp = temp * ψ.AR[n + kp] * ψ̃.AR[n + kp]
      end
      𝕙ᴿ[n] = temp + 𝕙ᴿ[n]
    end
  end
  Hᴿ = right_environment(hᴿ, 𝕙ᴿ, ψ; tol=tol)
  return Hᴿ, eᴿ
end

#In principle, could share even more code with vumps_mpo or with parallel
function tdvp_iteration_sequential(
  solver::Function,
  ∑h::InfiniteSum{MPO},
  ψ::InfiniteCanonicalMPS;
  (ϵᴸ!)=fill(1e-15, nsites(ψ)),
  (ϵᴿ!)=fill(1e-15, nsites(ψ)),
  time_step,
  solver_tol=(x -> x / 100),
  eager=true,
)
  Nsites = nsites(ψ)
  range_∑h = nrange(∑h, 1)
  ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
  _solver_tol = solver_tol(ϵᵖʳᵉˢ)

  ψ = copy(ψ)
  C̃ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  Ãᶜ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  Ãᴸ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  Ãᴿ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  eᴸ = Vector{Float64}(undef, Nsites)
  eᴿ = Vector{Float64}(undef, Nsites)

  for n in 1:Nsites
    ψᴴ = dag(ψ)
    ψ′ = ψᴴ'
    # XXX: make this prime the center sites
    ψ̃ = prime(linkinds, ψᴴ)

    l = linkinds(only, ψ.AL)
    l′ = linkinds(only, ψ′.AL)
    r = linkinds(only, ψ.AR)
    r′ = linkinds(only, ψ′.AR)
    s = siteinds(only, ψ)
    δʳ(n) = δ(dag(r[n]), prime(r[n]))
    δˡ(n) = δ(l[n], l′[n])
    δˢ(n) = δ(dag(s[n]), prime(s[n]))

    Hᴸ, eᴸ = left_environment(∑h, ψ; tol=_solver_tol)
    Hᴿ, eᴿ = right_environment(∑h, ψ; tol=_solver_tol)

    Cvalsₙ₋₁, Cvecsₙ₋₁, Cinfoₙ₋₁ = solver(
      Hᶜ(∑h, Hᴸ, Hᴿ, ψ, n - 1), time_step, ψ.C[n - 1], _solver_tol, eager
    )
    Cvalsₙ, Cvecsₙ, Cinfoₙ = solver(
      Hᶜ(∑h, Hᴸ, Hᴿ, ψ, n), time_step, ψ.C[n], _solver_tol, eager
    )
    Avalsₙ, Avecsₙ, Ainfoₙ = solver(
      Hᴬᶜ(∑h, Hᴸ, Hᴿ, ψ, n), time_step, ψ.AL[n] * ψ.C[n], _solver_tol, eager
    )

    C̃[n - 1] = Cvecsₙ₋₁
    C̃[n] = Cvecsₙ
    Ãᶜ[n] = Avecsₙ

    Ãᴸ[n] = ortho_polar(Ãᶜ[n], C̃[n])
    Ãᴿ[n] = ortho_polar(Ãᶜ[n], C̃[n - 1])

    # Update state for next iteration
    #ψ = InfiniteCanonicalMPS(Ãᴸ, C̃, Ãᴿ)
    ψ.AL[n] = Ãᴸ[n]
    ψ.AR[n] = Ãᴿ[n]
    ψ.C[n - 1] = C̃[n - 1]
    ψ.C[n] = C̃[n]
  end

  for n in 1:Nsites
    # Fix the sign
    C̃[n] /= sign((dag(Ãᶜ[n]) * (Ãᴸ[n] * C̃[n]))[])
    C̃[n - 1] /= sign((dag(Ãᶜ[n]) * (C̃[n - 1] * Ãᴿ[n]))[])

    ϵᴸ![n] = norm(Ãᶜ[n] - Ãᴸ[n] * C̃[n])
    ϵᴿ![n] = norm(Ãᶜ[n] - C̃[n - 1] * Ãᴿ[n])

    if ϵᴸ![n] > 1
      @warn "Left precision error $(ϵᴸ![n]) is too high!"
    elseif ϵᴿ![n] > 1
      @warn "Right precision error $(ϵᴿ![n]) is too high!"
    end
  end
  return ψ, (eᴸ, eᴿ)
end

function tdvp_iteration_parallel(
  solver::Function,
  ∑h::InfiniteSum{MPO},
  ψ::InfiniteCanonicalMPS;
  (ϵᴸ!)=fill(1e-15, nsites(ψ)),
  (ϵᴿ!)=fill(1e-15, nsites(ψ)),
  time_step,
  solver_tol=(x -> x / 100),
  eager=true,
)
  Nsites = nsites(ψ)
  range_∑h = nrange(∑h, 1)
  ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
  _solver_tol = solver_tol(ϵᵖʳᵉˢ)
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  # XXX: make this prime the center sites
  ψ̃ = prime(linkinds, ψᴴ)

  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)
  s = siteinds(only, ψ)
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˡ(n) = δ(l[n], l′[n])
  δˢ(n) = δ(dag(s[n]), prime(s[n]))
  Hᴸ, eᴸ = left_environment(∑h, ψ; tol=_solver_tol)
  Hᴿ, eᴿ = right_environment(∑h, ψ; tol=_solver_tol)

  C̃ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  Ãᶜ = InfiniteMPS(Vector{ITensor}(undef, Nsites))
  for n in 1:Nsites
    Cvalsₙ, Cvecsₙ, Cinfoₙ = solver(
      Hᶜ(∑h, Hᴸ, Hᴿ, ψ, n), time_step, ψ.C[n], _solver_tol, eager
    )
    Avalsₙ, Avecsₙ, Ainfoₙ = solver(
      Hᴬᶜ(∑h, Hᴸ, Hᴿ, ψ, n), time_step, ψ.AL[n] * ψ.C[n], _solver_tol, eager
    )

    C̃[n] = Cvecsₙ
    Ãᶜ[n] = Avecsₙ
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
