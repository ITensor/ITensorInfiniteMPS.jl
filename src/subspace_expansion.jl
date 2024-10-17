function replaceind_indval(IV::Tuple, iĩ::Pair)
  i, ĩ = iĩ
  return ntuple(n -> first(IV[n]) == i ? ĩ => last(IV[n]) : IV[n], length(IV))
end

function generate_twobody_nullspace(
  ψ::InfiniteCanonicalMPS, H::InfiniteSum{MPO}, b::Tuple{Int,Int}; atol=1e-2
)
  n1, n2 = b
  lⁿ¹ = commoninds(ψ.AL[n1], ψ.C[n1])
  rⁿ¹ = commoninds(ψ.AR[n2], ψ.C[n1])
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  s = siteinds(only, ψ)
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˢ(n) = δ(dag(s[n]), prime(s[n]))
  δˡ(n) = δ(l[n], dag(prime(l[n])))

  range_H = nrange(ψ, H[1])

  @assert range_H > 1 "Subspace expansion for InfiniteSum{MPO} is not defined for purely local Hamiltonians"

  if range_H > 2
    ψᴴ = dag(ψ)
    ψ′ = prime(ψᴴ)
  end

  if range_H == 2
    ψH2 = (ψ.AL[n1] * ψ.C[n1] * H[n1][1]) * (ψ.AR[n2] * H[n1][2])
    ψH2 = noprime(ψH2)
  else   # Should be a better version now
    ψH2 =
      ψ.AR[n2 + range_H - 2] * H[n1][end] * (ψ′.AR[n2 + range_H - 2] * δʳ(n2 + range_H - 2))
    common_sites = findsites(ψ, H[(n1, n2)])
    idx = length(common_sites) - 1
    for j in reverse(1:(range_H-3))
      if n2 + j == common_sites[idx]
        ψH2 = ψH2 * ψ.AR[n2 + j] * H[n1][idx] * ψ′.AR[n2 + j]
        idx -= 1
      else
        ψH2 = ψH2 * ψ.AR[n2 + j] * (δˢ(n2 + j) * ψ′.AR[n2 + j])
      end
    end
    if common_sites[idx] == n2
      ψH2 = ψH2 * ψ.AR[n2] * H[n1][idx]
      idx -= 1
    else
      ψH2 = ψH2 * (ψ.AR[n2] * δˢ(n2))
    end
    if common_sites[idx] == n1
      ψH2 = ψH2 * (ψ.AL[n1] * ψ.C[n1]) * H[n1][idx]
      idx -= 1
    else
      ψH2 = ψH2 * ((ψ.AL[n1] * δˢ(n1)) * ψ.C[n1])
    end

    ψH2 = noprime(ψH2)
    for n in 1:(range_H-2)
      temp_H2_right = δʳ(n2 + range_H - 2 - n)
      common_sites = findsites(ψ, H[n1 - n])
      idx = length(common_sites)
      for j in (n2+range_H-2-n):-1:(n2+1)
        if j == common_sites[idx]
          temp_H2_right = temp_H2_right * ψ.AR[j] * H[n1 - n][idx] * ψ′.AR[j]
          idx -= 1
        else
          temp_H2_right = temp_H2_right * ψ.AR[j] * (δˢ(j) * ψ′.AR[j])
        end
      end
      if common_sites[idx] == n2
        temp_H2_right = temp_H2_right * ψ.AR[n2] * H[n1 - n][idx]
        idx -= 1
      else
        temp_H2_right = temp_H2_right * ψ.AR[n2] * δˢ(n2)
      end
      if common_sites[idx] == n1
        temp_H2_right = temp_H2_right * (ψ.AL[n1] * ψ.C[n1]) * H[n1 - n][idx]
        idx -= 1
      else
        temp_H2_right = temp_H2_right * ((ψ.AL[n1] * δˢ(n1)) * ψ.C[n1])
      end
      idx = n - idx + 1
      temp_H2_left = δˡ(n1 - n - 1)
      for j in reverse(1:n)
        if n1 - j == common_sites[idx]
          temp_H2_left = temp_H2_left * ψ.AL[n1 - j] * H[n1 - n][idx] * ψ′.AL[n1 - j]
          idx += 1
        else
          temp_H2_left = temp_H2_left * (ψ.AL[n1 - j] * δˢ(n1 - j)) * ψ′.AL[n1 - j]
        end
      end
      ψH2 = ψH2 + noprime(temp_H2_left * temp_H2_right)
    end
  end
  return ψH2
end

function generate_twobody_nullspace(
  ψ::InfiniteCanonicalMPS, H::InfiniteSum{ITensor}, b::Tuple{Int,Int}; atol=1e-2
)
  n1, n2 = b
  lⁿ¹ = commoninds(ψ.AL[n1], ψ.C[n1])
  rⁿ¹ = commoninds(ψ.AR[n2], ψ.C[n1])
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  s = siteinds(only, ψ)
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˢ(n) = δ(dag(s[n]), prime(s[n]))
  δˡ(n) = δ(l[n], dag(prime(l[n])))

  range_H = nrange(H, 1)
  @assert range_H == 2 "Subspace expansion for InfiniteSum{ITensor} currently only works for 2-local Hamiltonians"

  if range_H == 2
    ψH2 = noprime(ψ.AL[n1] * H[n1] * ψ.C[n1] * ψ.AR[n2])
  end
  return ψH2
end

function generate_twobody_nullspace(
  ψ::InfiniteCanonicalMPS, H::InfiniteBlockMPO, b::Tuple{Int,Int}; atol=1e-2
)
  n_1, n_2 = b
  L, _ = left_environment(H, ψ)
  R, _ = right_environment(H, ψ)

  dₕ = length(L[n_1 - 1])
  temp_L = similar(L[n_1 - 1])
  for i in 1:dₕ
    non_empty_idx = dₕ
    while isempty(H[n_1][non_empty_idx, i]) && non_empty_idx >= i
      non_empty_idx -= 1
    end
    @assert non_empty_idx != i - 1 "Empty MPO"
    temp_L[i] =
      L[n_1 - 1][non_empty_idx] * (ψ.AL[n_1] * ψ.C[n_1]) * H[n_1][non_empty_idx, i]
    for j in reverse(i:(non_empty_idx-1))
      if !isempty(H[n_1][j, i])
        temp_L[i] += L[n_1 - 1][j] * (ψ.AL[n_1] * ψ.C[n_1]) * H[n_1][j, i]
      end
    end
  end
  temp_R = similar(R[n_1 + 2])
  for i in 1:dₕ
    non_empty_idx = 1
    while isempty(H[n_1 + 1][i, non_empty_idx]) && non_empty_idx <= i
      non_empty_idx += 1
    end
    @assert non_empty_idx != i + 1 "Empty MPO"
    temp_R[i] = H[n_1 + 1][i, non_empty_idx] * (ψ.AR[n_1 + 1] * R[n_1 + 2][non_empty_idx])
    for j in (non_empty_idx+1):i
      if !isempty(H[n_1 + 1][i, j])
        temp_R[i] += H[n_1 + 1][i, j] * (ψ.AR[n_1 + 1] * R[n_1 + 2][j])
      end
    end
  end
  ψH2 = temp_L[1] * temp_R[1]
  for j in 2:dₕ
    ψH2 += temp_L[j] * temp_R[j]
  end
  return noprime(ψH2)
end
# atol controls the tolerance cutoff for determining which eigenvectors are in the null
# space of the isometric MPS tensors. Setting to 1e-2 since we only want to keep
# the eigenvectors corresponding to eigenvalues of approximately 1.
function subspace_expansion(
  ψ::InfiniteCanonicalMPS, H, b::Tuple{Int,Int}; maxdim, cutoff, atol=1e-2, kwargs...
)
  n1, n2 = b
  lⁿ¹ = commoninds(ψ.AL[n1], ψ.C[n1])
  rⁿ¹ = commoninds(ψ.AR[n2], ψ.C[n1])
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  s = siteinds(only, ψ)
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˢ(n) = δ(dag(s[n]), prime(s[n]))
  δˡ(n) = δ(l[n], dag(prime(l[n])))

  dˡ = dim(lⁿ¹)
  dʳ = dim(rⁿ¹)
  @assert dˡ == dʳ
  if dˡ ≥ maxdim
    println(
      "Current bond dimension at bond $b is $dˡ while desired maximum dimension is $maxdim, skipping bond dimension increase at $b",
    )
    flush(stdout)
    flush(stderr)
    return (ψ.AL[n1], ψ.AL[n2]), ψ.C[n1], (ψ.AR[n1], ψ.AR[n2])
  end
  maxdim -= dˡ

  NL = nullspace(ψ.AL[n1], lⁿ¹; atol=atol)
  NR = nullspace(ψ.AR[n2], rⁿ¹; atol=atol)
  nL = uniqueinds(NL, ψ.AL[n1])
  nR = uniqueinds(NR, ψ.AR[n2])

  ψHN2 = generate_twobody_nullspace(ψ, H, b; atol=atol) * NL * NR

  #Added due to crash during testing
  if norm(ψHN2.tensor) < 1e-12
    println(
      "The two-site subspace expansion produced a zero-norm expansion at $b. This is likely due to the long-range nature of the QN conserving Hamiltonian.",
    )
    flush(stdout)
    flush(stderr)
    return (ψ.AL[n1], ψ.AL[n2]), ψ.C[n1], (ψ.AR[n1], ψ.AR[n2])
  end

  U, S, V = svd(ψHN2, nL; maxdim=maxdim, cutoff=cutoff, kwargs...)
  if dim(S) == 0 #Crash before reaching this point
    return (ψ.AL[n1], ψ.AL[n2]), ψ.C[n1], (ψ.AR[n1], ψ.AR[n2])
  end
  #@show S[end, end]
  NL *= dag(U)
  NR *= dag(V)

  ALⁿ¹, newl = ITensors.directsum(
    ψ.AL[n1] => uniqueinds(ψ.AL[n1], NL),
    dag(NL) => uniqueinds(NL, ψ.AL[n1]);
    tags=("Left",),
  )
  ARⁿ², newr = ITensors.directsum(
    ψ.AR[n2] => uniqueinds(ψ.AR[n2], NR),
    dag(NR) => uniqueinds(NR, ψ.AR[n2]);
    tags=("Right",),
  )

  C = ITensor(dag(newl)..., dag(newr)...)
  ψCⁿ¹ = permute(ψ.C[n1], lⁿ¹..., rⁿ¹...)
  for I in eachindex(ψ.C[n1])
    v = ψCⁿ¹[I]
    if !iszero(v)
      C[I] = ψCⁿ¹[I]
    end
  end
  if nsites(ψ) == 1
    ALⁿ² = ITensor(commoninds(C, ARⁿ²), uniqueinds(ALⁿ¹, ψ.AL[n1 - 1])...)
    il = only(uniqueinds(ALⁿ¹, ALⁿ²))
    ĩl = only(uniqueinds(ALⁿ², ALⁿ¹))
    for IV in eachindval(inds(ALⁿ¹))
      ĨV = replaceind_indval(IV, il => ĩl)
      v = ALⁿ¹[IV...]
      if !iszero(v)
        ALⁿ²[ĨV...] = v
      end
    end
    ARⁿ¹ = ITensor(commoninds(C, ALⁿ¹), uniqueinds(ARⁿ², ψ.AR[n2 + 1])...)
    ir = only(uniqueinds(ARⁿ², ARⁿ¹))
    ĩr = only(uniqueinds(ARⁿ¹, ARⁿ²))
    for IV in eachindval(inds(ARⁿ²))
      ĨV = replaceind_indval(IV, ir => ĩr)
      v = ARⁿ²[IV...]
      if !iszero(v)
        ARⁿ¹[ĨV...] = v
      end
    end

    CL = combiner(newl; tags=tags(only(lⁿ¹)))
    newind = uniqueinds(CL, newl)
    newind = replacetags(newind, tags(only(newind)), tags(r[n1 - 1]))
    temp = δ(dag(newind), newr)
    ALⁿ² *= CL
    ALⁿ² *= temp
    CR = combiner(newr; tags=tags(only(rⁿ¹)))
    newind = uniqueinds(CR, newr)
    newind = replacetags(newind, tags(only(newind)), tags(l[n2]))
    temp = δ(dag(newind), newl)
    ARⁿ¹ *= CR
    ARⁿ¹ *= temp
    C = (C * dag(CL)) * dag(CR)
    return (ALⁿ¹, ALⁿ²), C, (ARⁿ¹, ARⁿ²)
  else
    # Also expand the dimension of the neighboring MPS tensors
    ALⁿ² = ITensor(dag(newl)..., uniqueinds(ψ.AL[n2], ψ.AL[n1])...)

    il = only(uniqueinds(ψ.AL[n2], ALⁿ²))
    ĩl = only(uniqueinds(ALⁿ², ψ.AL[n2]))
    for IV in eachindval(inds(ψ.AL[n2]))
      ĨV = replaceind_indval(IV, il => ĩl)
      v = ψ.AL[n2][IV...]
      if !iszero(v)
        ALⁿ²[ĨV...] = v
      end
    end

    ARⁿ¹ = ITensor(dag(newr)..., uniqueinds(ψ.AR[n1], ψ.AR[n2])...)

    ir = only(uniqueinds(ψ.AR[n1], ARⁿ¹))
    ĩr = only(uniqueinds(ARⁿ¹, ψ.AR[n1]))
    for IV in eachindval(inds(ψ.AR[n1]))
      ĨV = replaceind_indval(IV, ir => ĩr)
      v = ψ.AR[n1][IV...]
      if !iszero(v)
        ARⁿ¹[ĨV...] = v
      end
    end

    CL = combiner(newl; tags=tags(only(lⁿ¹)))
    CR = combiner(newr; tags=tags(only(rⁿ¹)))
    ALⁿ¹ *= CL
    ALⁿ² *= dag(CL)
    ARⁿ² *= CR
    ARⁿ¹ *= dag(CR)
    C = (C * dag(CL)) * dag(CR)
    return (ALⁿ¹, ALⁿ²), C, (ARⁿ¹, ARⁿ²)
  end

  # TODO: delete or only print when verbose
  ## ψ₂ = ψ.AL[n1] * ψ.C[n1] * ψ.AR[n2]
  ## ψ̃₂ = ALⁿ¹ * C * ARⁿ²
  ## local_energy(ψ, H) = (noprime(ψ * H) * dag(ψ))[]

end

function subspace_expansion(ψ, H; kwargs...)
  ψ = copy(ψ)
  N = nsites(ψ)
  AL = ψ.AL
  C = ψ.C
  AR = ψ.AR
  for n in 1:N
    n1, n2 = n, n + 1
    ALⁿ¹², Cⁿ¹, ARⁿ¹² = subspace_expansion(ψ, H, (n1, n2); kwargs...)
    ALⁿ¹, ALⁿ² = ALⁿ¹²
    ARⁿ¹, ARⁿ² = ARⁿ¹²
    if N == 1
      AL[n1] = ALⁿ²
      C[n1] = Cⁿ¹
      AR[n2] = ARⁿ¹
    else
      AL[n1] = ALⁿ¹
      AL[n2] = ALⁿ²
      C[n1] = Cⁿ¹
      AR[n1] = ARⁿ¹
      AR[n2] = ARⁿ²
    end
    ψ = InfiniteCanonicalMPS(AL, C, AR)
  end
  return ψ
end
