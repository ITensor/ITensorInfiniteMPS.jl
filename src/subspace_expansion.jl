function replaceind_indval(IV::Tuple, iĩ::Pair)
  i, ĩ = iĩ
  return ntuple(n -> first(IV[n]) == i ? ĩ => last(IV[n]) : IV[n], length(IV))
end

function subspace_expansion(ψ::InfiniteCanonicalMPS, H, b::Tuple{Int,Int}; kwargs...)
  n1, n2 = b
  lⁿ¹ = commoninds(ψ.AL[n1], ψ.C[n1])
  rⁿ¹ = commoninds(ψ.AR[n2], ψ.C[n1])
  NL = nullspace(ψ.AL[n1], lⁿ¹; atol=1e-15)
  NR = nullspace(ψ.AR[n2], rⁿ¹; atol=1e-15)
  nL = uniqueinds(NL, ψ.AL[n1])
  nR = uniqueinds(NR, ψ.AR[n2])

  @show prime(NL, uniqueinds(NL, ψ.AL[n1])) * dag(NL)
  @show norm(ψ.AL[n1] * dag(NL))

  ψH2 = noprime(ψ.AL[n1] * H[(n1, n2)] * ψ.C[n1] * ψ.AR[n2])
  ψHN2 = ψH2 * dag(NL) * dag(NR)

  @show inds(ψHN2)
  U, S, V = svd(ψHN2, nL; kwargs...)
  NL *= U
  NR *= V

  ALⁿ¹, l = ITensors.directsum(ψ.AL[n1], NL, uniqueinds(ψ.AL[n1], NL), uniqueinds(NL, ψ.AL[n1]); tags=("Left",))
  ARⁿ², r = ITensors.directsum(ψ.AR[n2], NR, uniqueinds(ψ.AR[n2], NR), uniqueinds(NR, ψ.AR[n2]); tags=("Right",))

  C = ITensor(dag(l)..., dag(r)...)
  for I in eachindex(ψ.C[n1])
    v = ψ.C[n1][I]
    if !iszero(v)
      C[I] = ψ.C[n1][I]
    end
  end

  # Also expand the dimension of the neighboring MPS tensors
  ALⁿ² = ITensor(dag(l)..., uniqueinds(ψ.AL[n2], ψ.AL[n1])...)

  il = only(uniqueinds(ψ.AL[n2], ALⁿ²))
  ĩl = only(uniqueinds(ALⁿ², ψ.AL[n2]))
  for IV in eachindval(inds(ψ.AL[n2]))
    ĨV = replaceind_indval(IV, il => ĩl)
    v = ψ.AL[n2][IV...]
    if !iszero(v)
      ALⁿ²[ĨV...] = v
    end
  end

  ARⁿ¹ = ITensor(dag(r)..., uniqueinds(ψ.AR[n1], ψ.AR[n2])...)

  ir = only(uniqueinds(ψ.AR[n1], ARⁿ¹))
  ĩr = only(uniqueinds(ARⁿ¹, ψ.AR[n1]))
  for IV in eachindval(inds(ψ.AR[n1]))
    ĨV = replaceind_indval(IV, ir => ĩr)
    v = ψ.AR[n1][IV...]
    if !iszero(v)
      ARⁿ¹[ĨV...] = v
    end
  end

  CL = combiner(l; tags=tags(only(lⁿ¹)))
  CR = combiner(r; tags=tags(only(rⁿ¹)))
  ALⁿ¹ *= CL
  ALⁿ² *= dag(CL)
  ARⁿ² *= CR
  ARⁿ¹ *= dag(CR)
  C = (C * dag(CL)) * dag(CR)

  ψ₂ = ψ.AL[n1] * ψ.C[n1] * ψ.AR[n2]
  ψ̃₂ = ALⁿ¹ * C * ARⁿ²
  local_energy(ψ, H) = (noprime(ψ * H) * dag(ψ))[]
  @show local_energy(ψ₂, H[(n1, n2)])
  @show local_energy(ψ̃₂, H[(n1, n2)])
  return (ALⁿ¹, ALⁿ²), C, (ARⁿ¹, ARⁿ²)
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
    AL[n1] = ALⁿ¹
    AL[n2] = ALⁿ²
    C[n1] = Cⁿ¹
    AR[n1] = ARⁿ¹
    AR[n2] = ARⁿ²
    ψ = InfiniteCanonicalMPS(AL, C, AR)
  end
  return ψ
end

