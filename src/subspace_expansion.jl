function replaceind_indval(IV::Tuple, iĩ::Pair)
  i, ĩ = iĩ
  return ntuple(n -> first(IV[n]) == i ? ĩ => last(IV[n]) : IV[n], length(IV))
end

function subspace_expansion(
  ψ::InfiniteCanonicalMPS, H, b::Tuple{Int,Int}; maxdim, cutoff, kwargs...
)
  n1, n2 = b
  lⁿ¹ = commoninds(ψ.AL[n1], ψ.C[n1])
  rⁿ¹ = commoninds(ψ.AR[n2], ψ.C[n1])

  dˡ = dim(lⁿ¹)
  dʳ = dim(rⁿ¹)
  @assert dˡ == dʳ
  if dˡ ≥ maxdim
    println(
      "Current bond dimension at bond $b is $dˡ while desired maximum dimension is $maxdim, skipping bond dimension increase",
    )
    return (ψ.AL[n1], ψ.AL[n2]), ψ.C[n1], (ψ.AR[n1], ψ.AR[n2])
  end
  maxdim -= dˡ

  # Returns `NL` such that `norm(ψ.AL[n1] * NL) ≈ 0`
  NL = nullspace(ψ.AL[n1], lⁿ¹; atol=1e-15)
  NR = nullspace(ψ.AR[n2], rⁿ¹; atol=1e-15)

  nL = uniqueinds(NL, ψ.AL[n1])
  nR = uniqueinds(NR, ψ.AR[n2])

  ψH2 = noprime(ψ.AL[n1] * H[(n1, n2)] * ψ.C[n1] * ψ.AR[n2])
  ψHN2 = ψH2 * NL * NR

  U, S, V = svd(ψHN2, nL; maxdim=maxdim, cutoff=cutoff, kwargs...)
  @show S[end, end]
  NL *= dag(U)
  NR *= dag(V)

  ALⁿ¹, l = ITensors.directsum(
    ψ.AL[n1], dag(NL), uniqueinds(ψ.AL[n1], NL), uniqueinds(NL, ψ.AL[n1]); tags=("Left",)
  )
  ARⁿ², r = ITensors.directsum(
    ψ.AR[n2], dag(NR), uniqueinds(ψ.AR[n2], NR), uniqueinds(NR, ψ.AR[n2]); tags=("Right",)
  )

  C = ITensor(dag(l)..., dag(r)...)
  ψCⁿ¹ = permute(ψ.C[n1], lⁿ¹..., rⁿ¹...)
  for I in eachindex(ψ.C[n1])
    v = ψCⁿ¹[I]
    if !iszero(v)
      C[I] = ψCⁿ¹[I]
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

  # TODO: delete or only print when verbose
  ## ψ₂ = ψ.AL[n1] * ψ.C[n1] * ψ.AR[n2]
  ## ψ̃₂ = ALⁿ¹ * C * ARⁿ²
  ## local_energy(ψ, H) = (noprime(ψ * H) * dag(ψ))[]

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
