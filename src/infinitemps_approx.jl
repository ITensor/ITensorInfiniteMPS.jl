function siteind(ψ::InfiniteMPS, n::Integer)
  return uniqueind(ψ[n], ψ[n - 1], ψ[n + 1])
end

siteind(ψ::InfiniteCanonicalMPS, n::Integer) = siteind(ψ.AL, n)

function default_middle_range(N::Integer, ncenter::Integer)
  return round(Int, N / 2 - ncenter / 2 + 1, RoundDown):round(
    Int, N / 2 + ncenter / 2, RoundDown
  )
end

# Get the closest infinite MPS approximation to the finite MPS
function infinitemps_approx(
  ψ::MPS;
  nsites=1,
  nrange=default_middle_range(length(ψ), nsites),
  nsweeps=10,
  outputlevel=0,
)
  N = length(ψ)

  # nsites are the number of site in the unit cell
  # of the infinite MPS
  @assert length(nrange) == nsites

  s = siteinds(ψ)

  # The site we are interested in now
  ψ = orthogonalize(ψ, first(nrange))
  χ = maximum(n -> linkdim(ψ, n), nrange)

  site_tags = ITensors.commontags(siteinds(ψ))

  # The sites of the infinite MPS
  A = InfiniteMPS(
    [settags(s[nrange[n]], addtags(site_tags, "n=$n")) for n in 1:length(nrange)]; space=χ
  )
  # XXX: Why doesn't this work?
  # randn!.(A)
  for n in 1:N
    A[n] = random_itensor(inds(A[n]))
  end
  ψ∞ = orthogonalize(A, :)

  # Site indices of the infinite MPS
  # corresponding to those of the finite MPS
  inf_range = (-first(nrange) + 2):(-first(nrange) + N + 1)

  s∞ = [siteind(ψ∞, n) for n in inf_range]

  ψ = replace_siteinds(ψ, s∞)

  for sweep in 1:nsweeps
    # XXX: these are just used for testing
    TL = ITensor(1)
    for n in 1:N
      TL = TL * ψ[n] * ψ∞.AL[inf_range[n]]
    end
    TR = ITensor(1)
    for n in reverse(1:N)
      TR = TR * ψ[n] * ψ∞.AR[inf_range[n]]
    end

    L0 = random_itensor(linkind(ψ∞.AL, first(inf_range) - 1 => first(inf_range)))
    vals, vecs, _ = eigsolve(transpose(ITensorMap(ψ∞.AL[inf_range], ψ)), L0)
    L0 = vecs[1]
    λL0 = vals[1]
    LTL = L0 * TL
    LTL = replaceinds(LTL, inds(LTL) => inds(L0))

    RN = random_itensor(linkind(ψ∞.AR, last(inf_range) => last(inf_range) + 1))
    vals, vecs, _ = eigsolve(ITensorMap(ψ∞.AR[inf_range], ψ), RN)
    RN = vecs[1]
    λRN = vals[1]

    TRR = TR * RN
    TRR = replaceinds(TRR, inds(TRR) => inds(RN))

    L = OffsetVector(Vector{ITensor}(undef, N + 1), -1)
    R = OffsetVector(Vector{ITensor}(undef, N + 1), -1)
    L[0] = L0
    R[N] = RN

    for n in 1:N
      L[n] = L[n - 1] * ψ∞.AL[inf_range[n]] * dag(ψ[n])
    end
    for n in reverse(0:(N - 1))
      R[n] = R[n + 1] * ψ∞.AR[inf_range[n + 1]] * dag(ψ[n + 1])
    end

    AL = Vector{ITensor}(undef, nsites)
    AR = Vector{ITensor}(undef, nsites)
    C = Vector{ITensor}(undef, nsites)
    for n′ in eachindex(nrange)
      n = nrange[n′]
      C1 = L[n - 1] * R[n - 1]
      C2 = L[n] * R[n]
      C12 = L[n - 1] * dag(ψ[n]) * R[n]
      U, P = polar(C12 * dag(C2), uniqueinds(C12, C2))
      u = commoninds(U, P)
      U = noprime(U, u)
      AL[n′] = U
      # XXX: compute AR from a right polar decomposition of C2
      AR[n′] = replacetags(U, "Left" => "Right")
      C[n′] = C2 / norm(C2)
    end
    ψL = InfiniteMPS(AL)
    ψR = InfiniteMPS(AR)
    ψC = InfiniteMPS(C)
    ψ∞ = InfiniteCanonicalMPS(ψL, ψC, ψR)
    # This is a measure of the overlap
    if outputlevel > 0
      @show abs(λL0)^(1 / N), abs(λRN)^(1 / N)
    end
  end
  return ψ∞
end
