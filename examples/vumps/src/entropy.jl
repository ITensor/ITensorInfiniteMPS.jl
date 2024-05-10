using ITensors, ITensorMPS
using ITensorInfiniteMPS

# calculate -\sum_i rho_i log(rho_i)
function entropy(S::ITensor)
  @assert order(S) == 2
  SvN, norm = 0.0, 0.0
  for n in 1:dim(S, 1)
    p = S[n, n]^2
    SvN -= p * log(p)
    norm += p
  end
  return SvN, norm
end

# calculate von Neumann entropy of a MPS cut between sites b-1 and b
function entropy(ψ::MPS, b)
  ψ = orthogonalize(ψ, b)
  U, S, V = svd(ψ[b], (linkind(ψ, b - 1), siteind(ψ, b)))
  SvN, norm = entropy(S)
  @assert norm ≈ 1.0
  return SvN
end

# calculate von Neumann entropy of an infinite MPS at cut b of the unit cell
function entropy(ψ::InfiniteCanonicalMPS, b)
  #calculate entropy
  C = ψ.C[b]
  Ũ, S, Ṽ = svd(C, inds(C)[1])
  SvN, norm = entropy(S)
  @assert norm ≈ 1.0
  return SvN
end
