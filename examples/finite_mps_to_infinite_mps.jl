using ITensors
using ITensorInfiniteMPS

# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ
function ising_mpo(s; J, h)
  N = length(s)
  a = AutoMPO()
  for n in 1:N-1
    a .+= -J, "X", n, "X", n+1
  end
  for n in 1:N
    a .+= -h, "Z", n
  end
  H = MPO(a, s)
end

function ising_infinitempo_tensor(s⃗, l, r; J, h)
  s = only(s⃗)
  dₗ = 3 # The link dimension of the TFI
  Hmat = fill(ITensor(s', dag(s)), dₗ, dₗ)
  Hmat[1, 1] = op("Id", s)
  Hmat[2, 2] = op("Id", s)
  Hmat[3, 3] = op("Id", s)
  Hmat[2, 1] = -J * op("X", s)
  Hmat[3, 2] = -J * op("X", s)
  Hmat[3, 1] = -h * op("Z", s)
  H = emptyITensor()
  for i in 1:dₗ, j in 1:dₗ
    H += Hmat[i,j] * setelt(l => i) * setelt(r => j)
  end
  return H
end

# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ
function ising_infinitempo(s; J, h)
  N = length(s)
  # Use a finite MPO to create the infinite MPO
  H = InfiniteMPO(N)
  # For the finite unit cell
  s¹ = addtags.(s, "c=1")
  l¹ = [Index(3, "Link,l=$n,c=1") for n in 1:N]
  for n in 1:N
    # TODO: use a CelledVector here to make the code logic simpler
    l¹ₙ = n == 1 ? replacetags(dag(l¹[N]), "c=1" => "c=0") : l¹[n-1]
    r¹ₙ = l¹[n]
    H[n] = ising_infinitempo_tensor(s¹[n], l¹ₙ, r¹ₙ; J = J, h = h)
  end
  return H
end

# H = -J X₁X₂ - h Z₁
function ising_localop(s1::Index, s2::Index; J, h)
  return -J * op("X", s1) * op("X", s2) - h * op("Z", s1) * op("Id", s2)
end
# Version accepting IndexSet
ising_localop(s1, s2; kwargs...) = ising_localop(Index(s1), Index(s2); kwargs...)

function ising_infinitesumlocalops(s; J, h)
  N = length(s)
  H = InfiniteSumLocalOps(N)
  s∞ = CelledVector(s)
  return InfiniteSumLocalOps([ising_localop(s∞[n], s∞[n+1]; J = J, h = h) for n in 1:N])
end

let
  N = 100
  s = siteinds("S=1/2", N)

  J = 1.0
  h = 1.5
  H = ising_mpo(s; J = J, h = h)
  ψ0 = randomMPS(s)

  sweeps = Sweeps(10)
  maxdim!(sweeps, 10)
  cutoff!(sweeps, 1E-10)
  energy, ψ = dmrg(H, ψ0, sweeps)
  @show energy/N

  n1 = N ÷ 2
  n2 = n1 + 1

  ψ = orthogonalize(ψ, n1)

  println("\nFinite MPS energy on bond (n1, n2) = $((n1, n2))")

  ϕ12 = ψ[n1] * ψ[n2]
  ham12 = ising_localop(s[n1], s[n2]; J = J, h = h)
  @show (ϕ12 * ham12 * prime(dag(ϕ12), commoninds(ϕ12, ham12)))[]

  # Get an infinite MPS that approximates
  # the finite MPS in the bulk.
  # `nsites` sets the number of sites in the unit
  # cell of the infinite MPS.
  nsites = 1
  ψ∞ = infinitemps_approx(ψ; nsites = nsites, nsweeps = 10)

  println("\nInfinite MPS energy on a few different bonds")
  println("Infinite MPS has unit cell of size $nsites")

  # Compute a few energies
  for n1 in 1:3
    n2 = n1 + 1
    @show n1, n2
    ϕ12 = ψ∞.AL[n1] * ψ∞.AL[n2] * ψ∞.C[n2]
    s1 = siteind(ψ∞, n1)
    s2 = siteind(ψ∞, n2)
    ham12 = ising_localop(s1, s2; J = J, h = h)
    ϕ12ᴴ = prime(dag(ϕ12), commoninds(ϕ12, ham12))
    @show (ϕ12 * ham12 * ϕ12ᴴ)[] / norm(ϕ12)
  end

end

