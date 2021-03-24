using ITensors
using ITensorInfiniteMPS

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

function ising_local_ham(s1::Index, s2::Index; J, h)
  return -J * op("X", s1) * op("X", s2) - h * op("Z", s1) * op("Id", s2)
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
  ham12 = ising_local_ham(s[n1], s[n2]; J = J, h = h)
  @show (ϕ12 * ham12 * prime(dag(ϕ12), commoninds(ϕ12, ham12)))[]

  # Get an infinite MPS that approximates
  # the finite MPS in the bulk.
  # `nsites` sets the number of sites in the unit
  # cell of the infinite MPS.
  nsites = 2
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
    ham12 = ising_local_ham(s1, s2; J = J, h = h)
    ϕ12ᴴ = prime(dag(ϕ12), commoninds(ϕ12, ham12))
    @show (ϕ12 * ham12 * ϕ12ᴴ)[] / norm(ϕ12)
  end
end

