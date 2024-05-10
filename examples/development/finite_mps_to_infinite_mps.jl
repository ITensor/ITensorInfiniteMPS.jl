using ITensors, ITensorMPS
using ITensorInfiniteMPS

# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ
function ising_opsum_infinite(N; J, h)
  os = OpSum()
  for n in 1:N
    os -= J, "X", n, "X", n + 1
  end
  for n in 1:N
    os -= h, "Z", n
  end
  return os
end

function ising_opsum_finite(N; J, h)
  return ising_opsum_infinite(N - 1; J, h) + (-h, "Z", N)
end

function main(; N, J, h, nsites)
  s = siteinds("S=1/2", N)

  H = MPO(ising_opsum_finite(N; J=J, h=h), s)
  ψ0 = randomMPS(s)

  energy, ψ = dmrg(H, ψ0; nsweeps=10, cutoff=1e-10)
  @show energy / N

  n1 = N ÷ 2
  n2 = n1 + 1

  ψ = orthogonalize(ψ, n1)

  println("\nFinite MPS energy on bond (n1, n2) = $((n1, n2))")

  ϕ12 = ψ[n1] * ψ[n2]
  ham12 = contract(MPO(ising_opsum_infinite(1; J=J, h=h), [s[n1], s[n2]]))
  @show inner(ϕ12, apply(ham12, ϕ12))

  # Get an infinite MPS that approximates
  # the finite MPS in the bulk.
  # `nsites` sets the number of sites in the unit
  # cell of the infinite MPS.
  ψ∞ = infinitemps_approx(ψ; nsites=nsites, nsweeps=10)

  println("\nInfinite MPS energy on a few different bonds")
  println("Infinite MPS has unit cell of size $nsites")

  # Compute a few energies
  for n1 in 1:3
    n2 = n1 + 1
    @show n1, n2
    ϕ12 = ψ∞.AL[n1] * ψ∞.AL[n2] * ψ∞.C[n2]
    s1 = siteind(ψ∞, n1)
    s2 = siteind(ψ∞, n2)
    ham12 = contract(MPO(ising_opsum_infinite(1; J=J, h=h), [s1, s2]))
    @show inner(ϕ12, apply(ham12, ϕ12)) / norm(ϕ12)
  end

  return nothing
end

main(; N=100, J=1.0, h=1.5, nsites=1)
