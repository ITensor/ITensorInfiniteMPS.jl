using ITensors, ITensorMPS
using ITensorInfiniteMPS
using Random

Random.seed!(1234)

N = 100
s = siteinds("S=1/2", N)

J = 1.0
h = 1.5
model = Model("ising")
H = MPO(model, s; J=J, h=h)
ψ0 = randomMPS(s)

sweeps = Sweeps(10)
maxdim!(sweeps, 10)
cutoff!(sweeps, 1E-10)
energy, ψ = dmrg(H, ψ0, sweeps)
@show energy / N

n1 = N ÷ 2
n2 = n1 + 1

ψ = orthogonalize(ψ, n1)

println("\nFinite MPS energy on bond (n1, n2) = $((n1, n2))")

ϕ12 = ψ[n1] * ψ[n2]
ham12 = ITensor(model, s[n1], s[n2]; J=J, h=h)
@show (ϕ12 * ham12 * prime(dag(ϕ12), commoninds(ϕ12, ham12)))[]

# Get an infinite MPS that approximates
# the finite MPS in the bulk.
# `Nsites` sets the number of sites in the unit
# cell of the infinite MPS.
Nsites = 2
# XXX: not outputing a right orthogonal ψ∞.AR
ψ∞ = infinitemps_approx(ψ; nsites=Nsites, nsweeps=10)

# XXX: this is segfaulting for some reason
#ψ∞ = orthogonalize(ψ∞.AL, :)
C, ψR = ITensorInfiniteMPS.right_orthogonalize(ψ∞.AL)
ψ∞ = InfiniteCanonicalMPS(ψ∞.AL, C, ψR)

println("\nInfinite MPS energy on a few different bonds")
println("Infinite MPS has unit cell of size $Nsites")

# Compute a few energies
for n1 in 1:3
  global n2 = n1 + 1
  @show n1, n2
  global ϕ12 = ψ∞.AL[n1] * ψ∞.AL[n2] * ψ∞.C[n2]
  global s1 = siteind(ψ∞, n1)
  global s2 = siteind(ψ∞, n2)
  global ham12 = ITensor(model, s1, s2; J=J, h=h)
  global ϕ12ᴴ = prime(dag(ϕ12), commoninds(ϕ12, ham12))
  @show (ϕ12 * ham12 * ϕ12ᴴ)[] / norm(ϕ12)
end

s¹ = siteinds(ψ∞, Cell(1))

Σ∞h = InfiniteSum{MPO}(model, s¹; J=J, h=h)

ψ∞_opt = vumps(Σ∞h, ψ∞)

χ = 6
ψ∞ = InfiniteMPS(only.(s¹); space=χ)
randn!.(ψ∞)

ψ∞ = orthogonalize(ψ∞, :)
@show norm(prod(ψ∞.AL[1:Nsites]) * ψ∞.C[Nsites] - ψ∞.C[0] * prod(ψ∞.AR[1:Nsites]))

vumps(Σ∞h, ψ∞; niter=10)
