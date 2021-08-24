using ITensors
using ITensorInfiniteMPS

##############################################################################
# VUMPS parameters
#

maxdim = 30 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 100 # Maximum number of iterations of the VUMPS algorithm at each bond dimension
outer_iters = 6 # Number of times to increase the bond dimension

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#

N = 2 # Number of sites in the unit cell

heisenberg_space_shift(q̃nf, q̃sz) = [QN("Sz", 1 - q̃sz) => 1, QN("Sz", -1 - q̃sz) => 1]

heisenberg_space = fill(heisenberg_space_shift(1, 0), N)
s = infsiteinds("S=1/2", N; space=heisenberg_space)
initstate(n) = isodd(n) ? "↑" : "↓"
ψ = InfMPS(s, initstate)

model = Model"heisenberg"()

# Form the Hamiltonian
H = InfiniteITensorSum(model, s)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

vumps_kwargs = (tol=1e-8, maxiter=max_vumps_iters)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)
ψ = vumps(H, ψ; vumps_kwargs...)

# Alternate steps of running VUMPS and increasing the bond dimension
for _ in 1:outer_iters
  println("\nIncrease bond dimension")
  global ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
  println("Run VUMPS with new bond dimension")
  global ψ = vumps(H, ψ; vumps_kwargs...)
end

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

function ITensors.expect(ψ::InfiniteCanonicalMPS, o, n)
  return (noprime(ψ.AL[n] * ψ.C[n] * op(o, s[n])) * dag(ψ.AL[n] * ψ.C[n]))[]
end

function expect_two_site(ψ::InfiniteCanonicalMPS, h::ITensor, n1n2)
  n1, n2 = n1n2
  ϕ = ψ.AL[n1] * ψ.AL[n2] * ψ.C[n2]
  return (noprime(ϕ * h) * dag(ϕ))[]
end

Sz = [expect(ψ, "Sz", n) for n in 1:N]

bs = [(1, 2), (2, 3)]
energy_infinite = map(b -> expect_two_site(ψ, H[b], b), bs)

@show energy_infinite
@show Sz

#
# Compare to DMRG
#

Nfinite = 50
sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
Hfinite = MPO(model, sfinite)
ψfinite = randomMPS(sfinite, initstate)
@show flux(ψfinite)
sweeps = Sweeps(15)
setmaxdim!(sweeps, maxdim)
setcutoff!(sweeps, cutoff)
energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps)
@show energy_finite_total / Nfinite

function ITensors.expect(ψ, o)
  return (noprime(ψ * op(o, filterinds(ψ, "Site")...)) * dag(ψ))[]
end

nfinite = Nfinite ÷ 2
orthogonalize!(ψfinite, nfinite)
Sz1_finite = expect(ψfinite[nfinite], "Sz")
orthogonalize!(ψfinite, nfinite + 1)
Sz2_finite = expect(ψfinite[nfinite + 1], "Sz")

@show Sz1_finite, Sz2_finite
