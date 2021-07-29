using ITensors
using ITensorInfiniteMPS

# Unit cell size
N = 2

electron_space_shift(q̃nf, q̃sz) = [
  QN(("Nf", 0 - q̃nf, -1), ("Sz", 0 - q̃sz)) => 1,
  QN(("Nf", 1 - q̃nf, -1), ("Sz", 1 - q̃sz)) => 1,
  QN(("Nf", 1 - q̃nf, -1), ("Sz", -1 - q̃sz)) => 1,
  QN(("Nf", 2 - q̃nf, -1), ("Sz", 0 - q̃sz)) => 1]

electron_space = fill(electron_space_shift(1, 0), N)
s = infsiteinds("Electron", N; space=electron_space)
initstate(n) = isodd(n) ? "↑" : "↓"
ψ = InfMPS(s, initstate)

model_params = (t=1.0, U=8.0, V=0.0)
model = Model"hubbard"()

# Form the Hamiltonian
H = InfiniteITensorSum(model, s; model_params...)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

cutoff = 1e-8
maxdim = 100
environment_iterations = 20
niter = 20
vumps_kwargs = (environment_iterations=environment_iterations, niter=niter)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

# For now, to increase the bond dimension you must alternate
# between steps of VUMPS and subspace expansion (which outputs
# a new state that is equal to the original state but with
# a larger bond dimension)
ψ = vumps(H, ψ; vumps_kwargs...)
ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
ψ = vumps(H, ψ; vumps_kwargs...)
ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
ψ = vumps(H, ψ; vumps_kwargs...)
ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
ψ = vumps(H, ψ; vumps_kwargs...)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

ITensors.expect(ψ::InfiniteCanonicalMPS, o, n) = (noprime(ψ.AL[n] * ψ.C[n] * op(o, s[n])) * dag(ψ.AL[n] * ψ.C[n]))[]

function expect_two_site(ψ::InfiniteCanonicalMPS, h::ITensor, n1n2)
  n1, n2 = n1n2
  ϕ = ψ.AL[n1] * ψ.AL[n2] * ψ.C[n2]
  return (noprime(ϕ * h) * dag(ϕ))[]
end

function expect_two_site(ψ::MPS, h::ITensor, n1n1)
  n1, n2 = n1n2
  ψ = orthogonalize(ψ, n1)
  ϕ = ψ[n1] * ψ[n2]
  return (noprime(ϕ * h) * dag(ϕ))[]
end

Nup = [expect(ψ, "Nup", n) for n in 1:N]
Ndn = [expect(ψ, "Ndn", n) for n in 1:N]
Sz = [expect(ψ, "Sz", n) for n in 1:N]

bs = [(1, 2), (2, 3)]
energy_infinite = map(b -> expect_two_site(ψ, H[b], b), bs)

@show energy_infinite
@show Nup
@show Ndn
@show Nup .+ Ndn
@show Sz

#
# Compare to DMRG
#

Nfinite = 100
sfinite = siteinds("Electron", Nfinite; conserve_qns=true)
Hfinite = MPO(model, sfinite; model_params...)
ψfinite = randomMPS(sfinite, initstate)
@show flux(ψfinite)
sweeps = Sweeps(10)
setmaxdim!(sweeps, 100)
setcutoff!(sweeps, 1E-10)
energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps)
@show energy_finite_total / Nfinite

nfinite = Nfinite ÷ 2 - 1
bsfinite = [(nfinite, nfinite + 1), (nfinite + 1, nfinite + 2)]
hfinite(b) = ITensor(model, sfinite[b[1]], sfinite[b[2]]; model_params...)
energy_finite = map(b -> expect_two_site(ψfinite, hfinite(b)), bsfinite)

Nup_finite = ITensors.expect(ψfinite, "Nup")[nfinite:(nfinite + 1)]
Ndn_finite = ITensors.expect(ψfinite, "Ndn")[nfinite:(nfinite + 1)]
Sz_finite = ITensors.expect(ψfinite, "Sz")[nfinite:(nfinite + 1)]

@show energy_finite
@show Nup_finite
@show Ndn_finite
@show Nup_finite .+ Ndn_finite
@show Sz_finite

