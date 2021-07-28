using ITensors
using ITensorInfiniteMPS

include("models.jl")
include("infinitecanonicalmps.jl")

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

t = 1.0
U = 1.0
V = 0.5
model = Model(:hubbard)

# Form the Hamiltonian
H = InfiniteITensorSum(model, s; t=t, U=U, V=V)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

cutoff = 1e-8
maxdim = 100
environment_iterations = 20
niter = 20
vumps_kwargs = (environment_iterations=environment_iterations, niter=niter)
ψ = subspace_expansion(ψ, H; cutoff=cutoff, maxdim=maxdim)
ψ = vumps(H, ψ; vumps_kwargs...)
ψ = subspace_expansion(ψ, H; cutoff=cutoff, maxdim=maxdim)
ψ = vumps(H, ψ; vumps_kwargs...)
ψ = subspace_expansion(ψ, H; cutoff=cutoff, maxdim=maxdim)
ψ = vumps(H, ψ; vumps_kwargs...)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

expect(ψ, o, n) = (noprime(ψ.AL[n] * ψ.C[n] * op(o, s[n])) * dag(ψ.AL[n] * ψ.C[n]))[]

Nup = [expect(ψ, "Nup", n) for n in 1:N]
Ndn = [expect(ψ, "Ndn", n) for n in 1:N]
Sz = [expect(ψ, "Sz", n) for n in 1:N]

