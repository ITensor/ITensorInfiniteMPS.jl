using ITensors
using ITensorInfiniteMPS

include("models.jl")
include("infinitecanonicalmps.jl")

N = 2

heisenberg_space_shift(q̃nf, q̃sz) = [QN("Sz", 1 - q̃sz) => 1, QN("Sz", -1 - q̃sz) => 1]

electron_space = fill(electron_space_shift(1, 0), N)
s = infsiteinds("Electron", N; space=electron_space)
initstate(n) = isodd(n) ? "↑" : "↓"
ψ = InfMPS(s, initstate)

model = Model(:heisenberg)

# Form the Hamiltonian
H = InfiniteITensorSum(model, s)

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
