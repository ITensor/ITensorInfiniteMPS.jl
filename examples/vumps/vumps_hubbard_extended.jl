using ITensors
using ITensorInfiniteMPS
using Random

Random.seed!(1234)

include("infinitecanonicalmps.jl")
include("models.jl")

# Number of sites in the unit cell
N = 2

χ1 = [QN("Sz",0) => 1,
      QN("Sz",1) => 1,
      QN("Sz",-1) => 1,
      QN("Sz",0) => 1]
χ2 = [QN("Sz",0) => 1,
      QN("Sz",1) => 1,
      QN("Sz",-1) => 1,
      QN("Sz",0) => 1]
space = [χ1, χ2]
s = infsiteinds("Electron", N; space=space)
t = 1.0
U = 1.0
V = 0.5
model = Model(:hubbard)

# Form the Hamiltonian
H = InfiniteITensorSum(model, s; t=t, U=U, V=V)

state(n) = isodd(n) ? "↑" : "↓"
ψ = InfMPS(s, state)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

cutoff = 1e-8
maxdim = 100
#ψ = subspace_expansion(ψ, H; cutoff=cutoff, maxdim=maxdim)
ψ = vumps(H, ψ; niter=5)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

