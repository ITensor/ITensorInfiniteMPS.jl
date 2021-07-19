using ITensors
using ITensorInfiniteMPS
using Random

Random.seed!(1234)

include("models.jl")

# Number of sites in the unit cell
N = 2
space = [QN("SzParity", 1, 2) => 1, QN("SzParity", 0, 2) => 1]
s = _siteinds("Electron", N; space=space)
t = 1.0
U = 1.0
V = 0.5
model = Model(:hubbard)

# Form the Hamiltonian
Σ∞h = InfiniteITensorSum(model, s; t=t, U=U, V=V)

QN(("Nf",10,-1),("Sz",0))

χ = [QN(("Nf",0,-1),("Sz",0)) => 1,
QN(("Nf",1,-1),("Sz",1)) => 1,
QN(("Nf",1,-1),("Sz",-1)) => 1,
QN(("Nf",2,-1),("Sz",0)) => 1]

χ = [QN("SzParity", 1, 2) => 2, QN("SzParity", 0, 2) => 2]
ψ = InfiniteMPS(s; space=χ)
for n in 1:N
  for b in nzblocks(QN(), inds(ψ[n]))
    # Need to overwrite the tensor, since modifying
    # it in-place doesn' work when not modifying the storage directly.
    ψ[n] = insertblock!(ψ[n], b)
  end
end
randn!.(ψ)
ψ = orthogonalize(ψ, :)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

ψ = vumps(Σ∞h, ψ; niter=10)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

