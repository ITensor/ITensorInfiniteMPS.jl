using ITensors
using ITensorInfiniteMPS
using Random

Random.seed!(1234)

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
s = _siteinds("Electron", N; space=space)
t = 1.0
U = 1.0
V = 0.5
model = Model(:hubbard)

# Form the Hamiltonian
Σ∞h = InfiniteITensorSum(model, s; t=t, U=U, V=V)

d = 3
χ = [QN("Sz",0) => d,
     QN("Sz",1) => d,
     QN("Sz",2) => d,
     QN("Sz",-1) => d,
     QN("Sz",-2) => d]
space = χ1
ψ = InfiniteMPS(s; space=space)
flux = QN("Sz",0)
for n in 1:N
  for b in nzblocks(flux, inds(ψ[n]))
    # Need to overwrite the tensor, since modifying
    # it in-place doesn' work when not modifying the storage directly.
    ψ[n] = insertblock!(ψ[n], b)
  end
end
randn!.(ψ)
ψ = orthogonalize(ψ, :)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

ψ = vumps(Σ∞h, ψ; niter=5)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

#
# Bond dimension increase
#

using LinearAlgebra

n1, n2 = 1, 2
NL = nullspace(ψ.AL[n1], commoninds(ψ.AL[n1], ψ.C[n1]); atol=1e-15)
NR = nullspace(ψ.AR[n2], commoninds(ψ.AR[n2], ψ.C[n1]); atol=1e-15)
nL = uniqueinds(NL, ψ.AL[n1])
nR = uniqueinds(NR, ψ.AR[n2])

@show prime(NL, uniqueinds(NL, ψ.AL[n1])) * dag(NL)
@show ψ.AL[n1] * dag(NL)

ψH2 = noprime(ψ.AL[n1] * Σ∞h[(n1, n2)] * ψ.C[n1] * ψ.AR[n2])
ψHN2 = ψH2 * dag(NL) * dag(NR)

@show inds(ψHN2)
U, S, V = svd(ψHN2, nL)
#NL = NL * U
#NR = NR * V

AL, l = ITensors.directsum(ψ.AL[n1], NL, uniqueinds(ψ.AL[n1], NL), uniqueinds(NL, ψ.AL[n1]); tags=("Left",))
#CL = combiner(l)
#AL *= CL
AR, r = ITensors.directsum(ψ.AR[n2], NR, uniqueinds(ψ.AR[n2], NR), uniqueinds(NR, ψ.AR[n2]); tags=("Right",))
#CR = combiner(r)
#AR *= CR



