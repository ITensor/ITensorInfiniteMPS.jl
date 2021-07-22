using ITensors
using ITensorInfiniteMPS
using Random

Random.seed!(1234)

include("models.jl")
include("subspace_expansion.jl")

# Number of sites in the unit cell
N = 2

χ1 = [QN("SzParity",1,2) => 1,
      QN("SzParity",0,2) => 1]
χ2 = [QN("SzParity",1,2) => 1,
      QN("SzParity",0,2) => 1]
space = [χ1, χ2]

##using ITensorInfiniteMPS: celltags
# Make an infinite set of repeated site indices
s = _siteinds("S=1/2", N; space=space)
##s = addtags.(s, (celltags(1),))
##s = CelledVector(s)

J = 1.0
h = 1.1
model = Model(:ising)

# Form the Hamiltonian
H = InfiniteITensorSum(model, s; J=J, h=h)

ψL = InfiniteMPS(s)
ψR = InfiniteMPS(s; linksdir=ITensors.In)

ψL.data[1][1,2,1] = 1.0
ψL.data[2][1,2,1] = 1.0
ψR.data[1][1,2,1] = 1.0
ψR.data[2][1,2,1] = 1.0

function linkinds_dict(ψ::InfiniteMPS)
  N = nsites(ψ)
  return Dict([(n, n + 1) => linkinds(ψ, (n, n + 1)) for n in 1:N])
end

l = linkinds_dict(ψL)
r = linkinds_dict(ψR)

b = n1, n2 = 1, 2
C1 = randomITensor(dag(l[b])..., r[b]...)
b = n1, n2 = 2, 3
C2 = randomITensor(dag(l[b])..., r[b]...)
ψC = InfiniteMPS([C1, C2])

ψ = InfiniteCanonicalMPS(ψL, ψC, ψR)

ψ = vumps(H, ψ; niter=10)

function subspace_expansion(ψ)
  N = nsites(ψ)
  AL = ψ.AL
  C = ψ.C
  AR = ψ.AR
  for n in 1:N
    n1, n2 = n, n + 1
    ALⁿ¹², Cⁿ¹, ARⁿ¹² = subspace_expansion(ψ, (n1, n2))
    ALⁿ¹, ALⁿ² = ALⁿ¹²
    ARⁿ¹, ARⁿ² = ARⁿ¹²
    AL[n1] = ALⁿ¹
    AL[n2] = ALⁿ²
    C[n1] = Cⁿ¹
    AR[n1] = ARⁿ¹
    AR[n2] = ARⁿ²
    ψ = InfiniteCanonicalMPS(AL, C, AR)
  end
  return ψ
end

ψ̃ = subspace_expansion(ψ)

ψ̃ = vumps(H, ψ̃; niter=10)

ψ̃ = subspace_expansion(ψ̃)

ψ̃ = vumps(H, ψ̃; niter=10)

ψ̃ = subspace_expansion(ψ̃)

ψ̃ = vumps(H, ψ̃; niter=10)

