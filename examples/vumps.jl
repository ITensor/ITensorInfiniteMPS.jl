using ITensors
using ITensorInfiniteMPS
using Random

# Helper function to make an MPO
import ITensors: op
op(::OpName"Zero", ::SiteType"S=1/2", s::Index) =
  ITensor(s', dag(s))

function tfi_mpo(s, l, r; J = 1.0, h)
  #h = emptyITensor(s', dag(s), l, r)
  dₗ = 3 # The link dimension of the TFI
  Hmat = fill(op("Zero", s), dₗ, dₗ)
  Hmat[1, 1] = op("Id", s)
  Hmat[2, 2] = op("Id", s)
  Hmat[3, 3] = op("Id", s)
  Hmat[2, 1] = -J * op("X", s)
  Hmat[3, 2] = -J * op("X", s)
  Hmat[3, 1] = -h * op("Z", s)
  H = emptyITensor(Any)
  for i in 1:dₗ, j in 1:dₗ
    H += Hmat[i,j] * setelt(l => i) * setelt(r => j)
  end
  return H
end

Random.seed!(1234)

N = 1
s = siteinds("S=1/2", N)
χ = 2
@assert iseven(χ)
if any(hasqns, s)
  space = (("SzParity", 1, 2) => χ ÷ 2) ⊕ (("SzParity", 0, 2) => χ ÷ 2)
else
  space = 2
end
ψ = InfiniteMPS(ComplexF64, s; space = space)
randn!.(ψ)

# Use a finite MPO to create the infinite MPO
H = InfiniteMPO(N)

# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ
J = 1.0
h = 2.0

# For the finite unit cell
s¹ = addtags.(s, "c=1")
l¹ = [Index(3, "Link,l=$n,c=1") for n in 1:N]
for n in 1:N
  # TODO: use a CelledVector here to make the code logic simpler
  l¹ₙ = n == 1 ? replacetags(dag(l¹[N]), "c=1" => "c=0") : l¹[n-1]
  r¹ₙ = l¹[n]
  H[n] = tfi_mpo(s¹[n], l¹ₙ, r¹ₙ; J = J, h = h)
end

vumps(H, ψ)

