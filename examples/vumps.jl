using ITensors
using ITensorInfiniteMPS
using Random

Random.seed!(1234)

N = 2
s = siteinds("S=1/2", N)
χ = 6
@assert iseven(χ)
if hasqns(s)
  space = (("SzParity", 1, 2) => χ ÷ 2) ⊕ (("SzParity", 0, 2) => χ ÷ 2)
else
  space = 2
end
ψ = InfiniteMPS(ComplexF64, s; space = space)
randn!.(ψ)

H = InfiniteMPO(N)

vumps(H, ψ)
