using ITensors, ITensorMPS
using ITensorInfiniteMPS
using Random

Random.seed!(1234)

# n-site unit cell
nsites = 4
s = siteinds("S=1/2", nsites; conserve_szparity=true)
χ = 10
@assert iseven(χ)
space = (("SzParity", 1, 2) => χ ÷ 2) ⊕ (("SzParity", 0, 2) => χ ÷ 2)
ψ = InfiniteMPS(ComplexF64, s; space=space)
for n in 1:nsites
  ψ[n] = random_itensor(inds(ψ[n]))
end

ψ = orthogonalize(ψ, :)
@show norm(contract(ψ.AL[1:nsites]) * ψ.C[nsites] - ψ.C[0] * contract(ψ.AR[1:nsites]))
