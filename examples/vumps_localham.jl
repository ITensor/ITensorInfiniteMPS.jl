using ITensors
using ITensorInfiniteMPS
using Random

Random.seed!(1234)

include("models.jl")

function _siteind(site_tag, n::Int; space)
  return addtags(Index(space, "Site,n=$n"), site_tag)
end
function _siteinds(site_tag, N::Int; space)
  return [_siteind(site_tag, n; space=space) for n in 1:N]
end

# Number of sites in the unit cell
N = 4
space = [QN() => 2]
#space = 2
s = _siteinds("S=1/2", N; space=space)
J = 1.0
h = 1.5
model = Model(:ising)

# Form the Hamiltonian
Σ∞h = InfiniteITensorSum(model, s; J=J, h=h)

#χ = 6
χ = [QN() => 6]
ψ = InfiniteMPS(s; space=χ)
randn!.(ψ)
ψ = orthogonalize(ψ, :)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

ψ = vumps(Σ∞h, ψ; niter=10)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

