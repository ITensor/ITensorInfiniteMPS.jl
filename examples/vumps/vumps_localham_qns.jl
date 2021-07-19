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
N = 2
space = [QN("SzParity", 1, 2) => 1, QN("SzParity", 0, 2) => 1]
s = _siteinds("S=1/2", N; space=space)
J = 1.0
h = 1.5
model = Model(:ising)

# Form the Hamiltonian
Σ∞h = InfiniteITensorSum(model, s; J=J, h=h)

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

