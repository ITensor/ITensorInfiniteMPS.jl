using ITensors, ITensorMPS
using ITensorInfiniteMPS
using Random

Random.seed!(1234)

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
model_kwargs = (J=1.0, h=1.5)
model = Model(:ising)
initstate(n) = isodd(n) ? "↑" : "↓"

# Finite MPS
Nfinite = 100
sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
Hfinite = MPO(model, sfinite; model_kwargs...)
ψfinite = random_mps(sfinite, initstate)
sweeps = Sweeps(20)
setmaxdim!(sweeps, 1)
setcutoff!(sweeps, 1E-10)
energy_finite, ψfinite = dmrg(Hfinite, ψfinite, sweeps)
@show energy_finite / Nfinite

# Form the Hamiltonian
Σ∞h = InfiniteSum{ITensor}(model, s; model_kwargs...)

d = 1
χ = [QN("SzParity", 1, 2) => d, QN("SzParity", 0, 2) => d]
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

ψ = vumps(Σ∞h, ψ; niter=30, environment_iterations=20)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

function energy(ψ1, ψ2, h)
  ϕ = ψ1 * ψ2
  return (noprime(ϕ * prod(h)) * dag(ϕ))[]
end

function expect(ψ, o)
  return (noprime(ψ * op(o, filterinds(ψ, "Site")...)) * dag(ψ))[]
end

nfinite = Nfinite ÷ 2
orthogonalize!(ψfinite, nfinite)
hnfinite = ITensor(model, sfinite[nfinite], sfinite[nfinite + 1]; model_kwargs...)
energy_finite = energy(ψfinite[nfinite], ψfinite[nfinite + 1], hnfinite)
energy_infinite = energy(ψ.AL[1], ψ.AL[2] * ψ.C[2], Σ∞h[(1, 2)])
@show energy_finite, energy_infinite
@show abs(energy_finite - energy_infinite)

@show expect(ψfinite[nfinite], "Sz")
@show expect(ψfinite[nfinite + 1], "Sz")
@show expect(ψ[1], "Sz")
@show expect(ψ[2], "Sz")
