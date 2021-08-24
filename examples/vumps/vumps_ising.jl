using ITensorInfiniteMPS
using ITensorInfiniteMPS.ITensors

N = 2

model = Model"ising"()
model_kwargs = (J=1.0, h=1.0)

function space_shifted(::Model"ising", q̃sz)
  return [QN("SzParity", 1 - q̃sz, 2) => 1, QN("SzParity", 0 - q̃sz, 2) => 1]
end

space_ = fill(space_shifted(model, 0), N)
s = infsiteinds("S=1/2", N; space=space_)
initstate(n) = "↑"
ψ = InfMPS(s, initstate)

# Form the Hamiltonian
H = InfiniteITensorSum(model, s; model_kwargs...)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

cutoff = 1e-8
maxdim = 20
maxiter = 100
outer_iters = 5
vumps_kwargs = (tol=1e-8, maxiter=maxiter)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)
ψ = vumps(H, ψ; vumps_kwargs...)

# Alternate steps of running VUMPS and increasing the bond dimension
for _ in 1:outer_iters
  println("\nIncrease bond dimension")
  global ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
  println("Run VUMPS with new bond dimension")
  global ψ = vumps(H, ψ; vumps_kwargs...)
end

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

#
# Compare to DMRG
#

Nfinite = 50
sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
Hfinite = MPO(model, sfinite; model_kwargs...)
ψfinite = randomMPS(sfinite, initstate)
@show flux(ψfinite)
sweeps = Sweeps(15)
setmaxdim!(sweeps, maxdim)
setcutoff!(sweeps, cutoff)
energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps)
@show energy_finite_total / Nfinite

function energy(ψ1, ψ2, h)
  ϕ = ψ1 * ψ2
  return (noprime(ϕ * h) * dag(ϕ))[]
end

function ITensors.expect(ψ, o)
  return (noprime(ψ * op(o, filterinds(ψ, "Site")...)) * dag(ψ))[]
end

nfinite = Nfinite ÷ 2
orthogonalize!(ψfinite, nfinite)
hnfinite = ITensor(model, sfinite[nfinite], sfinite[nfinite + 1]; model_kwargs...)
energy_finite = energy(ψfinite[nfinite], ψfinite[nfinite + 1], hnfinite)
energy_infinite = energy(ψ.AL[1], ψ.AL[2] * ψ.C[2], H[(1, 2)])
@show energy_finite, energy_infinite
@show abs(energy_finite - energy_infinite)

Sz1_finite = expect(ψfinite[nfinite], "Sz")
orthogonalize!(ψfinite, nfinite + 1)
Sz2_finite = expect(ψfinite[nfinite + 1], "Sz")
Sz1_infinite = expect(ψ.AL[1] * ψ.C[1], "Sz")
Sz2_infinite = expect(ψ.AL[2] * ψ.C[2], "Sz")

@show Sz1_finite, Sz2_finite
@show Sz1_infinite, Sz2_infinite
