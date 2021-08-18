using ITensors
using ITensorInfiniteMPS

# Unit cell size
N = 2

function electron_space_shift(q̃nf, q̃sz)
  return [
    QN(("Nf", 0 - q̃nf, -1), ("Sz", 0 - q̃sz)) => 1,
    QN(("Nf", 1 - q̃nf, -1), ("Sz", 1 - q̃sz)) => 1,
    QN(("Nf", 1 - q̃nf, -1), ("Sz", -1 - q̃sz)) => 1,
    QN(("Nf", 2 - q̃nf, -1), ("Sz", 0 - q̃sz)) => 1,
  ]
end

electron_space = fill(electron_space_shift(1, 0), N)
s = infsiteinds("Electron", N; space=electron_space)
initstate(n) = isodd(n) ? "↑" : "↓"
ψ = InfMPS(s, initstate)

model = Model"hubbard"()
model_params = (t=1.0, U=0.0, V=0.0)
@show model, model_params

# Form the Hamiltonian
H = InfiniteITensorSum(model, s; model_params...)

# Check translational invariance
println("\nCheck translational invariance of initial infinite MPS")
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

cutoff = 1e-8
maxdim = 100
outputlevel = 1
environment_iterations = 100 # Number of iterations used to sum up the Hamiltonian terms (summing a geometric series with a recursive formula, one term at a time)
vumps_iters = 100 # Number of VUMPS iterations at a given bond dimension
outer_iters = 3 # Number of times to increase the bond dimension then run vumps_iters VUMPS iterations
vumps_kwargs = (
  environment_iterations=environment_iterations, niter=vumps_iters, outputlevel=outputlevel
)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

# For now, to increase the bond dimension you must alternate
# between steps of VUMPS and subspace expansion (which outputs
# a new state that is equal to the original state but with
# a larger bond dimension)

println("\nRun VUMPS on initial product state, unit cell size $N")
ψ = vumps(H, ψ; vumps_kwargs...)

for _ in 1:outer_iters
  println("\nIncrease bond dimension")
  global ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
  println("Run VUMPS with new bond dimension")
  global ψ = vumps(H, ψ; vumps_kwargs...)
end

# Check translational invariance
println("\nCheck translational invariance of optimized infinite MPS")
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

function ITensors.expect(ψ::InfiniteCanonicalMPS, o, n)
  return (noprime(ψ.AL[n] * ψ.C[n] * op(o, s[n])) * dag(ψ.AL[n] * ψ.C[n]))[]
end

function expect_two_site(ψ::InfiniteCanonicalMPS, h::ITensor, n1n2)
  n1, n2 = n1n2
  ϕ = ψ.AL[n1] * ψ.AL[n2] * ψ.C[n2]
  return (noprime(ϕ * h) * dag(ϕ))[]
end

function expect_two_site(ψ::MPS, h::ITensor, n1n2)
  n1, n2 = n1n2
  ψ = orthogonalize(ψ, n1)
  ϕ = ψ[n1] * ψ[n2]
  return (noprime(ϕ * h) * dag(ϕ))[]
end

Nup = [expect(ψ, "Nup", n) for n in 1:N]
Ndn = [expect(ψ, "Ndn", n) for n in 1:N]
Sz = [expect(ψ, "Sz", n) for n in 1:N]

bs = [(1, 2), (2, 3)]
energy_infinite = map(b -> expect_two_site(ψ, H[b], b), bs)

#
# Compare to DMRG
#

Nfinite = 40
sfinite = siteinds("Electron", Nfinite; conserve_qns=true)
Hfinite = MPO(model, sfinite; model_params...)
ψfinite = randomMPS(sfinite, initstate; linkdims=10)
println("\nQN sector of starting finite MPS")
@show flux(ψfinite)
sweeps = Sweeps(30)
setmaxdim!(sweeps, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 50)
setcutoff!(sweeps, 1E-8)
println("\nRun DMRG on $Nfinite sites")
energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps)
println("\nEnergy density")
@show energy_finite_total / Nfinite

nfinite = Nfinite ÷ 2 - 1
bsfinite = [(nfinite, nfinite + 1), (nfinite + 1, nfinite + 2)]
hfinite(b) = ITensor(model, sfinite[b[1]], sfinite[b[2]]; model_params...)
energy_finite = map(b -> expect_two_site(ψfinite, hfinite(b), b), bsfinite)

Nup_finite = ITensors.expect(ψfinite, "Nup")[nfinite:(nfinite + 1)]
Ndn_finite = ITensors.expect(ψfinite, "Ndn")[nfinite:(nfinite + 1)]
Sz_finite = ITensors.expect(ψfinite, "Sz")[nfinite:(nfinite + 1)]

println("\nResults from VUMPS")
@show energy_infinite
@show Nup
@show Ndn
@show Nup .+ Ndn
@show Sz

println("\nResults from DMRG")
@show energy_finite
@show Nup_finite
@show Ndn_finite
@show Nup_finite .+ Ndn_finite
@show Sz_finite

nothing
