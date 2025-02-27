using ITensors, ITensorMPS
using ITensorInfiniteMPS

base_path = joinpath(pkgdir(ITensorInfiniteMPS), "examples", "vumps", "src")
src_files = ["vumps_subspace_expansion.jl", "entropy.jl"]
for f in src_files
  include(joinpath(base_path, f))
end

##############################################################################
# VUMPS parameters
#

maxdim = 50 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 200 # Maximum number of iterations of the VUMPS algorithm at each bond dimension
vumps_tol = 1e-5
outer_iters = 5 # Number of times to increase the bond dimension
localham_type = MPO # or ITensor
conserve_qns = true
eager = true

model_params = (t=1.0, U=10.0, V=0.0)

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#

N = 2 # Unit cell size

@show N
@show localham_type

initstate(n) = isodd(n) ? "↑" : "↓"
s = infsiteinds("Electron", N; initstate, conserve_qns)
ψ = InfMPS(s, initstate)

model = Model"hubbard"()
@show model, model_params

# Form the Hamiltonian
H = InfiniteSum{localham_type}(model, s; model_params...)

# Check translational invariance
println("\nCheck translational invariance of initial infinite MPS")
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

outputlevel = 1
vumps_kwargs = (tol=vumps_tol, maxiter=max_vumps_iters, outputlevel, eager)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

# For now, to increase the bond dimension you must alternate
# between steps of VUMPS and subspace expansion (which outputs
# a new state that is equal to the original state but with
# a larger bond dimension)

println("\nRun VUMPS on initial product state, unit cell size $N")
ψ = vumps_subspace_expansion(H, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs)
ψ = orthogonalize(ψ.AL, :; tol=1e-14) # ensure translation invariance

# Check translational invariance
println("\nCheck translational invariance of optimized infinite MPS")
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

function expect_two_site(ψ::InfiniteCanonicalMPS, h::ITensor, n1n2)
  n1, n2 = n1n2
  ϕ = ψ.AL[n1] * ψ.AL[n2] * ψ.C[n2]
  return (noprime(ϕ * h) * dag(ϕ))[]
end

function expect_two_site(ψ::InfiniteCanonicalMPS, h::MPO, n1n2)
  return expect_two_site(ψ, prod(h), n1n2)
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

Nfinite = 100
sfinite = siteinds("Electron", Nfinite; conserve_qns)
Hfinite = MPO(model, sfinite; model_params...)
ψfinite = random_mps(sfinite, initstate; linkdims=10)
println("\nQN sector of starting finite MPS")
@show flux(ψfinite)

nsweeps = 15
maxdims =
  min.(maxdim, [2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 50])
@show maxdims

## setmaxdim!(sweeps, maxdims...)
## setcutoff!(sweeps, cutoff)

println("\nRun DMRG on $Nfinite sites")
energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite; nsweeps, maxdim=maxdims, cutoff)
println("\nEnergy density")
@show energy_finite_total / Nfinite

nfinite = Nfinite ÷ 2 - 1
bsfinite = [(nfinite, nfinite + 1), (nfinite + 1, nfinite + 2)]
hfinite(b) = ITensor(model, [sfinite[b[1]], sfinite[b[2]]]; model_params...)
energy_finite = map(b -> expect_two_site(ψfinite, hfinite(b), b), bsfinite)

Nup_finite = ITensorMPS.expect(ψfinite, "Nup")[nfinite:(nfinite + 1)]
Ndn_finite = ITensorMPS.expect(ψfinite, "Ndn")[nfinite:(nfinite + 1)]
Sz_finite = ITensorMPS.expect(ψfinite, "Sz")[nfinite:(nfinite + 1)]

energy_exact = reference(model, Observable("energy"); U=model_params.U / model_params.t)

corr_infinite = correlation_matrix(finite_mps(ψ, 1:10), "Cdagup", "Cup"; sites=2:11)
corr_finite = correlation_matrix(
  ψfinite, "Cdagup", "Cup"; sites=Int(Nfinite / 2):Int(Nfinite / 2 + 9)
)

S_finite = [entropy(ψfinite, b) for b in (Nfinite ÷ 2):(Nfinite ÷ 2 + N - 1)]
S_infinite = [entropy(ψ, b) for b in 1:N]

println("\nResults from VUMPS")
@show energy_infinite
@show energy_exact
@show Nup
@show Ndn
@show Nup .+ Ndn
@show Sz
@show corr_infinite
@show S_infinite

println("\nResults from DMRG")
@show energy_finite
@show Nup_finite
@show Ndn_finite
@show Nup_finite .+ Ndn_finite
@show Sz_finite
@show corr_finite
@show S_finite

nothing
