using ITensorInfiniteMPS
using ITensors

##############################################################################
# VUMPS/TDVP parameters
#

maxdim = 20 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 10 # Maximum number of iterations of the VUMPS/TDVP algorithm at a fixed bond dimension
tol = 1e-5 # Precision error tolerance for outer loop of VUMPS or TDVP
outer_iters = 5 # Number of times to increase the bond dimension
time_step = -Inf # -Inf corresponds to VUMPS, finite time_step corresponds to TDVP
solver_tol = (x -> x / 100) # Tolerance for the local solver (eigsolve in VUMPS and exponentiate in TDVP)
multisite_update_alg = "parallel" # Choose between ["sequential", "parallel"]. Only parallel works with TDVP.
conserve_qns = true # Whether or not to conserve spin parity
nsite = 2 # Number of sites in the unit cell

# Parameters of the transverse field Ising model
model_params = (J=1.0, h=0.9)

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#

model = Model"ising"()

function space_shifted(::Model"ising", q̃sz; conserve_qns=true)
  if conserve_qns
    return [QN("SzParity", 1 - q̃sz, 2) => 1, QN("SzParity", 0 - q̃sz, 2) => 1]
  else
    return [QN() => 2]
  end
end

# Shift the QNs by 1 so that the flux of the unit cell is zero
# and therefore the flux density of the uniform state is zero
# to avoid diverging flux in the thermodynamic limit.
space_ = fill(space_shifted(model, 1; conserve_qns=conserve_qns), nsite)
s = infsiteinds("S=1/2", nsite; space=space_)
initstate(n) = "↑"
ψ = InfMPS(s, initstate)

# Form the Hamiltonian
H = InfiniteSum{MPO}(model, s; model_params...)

# Check translational invariance
@show norm(contract(ψ.AL[1:nsite]..., ψ.C[nsite]) - contract(ψ.C[0], ψ.AR[1:nsite]...))

vumps_kwargs = (
  tol=tol,
  maxiter=max_vumps_iters,
  solver_tol=solver_tol,
  multisite_update_alg=multisite_update_alg,
)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

# Alternate steps of running VUMPS and increasing the bond dimension
@time for _ in 1:outer_iters
  println("\nIncrease bond dimension")
  global ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
  println("Run VUMPS with new bond dimension")
  global ψ = tdvp(H, ψ; time_step=time_step, vumps_kwargs...)
end

# Check translational invariance
@show norm(contract(ψ.AL[1:nsite]..., ψ.C[nsite]) - contract(ψ.C[0], ψ.AR[1:nsite]...))

#
# Compare to DMRG
#

nsite_finite = 100
s_finite = siteinds("S=1/2", nsite_finite; conserve_szparity=true)
H_finite = MPO(model, s_finite; model_params...)
ψ_finite = randomMPS(s_finite, initstate)
@show flux(ψ_finite)
sweeps = Sweeps(10)
setmaxdim!(sweeps, maxdim)
setcutoff!(sweeps, cutoff)
energy_finite_total, ψ_finite = @time dmrg(H_finite, ψ_finite, sweeps)
@show energy_finite_total / nsite_finite

function energy_local(ψ1, ψ2, h)
  ϕ = ψ1 * ψ2
  return (noprime(ϕ * prod(h)) * dag(ϕ))[]
end

function ITensors.expect(ψ, o)
  return (noprime(ψ * op(o, filterinds(ψ, "Site")...)) * dag(ψ))[]
end

# Exact energy at criticality: 4/pi = 1.2732395447351628

n_finite = nsite_finite ÷ 2
orthogonalize!(ψ_finite, n_finite)
hn_finite = ITensor(model, s_finite[n_finite], s_finite[n_finite + 1]; model_params...)
energy_finite = energy_local(ψ_finite[n_finite], ψ_finite[n_finite + 1], hn_finite)
energy_infinite = energy_local(ψ.AL[1], ψ.AL[2] * ψ.C[2], H[(1, 2)])
@show energy_finite, energy_infinite
@show abs(energy_finite - energy_infinite)

energy_exact = reference(model, Observable("energy"); model_params...)
@show energy_exact

Sz1_finite = expect(ψ_finite[n_finite], "Sz")
orthogonalize!(ψ_finite, n_finite + 1)
Sz2_finite = expect(ψ_finite[n_finite + 1], "Sz")
Sz1_infinite = expect(ψ.AL[1] * ψ.C[1], "Sz")
Sz2_infinite = expect(ψ.AL[2] * ψ.C[2], "Sz")

@show Sz1_finite, Sz2_finite
@show Sz1_infinite, Sz2_infinite
