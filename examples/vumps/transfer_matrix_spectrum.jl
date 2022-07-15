using ITensors
using ITensorInfiniteMPS

include(
  joinpath(pkgdir(ITensorInfiniteMPS), "examples", "src", "vumps_subspace_expansion.jl")
)

##############################################################################
# VUMPS parameters
#

maxdim = 10 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 20 # Maximum number of iterations of the VUMPS algorithm at a fixed bond dimension
tol = 1e-5 # Precision error tolerance for outer loop of VUMPS or TDVP
outer_iters = 5 # Number of times to increase the bond dimension
time_step = -Inf # -Inf corresponds to VUMPS
solver_tol = (x -> x / 100) # Tolerance for the local solver (eigsolve in VUMPS and exponentiate in TDVP)
multisite_update_alg = "parallel" # Choose between ["sequential", "parallel"]
conserve_qns = true
N = 2 # Number of sites in the unit cell (1 site unit cell is currently broken)

# Parameters of the transverse field Ising model
model_params = (J=1.0, h=0.9)

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#

model = Model("ising")

initstate(n) = "↑"
s = infsiteinds("S=1/2", N; initstate, conserve_szparity=conserve_qns)
ψ = InfMPS(s, initstate)

# Form the Hamiltonian
H = InfiniteSum{MPO}(model, s; model_params...)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

vumps_kwargs = (
  tol=tol,
  maxiter=max_vumps_iters,
  solver_tol=solver_tol,
  multisite_update_alg=multisite_update_alg,
)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

ψ = tdvp_subspace_expansion(
  H, ψ; time_step, outer_iters, subspace_expansion_kwargs, vumps_kwargs
)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

#
# Compare to DMRG
#

Nfinite = 100
sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=conserve_qns)
Hfinite = MPO(model, sfinite; model_params...)
ψfinite = randomMPS(sfinite, initstate)
@show flux(ψfinite)
sweeps = Sweeps(10)
setmaxdim!(sweeps, maxdim)
setcutoff!(sweeps, cutoff)
energy_finite_total, ψfinite = @time dmrg(Hfinite, ψfinite, sweeps)
@show energy_finite_total / Nfinite

function energy_local(ψ1, ψ2, h)
  ϕ = ψ1 * ψ2
  return inner(ϕ, apply(h, ϕ))
end

function ITensors.expect(ψ::ITensor, o::String)
  return inner(ψ, apply(op(o, filterinds(ψ, "Site")...), ψ))
end

# Exact energy at criticality: 4/pi = 1.2732395447351628

nfinite = Nfinite ÷ 2
orthogonalize!(ψfinite, nfinite)
hnfinite = ITensor(model, sfinite[nfinite], sfinite[nfinite + 1]; model_params...)
energy_finite = energy_local(ψfinite[nfinite], ψfinite[nfinite + 1], hnfinite)
energy_infinite = energy_local(ψ.AL[1], ψ.AL[2] * ψ.C[2], contract(H[(1, 2)]))
@show energy_finite, energy_infinite
@show abs(energy_finite - energy_infinite)

energy_exact = reference(model, Observable("energy"); model_params...)
@show energy_exact

Sz1_finite = expect(ψfinite[nfinite], "Sz")
orthogonalize!(ψfinite, nfinite + 1)
Sz2_finite = expect(ψfinite[nfinite + 1], "Sz")
Sz1_infinite = expect(ψ.AL[1] * ψ.C[1], "Sz")
Sz2_infinite = expect(ψ.AL[2] * ψ.C[2], "Sz")

@show Sz1_finite, Sz2_finite
@show Sz1_infinite, Sz2_infinite

##############################################################################
# Compute eigenspace of the transfer matrix
#

using KrylovKit: eigsolve
using LinearAlgebra

T = TransferMatrix(ψ.AL)
Tᵀ = transpose(T)
vⁱᴿ = randomITensor(dag(input_inds(T)))
vⁱᴸ = randomITensor(dag(input_inds(Tᵀ)))

neigs = 10
tol = 1e-10
λ⃗ᴿ, v⃗ᴿ, right_info = eigsolve(T, vⁱᴿ, neigs, :LM; tol=tol)
λ⃗ᴸ, v⃗ᴸ, left_info = eigsolve(Tᵀ, vⁱᴸ, neigs, :LM; tol=tol)

println("\n##########################################")
println("Check transfer matrix left and right fixed points")
@show norm(T(v⃗ᴿ[1]) - λ⃗ᴿ[1] * v⃗ᴿ[1])
@show norm(Tᵀ(v⃗ᴸ[1]) - λ⃗ᴸ[1] * v⃗ᴸ[1])

@show λ⃗ᴿ
@show λ⃗ᴸ
@show flux.(v⃗ᴿ)

neigs = min(length(v⃗ᴸ), length(v⃗ᴿ))
v⃗ᴸ = v⃗ᴸ[1:neigs]
v⃗ᴿ = v⃗ᴿ[1:neigs]

# Normalize the vectors
N⃗ = [(translatecelltags(v⃗ᴸ[n], 1) * v⃗ᴿ[n])[] for n in 1:neigs]

v⃗ᴿ ./= sqrt.(N⃗)
v⃗ᴸ ./= sqrt.(N⃗)

# Compare to full eigendecomposition
# Note the this obtains eigenvectors from all QN
# sectors so will include vectors not found by
# `eigsolve` above, which only includes eigenvectors
# in the trivial QN sector.
Tfull = prod(T)
DV = eigen(Tfull, input_inds(T), output_inds(T))

println("\nCheck full diagonalization on transfer matrix")
@show norm(Tfull * DV.V - DV.Vt * DV.D)

d = diag(array(DV.D))

p = sortperm(d; by=abs, rev=true)
@show p[1:neigs]
@show d[p[1:neigs]]
