using ITensors
using ITensorInfiniteMPS

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

model = Model"ising"()

function space_shifted(::Model"ising", q̃sz; conserve_qns=true)
  if conserve_qns
    return [QN("SzParity", 1 - q̃sz, 2) => 1, QN("SzParity", 0 - q̃sz, 2) => 1]
  else
    return [QN() => 2]
  end
end

space_ = fill(space_shifted(model, 0; conserve_qns=conserve_qns), N)
s = infsiteinds("S=1/2", N; space=space_)
initstate(n) = "↑"
ψ = InfMPS(s, initstate)

# Form the Hamiltonian
H = InfiniteITensorSum(model, s; model_params...)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

vumps_kwargs = (
  tol=tol,
  maxiter=max_vumps_iters,
  solver_tol=solver_tol,
  multisite_update_alg=multisite_update_alg,
)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)
#ψ = tdvp(H, ψ; time_step=time_step, vumps_kwargs...)

# Alternate steps of running VUMPS and increasing the bond dimension
@time for _ in 1:outer_iters
  println("\nIncrease bond dimension")
  global ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
  println("Run VUMPS with new bond dimension")
  global ψ = tdvp(H, ψ; time_step=time_step, vumps_kwargs...)
end

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

#
# Compare to DMRG
#

Nfinite = 100
sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
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
  return (noprime(ϕ * h) * dag(ϕ))[]
end

function ITensors.expect(ψ, o)
  return (noprime(ψ * op(o, filterinds(ψ, "Site")...)) * dag(ψ))[]
end

# Exact energy at criticality: 4/pi = 1.2732395447351628

nfinite = Nfinite ÷ 2
orthogonalize!(ψfinite, nfinite)
hnfinite = ITensor(model, sfinite[nfinite], sfinite[nfinite + 1]; model_params...)
energy_finite = energy_local(ψfinite[nfinite], ψfinite[nfinite + 1], hnfinite)
energy_infinite = energy_local(ψ.AL[1], ψ.AL[2] * ψ.C[2], H[(1, 2)])
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

using Arpack
using KrylovKit
using LinearAlgebra

T = TransferMatrix(ψ.AL)
Tᵀ = transpose(T)
vⁱᴿ = randomITensor(dag(input_inds(T)))
vⁱᴸ = randomITensor(dag(input_inds(Tᵀ)))

neigs = 10
tol = 1e-10
λ⃗ᴿ, v⃗ᴿ, right_info = eigsolve(T, vⁱᴿ, neigs, :LM; tol=tol)
λ⃗ᴸ, v⃗ᴸ, left_info = eigsolve(Tᵀ, vⁱᴸ, neigs, :LM; tol=tol)

@show norm(T(v⃗ᴿ[1]) - λ⃗ᴿ[1] * v⃗ᴿ[1])
@show norm(Tᵀ(v⃗ᴸ[1]) - λ⃗ᴸ[1] * v⃗ᴸ[1])

@show λ⃗ᴿ
@show λ⃗ᴸ
@show flux.(v⃗ᴿ)

neigs = length(v⃗ᴿ)

# Normalize the vectors
N⃗ = [(translatecell(v⃗ᴸ[n], 1) * v⃗ᴿ[n])[] for n in 1:neigs]

v⃗ᴿ ./= sqrt.(N⃗)
v⃗ᴸ ./= sqrt.(N⃗)

# Form a second starting vector orthogonal to v⃗ᴿ[1]
# This doesn't work. TODO: project out v⃗ᴿ[1], v⃗ᴸ[1] from T
#λ⃗ᴿ², v⃗ᴿ², right_info_2 = eigsolve(T, vⁱᴿ², neigs, :LM; tol=tol)

# Projector onto the n-th eigenstate
function proj(v⃗ᴸ, v⃗ᴿ, n)
  Lⁿ = v⃗ᴸ[n]
  Rⁿ = v⃗ᴿ[n]
  return ITensorMap(
    [translatecell(Lⁿ, 1), translatecell(Rⁿ, -1)]; input_inds=inds(Rⁿ), output_inds=inds(Lⁿ)
  )
end

P⃗ = [proj(v⃗ᴸ, v⃗ᴿ, n) for n in 1:neigs]
T⁻P = T - sum(P⃗)

#vⁱᴿ² = vⁱᴿ - (translatecell(v⃗ᴸ[1], 1) * vⁱᴿ)[] / norm(v⃗ᴿ[1]) * v⃗ᴿ[1]
#@show norm(dag(vⁱᴿ²) * v⃗ᴿ[1])

λ⃗ᴾᴿ, v⃗ᴾᴿ, right_info = eigsolve(T⁻P, vⁱᴿ, neigs, :LM; tol=tol)
@show λ⃗ᴾᴿ

vⁱᴿ⁻ᵈᵃᵗᵃ = vec(array(vⁱᴿ))
λ⃗ᴿᴬ, v⃗ᴿ⁻ᵈᵃᵗᵃ = Arpack.eigs(T; v0=vⁱᴿ⁻ᵈᵃᵗᵃ, nev=neigs)

## XXX: this is giving an error about trying to set the element of the wrong QN block for:
## maxdim = 5
## cutoff = 1e-12
## max_vumps_iters = 10
## outer_iters = 10
## model_params = (J=1.0, h=0.8)
##
## v⃗ᴿᴬ = [itensor(v⃗ᴿ⁻ᵈᵃᵗᵃ[:, n], input_inds(T); tol=1e-4) for n in 1:length(λ⃗ᴿᴬ)]
## @show flux.(v⃗ᴿᴬ)

@show λ⃗ᴿᴬ

# Full eigendecomposition

Tfull = prod(T)
DV = eigen(Tfull, input_inds(T), output_inds(T))

@show norm(Tfull * DV.V - DV.Vt * DV.D)

d = diag(array(DV.D))

p = sortperm(d; by=abs, rev=true)
@show p[1:neigs]
@show d[p[1:neigs]]

println("Error if ED with Arpack")
@show d[p[1:neigs]] - λ⃗ᴿᴬ
