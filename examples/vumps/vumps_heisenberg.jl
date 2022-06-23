using ITensors
using ITensorInfiniteMPS

include(joinpath(@__DIR__, "src", "vumps_subspace_expansion.jl"))

##############################################################################
# VUMPS parameters
#

maxdim = 100 # Maximum bond dimension
cutoff = 1e-8 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 100 # Maximum number of iterations of the VUMPS algorithm at each bond dimension
vumps_tol = 1e-5
conserve_qns = true
outer_iters = 6 # Number of times to increase the bond dimension

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#

N = 2 # Number of sites in the unit cell

initstate(n) = isodd(n) ? "↑" : "↓"
s = infsiteinds("S=1/2", N; conserve_qns, initstate)
ψ = InfMPS(s, initstate)

model = Model("heisenberg")

# Form the Hamiltonian
H = InfiniteSum{MPO}(model, s)

# Check translational invariance
println("\nCheck translation invariance of the initial VUMPS state")
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

vumps_kwargs = (tol=vumps_tol, maxiter=max_vumps_iters, solver_tol=(x -> x / 1000))
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

ψ = vumps_subspace_expansion(H, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs)

# Check translational invariance
println("\nCheck translation invariance of the final VUMPS state")
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

function expect_two_site(ψ::InfiniteCanonicalMPS, h::ITensor, n1n2)
  n1, n2 = n1n2
  ϕ = ψ.AL[n1] * ψ.AL[n2] * ψ.C[n2]
  return inner(ϕ, apply(h, ϕ))
end

function expect_two_site(ψ::InfiniteCanonicalMPS, h::MPO, n1n2)
  return expect_two_site(ψ, contract(h), n1n2)
end

Sz = [expect(ψ, "Sz", n) for n in 1:N]

bs = [(1, 2), (2, 3)]
energy_infinite = map(b -> expect_two_site(ψ, H[b], b), bs)

energy_exact = reference(model, Observable("energy"))

#
# Compare to DMRG
#

Nfinite = 100
sfinite = siteinds("S=1/2", Nfinite; conserve_qns)
Hfinite = MPO(model, sfinite)
ψfinite = randomMPS(sfinite, initstate; linkdims=10)
@show flux(ψfinite)

nsweeps = 10
println("\nRunning finite DMRG for model $model on $Nfinite sites with $nsweeps")
energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite; nsweeps, maxdim, cutoff)

energy_exact_finite = reference(model, Observable("energy"); N=Nfinite)

function ITensors.expect(ψ::ITensor, o::String)
  return inner(ψ, apply(op(o, filterinds(ψ, "Site")...), ψ))
end

nfinite = Nfinite ÷ 2
orthogonalize!(ψfinite, nfinite)
Sz1_finite = expect(ψfinite[nfinite], "Sz")
orthogonalize!(ψfinite, nfinite + 1)
Sz2_finite = expect(ψfinite[nfinite + 1], "Sz")

println("\nCompare energy")
@show energy_finite_total / Nfinite
@show energy_infinite
@show energy_exact_finite
@show energy_exact

println("\nCompare Sz")
@show Sz1_finite, Sz2_finite
@show Sz
