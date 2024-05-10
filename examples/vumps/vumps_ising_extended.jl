using ITensors, ITensorMPS
using ITensorInfiniteMPS

include(
  joinpath(
    pkgdir(ITensorInfiniteMPS), "examples", "vumps", "src", "vumps_subspace_expansion.jl"
  ),
)

##############################################################################
# VUMPS parameters
#

maxdim = 64 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 10 # Maximum number of iterations of the VUMPS algorithm at each bond dimension
vumps_tol = 1e-6
conserve_qns = true
outer_iters = 10 # Number of times to increase the bond dimension
eager = true

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#
N = 2 # Number of sites in the unit cell
J = -1.0
J₂ = -0.2
h = 1.0;

initstate(n) = "↑"
s = infsiteinds("S=1/2", N; initstate, conserve_szparity=conserve_qns)
ψ = InfMPS(s, initstate);

model = Model("ising_extended");
H = InfiniteSum{MPO}(model, s; J=J, J₂=J₂, h=h);

@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...));

vumps_kwargs = (tol=vumps_tol, maxiter=max_vumps_iters, eager)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

ψ = vumps_subspace_expansion(H, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs)

Sz_infinite = [expect(ψ, "Sz", n) for n in 1:N]
energy_infinite = expect(ψ, H)

Nfinite = 100
sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=conserve_qns)
Hfinite = MPO(model, sfinite; J=J, J₂=J₂, h=h)
ψfinite = randomMPS(sfinite, initstate; linkdims=10)
@show flux(ψfinite)
dmrg_kwargs = (nsweeps=10, maxdim, cutoff)
energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite; dmrg_kwargs...)

nfinite = Nfinite ÷ 2
Sz_finite = expect(ψfinite, "Sz")[nfinite:(nfinite + N - 1)]

println("\nEnergy")
@show energy_infinite
@show energy_finite_total / Nfinite
@show reference(model, Observable("energy"); J=J, h=h, J₂=J₂)

println("\nSz")
@show Sz_infinite
@show Sz_finite
