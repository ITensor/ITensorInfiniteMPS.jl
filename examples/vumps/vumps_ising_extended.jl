using ITensors
using ITensorInfiniteMPS

##############################################################################
# VUMPS parameters
#

maxdim = 64 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 100 # Maximum number of iterations of the VUMPS algorithm at each bond dimension
vumps_tol = 1e-6
outer_iters = 10 # Number of times to increase the bond dimension

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#
N = 3# Number of sites in the unit cell
J = -1.0
J₂ = -0.2
h = 1.0;

function space_shifted(q̃sz)
  return [QN("SzParity", 1 - q̃sz, 2) => 1, QN("SzParity", 0 - q̃sz, 2) => 1]
end

space_ = fill(space_shifted(1), N);
s = infsiteinds("S=1/2", N; space=space_)
initstate(n) = "↑"
ψ = InfMPS(s, initstate);

model = Model("ising_extended");
H = InfiniteSum{MPO}(model, s; J=J, J₂=J₂, h=h);

@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...));

vumps_kwargs = (tol=vumps_tol, maxiter=max_vumps_iters)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)
ψ_0 = @time vumps(H, ψ; vumps_kwargs...)

@time for j in 1:outer_iters
  println("\nIncrease bond dimension")
  ψ_1 = @time subspace_expansion(ψ_0, H; subspace_expansion_kwargs...)
  println("Run VUMPS with new bond dimension")
  global ψ_0 = @time vumps(H, ψ_1; vumps_kwargs...)
end

Sz = [expect(ψ_0, "Sz", n) for n in 1:N]
energy_infinite = expect(ψ_0, H)

Nfinite = 100
sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
Hfinite = MPO(model, sfinite; J=J, J₂=J₂, h=h)
ψfinite = randomMPS(sfinite, initstate; linkdims=10)
@show flux(ψfinite)
sweeps = Sweeps(15)
setmaxdim!(sweeps, maxdim)
setcutoff!(sweeps, cutoff)
energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps)

nfinite = Nfinite ÷ 2
Sz_finite = expect(ψfinite, "Sz")[nfinite:(nfinite + N - 1)]

@show (
  energy_infinite,
  energy_finite_total / Nfinite,
  reference(model, Observable("energy"); J=J, h=h, J₂=J₂),
)

@show (Sz, Sz_finite)
