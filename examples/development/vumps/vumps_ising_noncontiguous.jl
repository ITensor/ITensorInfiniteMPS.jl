using ITensors, ITensorMPS
using ITensorInfiniteMPS

##############################################################################
# VUMPS parameters
#

maxdim = 64 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 100 # Maximum number of iterations of the VUMPS algorithm at each bond dimension
vumps_tol = 1e-6
outer_iters = 4 # Number of times to increase the bond dimension

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#
N = 2# Number of sites in the unit cell
J = 1.0
h = 1.0;

initstate(n) = "↑"
s = infsiteinds("S=1/2", N; initstate)
ψ = InfMPS(s, initstate);

model = Model("ising");
H = InfiniteSum{MPO}(model, s; J=J, h=h);
#to test the case where the range is larger than the unit cell size
for x in 1:N
  temp = MPO(3)
  temp[1] = H[x][1]
  temp[2] = H[x][2]
  temp[3] = delta(prime(s[x + 3]), dag(s[x + 3]))
  H.data[x] = temp
end

@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...));

vumps_kwargs = (tol=vumps_tol, maxiter=max_vumps_iters)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)
ψ_0 = vumps(H, ψ; vumps_kwargs...)

for j in 1:outer_iters
  println("\nIncrease bond dimension")
  ψ_1 = subspace_expansion(ψ_0, H; subspace_expansion_kwargs...)
  println("Run VUMPS with new bond dimension")
  global ψ_0 = vumps(H, ψ_1; vumps_kwargs...)
end

Sz = [expect(ψ_0, "Sz", n) for n in 1:N]
energy_infinite = expect(ψ_0, H)

Nfinite = 100
sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
Hfinite = MPO(model, sfinite; J=J, h=h)
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
  reference(model, Observable("energy"); J=J, h=h),
)

@show (Sz, Sz_finite)
