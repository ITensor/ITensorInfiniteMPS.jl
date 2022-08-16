using ITensors
using ITensorInfiniteMPS
using Random

# Parameters
cutoff = 1e-8
maxdim = 100
tol = 1e-8
maxiter = 20
outer_iters = 3

N = 2
model = Model("ising")

initstate(n) = "↑"
s = infsiteinds("S=1/2", N; initstate)
ψ = InfMPS(s, initstate)

model_params = (J=-1.0, h=0.9)
vumps_kwargs = (multisite_update_alg="sequential", tol=tol, maxiter=maxiter)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

H = InfiniteSum{ITensor}(model, s; model_params...)
# Alternate steps of running VUMPS and increasing the bond dimension
ψ1 = vumps(H, ψ; vumps_kwargs...)
for _ in 1:outer_iters
  global ψ1 = subspace_expansion(ψ1, H; subspace_expansion_kwargs...)
  global ψ1 = vumps(H, ψ1; vumps_kwargs...)
end

Hmpo = InfiniteMPOMatrix(model, s; model_params...)
# Alternate steps of running VUMPS and increasing the bond dimension
ψ2 = vumps(Hmpo, ψ; vumps_kwargs...)
for _ in 1:outer_iters
  global ψ2 = subspace_expansion(ψ2, Hmpo; subspace_expansion_kwargs...)
  global ψ2 = vumps(Hmpo, ψ2; vumps_kwargs...)
end

SzSz = prod(op("Sz", s[1]) * op("Sz", s[2]))
