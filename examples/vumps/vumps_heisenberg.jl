using ITensors
using ITensorInfiniteMPS

N = 2

heisenberg_space_shift(q̃nf, q̃sz) = [QN("Sz", 1 - q̃sz) => 1, QN("Sz", -1 - q̃sz) => 1]

heisenberg_space = fill(heisenberg_space_shift(1, 0), N)
s = infsiteinds("S=1/2", N; space=heisenberg_space)
initstate(n) = isodd(n) ? "↑" : "↓"
ψ = InfMPS(s, initstate)

model = Model(:heisenberg)

# Form the Hamiltonian
H = InfiniteITensorSum(model, s)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

cutoff = 1e-8
maxdim = 100
environment_iterations = 300
niter = 200
outer_iters = 6
vumps_kwargs = (environment_iterations=environment_iterations, niter=niter)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

println("\nRun VUMPS on initial product state, unit cell size $N")
ψ = vumps(H, ψ; vumps_kwargs...)

for _ in 1:outer_iters
  println("\nIncrease bond dimension")
  global ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
  println("Run VUMPS with new bond dimension")
  global ψ = vumps(H, ψ; vumps_kwargs...)
end

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

function ITensors.expect(ψ::InfiniteCanonicalMPS, o, n)
  return (noprime(ψ.AL[n] * ψ.C[n] * op(o, s[n])) * dag(ψ.AL[n] * ψ.C[n]))[]
end

function expect_two_site(ψ::InfiniteCanonicalMPS, h::ITensor, n1n2)
  n1, n2 = n1n2
  ϕ = ψ.AL[n1] * ψ.AL[n2] * ψ.C[n2]
  return (noprime(ϕ * h) * dag(ϕ))[]
end

Sz = [expect(ψ, "Sz", n) for n in 1:N]

bs = [(1, 2), (2, 3)]
energy_infinite = map(b -> expect_two_site(ψ, H[b], b), bs)

@show energy_infinite
@show Sz

