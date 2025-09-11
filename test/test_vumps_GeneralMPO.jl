using ITensors, ITensorMPS
using ITensorInfiniteMPS

base_path = joinpath(pkgdir(ITensorInfiniteMPS), "examples", "vumps", "src")
src_files = ["vumps_subspace_expansion.jl", "entropy.jl"]
for f in src_files
  include(joinpath(base_path, f))
end


"""
build the transverse field Ising model with exponential interactions in the thermodynamic limit 𝑁 → ∞
use convention H = -J ∑_{n,m=−∞}^∞ σˣₙ σˣₘ ⋅ λ^(-|n-m-1|)  - hz ∑_{n=−∞}^∞ σᶻₙ
with 
"""
function InfiniteExpHTFI( 
  sites::CelledVector{<:Index},
  λ, # exponential decay base
  J, # interaction kinetic (tunneling) coupling
  hz; # interaction strength of the transverse field
  kwargs...
)
  if abs(λ) > 1.0
    throw(ArgumentError("cannot implement exponential decay with base larger than 1, λ = $(λ)!"))
  end

  link_dimension = 3

  N = length(sites)

  EType = eltype(union(λ, J, hz))

  linkindices = CelledVector(
    hasqns(sites.data) ?
    [Index([QN("SzParity",1,2) => 1], "Link,c=1,n=$n") for n in 1:N] : [Index(1, "Link,c=1,n=$(n)") for n in 1:N]
  )
  
  mpos = [Matrix{ITensor}(undef, link_dimension, link_dimension) for i in 1:N]
  for n in 1:N
    # define local matrix Hmat with empty tensors as local operators
    Hmat = fill(
      ITensor(EType, dag(sites[n]), prime(sites[n])), link_dimension, link_dimension
    )
    # left link index ll with daggered QN conserving direction (if applicable)
    ll = dag(linkindices[n-1])
    # right link index rl
    rl = linkindices[n]

    # add both Identities as netral elements in the MPO
    # replace all known tensors from empty to known interactions
    Hmat[1,1] = op("Id", sites[n])
    Hmat[3,3] = op("Id", sites[n])
    # local nearest neighbour and exp. decaying interaction terms
    Hmat[2,1] = op("X", sites[n])
    if !iszero(λ)
      Hmat[2,2] = op("Id", sites[n]) * λ  # λ Id,  on the diagonal
    end
    Hmat[3,2] = op("X", sites[n]) * -J # Jxx σˣ
    if !iszero(hz)
      Hmat[3,1] = op("Z", sites[n]) * -hz # hz σᶻ
    end

    # add all missing links that connect the interaction
    # operators in the unit cell
    Hmat[2,1] = setelt(ll[1]) * Hmat[2,1]
    Hmat[1,2] = Hmat[1,2] * setelt(rl[1])
    Hmat[2,2] = setelt(ll[1]) * Hmat[2,2] * setelt(rl[1])
    Hmat[3,2] = setelt(rl[1]) * Hmat[3,2]
    Hmat[2,3] = setelt(ll[1]) * Hmat[2,3]
    mpos[n] = Hmat
  end

  return InfiniteBlockMPO(mpos, sites.translator)
end


# sanity check if I can construct the MPO for the normal TFI correctly
"""
build the transverse field Ising model with nearest neighbour interactions in the thermodynamic limit 𝑁 → ∞
use convention H = -J ∑_{n=−∞}^∞ σˣₙ σˣₙ₊₁  - hz ∑_{n=−∞}^∞ σᶻₙ
with 
"""
function InfiniteHTFI( 
  sites::CelledVector{<:Index},
  kinetic_coupling,
  hz; # interaction kinetic (tunneling) coupling
  kwargs...
)
  if abs(λ) > 1.0
    throw(ArgumentError("cannot implement exponential decay with base larger than 1, λ = $(λ)!"))
  end
  link_dimension = 3

  N = length(sites)

  EType = eltype(union(J, hz))

  linkindices = CelledVector(
    hasqns(sites.data) ?
    [Index([QN("SzParity",1,2) => 1], "Link,c=1,n=$n") for n in 1:N] : [Index(1, "Link,c=1,n=$(n)") for n in 1:N]
  )
  
  mpos = [Matrix{ITensor}(undef, link_dimension, link_dimension) for i in 1:N]
  for n in 1:N
    # define local matrix Hmat with empty tensors as local operators
    Hmat = fill(
      ITensor(EType, dag(sites[n]), prime(sites[n])), link_dimension, link_dimension
    )
    # left link index ll with daggered QN conserving direction (if applicable)
    ll = dag(linkindices[n-1])
    # right link index rl
    rl = linkindices[n]

    # add both Identities as netral elements in the MPO
    # replace all known tensors from empty to known interactions
    Hmat[1,1] = op("Id", sites[n])
    Hmat[3,3] = op("Id", sites[n])
    # local nearest neighbour and exp. decaying interaction terms
    Hmat[2,1] = op("X", sites[n])
    Hmat[3,2] = op("X", sites[n]) * -J # Jxx σˣ
    if !iszero(hz)
      Hmat[3,1] = op("Z", sites[n]) * -hz # hz σᶻ
    end

    # add all missing links that connect the 
    # interaction operators in the unit cell
    Hmat[2,1] = setelt(ll[1]) * Hmat[2,1]
    # Hmat[1,2] = Hmat[1,2] * setelt(rl[1])
    Hmat[3,2] = setelt(rl[1]) * Hmat[3,2]
    # Hmat[2,3] = setelt(ll[1]) * Hmat[2,3]
    mpos[n] = Hmat
  end
  return InfiniteBlockMPO(mpos, sites.translator)
end

function expect_two_site(ψ::InfiniteCanonicalMPS, h::ITensor, n1n2)
  n1, n2 = n1n2
  ϕ = ψ.AL[n1] * ψ.AL[n2] * ψ.C[n2]
  return inner(ϕ, apply(h, ϕ))
end


function expect_two_site(ψ::InfiniteCanonicalMPS, h::MPO, n1n2)
  return expect_two_site(ψ, contract(h), n1n2)
end


function energy_local(ψ1, ψ2, h::ITensor)
  ϕ = ψ1 * ψ2
  return (noprime(ϕ * h) * dag(ϕ))[]
end

energy_local(ψ1, ψ2, h::MPO) = energy_local(ψ1, ψ2, prod(h))
# Check translational invariance
function ITensorMPS.expect(ψ, o)
  return (noprime(ψ * op(o, filterinds(ψ, "Site")...)) * dag(ψ))[]
end


maxdim = 16 # Maximum bond dimension
cutoff = 1e-10 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 100 # Maximum number of iterations of the VUMPS/TDVP algorithm at a fixed bond dimension
tol = 1e-8 # Precision error tolerance for outer loop of VUMPS or TDVP
outer_iters = 4 # Number of times to increase the bond dimension
time_step = -Inf # -Inf corresponds to VUMPS, finite time_step corresponds to TDVP
solver_tol = (x -> x / 100) # Tolerance for the local solver (eigsolve in VUMPS and exponentiate in TDVP)
multisite_update_alg = "parallel" # Choose between ["sequential", "parallel"]. Only parallel works with TDVP.
conserve_qns = true # Whether or not to conserve spin parity


nsite = 2 # Number of sites in the unit cell
initstate(n) = "↑"
s = infsiteinds("S=1/2", nsite; initstate, conserve_szparity=conserve_qns)

ψ = InfMPS(s, initstate)

@show norm(contract(ψ.AL[1:nsite]..., ψ.C[nsite]) - contract(ψ.C[0], ψ.AR[1:nsite]...))

# J  = -0.25
# hz = 2.0
# λ  = 0.4

J  = 1
hz = 1.0
λ  = 0.4

# H_test = InfiniteHTFI(s,J,hz)
H_test = InfiniteHTFI(s,J,hz)
H_test0 = InfiniteExpHTFI(s,0.0,J,hz)
H_test_exp = InfiniteExpHTFI(s,λ,J,hz)

vumps_kwargs = (
  tol=tol,
  maxiter=max_vumps_iters,
  solver_tol=solver_tol,
  multisite_update_alg=multisite_update_alg,
)

subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

H_ref = InfiniteSum{MPO}(Model("ising"), s; J=J, h=hz)


ψ_ref  = vumps_subspace_expansion(H_ref, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs)

ψ_test_NN = vumps_subspace_expansion(H_test, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs)
ψ_test0_NN = vumps_subspace_expansion(H_test0, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs)


E_ref = energy_local(ψ_ref.AL[1], ψ_ref.AL[2] * ψ_ref.C[2], H_ref[(1, 2)])

L_test, energy_test = ITensorInfiniteMPS.left_environment(H_test, ψ_test_NN; tol=1e-10);
L_test0, energy_test0 = ITensorInfiniteMPS.left_environment(H_test0, ψ_test0_NN; tol=1e-10);

energy_test  /= length(ψ_test_NN)
energy_test0 /= length(ψ_test0_NN)

@test energy_test ≈ E_ref
@test energy_test0 ≈ E_ref

@show E_ref        + 4/π
@show energy_test  + 4/π
@show energy_test0 + 4/π


# energy_local(ψ_NN.AL[1], ψ_NN.AL[2] * ψ.C[2], H_test[(1, 2)])
# energy_local(ψ_test_NN,ψ_test_NN,H_test)

function ITensorMPS.expect(ψ, o)
  return (noprime(ψ * op(o, filterinds(ψ, "Site")...)) * dag(ψ))[]
end


@test isapprox(expect(ψ_ref.AL[1] * ψ_ref.C[1], "Z"),  expect(ψ_test_NN.AL[1] * ψ_test_NN.C[1], "Z"); rtol=1e-6)
@test isapprox(expect(ψ_ref.AL[1] * ψ_ref.C[1], "X"),  expect(ψ_test_NN.AL[1] * ψ_test_NN.C[1], "X"); atol=1e-6)

@test isapprox(expect(ψ_ref.AL[1] * ψ_ref.C[1], "Z"),  expect(ψ_test0_NN.AL[1] * ψ_test0_NN.C[1], "Z"); rtol=1e-6)
@test isapprox(expect(ψ_ref.AL[1] * ψ_ref.C[1], "X"),  expect(ψ_test0_NN.AL[1] * ψ_test0_NN.C[1], "X"); rtol=1e-6)

entropy(ψ_ref,1)
entropy(ψ_test_NN,1)
entropy(ψ_test0_NN,1)

@test isapprox(entropy(ψ_ref,1), entropy(ψ_test_NN,1); rtol=1e-6)
@test isapprox(entropy(ψ_ref,1), entropy(ψ_test0_NN,1); rtol=1e-6)

##################
#### actually exponential interactions
#################
ψ_exp = vumps_subspace_expansion(H_test_exp, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs)

L_test, energy_test_exp = ITensorInfiniteMPS.left_environment(H_test_exp, ψ_exp; tol=1e-10);


E_gs_04_MPSKit = -1.8177300191088515 # computed at χ=16, λ=0.4, Jₓₓ=1.0, hz=1.0
E_gs_04_ITensor = energy_test_exp / length(ψ_exp)
@test E_gs_04_MPSKit ≈ E_gs_04_ITensor

