using ITensors, ITensorMPS
using ITensorInfiniteMPS
using ITensors: δ, dag, prime
using Test
using Random

base_path = joinpath(pkgdir(ITensorInfiniteMPS), "examples", "vumps", "src")
src_files = ["vumps_subspace_expansion.jl", "entropy.jl"]
for f in src_files
  include(joinpath(base_path, f))
end

"""
Build the transverse field Ising model with exponentially decaying interactions in the
thermodynamic limit 𝑁 → ∞,

    H = -J ∑_j ∑_{n>0} λ^(n-1) σˣ_j σˣ_{j+n} - hz ∑_j σᶻ_j

This is Eq. (C15) of Zauner-Stijl et al., arXiv:1701.07035. `λ = 0` reduces to the
nearest neighbour transverse field Ising model.
"""
function InfiniteExpHTFI(
  sites::CelledVector{<:Index},
  λ, # exponential decay base
  J, # interaction kinetic (tunneling) coupling
  hz; # interaction strength of the transverse field
  kwargs...,
)
  if abs(λ) >= 1.0
    throw(
      ArgumentError("cannot implement exponential decay with base larger than 1, λ = $(λ)!")
    )
  end

  link_dimension = 3

  N = length(sites)

  EType = eltype(union(λ, J, hz))

  linkindices = CelledVector(
    if hasqns(sites.data)
      [Index([QN("SzParity", 1, 2) => 1], "Link,c=1,n=$n") for n in 1:N]
    else
      [Index(1, "Link,c=1,n=$(n)") for n in 1:N]
    end,
  )

  mpos = [Matrix{ITensor}(undef, link_dimension, link_dimension) for i in 1:N]
  for n in 1:N
    # define local matrix Hmat with empty tensors as local operators
    Hmat = fill(
      ITensor(EType, dag(sites[n]), prime(sites[n])), link_dimension, link_dimension
    )
    # left link index ll with daggered QN conserving direction (if applicable)
    ll = dag(linkindices[n - 1])
    # right link index rl
    rl = linkindices[n]

    # add both Identities as netral elements in the MPO
    # replace all known tensors from empty to known interactions
    Hmat[1, 1] = op("Id", sites[n])
    Hmat[3, 3] = op("Id", sites[n])
    # local nearest neighbour and exp. decaying interaction terms
    Hmat[2, 1] = op("X", sites[n])
    if !iszero(λ)
      Hmat[2, 2] = op("Id", sites[n]) * λ  # λ Id,  on the diagonal
    end
    Hmat[3, 2] = op("X", sites[n]) * -J # Jxx σˣ
    if !iszero(hz)
      Hmat[3, 1] = op("Z", sites[n]) * -hz # hz σᶻ
    end

    # add all missing links that connect the interaction
    # operators in the unit cell
    Hmat[2, 1] = setelt(ll[1]) * Hmat[2, 1]
    Hmat[1, 2] = Hmat[1, 2] * setelt(rl[1])
    Hmat[2, 2] = setelt(ll[1]) * Hmat[2, 2] * setelt(rl[1])
    Hmat[3, 2] = setelt(rl[1]) * Hmat[3, 2]
    Hmat[2, 3] = setelt(ll[1]) * Hmat[2, 3]
    mpos[n] = Hmat
  end

  return InfiniteBlockMPO(mpos, sites.translator)
end

# sanity check if I can construct the MPO for the normal TFI correctly
"""
Build the transverse field Ising model with nearest neighbour interactions in the
thermodynamic limit 𝑁 → ∞,

    H = -J ∑_j σˣ_j σˣ_{j+1} - hz ∑_j σᶻ_j

Same as `InfiniteExpHTFI` with `λ = 0`, but written without the (then unused) diagonal
block, so that it also exercises the `T^{aa} = 0` branch of the environment solver.
"""
function InfiniteHTFI(
  sites::CelledVector{<:Index},
  J, # interaction kinetic (tunneling) coupling
  hz; # interaction strength of the transverse field
  kwargs...,
)
  link_dimension = 3

  N = length(sites)

  EType = eltype(union(J, hz))

  linkindices = CelledVector(
    if hasqns(sites.data)
      [Index([QN("SzParity", 1, 2) => 1], "Link,c=1,n=$n") for n in 1:N]
    else
      [Index(1, "Link,c=1,n=$(n)") for n in 1:N]
    end,
  )

  mpos = [Matrix{ITensor}(undef, link_dimension, link_dimension) for i in 1:N]
  for n in 1:N
    # define local matrix Hmat with empty tensors as local operators
    Hmat = fill(
      ITensor(EType, dag(sites[n]), prime(sites[n])), link_dimension, link_dimension
    )
    # left link index ll with daggered QN conserving direction (if applicable)
    ll = dag(linkindices[n - 1])
    # right link index rl
    rl = linkindices[n]

    # add both Identities as netral elements in the MPO
    # replace all known tensors from empty to known interactions
    Hmat[1, 1] = op("Id", sites[n])
    Hmat[3, 3] = op("Id", sites[n])
    # local nearest neighbour and exp. decaying interaction terms
    Hmat[2, 1] = op("X", sites[n])
    Hmat[3, 2] = op("X", sites[n]) * -J # Jxx σˣ
    if !iszero(hz)
      Hmat[3, 1] = op("Z", sites[n]) * -hz # hz σᶻ
    end

    # add all missing links that connect the
    # interaction operators in the unit cell
    Hmat[2, 1] = setelt(ll[1]) * Hmat[2, 1]
    Hmat[3, 2] = setelt(rl[1]) * Hmat[3, 2]
    mpos[n] = Hmat
  end
  return InfiniteBlockMPO(mpos, sites.translator)
end

"""
    reference_energy(ψ, λ, J, hz)

The energy per unit cell of `ψ` for the Hamiltonian of `InfiniteExpHTFI`, obtained by
summing the exponential tail explicitly with powers of the MPS transfer matrix,

    e = ∑_j [ -hz ⟨σᶻ_j⟩ - J ∑_{n>0} λ^(n-1) ⟨σˣ_j σˣ_{j+n}⟩ ]

No linear solver is involved, so this is an independent check of the environments
returned by `left_environment` / `right_environment`.
"""
function reference_energy(ψ::InfiniteCanonicalMPS, λ, J, hz)
  ψ′ = dag(ψ)'
  l = linkinds(ITensorInfiniteMPS.only, ψ.AL)
  l′ = linkinds(ITensorInfiniteMPS.only, ψ′.AL)
  r = linkinds(ITensorInfiniteMPS.only, ψ.AR)
  s = siteinds(ITensorInfiniteMPS.only, ψ)
  δˡ(n) = δ(l[n], l′[n])
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˢ(n) = δ(dag(s[n]), prime(s[n]))
  X(n) = op("X", s[n])
  Z(n) = op("Z", s[n])

  # truncate the geometric series once λ^n drops below machine precision
  nmax = iszero(λ) ? 1 : ceil(Int, log(1e-17) / log(abs(λ)))

  e = zero(promote_type(ITensorMPS.promote_itensor_eltype(ψ), typeof(λ)))
  for j in 1:nsites(ψ)
    e += -hz * (δˡ(j - 1) * ψ.AL[j] * Z(j) * ψ′.AL[j] * ψ.C[j] * δʳ(j) * ψ′.C[j])[]
    v = δˡ(j - 1) * ψ.AL[j] * X(j) * ψ′.AL[j]
    for n in 1:nmax
      k = j + n
      w = v * ψ.AL[k] * X(k) * ψ′.AL[k]
      e += -J * λ^(n - 1) * (w * ψ.C[k] * δʳ(k) * ψ′.C[k])[]
      v = v * ψ.AL[k] * δˢ(k) * ψ′.AL[k]
    end
  end
  return e
end

# A random, non symmetric infinite MPS in mixed canonical form. When `s` carries QNs the
# link indices get both parity sectors, so that the state is not a product state and the
# environments are genuinely block sparse.
function random_inf_canonical_mps(s::CelledVector, χ::Int)
  N = length(s)
  ls = CelledVector(
    if hasqns(s.data)
      [
        Index(
          [QN("SzParity", 0, 2) => χ, QN("SzParity", 1, 2) => χ];
          dir=ITensors.Out,
          tags="Link,c=1,l=$n",
        ) for n in 1:N
      ]
    else
      [Index(χ, "Link,c=1,l=$n") for n in 1:N]
    end,
  )
  A = [random_itensor(ls[n - 1], s[n], dag(ls[n])) for n in 1:N]
  return orthogonalize(InfiniteMPS(A, translator(s)), :)
end

function energy_local(ψ1, ψ2, h::ITensor)
  ϕ = ψ1 * ψ2
  return (noprime(ϕ * h) * dag(ϕ))[]
end
energy_local(ψ1, ψ2, h::MPO) = energy_local(ψ1, ψ2, prod(h))

local_expect(ψ, o) = (noprime(ψ * op(o, filterinds(ψ, "Site")...)) * dag(ψ))[]

@testset "environments of an exponentially decaying InfiniteBlockMPO" begin
  # Regression test for the fixed point equations of appendix C 2 of arXiv:1701.07035.
  # A diagonal MPO block λ⋅Id with |λ| < 1 must be solved with Eq. (C21)/(C22), i.e.
  # (Lₐ|[1 - λ T] = (Y_Lₐ|, with no energy subtraction and no |R)(1| projector. Using the
  # regularized Eq. (C25a)/(C25b) there instead shifts (Lₐ| by (Y_Lₐ|R)/(1-λ) (1|, which
  # leaks into the energy through (C17) whenever ⟨σˣ⟩ ≠ 0.
  Random.seed!(1234)

  J, hz = 1.0, 1.1

  @testset "λ = $λ, unit cell = $N, conserve_qns = $conserve_qns" for λ in
                                                                     [0.0, 0.6, -0.5],
    N in [1, 2],
    conserve_qns in [false, true]

    s = infsiteinds("S=1/2", N; initstate=(n -> "↑"), conserve_szparity=conserve_qns)
    ψ = random_inf_canonical_mps(s, 4)
    H = InfiniteExpHTFI(s, λ, J, hz)

    e_ref = reference_energy(ψ, λ, J, hz)
    _, e_left = ITensorInfiniteMPS.left_environment(H, ψ; tol=1e-14)
    _, e_right = ITensorInfiniteMPS.right_environment(H, ψ; tol=1e-14)

    if conserve_qns
      # σˣ is parity odd, so ⟨σˣ⟩ vanishes by symmetry and the spurious shift of the
      # buggy solve is invisible here. The two point function ⟨σˣ_j σˣ_{j+n}⟩ is parity
      # even though, so the exponential channel is still exercised: check that the tail
      # actually moves the energy.
      @test abs(local_expect(ψ.AL[1] * ψ.C[1], "X")) < 1e-12
    else
      # a random dense state is deep in the symmetry broken sector, which is what makes
      # the spurious shift observable
      @test abs(local_expect(ψ.AL[1] * ψ.C[1], "X")) > 0.1
    end
    if !iszero(λ)
      @test abs(e_ref - reference_energy(ψ, 0.0, J, hz)) > 1e-3
    end

    @test e_left ≈ e_ref atol = 1e-10
    @test e_right ≈ e_ref atol = 1e-10
  end

  @testset "diverging geometric series is rejected" begin
    s = infsiteinds("S=1/2", 1; initstate=(n -> "↑"))
    @test_throws ArgumentError InfiniteExpHTFI(s, 1.0, J, hz)
  end
end

@testset "VUMPS with a nearest neighbour InfiniteBlockMPO" begin
  Random.seed!(1234)

  maxdim = 16 # Maximum bond dimension
  cutoff = 1e-10 # Singular value cutoff when increasing the bond dimension
  max_vumps_iters = 100 # Maximum number of iterations of the VUMPS/TDVP algorithm
  tol = 1e-8 # Precision error tolerance for outer loop of VUMPS or TDVP
  outer_iters = 4 # Number of times to increase the bond dimension
  solver_tol = (x -> x / 100) # Tolerance for the local solver
  multisite_update_alg = "parallel"

  nsite = 2 # Number of sites in the unit cell
  initstate(n) = "↑"

  @testset "conserve_qns = $conserve_qns" for conserve_qns in [true, false]
    s = infsiteinds("S=1/2", nsite; initstate, conserve_szparity=conserve_qns)
    ψ = InfMPS(s, initstate)

    @test norm(
      contract(ψ.AL[1:nsite]..., ψ.C[nsite]) - contract(ψ.C[0], ψ.AR[1:nsite]...)
    ) ≈ 0 atol = 1e-10

    J = 1
    hz = 1.1

    H_test = InfiniteHTFI(s, J, hz)
    H_test0 = InfiniteExpHTFI(s, 0.0, J, hz)

    vumps_kwargs = (
      tol=tol,
      maxiter=max_vumps_iters,
      solver_tol=solver_tol,
      multisite_update_alg=multisite_update_alg,
      outputlevel=0,
    )
    subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

    H_ref = InfiniteSum{MPO}(Model("ising"), s; J=J, h=hz)

    ψ_ref = vumps_subspace_expansion(
      H_ref, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs
    )
    ψ_test_NN = vumps_subspace_expansion(
      H_test, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs
    )
    ψ_test0_NN = vumps_subspace_expansion(
      H_test0, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs
    )

    E_ref = energy_local(ψ_ref.AL[1], ψ_ref.AL[2] * ψ_ref.C[2], H_ref[(1, 2)])

    _, energy_test = ITensorInfiniteMPS.left_environment(H_test, ψ_test_NN; tol=1e-10)
    _, energy_test0 = ITensorInfiniteMPS.left_environment(H_test0, ψ_test0_NN; tol=1e-10)

    energy_test /= nsite
    energy_test0 /= nsite

    @test energy_test ≈ E_ref rtol = 1e-6
    @test energy_test0 ≈ E_ref rtol = 1e-6

    E_exact = reference(Model("ising"), Observable("energy"); h=hz, J)

    @test isapprox(E_ref, E_exact; rtol=1e-6)
    @test isapprox(energy_test, E_exact; rtol=1e-6)
    @test isapprox(energy_test0, E_exact; rtol=1e-6)

    @test isapprox(
      local_expect(ψ_ref.AL[1] * ψ_ref.C[1], "Z"),
      local_expect(ψ_test_NN.AL[1] * ψ_test_NN.C[1], "Z");
      atol=1e-7,
    )
    @test isapprox(
      abs(local_expect(ψ_ref.AL[1] * ψ_ref.C[1], "X")),
      abs(local_expect(ψ_test_NN.AL[1] * ψ_test_NN.C[1], "X"));
      atol=1e-7,
    )
    @test isapprox(
      local_expect(ψ_ref.AL[1] * ψ_ref.C[1], "Z"),
      local_expect(ψ_test0_NN.AL[1] * ψ_test0_NN.C[1], "Z");
      atol=1e-7,
    )
    @test isapprox(
      abs(local_expect(ψ_ref.AL[1] * ψ_ref.C[1], "X")),
      abs(local_expect(ψ_test0_NN.AL[1] * ψ_test0_NN.C[1], "X"));
      atol=1e-7,
    )

    @test isapprox(entropy(ψ_ref, 1), entropy(ψ_test_NN, 1); rtol=1e-6)
    @test isapprox(entropy(ψ_ref, 1), entropy(ψ_test0_NN, 1); rtol=1e-6)
  end
end

@testset "VUMPS with an exponentially decaying InfiniteBlockMPO" begin
  Random.seed!(1234)

  nsite = 2
  initstate(n) = "↑"
  # the reference value below was computed at λ = 0.4, so the model has to match
  λ = 0.4
  J = 1
  hz = 1.1

  s = infsiteinds("S=1/2", nsite; initstate, conserve_szparity=false)
  ψ = InfMPS(s, initstate)

  H_test_exp = InfiniteExpHTFI(s, λ, J, hz)

  vumps_kwargs = (
    tol=1e-8,
    maxiter=100,
    solver_tol=(x -> x / 100),
    multisite_update_alg="parallel",
    outputlevel=0,
  )
  subspace_expansion_kwargs = (cutoff=1e-10, maxdim=16)

  ψ_exp = vumps_subspace_expansion(
    H_test_exp, ψ; outer_iters=4, subspace_expansion_kwargs, vumps_kwargs
  )

  _, energy_exp_L = ITensorInfiniteMPS.left_environment(H_test_exp, ψ_exp; tol=1e-12)
  _, energy_exp_R = ITensorInfiniteMPS.right_environment(H_test_exp, ψ_exp; tol=1e-12)

  # the environments must agree with the explicit geometric sum for the converged state
  @test energy_exp_L ≈ reference_energy(ψ_exp, λ, J, hz) atol = 1e-8
  @test energy_exp_L ≈ energy_exp_R atol = 1e-8

  # computed with MPSKit at χ=16, λ=0.4, Jₓₓ=1.0, hz=1.1
  E_gs_λ04_J1_h11_MPSKit = -1.8497463013720623
  @test energy_exp_L / nsite ≈ E_gs_λ04_J1_h11_MPSKit atol = 1e-6
end
