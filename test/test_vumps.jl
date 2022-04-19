using ITensors
using ITensorInfiniteMPS
using Test
using Random

@testset "vumps" begin
  Random.seed!(1234)

  model = Model"ising"()
  model_kwargs = (J=1.0, h=1.1)

  function space_shifted(::Model"ising", q̃sz; conserve_qns=true)
    if conserve_qns
      return [QN("SzParity", 1 - q̃sz, 2) => 1, QN("SzParity", 0 - q̃sz, 2) => 1]
    else
      return [QN() => 2]
    end
  end

  # VUMPS arguments
  cutoff = 1e-8
  maxdim = 20
  tol = 1e-8
  maxiter = 20
  outer_iters = 3

  initstate(n) = "↑"

  # DMRG arguments
  Nfinite = 100
  sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
  Hfinite = MPO(model, sfinite; model_kwargs...)
  ψfinite = randomMPS(sfinite, initstate)
  sweeps = Sweeps(20)
  setmaxdim!(sweeps, 10)
  setcutoff!(sweeps, 1E-10)
  energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps; outputlevel=0)

  @testset "VUMPS/TDVP with: multisite_update_alg = $multisite_update_alg, conserve_qns = $conserve_qns, nsites = $nsites, time_step = $time_step, localham_type = $localham_type" for multisite_update_alg in
                                                                                                                                                                                       [
      "sequential", "parallel"
    ],
    conserve_qns in [true, false],
    nsites in [1, 2, 3, 4],
    time_step in [-Inf, -0.5],
    localham_type in [ITensor, MPO]

    # ITensor VUMPS currently broken for unit cells > 2
    if (localham_type == ITensor) && (nsites > 2)
      continue
    end

    @show localham_type

    #if conserve_qns
    # Flux density is inconsistent for odd unit cells
    #  isodd(nsites) && continue
    #end

    space_ = fill(space_shifted(model, 1; conserve_qns=conserve_qns), nsites)
    s = infsiteinds("S=1/2", nsites; space=space_)
    ψ = InfMPS(s, initstate)

    # Form the Hamiltonian
    H = InfiniteSum{localham_type}(model, s; model_kwargs...)

    # Check translational invariance
    @test contract(ψ.AL[1:nsites]..., ψ.C[nsites]) ≈ contract(ψ.C[0], ψ.AR[1:nsites]...)

    vumps_kwargs = (
      multisite_update_alg=multisite_update_alg,
      tol=tol,
      maxiter=maxiter,
      outputlevel=0,
      time_step=time_step,
    )
    subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

    # Alternate steps of running VUMPS and increasing the bond dimension
    for _ in 1:outer_iters
      ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
      ψ = @time tdvp(H, ψ; vumps_kwargs...)
    end

    # Check translational invariance
    ## @test contract(ψ.AL[1:nsites]..., ψ.C[nsites]) ≈ contract(ψ.C[0], ψ.AR[1:nsites]...) rtol =
    ##   1e-6

    function energy(ψ1, ψ2, h::ITensor)
      ϕ = ψ1 * ψ2
      return (noprime(ϕ * h) * dag(ϕ))[]
    end

    energy(ψ1, ψ2, h::MPO) = energy(ψ1, ψ2, prod(h))

    function expect(ψ, o)
      return (noprime(ψ * op(o, filterinds(ψ, "Site")...)) * dag(ψ))[]
    end

    nfinite = Nfinite ÷ 2
    hnfinite1 = ITensor(model, sfinite[nfinite], sfinite[nfinite + 1]; model_kwargs...)
    hnfinite2 = ITensor(model, sfinite[nfinite + 1], sfinite[nfinite + 2]; model_kwargs...)

    orthogonalize!(ψfinite, nfinite)
    energy1_finite = energy(ψfinite[nfinite], ψfinite[nfinite + 1], hnfinite1)

    orthogonalize!(ψfinite, nfinite + 1)
    energy2_finite = energy(ψfinite[nfinite + 1], ψfinite[nfinite + 2], hnfinite2)

    energy1_infinite = energy(ψ.AL[1], ψ.AL[2] * ψ.C[2], H[(1, 2)])
    energy2_infinite = energy(ψ.AL[2], ψ.AL[3] * ψ.C[3], H[(2, 3)])

    orthogonalize!(ψfinite, nfinite)
    Sz1_finite = expect(ψfinite[nfinite], "Sz")

    orthogonalize!(ψfinite, nfinite + 1)
    Sz2_finite = expect(ψfinite[nfinite + 1], "Sz")

    Sz1_infinite = expect(ψ.AL[1] * ψ.C[1], "Sz")
    Sz2_infinite = expect(ψ.AL[2] * ψ.C[2], "Sz")

    @test energy1_finite ≈ energy1_infinite rtol = 1e-4
    @test energy2_finite ≈ energy2_infinite rtol = 1e-4
    @test Sz1_finite ≈ Sz2_finite rtol = 1e-5
    @test Sz1_infinite ≈ Sz2_infinite rtol = 1e-5
  end
end

@testset "vumps_ising_translator" begin
  Random.seed!(1234)

  model = Model"ising"()
  model_kwargs = (J=1.0, h=1.1)

  function space_shifted(::Model"ising", q̃sz; conserve_qns=true)
    if conserve_qns
      return [QN("SzParity", 1 - q̃sz, 2) => 1, QN("SzParity", 0 - q̃sz, 2) => 1]
    else
      return [QN() => 2]
    end
  end

  #Not a correct and valid Hamiltonian #nned to think about a good test (or just import Laughlin 13)
  temp_translatecell(i::Index, n::Integer) = ITensorInfiniteMPS.translatecelltags(i, n)

  # VUMPS arguments
  cutoff = 1e-8
  maxdim = 20
  tol = 1e-8
  maxiter = 20
  outer_iters = 3

  initstate(n) = "↑"

  # DMRG arguments
  Nfinite = 100
  sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
  Hfinite = MPO(model, sfinite; model_kwargs...)
  ψfinite = randomMPS(sfinite, initstate)
  sweeps = Sweeps(20)
  setmaxdim!(sweeps, 10)
  setcutoff!(sweeps, 1E-10)
  energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps; outputlevel=0)

  multisite_update_alg = "sequential"
  conserve_qns = true
  time_step = -Inf

  for nsite in 1:3
    space_ = fill(space_shifted(model, 1; conserve_qns=conserve_qns), nsite)
    s_bis = infsiteinds("S=1/2", nsite; space=space_)
    s = infsiteinds("S=1/2", nsite; space=space_, translator=temp_translatecell)
    ψ = InfMPS(s, initstate)

    # Form the Hamiltonian
    H = InfiniteSum{MPO}(model, s; model_kwargs...)

    # Check translational invariance
    @test contract(ψ.AL[1:nsite]..., ψ.C[nsite]) ≈ contract(ψ.C[0], ψ.AR[1:nsite]...)

    vumps_kwargs = (
      multisite_update_alg=multisite_update_alg,
      tol=tol,
      maxiter=maxiter,
      outputlevel=0,
      time_step=time_step,
    )
    subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

    # Alternate steps of running VUMPS and increasing the bond dimension
    for _ in 1:outer_iters
      ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
      ψ = tdvp(H, ψ; vumps_kwargs...)
    end

    # Check translational invariance
    ## @test contract(ψ.AL[1:nsites]..., ψ.C[nsites]) ≈ contract(ψ.C[0], ψ.AR[1:nsites]...) rtol =
    ##   1e-6

    function energy(ψ1, ψ2, h)
      ϕ = ψ1 * ψ2
      return (noprime(ϕ * h) * dag(ϕ))[]
    end

    function expect(ψ, o)
      return (noprime(ψ * op(o, filterinds(ψ, "Site")...)) * dag(ψ))[]
    end

    nfinite = Nfinite ÷ 2
    hnfinite1 = ITensor(model, sfinite[nfinite], sfinite[nfinite + 1]; model_kwargs...)
    hnfinite2 = ITensor(model, sfinite[nfinite + 1], sfinite[nfinite + 2]; model_kwargs...)

    orthogonalize!(ψfinite, nfinite)
    energy1_finite = energy(ψfinite[nfinite], ψfinite[nfinite + 1], hnfinite1)

    orthogonalize!(ψfinite, nfinite + 1)
    energy2_finite = energy(ψfinite[nfinite + 1], ψfinite[nfinite + 2], hnfinite2)

    energy1_infinite = energy(ψ.AL[1], ψ.AL[2] * ψ.C[2], prod(H[(1, 2)]))
    energy2_infinite = energy(ψ.AL[2], ψ.AL[3] * ψ.C[3], prod(H[(2, 3)]))

    orthogonalize!(ψfinite, nfinite)
    Sz1_finite = expect(ψfinite[nfinite], "Sz")

    orthogonalize!(ψfinite, nfinite + 1)
    Sz2_finite = expect(ψfinite[nfinite + 1], "Sz")

    Sz1_infinite = expect(ψ.AL[1] * ψ.C[1], "Sz")
    Sz2_infinite = expect(ψ.AL[2] * ψ.C[2], "Sz")

    @test energy1_finite ≈ energy1_infinite rtol = 1e-4
    @test energy2_finite ≈ energy2_infinite rtol = 1e-4
    @test Sz1_finite ≈ Sz2_finite rtol = 1e-5
    @test Sz1_infinite ≈ Sz2_infinite rtol = 1e-5

    #@test tags(s[nsite + 1]) == tags(s_bis[1 + nsite])
    @test ITensorInfiniteMPS.translator(ψ) == temp_translatecell
    @test ITensorInfiniteMPS.translator(s) == temp_translatecell
    @test ITensorInfiniteMPS.translator(H) == temp_translatecell
  end
end
