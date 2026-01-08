using ITensors, ITensorMPS
using ITensorInfiniteMPS
using Test
using Random

function expect_two_site(ψ1::ITensor, ψ2::ITensor, h::ITensor)
  ϕ = ψ1 * ψ2
  return inner(ϕ, apply(h, ϕ))
end

expect_two_site(ψ1::ITensor, ψ2::ITensor, h::MPO) = expect_two_site(ψ1, ψ2, contract(h))

function expect_one_site(ψ::ITensor, o::String)
  return inner(ψ, apply(op(o, filterinds(ψ, "Site")...), ψ))
end

@testset "vumps" begin
  Random.seed!(1234)

  model = Model("ising")
  model_kwargs = (J=1.0, h=1.1)

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
  ψfinite = random_mps(sfinite, initstate)
  nsweeps = 20
  energy_finite_total, ψfinite = dmrg(
    Hfinite, ψfinite; nsweeps, maxdim=10, cutoff=1e-10, outputlevel=0
  )

  @testset "VUMPS/TDVP with: multisite_update_alg = $multisite_update_alg, conserve_qns = $conserve_qns, nsites = $nsites, time_step = $time_step, localham_type = $localham_type" for multisite_update_alg in
                                                                                                                                                                                       [
      "sequential", "parallel"
    ],
    conserve_qns in [true, false], nsites in [1, 2, 3, 4], time_step in [-Inf, -0.5],
    localham_type in [ITensor, MPO]

    if (localham_type == ITensor) && (nsites > 2)
      # ITensor VUMPS currently broken for unit cells > 2
      continue
    end

    if nsites > 1 && isodd(nsites) && conserve_qns
      # Odd site greater than 1 not commensurate with conserving parity
      continue
    end

    @show localham_type

    s = infsiteinds("S=1/2", nsites; initstate, conserve_szparity=conserve_qns)
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

    nfinite = Nfinite ÷ 2
    hnfinite1 = ITensor(model, sfinite[nfinite], sfinite[nfinite + 1]; model_kwargs...)
    hnfinite2 = ITensor(model, sfinite[nfinite + 1], sfinite[nfinite + 2]; model_kwargs...)

    orthogonalize!(ψfinite, nfinite)
    energy1_finite = expect_two_site(ψfinite[nfinite], ψfinite[nfinite + 1], hnfinite1)

    orthogonalize!(ψfinite, nfinite + 1)
    energy2_finite = expect_two_site(ψfinite[nfinite + 1], ψfinite[nfinite + 2], hnfinite2)

    energy1_infinite = expect_two_site(ψ.AL[1], ψ.AL[2] * ψ.C[2], H[(1, 2)])
    energy2_infinite = expect_two_site(ψ.AL[2], ψ.AL[3] * ψ.C[3], H[(2, 3)])

    orthogonalize!(ψfinite, nfinite)
    Sz1_finite = expect_one_site(ψfinite[nfinite], "Sz")

    orthogonalize!(ψfinite, nfinite + 1)
    Sz2_finite = expect_one_site(ψfinite[nfinite + 1], "Sz")

    Sz1_infinite = expect_one_site(ψ.AL[1] * ψ.C[1], "Sz")
    Sz2_infinite = expect_one_site(ψ.AL[2] * ψ.C[2], "Sz")

    @test energy1_finite ≈ energy1_infinite rtol = 1e-4
    @test energy2_finite ≈ energy2_infinite rtol = 1e-4
    @test Sz1_finite ≈ Sz2_finite rtol = 1e-5
    @test Sz1_infinite ≈ Sz2_infinite rtol = 1e-5
  end
end

@testset "vumps_ising_translator" begin
  Random.seed!(1234)

  model = Model("ising")
  model_kwargs = (J=1.0, h=1.1)

  #Not a correct and valid Hamiltonian need to think about a good test (or just import Laughlin 13)
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
  ψfinite = random_mps(sfinite, initstate)
  nsweeps = 20
  energy_finite_total, ψfinite = dmrg(
    Hfinite, ψfinite; nsweeps, maxdim=10, cutoff=1e-10, outputlevel=0
  )

  multisite_update_alg = "sequential"
  conserve_qns = true
  time_step = -Inf

  for nsite in 1:3
    s_bis = infsiteinds("S=1/2", nsite; initstate)
    s = infsiteinds("S=1/2", nsite; initstate, translator=temp_translatecell)
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

    nfinite = Nfinite ÷ 2
    hnfinite1 = ITensor(model, sfinite[nfinite], sfinite[nfinite + 1]; model_kwargs...)
    hnfinite2 = ITensor(model, sfinite[nfinite + 1], sfinite[nfinite + 2]; model_kwargs...)

    orthogonalize!(ψfinite, nfinite)
    energy1_finite = expect_two_site(ψfinite[nfinite], ψfinite[nfinite + 1], hnfinite1)

    orthogonalize!(ψfinite, nfinite + 1)
    energy2_finite = expect_two_site(ψfinite[nfinite + 1], ψfinite[nfinite + 2], hnfinite2)

    energy1_infinite = expect_two_site(ψ.AL[1], ψ.AL[2] * ψ.C[2], prod(H[(1, 2)]))
    energy2_infinite = expect_two_site(ψ.AL[2], ψ.AL[3] * ψ.C[3], prod(H[(2, 3)]))

    orthogonalize!(ψfinite, nfinite)
    Sz1_finite = expect_one_site(ψfinite[nfinite], "Sz")

    orthogonalize!(ψfinite, nfinite + 1)
    Sz2_finite = expect_one_site(ψfinite[nfinite + 1], "Sz")

    Sz1_infinite = expect_one_site(ψ.AL[1] * ψ.C[1], "Sz")
    Sz2_infinite = expect_one_site(ψ.AL[2] * ψ.C[2], "Sz")

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
