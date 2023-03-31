using ITensors
using ITensorInfiniteMPS
using Test
using Random

function expect_three_site(ψ::MPS, h::ITensor, n::Int)
  ϕ = ψ[n] * ψ[n + 1] * ψ[n + 2]
  return inner(ϕ, (apply(h, ϕ)))
end

#Ref time is 21.6s with negligible compilation time
@testset "vumpsmpo_ising" begin
  Random.seed!(1234)

  model = Model("ising")
  model_kwargs = (J=1.0, h=1.2)

  # VUMPS arguments
  cutoff = 1e-8
  maxdim = 20
  tol = 1e-8
  maxiter = 50
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
  Szs_finite = expect(ψfinite, "Sz")

  nfinite = Nfinite ÷ 2
  hnfinite = ITensor(model, sfinite[nfinite], sfinite[nfinite + 1]; model_kwargs...)
  orthogonalize!(ψfinite, nfinite)
  energy_finite = expect_three_site(ψfinite, hnfinite, nfinite)

  @testset "VUMPS/TDVP with: multisite_update_alg = $multisite_update_alg, conserve_qns = $conserve_qns, nsites = $nsites" for multisite_update_alg in
                                                                                                                               [
      "sequential", "parallel"
    ],
    conserve_qns in [true, false],
    nsites in [1, 2],
    time_step in [-Inf]

    vumps_kwargs = (
      multisite_update_alg=multisite_update_alg,
      tol=tol,
      maxiter=maxiter,
      outputlevel=0,
      time_step=time_step,
    )
    subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

    s = infsiteinds("S=1/2", nsites; initstate, conserve_szparity=conserve_qns)
    ψ = InfMPS(s, initstate)

    Hmpo = InfiniteMPOMatrix(model, s; model_kwargs...)
    # Alternate steps of running VUMPS and increasing the bond dimension
    ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
    for _ in 1:outer_iters
      println("Subspace expansion")
      ψ = @time subspace_expansion(ψ, Hmpo; subspace_expansion_kwargs...)
      println("TDVP")
      ψ = @time tdvp(Hmpo, ψ; vumps_kwargs...)
    end

    @test norm(
      contract(ψ.AL[1:nsites]..., ψ.C[nsites]) - contract(ψ.C[0], ψ.AR[1:nsites]...)
    ) ≈ 0 atol = 1e-5
    #@test contract(ψ.AL[1:nsites]..., ψ.C[nsites]) ≈ contract(ψ.C[0], ψ.AR[1:nsites]...)

    H = InfiniteSum{MPO}(model, s; model_kwargs...)
    energy_infinite = expect(ψ, H)
    Szs_infinite = [expect(ψ, "Sz", n) for n in 1:nsites]

    @test energy_finite ≈ sum(energy_infinite) / nsites rtol = 1e-4
    @test Szs_finite[nfinite:(nfinite + nsites - 1)] ≈ Szs_infinite rtol = 1e-3
  end
end

@testset "vumpsmpo_extendedising" begin
  Random.seed!(1234)

  model = Model"ising_extended"()
  model_kwargs = (J=1.0, h=1.1, J₂=0.2)

  # VUMPS arguments
  cutoff = 1e-8
  maxdim = 20
  tol = 1e-8
  maxiter = 20
  outer_iters = 4

  initstate(n) = "↑"

  # DMRG arguments
  Nfinite = 100
  sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
  Hfinite = MPO(model, sfinite; model_kwargs...)
  ψfinite = randomMPS(sfinite, initstate)
  nsweeps = 20
  energy_finite_total, ψfinite = dmrg(
    Hfinite, ψfinite; nsweeps, maxdims=10, cutoff=1e-10, outputlevel=0
  )
  Szs_finite = expect(ψfinite, "Sz")

  nfinite = Nfinite ÷ 2
  hnfinite = ITensor(
    model, sfinite[nfinite], sfinite[nfinite + 1], sfinite[nfinite + 2]; model_kwargs...
  )
  orthogonalize!(ψfinite, nfinite)
  energy_finite = expect_three_site(ψfinite, hnfinite, nfinite)

  for multisite_update_alg in ["sequential"],
    conserve_qns in [true, false],
    nsites in [1, 2],
    time_step in [-Inf]

    vumps_kwargs = (
      multisite_update_alg=multisite_update_alg,
      tol=tol,
      maxiter=maxiter,
      outputlevel=0,
      time_step=time_step,
    )
    subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

    s = infsiteinds("S=1/2", nsites; conserve_szparity=conserve_qns, initstate)
    ψ = InfMPS(s, initstate)

    Hmpo = InfiniteMPOMatrix(model, s; model_kwargs...)
    # Alternate steps of running VUMPS and increasing the bond dimension
    ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
    for _ in 1:outer_iters
      ψ = subspace_expansion(ψ, Hmpo; subspace_expansion_kwargs...)
      ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
    end

    @test norm(
      contract(ψ.AL[1:nsites]..., ψ.C[nsites]) - contract(ψ.C[0], ψ.AR[1:nsites]...)
    ) ≈ 0 atol = 1e-5

    H = InfiniteSum{MPO}(model, s; model_kwargs...)
    energy_infinite = expect(ψ, H)
    Szs_infinite = [expect(ψ, "Sz", n) for n in 1:nsites]

    @test energy_finite ≈ sum(energy_infinite) / nsites rtol = 1e-4
    @test Szs_finite[nfinite:(nfinite + nsites - 1)] ≈ Szs_infinite rtol = 1e-3
  end
end

@testset "vumpsmpo_extendedising_translator" begin
  Random.seed!(1234)

  model = Model("ising_extended")
  model_kwargs = (J=1.0, h=1.1, J₂=0.2)

  # VUMPS arguments
  cutoff = 1e-8
  maxdim = 20
  tol = 1e-8
  maxiter = 20
  outer_iters = 4

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
  Szs_finite = expect(ψfinite, "Sz")

  nfinite = Nfinite ÷ 2
  hnfinite = ITensor(
    model, sfinite[nfinite], sfinite[nfinite + 1], sfinite[nfinite + 2]; model_kwargs...
  )
  orthogonalize!(ψfinite, nfinite)
  energy_finite = expect_three_site(ψfinite, hnfinite, nfinite)

  temp_translatecell(i::Index, n::Integer) = ITensorInfiniteMPS.translatecelltags(i, n)

  for multisite_update_alg in ["sequential"],
    conserve_qns in [true, false],
    nsite in [1, 2, 3],
    time_step in [-Inf]

    if nsite > 1 && isodd(nsite) && conserve_qns
      # Parity conservation not commensurate with odd number of sites.
      continue
    end

    vumps_kwargs = (
      multisite_update_alg=multisite_update_alg,
      tol=tol,
      maxiter=maxiter,
      outputlevel=0,
      time_step=time_step,
    )
    subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

    s_bis = infsiteinds("S=1/2", nsite; conserve_szparity=conserve_qns, initstate)
    s = infsiteinds(
      "S=1/2",
      nsite;
      conserve_szparity=conserve_qns,
      initstate,
      translator=temp_translatecell,
    )
    ψ = InfMPS(s, initstate)

    Hmpo = InfiniteMPOMatrix(model, s; model_kwargs...)
    # Alternate steps of running VUMPS and increasing the bond dimension
    ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
    for _ in 1:outer_iters
      ψ = subspace_expansion(ψ, Hmpo; subspace_expansion_kwargs...)
      ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
    end

    @test norm(
      contract(ψ.AL[1:nsite]..., ψ.C[nsite]) - contract(ψ.C[0], ψ.AR[1:nsite]...)
    ) ≈ 0 atol = 1e-5

    H = InfiniteSum{MPO}(model, s; model_kwargs...)
    energy_infinite = expect(ψ, H)
    Szs_infinite = [expect(ψ, "Sz", n) for n in 1:nsite]

    @test energy_finite ≈ sum(energy_infinite) / nsite rtol = 1e-4
    @test Szs_finite[nfinite:(nfinite + nsite - 1)] ≈ Szs_infinite rtol = 1e-3

    #@test tags(s[nsite + 1]) == tags(s_bis[1 + 2nsite])
    @test ITensorInfiniteMPS.translator(ψ) == temp_translatecell
    @test ITensorInfiniteMPS.translator(s) == temp_translatecell
    @test ITensorInfiniteMPS.translator(Hmpo) == temp_translatecell
  end
end
