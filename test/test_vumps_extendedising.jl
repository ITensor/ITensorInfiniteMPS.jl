using ITensors
using ITensorInfiniteMPS
using Test
using Random

@testset "vumps_extended_ising" begin
  Random.seed!(1234)

  model = Model"ising_extended"()
  model_kwargs = (J=1.0, h=1.1, J₂=0.2)
  initstate(n) = "↑"

  function space_shifted(::Model"ising_extended", q̃sz; conserve_qns=true)
    if conserve_qns
      return [QN("SzParity", 1 - q̃sz, 2) => 1, QN("SzParity", 0 - q̃sz, 2) => 1]
    else
      return [QN() => 2]
    end
  end

  # Compare to DMRG
  Nfinite = 100
  sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
  Hfinite = MPO(model, sfinite; model_kwargs...)
  ψfinite = randomMPS(sfinite, initstate)
  sweeps = Sweeps(20)
  setmaxdim!(sweeps, 30)
  setcutoff!(sweeps, 1E-10)
  energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps; outputlevel=0)
  Szs_finite = expect(ψfinite, "Sz")

  function energy(ψ, h, n)
    ϕ = ψ[n] * ψ[n + 1] * ψ[n + 2]
    return (noprime(ϕ * h) * dag(ϕ))[]
  end

  nfinite = Nfinite ÷ 2
  hnfinite = ITensor(
    model, sfinite[nfinite], sfinite[nfinite + 1], sfinite[nfinite + 2]; model_kwargs...
  )
  orthogonalize!(ψfinite, nfinite)
  energy_finite = energy(ψfinite, hnfinite, nfinite)

  cutoff = 1e-8
  maxdim = 100
  tol = 1e-8
  maxiter = 20
  outer_iters = 4
  for multisite_update_alg in ["sequential", "parallel"],
    conserve_qns in [true, false],
    nsites in [1, 2],
    time_step in [-Inf, -0.5]

    vumps_kwargs = (
      multisite_update_alg=multisite_update_alg,
      tol=tol,
      maxiter=maxiter,
      outputlevel=0,
      time_step=time_step,
    )
    subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

    space_ = fill(space_shifted(model, 1; conserve_qns=conserve_qns), nsites)
    s = infsiteinds("S=1/2", nsites; space=space_)
    H = InfiniteSum{MPO}(model, s; model_kwargs...)
    ψ = InfMPS(s, initstate)

    # Check translational invariance
    @test contract(ψ.AL[1:nsites]..., ψ.C[nsites]) ≈ contract(ψ.C[0], ψ.AR[1:nsites]...)

    ψ = tdvp(H, ψ; vumps_kwargs...)
    for _ in 1:outer_iters
      ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
      ψ = tdvp(H, ψ; vumps_kwargs...)
    end
    # Check translational invariance
    @test norm(
      contract(ψ.AL[1:nsites]..., ψ.C[nsites]) - contract(ψ.C[0], ψ.AR[1:nsites]...)
    ) ≈ 0 atol = 1e-5

    energy_infinite = expect(ψ, H)
    Szs_infinite = [expect(ψ, "Sz", n) for n in 1:nsites]

    @test energy_finite ≈ sum(energy_infinite) / nsites rtol = 1e-4
    @test Szs_finite[nfinite:(nfinite + nsites - 1)] ≈ Szs_infinite rtol = 1e-3
  end
end

# XXX: orthogonalize is broken right now
## @testset "ITensorInfiniteMPS.jl" begin
##   @testset "Mixed canonical gauge" begin
##     N = 10
##     s = siteinds("S=1/2", N; conserve_szparity=true)
##     χ = 6
##     @test iseven(χ)
##     space = (("SzParity", 1, 2) => χ ÷ 2) ⊕ (("SzParity", 0, 2) => χ ÷ 2)
##     ψ = InfiniteMPS(ComplexF64, s; space=space)
##     randn!.(ψ)
##
##     ψ = orthogonalize(ψ, :)
##     @test prod(ψ.AL[1:N]) * ψ.C[N] ≈ ψ.C[0] * prod(ψ.AR[1:N])
##   end
## end
