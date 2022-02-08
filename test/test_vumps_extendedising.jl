using ITensors
using ITensorInfiniteMPS
using Test
using Random

@testset "vumps_extended_ising" begin
  Random.seed!(1234)

  N = 2
  model = Model"ising_extended"()
  model_kwargs = (J=1.0, h=1.1, J₂=0.2)

  function space_shifted(q̃sz)
    return [QN("SzParity", 1 - q̃sz, 2) => 1, QN("SzParity", 0 - q̃sz, 2) => 1]
  end

  space_ = fill(space_shifted(1), N)
  s = infsiteinds("S=1/2", N; space=space_)
  initstate(n) = "↑"
  # Form the Hamiltonian
  H = InfiniteITensorSum(model, s; model_kwargs...)

  # Compare to DMRG
  Nfinite = 100
  sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
  Hfinite = MPO(model, sfinite; model_kwargs...)
  ψfinite = randomMPS(sfinite, initstate)
  sweeps = Sweeps(20)
  setmaxdim!(sweeps, 30)
  setcutoff!(sweeps, 1E-10)
  energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps; outputlevel=0)

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
  for multisite_update_alg in ["sequential", "parallel"]
    vumps_kwargs = (
      multisite_update_alg=multisite_update_alg, tol=tol, maxiter=maxiter, outputlevel=0
    )
    subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

    #
    # Alternate steps of running VUMPS and increasing the bond dimension
    ψ = InfMPS(s, initstate)

    # Check translational invariance
    @test contract(ψ.AL[1:N]..., ψ.C[N]) ≈ contract(ψ.C[0], ψ.AR[1:N]...)

    ψ = vumps(H, ψ; vumps_kwargs...)
    for _ in 1:outer_iters
      ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
      ψ = vumps(H, ψ; vumps_kwargs...)
    end

    # Check translational invariance
    @test norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...)) ≈ 0 atol =
      1e-5

    energy_infinite = expect(ψ, H)
    Sz1_finite, Sz2_finite = expect(ψfinite, "Sz")[(Nfinite ÷ 2):(Nfinite ÷ 2 + 1)]
    Sz1_infinite, Sz2_infinite = [expect(ψ, "Sz", n) for n in 1:N]

    @test energy_finite ≈ sum(energy_infinite) / N rtol = 1e-4
    @test Sz1_finite ≈ Sz1_infinite rtol = 1e-3
    @test Sz1_finite ≈ Sz2_finite rtol = 1e-5
    @test Sz1_infinite ≈ Sz2_infinite rtol = 1e-5
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
