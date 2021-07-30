using ITensors
using ITensorInfiniteMPS
using Test

@testset "ITensorInfiniteMPS.jl" begin
  @testset "Mixed canonical gauge" begin
    N = 10
    s = siteinds("S=1/2", N; conserve_szparity=true)
    χ = 6
    @test iseven(χ)
    space = (("SzParity", 1, 2) => χ ÷ 2) ⊕ (("SzParity", 0, 2) => χ ÷ 2)
    ψ = InfiniteMPS(ComplexF64, s; space=space)
    randn!.(ψ)

    ψ = orthogonalize(ψ, :)
    @test prod(ψ.AL[1:N]) * ψ.C[N] ≈ ψ.C[0] * prod(ψ.AR[1:N])
  end
end
