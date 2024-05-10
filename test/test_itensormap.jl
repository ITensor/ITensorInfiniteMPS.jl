using ITensors, ITensorMPS
using ITensorInfiniteMPS
using Test
using Random

@testset "ITensorMap" begin
  i = Index(2)
  A1 = randomITensor(i'', dag(i))
  A2 = randomITensor(i', dag(i''))
  B1 = randomITensor(i'', dag(i))
  B2 = randomITensor(i', dag(i''))
  c1 = 2.3
  c2 = 3.4

  MA = ITensorMap([A1, A2])
  MB = ITensorMap([B1, B2])
  v = randomITensor(i)

  @test MA(v) ≈ noprime(A2 * A1 * v)
  @test MB(v) ≈ noprime(B2 * B1 * v)

  c1⨉MA = c1 * MA
  c2⨉MB = c2 * MB
  @test c1⨉MA isa ITensorMap
  @test c2⨉MB isa ITensorMap
  @test c1⨉MA(v) ≈ noprime(c1 * A2 * A1 * v)
  @test c2⨉MB(v) ≈ noprime(c2 * B2 * B1 * v)

  MA⁺MB = MA + MB
  @test MA⁺MB isa ITensorInfiniteMPS.ITensorMapSum
  @test MA⁺MB(v) ≈ noprime(A2 * A1 * v + B2 * B1 * v)

  c1MA⁺c2MB = c1 * MA + c2 * MB
  @test c1MA⁺c2MB isa ITensorInfiniteMPS.ITensorMapSum
  @test c1MA⁺c2MB(v) ≈ noprime(c1 * A2 * A1 * v + c2 * B2 * B1 * v)

  ⁻MA = -MA
  @test ⁻MA isa ITensorMap
  @test ⁻MA(v) ≈ noprime(-A2 * A1 * v)

  MA⁻MB = MA - MB
  @test MA⁻MB isa ITensorInfiniteMPS.ITensorMapSum
  @test MA⁻MB(v) ≈ noprime(A2 * A1 * v - B2 * B1 * v)
end
