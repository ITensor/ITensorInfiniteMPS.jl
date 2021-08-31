using Test

@testset "ITensorInfiniteMPS.jl" begin
  @testset "$file" for file in readdir(@__DIR__)
    if startswith(file, "test_") && endswith(file, ".jl")
      println("Running $file")
      include(file)
    end
  end
end

