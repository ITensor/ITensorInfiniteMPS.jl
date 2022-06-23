using Test
using Suppressor

@testset "examples" begin
  @testset "$file" for file in joinpath(readdir(@__DIR__), "..", "examples", "vumps")
    if endswith(file, ".jl")
      println("Running $file")
      @suppress include(file)
    end
  end
end
