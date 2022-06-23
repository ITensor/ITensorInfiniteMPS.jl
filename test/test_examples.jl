using Test
using Suppressor

examples_dir = joinpath(@__DIR__, "..", "examples", "vumps")
@testset "examples" begin
  @testset "$file" for file in readdir(examples_dir; join=true)
    if endswith(file, ".jl")
      println("Running $file")
      @suppress include(file)
    end
  end
end
