using Test
using Suppressor

# Each example is included into its own throwaway module. The examples are scripts that
# assign at top level, so including them directly into `Main` leaks every one of their
# globals into the shared namespace that `runtests.jl` includes all test files into,
# where a later test file can silently pick them up.
function include_isolated(file)
  mod = Module(gensym(basename(file)))
  # scripts refer to `include`/`@__DIR__` relative to themselves
  Base.eval(mod, :(include(path) = Base.include($mod, path)))
  return Base.include(mod, file)
end

examples_dir = joinpath(@__DIR__, "..", "examples", "vumps")
@testset "examples" begin
  @testset "$file" for file in readdir(examples_dir; join=true)
    if endswith(file, ".jl")
      println("Running $file")
      @suppress include_isolated(file)
    end
  end
end
