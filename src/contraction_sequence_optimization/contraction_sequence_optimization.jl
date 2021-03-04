
module ContractionSequenceOptimization

  using
    ITensors,
    IterTools

  export
    optimize_contraction_sequence

  include("utils.jl")
  include("three_tensors.jl")
  include("depth_first_constructive.jl")
  include("breadth_first_constructive.jl")

  # The integer type of the dimensions and costs.
  # Needs to be large to avoid overflow.
  const DimT = UInt128

  function optimize_contraction_sequence(T)
    if length(T) == 1
      return 0, Any[1]
    elseif length(T) == 2
      return 0, Any[1, 2]
    elseif length(T) == 3
      return optimize_contraction_sequence(T[1], T[2], T[3])
    elseif length(T) ≤ 6
      return depth_first_constructive(T)
    end
    return breadth_first_constructive(T)
  end

end # module ContractionSequenceOptimization

