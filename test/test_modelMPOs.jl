using ITensors, ITensorMPS
using ITensorInfiniteMPS
using Test

@testset verbose = true "Heisenberg Model Test" begin
  cell_widths = [2, 3, 4, 5]
  @testset "cell_width=$width" for width in cell_widths
    model = Model("heisenberg")
    os = ITensorInfiniteMPS.opsum_finite(model, width;)

    connections = []
    for term in os
      push!(connections, sort(ITensors.sites(term)))
    end
    for i in 1:(width - 1)
      @test [i, i + 1] ∈ connections
    end
  end
end

@testset verbose = true "Heisenberg2D Model Test" begin
  widths = [2, 3, 4, 5, 6]
  cell_widths = [2, 3, 4, 5]
  @testset "cell_width=$cell_width width=$width yperiodic=$yperiodic" for cell_width in
                                                                          cell_widths,
    width in widths,
    yperiodic in [false, true]

    model = Model("heisenberg2D")

    os = ITensorInfiniteMPS.opsum_finite(model, cell_width * width; width, yperiodic)

    connections = []
    for term in os
      push!(connections, ITensors.sites(term))
    end
    for col in 1:(cell_width - 1)
      for row in 1:(width - 1)
        i = (col - 1) * width + row
        @test [i, i + 1] ∈ connections
        @test [i, i + width] ∈ connections
      end
      # The wrap around bond of the cylinder. `unit_cell_terms` generates it as
      # ("S+", width, "S-", mod(width, width) + 1), so the sites come out descending,
      # unlike the other bonds of this testset.
      if yperiodic && width > 2
        i = (col - 1) * width + 1
        @test [i + width - 1, i] ∈ connections
      end
    end
    # the above forgets the last horizontal bond
    for col in 1:(cell_width - 1)
      i = (col - 1) * width + width
      @test [i, i + width] ∈ connections
    end
  end
end

@testset verbose = true "Ising Model Test" begin
  cell_widths = [2, 3, 4, 5]
  @testset "cell_width=$width" for width in cell_widths
    model = Model("ising")
    os = ITensorInfiniteMPS.opsum_finite(model, width;)

    connections = []
    for term in os
      push!(connections, sort(ITensors.sites(term)))
    end
    for i in 1:(width - 1)
      @test [i, i + 1] ∈ connections
      @test [i] ∈ connections
    end
    @test [width] ∈ connections
  end
end

@testset verbose = true "Hubbard Model Test" begin
  t, U = 1.0, 4.0
  cell_widths = [2, 3, 4, 5]
  @testset "cell_width=$width" for width in cell_widths
    model = Model("hubbard")
    os = ITensorInfiniteMPS.opsum_finite(model, width; t, U)

    connections = []
    for term in os
      push!(connections, ITensors.sites(term))
    end
    for i in 1:(width - 1)
      @test [i, i + 1] ∈ connections
      @test [i + 1, i] ∈ connections
      @test [i] ∈ connections
    end
    @test [width] ∈ connections
  end
end

nothing
