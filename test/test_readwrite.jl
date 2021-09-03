using ITensors
using ITensorInfiniteMPS
using HDF5
using Test

@testset "HDF5 Read and Write" begin
  function space_shifted(::Model"ising", q̃sz)
    return [QN("SzParity", 1 - q̃sz, 2) => 1, QN("SzParity", 0 - q̃sz, 2) => 1]
  end

  N = 2
  model = Model"ising"()
  space_ = fill(space_shifted(model, 0), N)
  s = infsiteinds("S=1/2", N; space=space_)
  initstate(n) = "↑"
  ψ = InfMPS(s, initstate)

  @testset "InfiniteCanonicalMPS" begin
    fo = h5open("data.h5", "w")
    write(fo, "ψ", ψ)
    close(fo)

    fi = h5open("data.h5", "r")
    ψr = read(fi, "ψ", InfiniteCanonicalMPS)
    close(fi)

    @test ψ.AL.data == ψr.AL.data
    @test ψ.C.data == ψr.C.data
    @test ψ.AR.data == ψr.AR.data
  end

  # Clean up the test hdf5 file
  rm("data.h5"; force=true)
end
