using ITensors, ITensorMPS
using ITensorInfiniteMPS
using ITensorInfiniteMPS.HDF5
using Test

@testset "HDF5 Read and Write" begin
  N = 2
  model = Model("ising")
  initstate(n) = "↑"
  s = infsiteinds("S=1/2", N; initstate)
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
