using ITensors
using ITensorInfiniteMPS

include("models.jl")
include("infinitecanonicalmps.jl")

s = infsiteinds("Electron", 2; conserve_sz=true)
initstate(n) = isodd(n) ? "↑" : "↓"
ψ = UniformMPS(s, initstate)

