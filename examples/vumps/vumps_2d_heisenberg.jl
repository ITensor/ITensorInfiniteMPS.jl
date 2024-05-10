using ITensorInfiniteMPS
using ITensors, ITensorMPS

include(
  joinpath(
    pkgdir(ITensorInfiniteMPS), "examples", "vumps", "src", "vumps_subspace_expansion.jl"
  ),
)

##############################################################################
# VUMPS parameters
#

maxdim = 256 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 100 # Maximum number of iterations of the VUMPS algorithm at each bond dimension
vumps_tol = 1e-4
conserve_qns = true
solver_tol = (x -> x / 10)
outer_iters = 10 # Number of times to increase the bond dimension
width = 4

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#

N = width # Number of sites in the unit cell

initstate(n) = isodd(n) ? "↑" : "↓"
s = infsiteinds("S=1/2", N; conserve_qns, initstate)
ψ = InfMPS(s, initstate)

function ITensorInfiniteMPS.unit_cell_terms(::Model"heisenberg2D"; width)
  opsum = OpSum()
  for i in 1:width
    # Vertical
    opsum += -0.5, "S+", i, "S-", mod(i + 1, width)
    opsum += -0.5, "S-", i, "S+", mod(i + 1, width)
    opsum += "Sz", i, "Sz", mod(i + 1, width)
    # Horizontal
    opsum += -0.5, "S+", i, "S-", i + width
    opsum += -0.5, "S-", i, "S+", i + width
    opsum += "Sz", i, "Sz", i + width
  end
  return opsum
end
model = Model("heisenberg2D")

# Form the Hamiltonian
H = InfiniteSum{MPO}(model, s; width)

# Check translational invariance
# println("\nCheck translation invariance of the initial VUMPS state")
# @show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

vumps_kwargs = (tol=vumps_tol, maxiter=max_vumps_iters, solver_tol)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

ψ = vumps_subspace_expansion(H, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs)

# Check translational invariance
# println()
# println("==============================================================")
# println()
# println("\nCheck translation invariance of the final VUMPS state")
# @show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

# Sz = [expect(ψ, "Sz", n) for n in 1:N]

energy_infinite = expect(ψ, H)
@show energy_infinite

## using JLD2
## jldsave("infmps.jld2"; ψ)
