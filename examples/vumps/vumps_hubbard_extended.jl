using ITensors
using ITensorInfiniteMPS
using Random

Random.seed!(1234)

include("models.jl")

# Number of sites in the unit cell
N = 2

χ1 = [QN("Sz",0) => 1,
      QN("Sz",1) => 1,
      QN("Sz",-1) => 1,
      QN("Sz",0) => 1]
χ2 = [QN("Sz",0) => 1,
      QN("Sz",1) => 1,
      QN("Sz",-1) => 1,
      QN("Sz",0) => 1]
space = [χ1, χ2]
s = _siteinds("Electron", N; space=space)
t = 1.0
U = 1.0
V = 0.5
model = Model(:hubbard)

# Form the Hamiltonian
Σ∞h = InfiniteITensorSum(model, s; t=t, U=U, V=V)

d = 3
χ = [QN("Sz",0) => d,
     QN("Sz",1) => d,
     QN("Sz",2) => d,
     QN("Sz",-1) => d,
     QN("Sz",-2) => d]
space = χ1
ψ = InfiniteMPS(s; space=space)
flux = QN("Sz",0)
for n in 1:N
  for b in nzblocks(flux, inds(ψ[n]))
    # Need to overwrite the tensor, since modifying
    # it in-place doesn' work when not modifying the storage directly.
    ψ[n] = insertblock!(ψ[n], b)
  end
end
randn!.(ψ)
ψ = orthogonalize(ψ, :)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

ψ = vumps(Σ∞h, ψ; niter=5)

# Check translational invariance
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

#
# Bond dimension increase
#

ψ.AL[1] * ψ.C[1] * ψ.AR[2] * Σ∞h[(1, 2)]

using LinearAlgebra
using ITensors.NDTensors

# Reshape into an order-2 ITensor
combine(::Order{2}, T::ITensor, inds::Index...) = combine(Order(2), T, inds)

function combine(::Order{2}, T::ITensor, inds)
  left_inds = commoninds(T, inds)
  right_inds = uniqueinds(T, inds)
  return combine(T, left_inds, right_inds)
end

function combine(T::ITensor, left_inds, right_inds)
  CL = combiner(left_inds)
  CR = combiner(right_inds)
  M = (T * CL) * CR
  return M, CL, CR
end

function LinearAlgebra.nullspace(::Order{2}, M::ITensor, left_inds, right_inds; kwargs...)
  @assert order(M) == 2
  M = permute(M, left_inds..., right_inds...)
  Mᵀ = tensor(M)
  Nᵀ = nullspace(Mᵀ; kwargs...)
  return itensor(Nᵀ)
end

function LinearAlgebra.nullspace(T::ITensor, inds...; kwargs...)
  M, CL, CR = combine(Order(2), T, inds...)
  @assert order(M) == 2
  cL = commoninds(M, CL)
  cR = commoninds(M, CR)
  N₂ = nullspace(Order(2), M, cL, cR; kwargs...)
  return N₂ * dag(CL) * dag(CR)
end

nullspace(ψ.AL[1], noncommoninds(ψ.AL[1], ψ.C[1]))



