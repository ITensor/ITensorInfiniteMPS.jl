using ITensors
using ITensorInfiniteMPS

struct Model{model} end
Model(model::Symbol) = Model{model}()

struct InfiniteITensorSum <: ITensorInfiniteMPS.AbstractInfiniteMPS
  data::CelledVector{ITensor}
end
InfiniteITensorSum(N::Int) = InfiniteITensorSum(Vector{ITensor}(undef, N))
InfiniteITensorSum(data::Vector{ITensor}) = InfiniteITensorSum(CelledVector(data))
Base.getindex(l::InfiniteITensorSum, n::Integer) = ITensors.data(l)[n]

# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ
function ITensors.MPO(::Model{:ising}, s; J, h)
  N = length(s)
  a = OpSum()
  for n in 1:N-1
    a .+= -J, "X", n, "X", n+1
  end
  for n in 1:N
    a .+= -h, "Z", n
  end
  return MPO(a, s)
end

# H = -J X₁X₂ - h Z₁
function ITensors.ITensor(::Model{:ising}, s1::Index, s2::Index; J, h)
  return -J * op("X", s1) * op("X", s2) - h * op("Z", s1) * op("Id", s2)
end
# Version accepting IndexSet
ITensors.ITensor(model::Model{:ising}, s1, s2; kwargs...) = ITensor(model, Index(s1), Index(s2); kwargs...)

function InfiniteITensorSum(model::Model{:ising}, s; J, h)
  N = length(s)
  H = InfiniteITensorSum(N)
  @show s[1]
  s∞ = CelledVector(s)
  @show s∞[1]
  @show s∞[2]
  @show s∞[3]
  return InfiniteITensorSum([ITensor(model, s∞[n], s∞[n+1]; J = J, h = h) for n in 1:N])
end

N = 100
s = siteinds("S=1/2", N)

J = 1.0
h = 1.5
model = Model(:ising)
H = MPO(model, s; J = J, h = h)
ψ0 = randomMPS(s)

sweeps = Sweeps(10)
maxdim!(sweeps, 10)
cutoff!(sweeps, 1E-10)
energy, ψ = dmrg(H, ψ0, sweeps)
@show energy/N

n1 = N ÷ 2
n2 = n1 + 1

ψ = orthogonalize(ψ, n1)

println("\nFinite MPS energy on bond (n1, n2) = $((n1, n2))")

ϕ12 = ψ[n1] * ψ[n2]
ham12 = ITensor(model, s[n1], s[n2]; J = J, h = h)
@show (ϕ12 * ham12 * prime(dag(ϕ12), commoninds(ϕ12, ham12)))[]

# Get an infinite MPS that approximates
# the finite MPS in the bulk.
# `nsites` sets the number of sites in the unit
# cell of the infinite MPS.
nsites = 2
ψ∞ = infinitemps_approx(ψ; nsites = nsites, nsweeps = 10)

println("\nInfinite MPS energy on a few different bonds")
println("Infinite MPS has unit cell of size $nsites")

# Compute a few energies
for n1 in 1:3
  global n2 = n1 + 1
  @show n1, n2
  global ϕ12 = ψ∞.AL[n1] * ψ∞.AL[n2] * ψ∞.C[n2]
  global s1 = siteind(ψ∞, n1)
  global s2 = siteind(ψ∞, n2)
  global ham12 = ITensor(model, s1, s2; J = J, h = h)
  global ϕ12ᴴ = prime(dag(ϕ12), commoninds(ϕ12, ham12))
  @show (ϕ12 * ham12 * ϕ12ᴴ)[] / norm(ϕ12)
end

#
# Test computing the expectation value of an infinite sum
# of local operators
#

s¹ = siteinds(ψ∞, Cell(1))

Σ∞h = InfiniteITensorSum(model, s¹; J = J, h = h)
ψ12 = ψ∞.AL[1] * ψ∞.AL[2]
h12 = Σ∞h[1]
s12 = commoninds(ψ12, h12)
@show ψ12 * h12 * prime(dag(ψ12), s12)

