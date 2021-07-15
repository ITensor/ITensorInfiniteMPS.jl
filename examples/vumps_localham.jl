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
# VUMPS code
#

# Make an ITensorMap representing the transfer matrix T|v> = |Tv>
function transfer_matrix(ψ::InfiniteMPS)
  N = nsites(ψ)
  ψᴴ = prime(linkinds, dag(ψ))

  # The unit cell range
  # Turn into function `eachcellindex(ψ::InfiniteMPS, cell::Integer = 1)`
  cell₁ = 1:N
  # A transfer matrix made from the 1st unit cell of
  # the infinite MPS
  # TODO: make a `Cell` Integer type and call as `ψ[Cell(1)]`
  # TODO: make a TransferMatrix wrapper that automatically
  # primes and daggers, so it can be called with:
  # T = TransferMatrix(ψ[Cell(1)])
  ψ₁ = ψ[cell₁]
  ψ₁ᴴ = ψᴴ[cell₁]
  T₀₁ = ITensorMap(ψ₁, ψ₁ᴴ;
                   input_inds = unioninds(commoninds(ψ[N], ψ[N+1]), commoninds(ψᴴ[N], ψᴴ[N+1])),
                   output_inds = unioninds(commoninds(ψ[1], ψ[0]), commoninds(ψᴴ[1], ψᴴ[0])))
  return T₀₁
end

# A TransferMatrix with the dominant space projected out
struct ProjectedTransferMatrix
  T::ITensorMap
  R::ITensor
end
function (T::ProjectedTransferMatrix)(v::ITensor)
  return v - T.T(v) + (v * T.R) * δ(inds(v))
end

# Also input C bond matrices to help compute the right fixed points
# of ψ (R ≈ C * dag(C))
function left_environment(H::InfiniteITensorSum, ψ::InfiniteMPS, C::InfiniteMPS)
  N = nsites(H)
  @assert N == nsites(ψ)
  # Solve using C25a
  # Solve:
  # (Lᵃ|[𝟏 - Tᴸ + |R)(𝟏|] = (YLᵃ| - (YLᵃ|R)(𝟏|
  b = YLᵃ - (YLᵃ * R) * δ(inds(YLᵃ))
  L⃗[a], _ = linsolve(ProjectedTransferMatrix(T, R), b)

  # Get error
  err_lhs = L⃗[a] - translatecell(L⃗[a] * ψ[1] * dag(prime(ψ[1], "Link")), -1) + L⃗[a] * R * δ(inds(L⃗[a]))
  err_rhs = YLᵃ - YLᵃ * R * δ(inds(YLᵃ))
  @show norm(err_lhs - err_rhs)
  return InfiniteMPS([L])
end

function right_environment(H::InfiniteITensorSum, ψ::InfiniteMPS, C::InfiniteMPS)
  N = nsites(H)
  @assert N == nsites(ψ)
  # Solve using C25b
  # Solve:
  # [𝟏 - Tᴿ + |𝟏)(L|]|Rᵃ) = |YRᵃ) - |𝟏)(L|YRᵃ)
  b = YRᵃ - (L * YRᵃ) * δ(inds(YRᵃ))
  R⃗[a], _ = linsolve(ProjectedTransferMatrix(T, L), b)

  # Get error
  err_lhs = R⃗[a] - translatecell(ψ[1] * dag(prime(ψ[1], "Link")) * R⃗[a], 1) + L * R⃗[a] * δ(inds(R⃗[a]))
  err_rhs = YRᵃ - L * YRᵃ * δ(inds(YRᵃ))
  @show norm(err_lhs - err_rhs)
  return InfiniteMPS([R])
end

vumps(H::InfiniteITensorSum, ψ::InfiniteMPS; kwargs...) = vumps(H, orthogonalize(ψ, :); kwargs...)

# Find the best orthogonal approximation given
# the center tensors AC and C
function ortho(AC::ITensor, C::ITensor)
  E = AC * dag(C)
  U, P = polar(E, uniqueinds(AC, C))
  l = commoninds(U, P)
  return noprime(U, l)
end

#function right_ortho(C::ITensor, AC::ITensor)
#  E = dag(C) * AC
#  U, P = polar(E, uniqueinds(C, AC))
#  l = commoninds(U, P)
#  return noprime(U, l)
#end

function vumps(H::InfiniteITensorSum, ψ::InfiniteCanonicalMPS; nsweeps = 10)
  for sweep in 1:nsweeps
    L = left_environment(H, ψ.AL, ψ.C)
    R = right_environment(H, ψ.AR, ψ.C)

    n = 1

    # 0-site effective Hamiltonian
    H⁰ = ITensorMap([L[n], R[n]])
    vals0, vecs0, info0 = eigsolve(H⁰, ψ.C[n])
    E0 = vals0[1]
    Cⁿ = vecs0[1]
    C = InfiniteMPS([Cⁿ])

    @show E0
    @show inds(Cⁿ)

    # 1-site effective Hamiltonian
    H¹ = ITensorMap([L[n-1], H[n], R[n]])
    vals1, vecs1, info1 = eigsolve(H¹, ψ.AL[n] * ψ.C[n]; ishermition = true)
    E1 = vals1[1]
    ACⁿ = vecs1[1]
    AC = InfiniteMPS([ACⁿ])

    @show E1
    @show inds(ACⁿ)

    ALⁿ = ortho(ACⁿ, Cⁿ)
    ψL = InfiniteMPS([ALⁿ])

    @show norm(ALⁿ * Cⁿ - ACⁿ)

    @show inds(ALⁿ)
    #ARⁿ = ortho(ACⁿ, translatecell(Cⁿ, -1))
    ARⁿ = replacetags(ALⁿ, "Left" => "Right")

    @show norm(translatecell(Cⁿ, -1) * ARⁿ - ACⁿ)
    ψR = InfiniteMPS([ARⁿ])

    ψ = InfiniteCanonicalMPS(ψL, C, ψR)
  end

  return ψ
end

#
# Test computing the expectation value of an infinite sum
# of local operators
#

s¹ = siteinds(ψ∞, Cell(1))

Σ∞h = InfiniteITensorSum(model, s¹; J = J, h = h)
ψ∞′ = ψ∞'
ψ12 = ψ∞.AL[1] * ψ∞.AL[2]
h12 = Σ∞h[1]
s12 = commoninds(ψ12, h12)
@show ψ12 * h12 * prime(dag(ψ12), s12)

## #
## # One VUMPS step
## #
## 
## L = left_environment(H, ψ.AL, ψ.C)
## R = right_environment(H, ψ.AR, ψ.C)
## 
## n = 1
## 
## # 0-site effective Hamiltonian
## H⁰ = ITensorMap([L[n], R[n]])
## vals0, vecs0, info0 = eigsolve(H⁰, ψ.C[n])
## E0 = vals0[1]
## Cⁿ = vecs0[1]
## C = InfiniteMPS([Cⁿ])
## 
## @show E0
## @show inds(Cⁿ)
## 
## # 1-site effective Hamiltonian
## H¹ = ITensorMap([L[n-1], H[n], R[n]])
## vals1, vecs1, info1 = eigsolve(H¹, ψ.AL[n] * ψ.C[n]; ishermition = true)
## E1 = vals1[1]
## ACⁿ = vecs1[1]
## AC = InfiniteMPS([ACⁿ])
## 
## @show E1
## @show inds(ACⁿ)
## 
## ALⁿ = ortho(ACⁿ, Cⁿ)
## ψL = InfiniteMPS([ALⁿ])
## 
## @show norm(ALⁿ * Cⁿ - ACⁿ)
## 
## @show inds(ALⁿ)
## #ARⁿ = ortho(ACⁿ, translatecell(Cⁿ, -1))
## ARⁿ = replacetags(ALⁿ, "Left" => "Right")
## 
## @show norm(translatecell(Cⁿ, -1) * ARⁿ - ACⁿ)
## ψR = InfiniteMPS([ARⁿ])


