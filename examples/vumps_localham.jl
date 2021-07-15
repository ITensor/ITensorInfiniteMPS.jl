using ITensors
using ITensorInfiniteMPS
using ITensorsVisualization
using Random

Random.seed!(1234)

using ITensorInfiniteMPS: InfiniteCanonicalMPS

struct Model{model} end
Model(model::Symbol) = Model{model}()

# For now, only represents nearest neighbor interactions
# on a linear chain
struct InfiniteITensorSum
  data::CelledVector{ITensor}
end
InfiniteITensorSum(N::Int) = InfiniteITensorSum(Vector{ITensor}(undef, N))
InfiniteITensorSum(data::Vector{ITensor}) = InfiniteITensorSum(CelledVector(data))
function Base.getindex(l::InfiniteITensorSum, n1n2::Tuple{Int,Int})
  n1, n2 = n1n2
  @assert n2 == n1 + 1
  return l.data[n1]
end

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
# `Nsites` sets the number of sites in the unit
# cell of the infinite MPS.
Nsites = 2
# XXX: not outputing a right orthogonal ψ∞.AR
ψ∞ = infinitemps_approx(ψ; nsites = Nsites, nsweeps = 10)

# XXX: this is segfaulting for some reason
#ψ∞ = orthogonalize(ψ∞.AL, :)
C, ψR = ITensorInfiniteMPS.right_orthogonalize(ψ∞.AL)
ψ∞ = InfiniteCanonicalMPS(ψ∞.AL, C, ψR)

println("\nInfinite MPS energy on a few different bonds")
println("Infinite MPS has unit cell of size $Nsites")

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
ψᴴ∞ = dag(ψ∞)
ψ′∞ = ψᴴ∞'
# XXX: make this prime the center sites
ψ̃∞ = prime(linkinds, ψᴴ∞)
## ψ12 = ψ∞.AL[1] * ψ∞.AL[2]
## h12 = Σ∞h[1]
## s12 = commoninds(ψ12, h12)
## @show ψ12 * h12 * prime(dag(ψ12), s12)

l = CelledVector([commoninds(ψ∞.AL[n], ψ∞.AL[n + 1]) for n in 1:Nsites])
l′ = CelledVector([commoninds(ψ′∞.AL[n], ψ′∞.AL[n + 1]) for n in 1:Nsites])
r = CelledVector([commoninds(ψ∞.AR[n], ψ∞.AR[n + 1]) for n in 1:Nsites])
r′ = CelledVector([commoninds(ψ′∞.AR[n], ψ′∞.AR[n + 1]) for n in 1:Nsites])

## l⁻¹⁰ = commoninds(ψ∞.AL[-1], ψ∞.AL[0])
## l⁰¹ = commoninds(ψ∞.AL[0], ψ∞.AL[1])
## l′⁻¹⁰ = commoninds(ψ′∞.AL[-1], ψ′∞.AL[0])
## l′⁰¹ = commoninds(ψ′∞.AL[0], ψ′∞.AL[1])
## 
## r⁻¹⁰ = commoninds(ψ∞.AR[-1], ψ∞.AR[0])
## r⁰¹ = commoninds(ψ∞.AR[0], ψ∞.AR[1])
## r¹² = commoninds(ψ∞.AR[1], ψ∞.AR[2])
## r²³ = commoninds(ψ∞.AR[2], ψ∞.AR[3])
## r′⁻¹⁰ = commoninds(ψ′∞.AR[-1], ψ′∞.AR[0])
## r′⁰¹ = commoninds(ψ′∞.AR[0], ψ′∞.AR[1])
## r′¹² = commoninds(ψ′∞.AR[1], ψ′∞.AR[2])
## r′²³ = commoninds(ψ′∞.AR[2], ψ′∞.AR[3])

hᴸ = InfiniteMPS([δ(only(l[n - 2]), only(l′[n - 2])) * ψ∞.AL[n - 1] * ψ∞.AL[n] * Σ∞h[(n - 1, n)] * dag(ψ′∞.AL[n - 1]) * dag(ψ′∞.AL[n]) for n in 1:Nsites])
#hᴸ²³ = @visualize δ(only(l[0]), only(l′[0])) * ψ∞.AL[1] * ψ∞.AL[2] * Σ∞h[(1, 2)] * dag(ψ′∞.AL[1]) * dag(ψ′∞.AL[2])

hᴿ = InfiniteMPS([δ(only(r[n + 2]), only(r′[n + 2])) * ψ∞.AR[n + 2] * ψ∞.AR[n + 1] * Σ∞h[(n + 1, n + 2)] * dag(ψ′∞.AR[n + 2]) * dag(ψ′∞.AR[n + 1]) for n in 1:Nsites])

#hᴸ¹² = @visualize δ(only(r[3]), only(r′[3])) * ψ∞.AR[3] * ψ∞.AR[2] * Σ∞h[(2, 3)] * dag(ψ′∞.AR[3]) * dag(ψ′∞.AR[2])
#hᴸ²³ = @visualize δ(only(r[4]), only(r′[4])) * ψ∞.AR[4] * ψ∞.AL[3] * Σ∞h[(3, 4)] * dag(ψ′∞.AL[4]) * dag(ψ′∞.AL[3])

# Shift the energy
## eᴸ¹² = @visualize hᴸ¹² * ψ∞.C[1] * δ(only(r¹²), only(r′¹²)) * ψ′∞.C[1]
## @show eᴸ¹²
## hᴸ¹² = hᴸ¹² - eᴸ¹² * δ(inds(hᴸ¹²))
## eᴸ²³ = @visualize hᴸ²³ * ψ∞.C[2] * δ(only(r²³), only(r′²³)) * ψ′∞.C[2]
## @show eᴸ²³
## hᴸ²³ = hᴸ²³ - eᴸ²³ * δ(inds(hᴸ²³))
## 
## hᴸ = InfiniteMPS([hᴸ¹², hᴸ²³])

eᴸ = [(hᴸ[n] * ψ∞.C[n] * δ(only(r[n]), only(r′[n])) * ψ′∞.C[n])[] for n in 1:Nsites]
eᴿ = [(hᴿ[n] * ψ∞.C[n] * δ(only(l[n]), only(l′[n])) * ψ′∞.C[n])[] for n in 1:Nsites]

@show eᴸ
@show eᴿ

for n in 1:Nsites
  hᴸ[n] -= eᴸ[n] * δ(inds(hᴸ[n]))
  hᴿ[n] -= eᴿ[n] * δ(inds(hᴿ[n]))
end

function left_environment_sum(hᴸ, ψ∞)
  ψ̃∞ = prime(linkinds, dag(ψ∞))
  Hᴸ²¹ = hᴸ[2]
  Hᴸ²¹ += hᴸ[1] * ψ∞.AL[2] * ψ̃∞.AL[2]
  Hᴸ²¹ += hᴸ[0] * ψ∞.AL[1] * ψ̃∞.AL[1] * ψ∞.AL[2] * ψ̃∞.AL[2]
  Hᴸ²¹ += hᴸ[-1] * ψ∞.AL[0] * ψ̃∞.AL[0] * ψ∞.AL[1] * ψ̃∞.AL[1] * ψ∞.AL[2] * ψ̃∞.AL[2]
  Hᴸ²¹ += hᴸ[-2] * ψ∞.AL[-1] * ψ̃∞.AL[-1] * ψ∞.AL[0] * ψ̃∞.AL[0] * ψ∞.AL[1] * ψ̃∞.AL[1] * ψ∞.AL[2] * ψ̃∞.AL[2]
  Hᴸ²¹ += hᴸ[-3] * ψ∞.AL[-2] * ψ̃∞.AL[-2] * ψ∞.AL[-1] * ψ̃∞.AL[-1] * ψ∞.AL[0] * ψ̃∞.AL[0] * ψ∞.AL[1] * ψ̃∞.AL[1] * ψ∞.AL[2] * ψ̃∞.AL[2]
  Hᴸ²¹ += hᴸ[-4] * ψ∞.AL[-3] * ψ̃∞.AL[-3] * ψ∞.AL[-2] * ψ̃∞.AL[-2] * ψ∞.AL[-1] * ψ̃∞.AL[-1] * ψ∞.AL[0] * ψ̃∞.AL[0] * ψ∞.AL[1] * ψ̃∞.AL[1] * ψ∞.AL[2] * ψ̃∞.AL[2]
  Hᴸ²¹ *= inv(sign(Hᴸ²¹[1, 1]))
  return Hᴸ²¹
end

function left_environment_recursive(hᴸ, ψ∞)
  ψ̃∞ = prime(linkinds, dag(ψ∞))
  # XXX: replace with `nsites`
  #N = nsites(ψ∞)
  N = length(ψ∞)
  Hᴸᴺ¹ = hᴸ[N]
  for _ in 1:10
    Hᴸᴺ¹ = translatecell(Hᴸᴺ¹, -1)
    for n in 1:N
      Hᴸᴺ¹ = Hᴸᴺ¹ * ψ∞.AL[n] * ψ̃∞.AL[n]
    end
    # Loop over the Hamiltonian terms in the unit cell
    for n in 1:N
      hᴸⁿ = hᴸ[n]
      for k in (n + 1):N
        hᴸⁿ = hᴸⁿ * ψ∞.AL[k] * ψ̃∞.AL[k]
      end
      Hᴸᴺ¹ += hᴸⁿ
    end
  end
  # Get the rest of the environments in the unit cell
  Hᴸ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴸ[N] = Hᴸᴺ¹
  Hᴸᴺ¹ = translatecell(Hᴸᴺ¹, -1)
  for n in 1:(N - 1)
    Hᴸ[n] = Hᴸ[n - 1] * ψ∞.AL[n] * ψ̃∞.AL[n] + hᴸ[n]
  end
  return Hᴸ
end

function right_environment_recursive(hᴿ, ψ∞)
  ψ̃∞ = prime(linkinds, dag(ψ∞))
  # XXX: replace with `nsites`
  #N = nsites(ψ∞)
  N = length(ψ∞)
  Hᴿᴺ¹ = hᴿ[0]
  for _ in 1:10
    Hᴿᴺ¹ = translatecell(Hᴿᴺ¹, 1)
    for n in reverse(1:N)
      Hᴿᴺ¹ = Hᴿᴺ¹ * ψ∞.AR[n] * ψ̃∞.AR[n]
    end
    # Loop over the Hamiltonian terms in the unit cell
    for n in reverse(0:(N - 1))
      hᴿⁿ = hᴿ[n]
      for k in reverse(1:n)
        hᴿⁿ = hᴿⁿ * ψ∞.AR[k] * ψ̃∞.AR[k]
      end
      Hᴿᴺ¹ += hᴿⁿ
    end
  end
  Hᴿᴺ¹ = translatecell(Hᴿᴺ¹, 1)
  # Get the rest of the environments in the unit cell
  Hᴿ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴿ[N] = Hᴿᴺ¹
  for n in reverse(1:(N - 1))
    Hᴿ[n] = Hᴿ[n + 1] * ψ∞.AR[n + 1] * ψ̃∞.AR[n + 1] + hᴿ[n]
  end
  return Hᴿ
end

Hᴸ = left_environment_recursive(hᴸ, ψ∞)
Hᴿ = right_environment_recursive(hᴿ, ψ∞)
for n in 1:Nsites
  @show n
  @show tr(Hᴸ[n] * ψ∞.C[n] * ψ′∞.C[n])
  @show tr(Hᴿ[n] * ψ∞.C[n] * ψ′∞.C[n])
end

n = 1
struct Hᶜ
  Σ∞h::InfiniteITensorSum
  Hᴸ::InfiniteMPS
  Hᴿ::InfiniteMPS
  ψ∞::InfiniteCanonicalMPS
end

function (H::Hᶜ)(v)
  Σ∞h = H.Σ∞h
  Hᴸ = H.Hᴸ
  Hᴿ = H.Hᴿ
  ψ∞ = H.ψ∞
  ψ′∞ = ψ∞'
  N = length(ψ∞)
  l = CelledVector([commonind(ψ∞.AL[n], ψ∞.AL[n + 1]) for n in 1:N])
  l′ = CelledVector([commonind(ψ′∞.AL[n], ψ′∞.AL[n + 1]) for n in 1:N])
  r = CelledVector([commonind(ψ∞.AR[n], ψ∞.AR[n + 1]) for n in 1:N])
  r′ = CelledVector([commonind(ψ′∞.AR[n], ψ′∞.AR[n + 1]) for n in 1:N])
  δˡ = δ(l[n], l′[n])
  δˡ⁻¹ = δ(l[n - 1], l′[n - 1])
  δʳ = δ(r[n], r′[n])
  δʳ⁺¹ = δ(r[n + 1], r′[n + 1])
  Hᶜᴸv = v * Hᴸ[n] * δʳ
  Hᶜᴿv = v * δˡ * Hᴿ[n]
  Hᶜʰv = v * ψ∞.AL[n] * δˡ⁻¹ * ψ′∞.AL[n] * Σ∞h[(n, n + 1)] * ψ∞.AR[n + 1] * δʳ⁺¹ * ψ′∞.AR[n + 1]
  Hᶜv = Hᶜᴸv + Hᶜʰv + Hᶜᴿv
  return Hᶜv * δˡ * δʳ
end

@show Hᶜ(Σ∞h, Hᴸ, Hᴿ, ψ∞)(ψ∞.C[Nsites])

#D, _ = eigen(Hᶜ; ishermitian=true)
#@show minimum(D)

# @show norm(Hᴸ²¹_sum - Hᴸ[2])

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


