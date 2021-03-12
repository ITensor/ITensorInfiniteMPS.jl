using ITensors
using ITensorsInfiniteMPS
using ITensorsVisualization
using KrylovKit
using Random
using LinearAlgebra

Random.seed!(1234)

# TODO: call as `orthogonalize(ψ, -∞)`
# TODO: could use commontags(ψ) as a default for left_tags
function right_orthogonalize(ψ::InfiniteMPS; left_tags = ts"Left", right_tags = ts"Right", tol::Real = 1e-12)
  # TODO: replace dag(ψ) with ψ'ᴴ?
  ψᴴ = prime(linkinds, dag(ψ))

  # The unit cell range
  # Turn into function `eachcellindex(ψ::InfiniteMPS, cell::Integer = 1)`
  cell₁ = 1:nsites(ψ)
  # A transfer matrix made from the 1st unit cell of
  # the infinite MPS
  # TODO: make a `Cell` Integer type and call as `ψ[Cell(1)]`
  # TODO: make a TransferMatrix wrapper that automatically
  # primes and daggers, so it can be called with:
  # T = TransferMatrix(ψ[Cell(1)])
  ψ₁ = ψ[cell₁]
  ψ₁ᴴ = ψᴴ[cell₁]
  T₀₁ = ITensorMap(ψ₁, ψ₁ᴴ)

  # TODO: make an optional initial state
  v₁ᴿᴺ = randomITensor(dag(input_inds(T₀₁)))

  # Start by getting the right eivenvector/eigenvalue of T₀₁
  # TODO: make a function `right_environments(::InfiniteMPS)` that computes
  # all of the right environments using `eigsolve` and shifting unit cells
  λ⃗₁ᴿᴺ, v⃗₁ᴿᴺ, eigsolve_info = eigsolve(T₀₁, v₁ᴿᴺ, 1, :LM; tol = tol)
  λ₁ᴿᴺ, v₁ᴿᴺ = λ⃗₁ᴿᴺ[1], v⃗₁ᴿᴺ[1]

  @assert imag(λ₁ᴿᴺ) / norm(λ₁ᴿᴺ) < 1e-15

  # Fix the phase of the diagonal to make Hermitian
  v₁ᴿᴺ .*= conj(sign(v₁ᴿᴺ[1, 1]))
  @assert ishermitian(v₁ᴿᴺ; rtol = tol)

  @show λ₁ᴿᴺ
  @show v₁ᴿᴺ

  # Initial guess for bond matrix such that:
  # ψ₁ * C₁ᴿᴺ = C₁ᴿᴺ * ψ₁ᴿ
  C₁ᴿᴺ = sqrt(v₁ᴿᴺ)

  @show inds(C₁ᴿᴺ)

  @show ψ.reverse

  @show left_tags
  @show right_tags

  C₁ᴿᴺ = replacetags(C₁ᴿᴺ, left_tags => right_tags; plev = 1)
  
  @show inds(C₁ᴿᴺ)

  C₁ᴿᴺ = noprime(C₁ᴿᴺ, right_tags)

  @show inds(C₁ᴿᴺ)

  if ψ.reverse
    #error("END")
  end

  normalize!(C₁ᴿᴺ)

  Cᴿ, ψᴿ, λᴿ = right_orthogonalize_polar(ψ, C₁ᴿᴺ; left_tags = left_tags, right_tags = right_tags)
  @assert λᴿ ≈ sqrt(real(λ₁ᴿᴺ))
  return Cᴿ, ψᴿ, λᴿ
end

function right_orthogonalize_polar(ψ::InfiniteMPS, Cᴿᴺ::ITensor; left_tags = ts"Left", right_tags = ts"Right")
  N = length(ψ)
  ψᴿ = InfiniteMPS(N; reverse = ψ.reverse)
  Cᴿ = InfiniteMPS(N; reverse = ψ.reverse)
  Cᴿ[N] = Cᴿᴺ
  λ = 1.0
  for n in reverse(1:N)
    sⁿ = uniqueinds(ψ[n], ψ[n-1], Cᴿ[n])
    # TODO: use "Right" tag for ψᴿ, Cᴿ
    lᴿⁿ = uniqueinds(Cᴿ[n], ψ[n])

    @show inds(ψ[n] * Cᴿ[n])
    @show sⁿ
    @show lᴿⁿ

    ψᴿⁿ, Cᴿⁿ⁻¹ = polar(ψ[n] * Cᴿ[n], (sⁿ..., lᴿⁿ...))
    @show n
    @show inds(ψᴿⁿ)
    @show ITensorsInfiniteMPS.cell(ψᴿ, n)
    ψᴿ[n] = ψᴿⁿ

    Cᴿ[n-1] = Cᴿⁿ⁻¹
    
    ψᴿ[n] = replacetags(ψᴿ[n], "" => right_tags; plev = 1)
    ψᴿ[n] = noprime(ψᴿ[n], right_tags)
    Cᴿ[n-1] = replacetags(Cᴿ[n-1], "" => right_tags; plev = 1)
    Cᴿ[n-1] = noprime(Cᴿ[n-1], right_tags)

    λⁿ = norm(Cᴿ[n-1])
    Cᴿ[n-1] /= λⁿ
    λ *= λⁿ
    @assert ψ[n] * Cᴿ[n] ≈ λⁿ * Cᴿ[n-1] * ψᴿ[n]
  end
  return Cᴿ, ψᴿ, λ
end

function left_orthogonalize(ψ::InfiniteMPS; left_tags = ts"Left", right_tags = ts"Right", tol::Real = 1e-12)
  Cᴸ, ψᴸ, λᴸ = right_orthogonalize(reverse(ψ); left_tags = right_tags, right_tags = left_tags, tol = tol)
  return reverse(ψᴸ), reverse(Cᴸ), λᴸ
end

#
# Example
#

N = 3
s = siteinds("S=1/2", N; conserve_szparity = true)
χ = 4
@assert iseven(χ)
space = (("SzParity", 1, 2) => χ ÷ 2) ⊕ (("SzParity", 0, 2) => χ ÷ 2)
ψ = InfiniteMPS(ComplexF64, s; space = space)
randn!.(ψ)

Cᴿ, ψᴿ, λᴿ = right_orthogonalize(ψ; left_tags = ts"", right_tags = ts"Right")
@show λᴿ
@show norm(prod(ψ[1:N]) * Cᴿ[N] - λᴿ * Cᴿ[0] * prod(ψᴿ[1:N]))

ψᴸ, Cᴸ, λᴸ = left_orthogonalize(ψᴿ; left_tags = ts"Left", right_tags = ts"Right")


