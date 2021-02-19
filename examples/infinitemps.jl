using ITensors
using ITensorsInfiniteMPS
using ITensorsVisualization
using KrylovKit
using Random
using LinearAlgebra

Random.seed!(1234)

# TODO: call as `orthogonalize(ψ, -∞)`
function right_orthogonalize(ψ::InfiniteMPS; tol::Real = 1e-12)
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
  @show eigsolve_info.numops
  λ₁ᴿᴺ, v₁ᴿᴺ = λ⃗₁ᴿᴺ[1], v⃗₁ᴿᴺ[1]

  @show λ₁ᴿᴺ

  # Fix the phase of the diagonal to make Hermitian
  v₁ᴿᴺ .*= conj(sign(v₁ᴿᴺ[1, 1]))
  @show ishermitian(v₁ᴿᴺ; rtol = tol)

  # Initial guess for bond matrix such that:
  # ψ₁ * C₁ᴿᴺ = C₁ᴿᴺ * ψ₁ᴿ
  C₁ᴿᴺ = sqrt(v₁ᴿᴺ)
  @show norm(replaceprime(C₁ᴿᴺ' * C₁ᴿᴺ, 2 => 1) - v₁ᴿᴺ)
  ψ₁ᴿ, C₁ᴿ = right_orthogonalize_polar(ψ, C₁ᴿᴺ)

  #return ψᴿ, Cᴿ
end

function right_orthogonalize_polar(ψ::InfiniteMPS, Cᴿᴺ::ITensor)
  N = length(ψ)
  ψᴿ = InfiniteMPS(N)
  Cᴿ = InfiniteMPS(N)
  Cᴿ[N] = Cᴿᴺ
  for n in reverse(1:N)
    sⁿ = uniqueinds(ψ[n], ψ[n-1], Cᴿ[n])
    # TODO: use "Right" tag for ψᴿ, Cᴿ
    lᴿⁿ = uniqueinds(Cᴿ[n], ψ[n])
    ψᴿ[n], Cᴿⁿ⁻¹ = polar(ψ[n] * Cᴿ[n], (sⁿ..., lᴿⁿ...))
    Cᴿ[n-1] = Cᴿⁿ⁻¹
    @show norm(Cᴿ[n-1] - swapprime(dag(Cᴿ[n-1]), 0 => 1))
    @show norm(ψ[n] * Cᴿ[n] - Cᴿ[n-1] * ψᴿ[n])
  end
  return ψᴿ, Cᴿ
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

ψᴿ, Cᴿ = right_orthogonalize(ψ, "" => "Right")
ψᴸ, Cᴸ = left_orthogonalize(ψᴿ, "Left" => "Right")


