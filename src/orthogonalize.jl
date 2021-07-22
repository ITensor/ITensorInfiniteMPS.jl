
# TODO: call as `orthogonalize(ψ, -∞)`
# TODO: could use commontags(ψ) as a default for left_tags
function right_orthogonalize(ψ::InfiniteMPS; left_tags = ts"Left", right_tags = ts"Right", tol::Real = 1e-12)
  # TODO: replace dag(ψ) with ψ'ᴴ?
  ψᴴ = prime(linkinds, dag(ψ))

  N = nsites(ψ)

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

  # TODO: make an optional initial state
  v₁ᴿᴺ = randomITensor(dag(input_inds(T₀₁)))

  # Start by getting the right eivenvector/eigenvalue of T₀₁
  # TODO: make a function `right_environments(::InfiniteMPS)` that computes
  # all of the right environments using `eigsolve` and shifting unit cells
  λ⃗₁ᴿᴺ, v⃗₁ᴿᴺ, eigsolve_info = eigsolve(T₀₁, v₁ᴿᴺ, 1, :LM; tol = tol)
  λ₁ᴿᴺ, v₁ᴿᴺ = λ⃗₁ᴿᴺ[1], v⃗₁ᴿᴺ[1]

  @show inds(v₁ᴿᴺ)

  if imag(λ₁ᴿᴺ) / norm(λ₁ᴿᴺ) > 1e-15
    @show λ₁ᴿᴺ
    error("Imaginary part of eigenvalue is large: imag(λ₁ᴿᴺ) / norm(λ₁ᴿᴺ) = $(imag(λ₁ᴿᴺ) / norm(λ₁ᴿᴺ))")
  end

  # Fix the phase of the diagonal to make Hermitian
  v₁ᴿᴺ .*= conj(sign(v₁ᴿᴺ[1, 1]))
  if !ishermitian(v₁ᴿᴺ; rtol = tol)
    @show λ₁ᴿᴺ
    @show v₁ᴿᴺ
    @show norm(v₁ᴿᴺ - swapinds(dag(v₁ᴿᴺ), reverse(Pair(inds(v₁ᴿᴺ)...))))
    error("v₁ᴿᴺ not hermitian")
  end
  if norm(imag(v₁ᴿᴺ)) > 1e-15
    println("Norm of the imaginary part $(norm(imag(v₁ᴿᴺ))) is larger than the tolerance value 1e-15. Keeping as complex.")
    @show norm(v₁ᴿᴺ - swapinds(dag(v₁ᴿᴺ), reverse(Pair(inds(v₁ᴿᴺ)...))))
  else
    v₁ᴿᴺ = real(v₁ᴿᴺ)
  end

  # Initial guess for bond matrix such that:
  # ψ₁ * C₁ᴿᴺ = C₁ᴿᴺ * ψ₁ᴿ
  C₁ᴿᴺ = sqrt(v₁ᴿᴺ)
  C₁ᴿᴺ = replacetags(C₁ᴿᴺ, left_tags => right_tags; plev = 1)
  C₁ᴿᴺ = noprime(C₁ᴿᴺ, right_tags)

  # Normalize the center matrix
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
    lᴿⁿ = uniqueinds(Cᴿ[n], ψ[n])
    ψᴿⁿ, Cᴿⁿ⁻¹ = polar(ψ[n] * Cᴿ[n], (sⁿ..., lᴿⁿ...))
    # TODO: set the tags in polar
    ψᴿⁿ = replacetags(ψᴿⁿ, left_tags => right_tags; plev = 1)
    ψᴿⁿ = noprime(ψᴿⁿ, right_tags)
    Cᴿⁿ⁻¹ = replacetags(Cᴿⁿ⁻¹, left_tags => right_tags; plev = 1)
    Cᴿⁿ⁻¹ = noprime(Cᴿⁿ⁻¹, right_tags)
    ψᴿ[n] = ψᴿⁿ
    Cᴿ[n-1] = Cᴿⁿ⁻¹
    λⁿ = norm(Cᴿ[n-1])
    Cᴿ[n-1] /= λⁿ
    λ *= λⁿ
    if !isapprox(ψ[n] * Cᴿ[n], λⁿ * Cᴿ[n-1] * ψᴿ[n]; atol=1e-14)
      @show norm(ψ[n] * Cᴿ[n] - λⁿ * Cᴿ[n-1] * ψᴿ[n])
      error("ψ[n] * Cᴿ[n] ≠ λⁿ * Cᴿ[n-1] * ψᴿ[n]")
    end
  end
  return Cᴿ, ψᴿ, λ
end

function left_orthogonalize(ψ::InfiniteMPS; left_tags = ts"Left", right_tags = ts"Right", tol::Real = 1e-12)
  Cᴸ, ψᴸ, λᴸ = right_orthogonalize(reverse(ψ); left_tags = right_tags, right_tags = left_tags, tol = tol)
  # Cᴸ has the unit cell shifted from what is expected
  Cᴸ = reverse(Cᴸ)
  Cᴸ_shift = copy(Cᴸ)
  for n in 1:nsites(Cᴸ)
    Cᴸ_shift[n] = Cᴸ[n+1]
  end
  return reverse(ψᴸ), Cᴸ_shift, λᴸ
end

# TODO: rename to `orthogonalize(ψ)`? With no limit specified, it is like orthogonalizing to over point.
# Alternatively, it could be called as `orthogonalize(ψ, :)`
function mixed_canonical(ψ::InfiniteMPS; left_tags = ts"Left", right_tags = ts"Right", tol::Real = 1e-12)
  _, ψᴿ, _ = right_orthogonalize(ψ; left_tags = ts"", right_tags = ts"Right")
  ψᴸ, C, λ = left_orthogonalize(ψᴿ; left_tags = ts"Left", right_tags = ts"Right")
  if λ ≉ one(λ)
    error("λ should be approximately 1 after orthogonalization, instead it is $λ")
  end
  return InfiniteCanonicalMPS(ψᴸ, C, ψᴿ)
end

ITensors.orthogonalize(ψ::InfiniteMPS, ::Colon; kwargs...) = mixed_canonical(ψ; kwargs...)
