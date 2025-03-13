using KrylovKit: schursolve, Arnoldi
# TODO: call as `orthogonalize(ψ, -∞)`
# TODO: could use commontags(ψ) as a default for left_tags
function right_orthogonalize(
  ψ::InfiniteMPS;
  left_tags=ts"Left",
  right_tags=ts"Right",
  tol::Real=1e-12,
  eager=true,
  ishermitian_kwargs=(; rtol=tol * 100),
)
  # A transfer matrix made from the 1st unit cell of the infinite MPS
  T = TransferMatrix(ψ)

  # TODO: make an optional initial state
  v₁ᴿᴺ = random_itensor(dag(input_inds(T)))

  # Start by getting the right eivenvector/eigenvalue of T
  # TODO: make a function `right_environments(::InfiniteMPS)` that computes
  # all of the right environments using `eigsolve` and shifting unit cells

  # original eigsolve function, switch to schur which enforces real
  #λ⃗₁ᴿᴺ, v⃗₁ᴿᴺ, eigsolve_info = eigsolve(T, v₁ᴿᴺ, 1, :LM; tol, eager)
  TT, v⃗₁ᴿᴺ, λ⃗₁ᴿᴺ, info = schursolve(T, v₁ᴿᴺ, 1, :LM, Arnoldi(; tol, eager))
  λ₁ᴿᴺ, v₁ᴿᴺ = λ⃗₁ᴿᴺ[1], v⃗₁ᴿᴺ[1]

  if info.converged == 0
    @warn "orthogonalize not converged after $(info.numiter) iterations"
  end

  if size(TT, 2) > 1 && TT[2, 1] != 0
    @warn("Non-unique largest eigenvector of transfer matrix found")
  end

  if imag(λ₁ᴿᴺ) / norm(λ₁ᴿᴺ) > 1e-15
    @show λ₁ᴿᴺ
    error(
      "Imaginary part of eigenvalue is large: imag(λ₁ᴿᴺ) / norm(λ₁ᴿᴺ) = $(imag(λ₁ᴿᴺ) / norm(λ₁ᴿᴺ))",
    )
  end

  # Fix the phase of the diagonal to make Hermitian
  v₁ᴿᴺ .*= conj(sign(v₁ᴿᴺ[1, 1]))
  if !ishermitian(v₁ᴿᴺ; ishermitian_kwargs...)
    @show norm(v₁ᴿᴺ - swapinds(dag(v₁ᴿᴺ), reverse(Pair(inds(v₁ᴿᴺ)...))))
    @warn("v₁ᴿᴺ is not hermitian, passed kwargs: $ishermitian_kwargs")
  end
  if norm(imag(v₁ᴿᴺ)) / norm(v₁ᴿᴺ) > 1e-13
    println(
      "Norm of the imaginary part $(norm(imag(v₁ᴿᴺ))) is larger than the tolerance value 1e-15. Keeping as complex.",
    )
    @show norm(v₁ᴿᴺ - swapinds(dag(v₁ᴿᴺ), reverse(Pair(inds(v₁ᴿᴺ)...))))
  else
    v₁ᴿᴺ = real(v₁ᴿᴺ)
  end

  # Initial guess for bond matrix such that:
  # ψ₁ * C₁ᴿᴺ = C₁ᴿᴺ * ψ₁ᴿ
  C₁ᴿᴺ = sqrt(v₁ᴿᴺ)
  C₁ᴿᴺ = replacetags(C₁ᴿᴺ, left_tags => right_tags; plev=1)
  C₁ᴿᴺ = noprime(C₁ᴿᴺ, right_tags)

  # Normalize the center matrix
  normalize!(C₁ᴿᴺ)

  Cᴿ, ψᴿ, λᴿ = right_orthogonalize_polar(
    ψ, C₁ᴿᴺ; left_tags=left_tags, right_tags=right_tags
  )
  @assert λᴿ ≈ sqrt(real(λ₁ᴿᴺ))
  return Cᴿ, ψᴿ, λᴿ
end

function right_orthogonalize_polar(
  ψ::InfiniteMPS, Cᴿᴺ::ITensor; left_tags=ts"Left", right_tags=ts"Right"
)
  N = length(ψ)
  ψᴿ = InfiniteMPS(N; reverse=ψ.reverse)
  Cᴿ = InfiniteMPS(N; reverse=ψ.reverse)
  Cᴿ[N] = Cᴿᴺ
  λ = 1.0
  for n in reverse(1:N)
    sⁿ = uniqueinds(ψ[n], ψ[n - 1], Cᴿ[n])
    lᴿⁿ = uniqueinds(Cᴿ[n], ψ[n])
    ψᴿⁿ, Cᴿⁿ⁻¹ = polar(ψ[n] * Cᴿ[n], (sⁿ..., lᴿⁿ...))
    # TODO: set the tags in polar
    ψᴿⁿ = replacetags(ψᴿⁿ, left_tags => right_tags; plev=1)
    ψᴿⁿ = noprime(ψᴿⁿ, right_tags)
    Cᴿⁿ⁻¹ = replacetags(Cᴿⁿ⁻¹, left_tags => right_tags; plev=1)
    Cᴿⁿ⁻¹ = noprime(Cᴿⁿ⁻¹, right_tags)
    ψᴿ[n] = ψᴿⁿ
    Cᴿ[n - 1] = Cᴿⁿ⁻¹
    λⁿ = norm(Cᴿ[n - 1])
    Cᴿ[n - 1] /= λⁿ
    λ *= λⁿ
    if !isapprox(ψ[n] * Cᴿ[n], λⁿ * Cᴿ[n - 1] * ψᴿ[n]; rtol=1e-10)
      @show norm(ψ[n] * Cᴿ[n] - λⁿ * Cᴿ[n - 1] * ψᴿ[n])
      error("ψ[n] * Cᴿ[n] ≠ λⁿ * Cᴿ[n-1] * ψᴿ[n]")
    end
  end
  return Cᴿ, ψᴿ, λ
end

function left_orthogonalize(
  ψ::InfiniteMPS; left_tags=ts"Left", right_tags=ts"Right", tol::Real=1e-12
)
  Cᴸ, ψᴸ, λᴸ = right_orthogonalize(
    reverse(ψ); left_tags=right_tags, right_tags=left_tags, tol=tol
  )
  # Cᴸ has the unit cell shifted from what is expected
  Cᴸ = reverse(Cᴸ)
  Cᴸ_shift = copy(Cᴸ)
  for n in 1:nsites(Cᴸ)
    Cᴸ_shift[n] = Cᴸ[n + 1]
  end
  return reverse(ψᴸ), Cᴸ_shift, λᴸ
end

# TODO: rename to `orthogonalize(ψ)`? With no limit specified, it is like orthogonalizing to over point.
# Alternatively, it could be called as `orthogonalize(ψ, :)`
function mixed_canonical(
  ψ::InfiniteMPS; left_tags=ts"Left", right_tags=ts"Right", tol::Real=1e-12
)
  _, ψᴿ, _ = right_orthogonalize(ψ; left_tags=ts"", right_tags)
  ψᴸ, C, λ = left_orthogonalize(ψᴿ; left_tags, right_tags)
  if λ ≉ one(λ)
    error("λ should be approximately 1 after orthogonalization, instead it is $λ")
  end
  return InfiniteCanonicalMPS(ψᴸ, C, ψᴿ)
end

ITensorMPS.orthogonalize(ψ::InfiniteMPS, ::Colon; kwargs...) = mixed_canonical(ψ; kwargs...)

#TODO put these functions somewhere else
function ortho_overlap(AC, C)
  AL, _ = polar(AC * dag(C), uniqueinds(AC, C))
  return noprime(AL)
end

function ortho_polar(AC, C)
  UAC, _ = polar(AC, uniqueinds(AC, C))
  UC, _ = polar(C, commoninds(C, AC))
  return noprime(UAC) * noprime(dag(UC))
end
