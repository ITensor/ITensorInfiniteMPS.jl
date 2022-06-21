struct Hᶜ{T}
  ∑h::InfiniteSum{T}
  Hᴸ::InfiniteMPS
  Hᴿ::InfiniteMPS
  ψ::InfiniteCanonicalMPS
  n::Int
end

struct Hᴬᶜ{T}
  ∑h::InfiniteSum{T}
  Hᴸ::InfiniteMPS
  Hᴿ::InfiniteMPS
  ψ::InfiniteCanonicalMPS
  n::Int
end

function (H::Hᶜ)(v)
  Hᶜv = H * v
  ## return Hᶜv * δˡ * δʳ
  return noprime(Hᶜv)
end

function (H::Hᴬᶜ)(v)
  Hᴬᶜv = H * v
  ## return Hᶜv * δˡ⁻¹ * δˢ * δʳ
  return noprime(Hᴬᶜv)
end

# Struct for use in linear system solver
struct Aᴸ
  ψ::InfiniteCanonicalMPS
  n::Int
end

function (A::Aᴸ)(x)
  ψ = A.ψ
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  ψ̃ = prime(linkinds, ψᴴ)
  n = A.n

  N = length(ψ)
  #@assert n == N

  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)

  xT = translatecell(translator(ψ), x, -1)
  for k in (n - N + 1):n
    xT = xT * ψ.AL[k] * ψ̃.AL[k]
  end
  δˡ = δ(l[n], l′[n])
  δʳ = δ(r[n], r′[n])
  xR = x * ψ.C[n] * ψ′.C[n] * dag(δʳ) * denseblocks(δˡ)
  return xT - xR
end

function left_environment(hᴸ, 𝕙ᴸ, ψ; tol=1e-15)
  ψ̃ = prime(linkinds, dag(ψ))
  N = nsites(ψ)

  Aᴺ = Aᴸ(ψ, N)
  Hᴸᴺ¹, info = linsolve(Aᴺ, 𝕙ᴸ[N], 1, -1; tol=tol)
  # Get the rest of the environments in the unit cell
  Hᴸ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴸ[N] = Hᴸᴺ¹
  for n in 1:(N - 1)
    Hᴸ[n] = Hᴸ[n - 1] * ψ.AL[n] * ψ̃.AL[n] + hᴸ[n]
  end
  return Hᴸ
end

# Struct for use in linear system solver
struct Aᴿ
  hᴿ::InfiniteMPS
  ψ::InfiniteCanonicalMPS
  n::Int
end

function (A::Aᴿ)(x)
  hᴿ = A.hᴿ
  ψ = A.ψ
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  ψ̃ = prime(linkinds, ψᴴ)
  n = A.n

  N = length(ψ)
  @assert n == N

  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)

  xT = x
  for k in reverse(1:N)
    xT = xT * ψ.AR[k] * ψ̃.AR[k]
  end
  xT = translatecell(translator(ψ), xT, 1)
  δˡ = δ(l[n], l′[n])
  δʳ = δ(r[n], r′[n])
  xR = x * ψ.C[n] * ψ′.C[n] * δˡ * denseblocks(dag(δʳ))
  return xT - xR
end

function right_environment(hᴿ, ψ; tol=1e-15)
  ψ̃ = prime(linkinds, dag(ψ))
  # XXX: replace with `nsites`
  #N = nsites(ψ)
  N = length(ψ)

  A = Aᴿ(hᴿ, ψ, N)
  Hᴿᴺ¹, info = linsolve(A, hᴿ[N], 1, -1; tol=tol)
  # Get the rest of the environments in the unit cell
  Hᴿ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴿ[N] = Hᴿᴺ¹
  for n in reverse(1:(N - 1))
    Hᴿ[n] = Hᴿ[n + 1] * ψ.AR[n + 1] * ψ̃.AR[n + 1] + hᴿ[n]
  end
  return Hᴿ
end

# TODO Generate all environments, why? Only one is needed in the sequential version
function right_environment(hᴿ, 𝕙ᴿ, ψ; tol=1e-15)
  ψ̃ = prime(linkinds, dag(ψ))
  N = nsites(ψ)

  A = Aᴿ(hᴿ, ψ, N)
  Hᴿᴺ¹, info = linsolve(A, 𝕙ᴿ[N], 1, -1; tol=tol)
  # Get the rest of the environments in the unit cell
  Hᴿ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴿ[N] = Hᴿᴺ¹
  for n in reverse(1:(N - 1))
    Hᴿ[n] = Hᴿ[n + 1] * ψ.AR[n + 1] * ψ̃.AR[n + 1] + hᴿ[n]
  end
  return Hᴿ
end

function tdvp_iteration(args...; multisite_update_alg="sequential", kwargs...)
  if multisite_update_alg == "sequential"
    return tdvp_iteration_sequential(args...; kwargs...)
  elseif multisite_update_alg == "parallel"
    return tdvp_iteration_parallel(args...; kwargs...)
  else
    error(
      "Multisite update algorithm multisite_update_alg = $multisite_update_alg not supported, use \"parallel\" or \"sequential\"",
    )
  end
end

function tdvp(
  solver::Function,
  ∑h,
  ψ;
  eager=true,
  maxiter=10,
  tol=1e-8,
  outputlevel=1,
  multisite_update_alg="sequential",
  time_step,
  solver_tol=(x -> x / 100),
)
  N = nsites(ψ)
  (ϵᴸ!) = fill(tol, nsites(ψ))
  (ϵᴿ!) = fill(tol, nsites(ψ))
  if outputlevel > 0
    println("Running VUMPS with multisite_update_alg = $multisite_update_alg")
    flush(stdout)
    flush(stderr)
  end
  for iter in 1:maxiter
    ψ, (eᴸ, eᴿ) = tdvp_iteration(
      solver,
      ∑h,
      ψ;
      (ϵᴸ!)=(ϵᴸ!),
      (ϵᴿ!)=(ϵᴿ!),
      multisite_update_alg=multisite_update_alg,
      solver_tol=solver_tol,
      time_step=time_step,
      eager,
    )
    ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
    maxdimψ = maxlinkdim(ψ[0:(N + 1)])
    if outputlevel > 0
      println(
        "VUMPS iteration $iter (out of maximum $maxiter). Bond dimension = $maxdimψ, energy = $((eᴸ, eᴿ)), ϵᵖʳᵉˢ = $ϵᵖʳᵉˢ, tol = $tol",
      )
      flush(stdout)
      flush(stderr)
    end
    if ϵᵖʳᵉˢ < tol
      println(
        "Precision error $ϵᵖʳᵉˢ reached tolerance $tol, stopping VUMPS after $iter iterations (of a maximum $maxiter).",
      )
      flush(stdout)
      flush(stderr)
      break
    end
  end
  return ψ
end

function vumps_solver(M, time_step, v₀, solver_tol, eager=true)
  λ⃗, v⃗, info = eigsolve(M, v₀, 1, :SR; ishermitian=true, tol=solver_tol, eager)
  return λ⃗[1], v⃗[1], info
end

return function tdvp_solver(M, time_step, v₀, solver_tol, eager=true)
  v, info = exponentiate(M, time_step, v₀; ishermitian=true, tol=solver_tol, eager)
  v = v / norm(v)
  return nothing, v, info
end

function vumps(
  args...;
  time_step=-Inf,
  eigsolve_tol=(x -> x / 100),
  solver_tol=eigsolve_tol,
  eager=true,
  kwargs...,
)
  @assert isinf(time_step) && time_step < 0
  println("Using VUMPS solver with time step $time_step")
  flush(stdout)
  flush(stderr)
  return tdvp(
    vumps_solver, args...; time_step=time_step, solver_tol=solver_tol, eager, kwargs...
  )
end

function tdvp(args...; time_step, solver_tol=(x -> x / 100), eager=true, kwargs...)
  solver = if !isinf(time_step)
    println("Using TDVP solver with time step $time_step")
    flush(stdout)
    flush(stderr)
    tdvp_solver
  elseif time_step < 0
    # Call VUMPS instead
    println("Using VUMPS solver with time step $time_step")
    flush(stdout)
    flush(stderr)
    vumps_solver
  else
    error("Time step $time_step not supported.")
  end
  return tdvp(solver, args...; time_step=time_step, solver_tol=solver_tol, eager, kwargs...)
end
