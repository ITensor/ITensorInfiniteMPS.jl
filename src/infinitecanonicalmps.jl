# TODO: Move to ITensors.jl
function setval(qnval::ITensors.QNVal, val::Int)
  return ITensors.QNVal(ITensors.name(qnval), val, ITensors.modulus(qnval))
end

# TODO: Move to ITensors.jl
function Base.:/(qnval::ITensors.QNVal, n::Int)
  if abs(qnval.modulus) <= 1
    div_val = ITensors.val(qnval) / n
    if !isinteger(div_val)
      error("Dividing $qnval by $n, the resulting QN value is not an integer")
    end
    return setval(qnval, Int(div_val))
  else
    if mod(qnval.val, gcd(qnval.modulus, n)) != 0
      error("Dividing $qnval by $n, no solution to the Chinese remainder theorem")
    end
    #We look for the inverse of n in the equation n x = qnval.val mod qn.modulus.
    #The Chinese remainder theorem guarantees it exists
    #We perform a brute force sieving solution -> should be ok for all reasonnable n
    sol = qnval.val
    for x in 1:n
      if mod(sol, n) != 0
        sol += qnval.modulus
      else
        break
      end
      x == n && error("Failed to find solution") #Bezut identity would be a better solver here
    end
    modulus = lcm(qnval.modulus, n)
    sol = mod(sol, modulus)
    sol = sol < abs(sol - modulus) ? sol : sol - modulus
    return setval(qnval, Int(sol / n))
  end
end

function Base.:*(qnval::ITensors.QNVal, n::Int)
  prod_val = ITensors.val(qnval) * n
  return setval(qnval, Int(prod_val))
end

# TODO: Move to ITensors.jl
function Base.:/(qn::QN, n::Int)
  return QN(map(qnval -> qnval / n, qn.data))
end

function Base.:*(qn::QN, n::Int)
  return QN(map(qnval -> qnval * n, qn.data))
end

# of Index (Tuple, Vector, ITensor, etc.)
indtype(i::Index) = typeof(i)
indtype(T::Type{<:Index}) = T
indtype(is::Tuple{Vararg{Index}}) = eltype(is)
indtype(is::Vector{<:Index}) = eltype(is)
indtype(A::ITensor...) = indtype(inds.(A))

indtype(tn1, tn2) = promote_type(indtype(tn1), indtype(tn2))
indtype(tn) = mapreduce(indtype, promote_type, tn)

function infsiteinds(s::Vector{<:Index}, translator=translatecelltags)
  return CelledVector(addtags(s, celltags(1)), translator)
end

shift_flux_to_zero(s::Vector{Index{Int}}, initestate::Function) = s
shift_flux_to_zero(s::Vector{Index{Int}}, flux_density::QN) = s

function shift_flux_to_zero(s::Vector{<:Index}, initstate::Function)
  return shift_flux_to_zero(s, flux(MPS(s, initstate)))
end

function shift_flux(qnblock::Pair{QN,Int}, flux_density::QN)
  return ((ITensors.qn(qnblock) - flux_density) => ITensors.blockdim(qnblock))
end
function shift_flux(space::Vector{Pair{QN,Int}}, flux_density::QN)
  return map(qnblock -> shift_flux(qnblock, flux_density), space)
end
function shift_flux(i::Index, flux_density::QN)
  return ITensors.setspace(i, shift_flux(space(i), flux_density))
end

function scale_flux(qn::ITensors.QNVal, flux_factor::Int)
  flux_factor == 1 && return qn #This test could occur earlier
  if -1 <= qn.modulus <= 1  #U(1) symmetry, should remain a U(1) symmetry, if qn.modulus = 0, it is a dummy term and the code does nothing
    return qn * flux_factor
  else #To get the same algebra, multiplying the modulus by flux_factor works
    return ITensors.QNVal(qn.name, qn.val * flux_factor, qn.modulus * flux_factor)
  end
end

function scale_flux(qn::ITensors.QNVal, flux_factor::Dict{ITensors.SmallString,Int})
  !haskey(flux_factor, qn.name) && return qn #Ensure that we do not touch empty QNs or QNs that we do not need to rescale because the shift is 0
  return scale_flux(qn, flux_factor[qn.name])
end

function scale_flux(qn::QN, flux_factor::Union{Int,Dict{ITensors.SmallString,Int}})
  return QN(map(qnval -> scale_flux(qnval, flux_factor), qn.data))
end

function scale_flux(
  qnblock::Pair{QN,Int}, flux_factor::Union{Int,Dict{ITensors.SmallString,Int}}
)
  return (scale_flux(ITensors.qn(qnblock), flux_factor) => ITensors.blockdim(qnblock))
end

function scale_flux(
  space::Vector{Pair{QN,Int}}, flux_factor::Union{Int,Dict{ITensors.SmallString,Int}}
)
  return map(qnblock -> scale_flux(qnblock, flux_factor), space)
end

function scale_flux(i::Index, flux_factor::Union{Int,Dict{ITensors.SmallString,Int}})
  return ITensors.setspace(i, scale_flux(space(i), flux_factor))
end

function shift_flux_to_zero(s::Vector{<:Index}, flux::QN)
  if iszero(flux)
    return s
  end
  #We are introducing a factor per QN
  n = length(s)
  multipliers = Dict{ITensors.SmallString,Int}() #multipliers is assigned using the names
  for qn in flux.data
    if qn.val == 0 #We do not need to multiply in this case. It also takes into account the empty QNs which have val and modulus 0
      continue
    end
    multipliers[qn.name] = abs(lcm(qn.val, n) ÷ qn.val)#default solution with positive multiplier
    if abs(qn.modulus) > 1
      #=
      for Zp symmetries, we use the Chinese remainder theorem to find an optimal solution.
      We try to find y such that  y = 0 mod n and y = qn.val mod qn.modulus. Then we just need to find x such that y = n x.

      This system admits a solution iff qn.val = 0 mod( gcd(qn.modulus, n)), which is not necessarily true.
      In order to avoid this problem, we introduce a global multiplier b: qn.val -> b qn.val, qn.modulus -> b qn.modulus.
      If b = n, we always have  n qn.val = 0 mod(  gcd(n qn.modulus, n) = n )
      We can do better: I try to find the smallest b such that:
      b qn.val = 0 mod gcd(n, qn.modulus b).

      Case when things changed: n = 3, initstate = +--, flux = 1 mod 2.
      Previous version: 1 is not divisible by 3, so we mutiply everything by 3. Gives QN ( 4 mod 6, 1 mod 6)
      Current version: gcd(2, 3) = 1, so a solution actually exists: 3 x 1 = 1 mod 2 => work with QN (0 mod 2, 1 mod 2)
      =#
      multipliers[qn.name] = abs(lcm(qn.val, n) ÷ qn.val)
      for b in 1:(multipliers[qn.name] - 1)
        if mod(n, b) != 0 #It is easy to show that the optimal b divides n
          continue
        end
        if mod(qn.val * b, gcd(n, b * qn.modulus)) == 0
          multipliers[qn.name] = b
          break
        end
      end
    end
  end
  s = map(sₙ -> scale_flux(sₙ, multipliers), s)
  flux = scale_flux(flux, multipliers)
  flux_density = flux / n
  return map(sₙ -> shift_flux(sₙ, flux_density), s)
end

function infsiteinds(
  site_tag, n::Int; translator=translatecelltags, initstate=nothing, kwargs...
)
  s = siteinds(site_tag, n; kwargs...)
  s = shift_flux_to_zero(s, initstate)
  return infsiteinds(s, translator)
end

function ITensorMPS.linkinds(ψ::InfiniteMPS)
  N = nsites(ψ)
  return CelledVector([linkinds(ψ, (n, n + 1)) for n in 1:N], translator(ψ))
end

function InfMPS(s::Vector, f::Function, translator::Function=translatecelltags)
  return InfMPS(infsiteinds(s, translator), f)
end

function indval(iv::Pair)
  return ind(iv) => val(iv)
end

zero_qn(i::Index{Int}) = nothing

function zero_qn(i::Index)
  return zero(qn(first(space(i))))
end

function insert_linkinds!(A; left_dir=ITensors.Out)
  # TODO: use `celllength` here
  N = nsites(A)
  l = CelledVector{indtype(A)}(undef, N, translator(A))
  n = N
  s = siteind(A, 1)
  dim = if hasqns(s)
    kwargs = (; dir=left_dir)
    qn_ln = zero_qn(s)
    [qn_ln => 1] #Default to 0 on the right
  else
    kwargs = ()
    1
  end
  l[N] = Index(dim, default_link_tags("l", n, 1); kwargs...)
  for n in 1:(N - 1)
    # TODO: is this correct?
    dim = if hasqns(s)
      qn_ln = flux(A[n]) * left_dir + qn_ln#Fixed a bug on flux conservation
      [qn_ln => 1]
    else
      1
    end
    l[n] = Index(dim, default_link_tags("l", n, 1); kwargs...)
  end
  for n in 1:N
    A[n] = A[n] * onehot(l[n - 1] => 1) * onehot(dag(l[n]) => 1)
  end

  @assert all(i -> flux(i) == zero_qn(s), A) "Flux not invariant under one unit cell translation, not implemented"

  return A
end

function UniformMPS(s::CelledVector, f::Function; left_dir=ITensors.Out)
  sᶜ¹ = s[Cell(1)]
  A = InfiniteMPS([ITensor(sⁿ) for sⁿ in sᶜ¹], translator(s))
  #A.data.translator = translator(s)
  N = length(sᶜ¹)
  for n in 1:N
    Aⁿ = A[n]
    Aⁿ[indval(s[n] => f(n))] = 1.0
    A[n] = Aⁿ
  end
  insert_linkinds!(A; left_dir=left_dir)
  return A
end

function InfMPS(s::CelledVector, f::Function)
  # TODO: rename cell_length
  N = length(s)
  ψL = UniformMPS(s, f; left_dir=ITensors.Out)
  ψR = UniformMPS(s, f; left_dir=ITensors.In)
  ψC = InfiniteMPS(N, translator(s))
  l = linkinds(ψL)
  r = linkinds(ψR)
  for n in 1:N
    ψCₙ = ITensor(dag(l[n])..., r[n]...)
    ψCₙ[l[n]... => 1, r[n]... => 1] = 1.0
    ψC[n] = ψCₙ
  end
  return ψ = InfiniteCanonicalMPS(ψL, ψC, ψR)
end

function finite_mps(ψ::InfiniteCanonicalMPS, range::AbstractRange)
  @assert isone(step(range))
  N = length(range)
  ψ_finite = ψ.AL[range]
  ψ_finite[N] *= ψ.C[last(range)]
  l0 = linkind(ψ.AL, first(range) - 1 => first(range))
  l̃0 = sim(l0)
  lN = linkind(ψ.AR, last(range) => last(range) + 1)
  l̃N = sim(lN)
  δl0 = δ(dag(l̃0), l0)
  δlN = δ(dag(l̃N), lN)
  ψ_finite[1] *= δl0
  ψ_finite[N] *= dag(δlN)
  ψ_finite = MPS([dag(δl0); [ψ_finiteᵢ for ψ_finiteᵢ in ψ_finite]; δlN])
  set_ortho_lims!(ψ_finite, (N + 1):(N + 1))
  return ψ_finite
end
function ITensorMPS.expect(ψ::InfiniteCanonicalMPS, o::String, n::Int)
  s = siteinds(only, ψ.AL)
  O = op(o, s[n])
  ϕ = ψ.AL[n] * ψ.C[n]
  return inner(ϕ, apply(O, ϕ))
end

function ITensorMPS.expect(ψ::InfiniteCanonicalMPS, h::MPO)
  l = linkinds(ITensorInfiniteMPS.only, ψ.AL)
  r = linkinds(ITensorInfiniteMPS.only, ψ.AR)
  s = siteinds(ITensorInfiniteMPS.only, ψ)
  δˢ(n) = ITensorInfiniteMPS.δ(dag(s[n]), prime(s[n]))
  δˡ(n) = ITensorInfiniteMPS.δ(l[n], prime(dag(l[n])))
  δʳ(n) = ITensorInfiniteMPS.δ(dag(r[n]), prime(r[n]))
  ψ′ = prime(dag(ψ))

  ns = ITensorInfiniteMPS.findsites(ψ, h)
  nrange = ns[end] - ns[1] + 1
  idx = 2
  temp_O = δˡ(ns[1] - 1) * ψ.AL[ns[1]] * ψ′.AL[ns[1]] * h[1]
  for n in (ns[1] + 1):(ns[1] + nrange - 1)
    if n == ns[idx]
      temp_O = temp_O * ψ.AL[n] * ψ′.AL[n] * h[idx]
      idx += 1
    else
      temp_O = temp_O * ψ.AL[n] * δˢ(n) * ψ′.AL[n]
    end
  end
  temp_O = temp_O * ψ.C[ns[end]] * δʳ(ns[end]) * ψ′.C[ns[end]]
  return temp_O[]
end

function ITensorMPS.expect(ψ::InfiniteCanonicalMPS, h::InfiniteSum)
  return [expect(ψ, h[j]) for j in 1:nsites(ψ)]
end
