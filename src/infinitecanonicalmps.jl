# Get the promoted type of the Index objects in a collection
# of Index (Tuple, Vector, ITensor, etc.)
indtype(i::Index) = typeof(i)
indtype(T::Type{<:Index}) = T
indtype(is::Tuple{Vararg{<:Index}}) = eltype(is)
indtype(is::Vector{<:Index}) = eltype(is)
indtype(A::ITensor...) = indtype(inds.(A))

indtype(tn1, tn2) = promote_type(indtype(tn1), indtype(tn2))
indtype(tn) = mapreduce(indtype, promote_type, tn)

# More general siteind that allows specifying
# the space
function _siteind(site_tag, n::Int; space)
  return addtags(Index(space, "Site,n=$n"), site_tag)
end

_siteinds(site_tag, N::Int; space) = __siteinds(site_tag, N, space)

function __siteinds(site_tag, N::Int, space::Vector)
  return [_siteind(site_tag, n; space=space[n]) for n in 1:N]
end

function __siteinds(site_tag, N::Int, space)
  return [_siteind(site_tag, n; space=space) for n in 1:N]
end

infsiteinds(s::Vector{<:Index}) = CelledVector(addtags(s, celltags(1)))
function infsiteinds(s::Vector{<:Index}, translator)
  return CelledVector(addtags(s, celltags(1)), translator)
end

function infsiteinds(site_tag, N::Int; space=nothing, translator=nothing, kwargs...)
  if !isnothing(space)
    s = _siteinds(site_tag, N; space=space)
  else
    # TODO: add a shift option
    s = siteinds(site_tag, N; kwargs...)
  end
  if !isnothing(translator)
    return infsiteinds(s, translator)
  else
    return infsiteinds(s)
  end
end

function ITensors.linkinds(ψ::InfiniteMPS)
  N = nsites(ψ)
  return CelledVector([linkinds(ψ, (n, n + 1)) for n in 1:N], translator(ψ))
end

function InfMPS(s::Vector, f::Function)
  return InfMPS(infsiteinds(s), f)
end

function InfMPS(s::Vector, f::Function, translator::Function)
  return InfMPS(infsiteinds(s, translator), f)
end

function indval(iv::Pair)
  return ind(iv) => val(iv)
end

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

function ITensors.expect(ψ::InfiniteCanonicalMPS, o, n)
  s = siteinds(only, ψ.AL)
  return (noprime(ψ.AL[n] * ψ.C[n] * op(o, s[n])) * dag(ψ.AL[n] * ψ.C[n]))[]
end

function ITensors.expect(ψ::InfiniteCanonicalMPS, h::MPO)
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

function ITensors.expect(ψ::InfiniteCanonicalMPS, h::InfiniteSum)
  return [expect(ψ, h[j]) for j in 1:nsites(ψ)]
end
