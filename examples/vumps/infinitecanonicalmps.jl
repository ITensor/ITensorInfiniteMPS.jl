using ITensorInfiniteMPS: celltags, default_link_tags

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
function infsiteinds(site_tag, N::Int; space=nothing, kwargs...)
  if !isnothing(space)
    s = _siteinds(site_tag, N; space=space)
  else
    # TODO: add a shift option
    s = siteinds(site_tag, N; kwargs...)
  end
  return infsiteinds(s)
end

function ITensors.linkinds(ψ::InfiniteMPS)
  N = nsites(ψ)
  return CelledVector([linkinds(ψ, (n, n + 1)) for n in 1:N])
end

function InfMPS(s::Vector, f::Function)
  return InfMPS(infsiteinds(s), f)
end

function indval(iv::Pair)
  return ind(iv) => val(iv)
end

function zero_qn(i::Index)
  return zero(qn(first(space(i))))
end

function insert_linkinds!(A; left_dir=ITensors.Out)
  # TODO: use `celllength` here
  N = length(A)
  l = CelledVector{indtype(A)}(undef, N)
  n = N
  qn_ln = zero_qn(siteind(A, 1))
  l[N] = Index([qn_ln => 1], default_link_tags("l", n, 1); dir=left_dir)
  for n in 1:(N - 1)
    # TODO: is this correct?
    qn_ln = (flux(A[n]) + qn_ln) * left_dir
    l[n] = Index([qn_ln => 1], default_link_tags("l", n, 1); dir=left_dir)
  end
  for n in 1:N
    A[n] = A[n] * onehot(l[n - 1] => 1) * onehot(dag(l[n]) => 1)
  end
  return A
end

function UniformMPS(s::CelledVector, f::Function; left_dir=ITensors.Out)
  sᶜ¹ = s[Cell(1)]
  A = InfiniteMPS([ITensor(sⁿ) for sⁿ in sᶜ¹])
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
  ψC = InfiniteMPS(N)
  l = linkinds(ψL)
  r = linkinds(ψR)
  for n in 1:N
    ψCₙ = ITensor(dag(l[n])..., r[n]...)
    ψCₙ[l[n]... => 1, r[n]... => 1] = 1.0
    ψC[n] = ψCₙ
  end
  ψ = InfiniteCanonicalMPS(ψL, ψC, ψR)
end

