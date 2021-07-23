
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

## # Make an AbstractInfiniteMPS from a set of site indices
## function (::Type{MPST})(ElT::Type, s::Vector{<:Index};
##                         linksdir = ITensors.Out,
##                         space = any(hasqns, s) ? [zero(qn(first(s), Block(1))) => 1] : 1,
##                         cell = 1) where {MPST <: AbstractInfiniteMPS}
##   s = addtags.(s, (celltags(cell),))
##   N = length(s)
##   s_hasqns = any(hasqns, s)
##   kwargs(n) = if s_hasqns
##     (tags = default_link_tags("l", n, cell), dir = linksdir)
##   else
##     # TODO: support non-QN constructor that accepts `dir`
##     (tags = default_link_tags("l", n, cell),)
##   end
##   l₀ = [Index(space; kwargs(n)...) for n in 1:N]
##   l₋₁ᴺ = replacetags(l₀[N], celltags(cell) => celltags(cell-1))
##   l = OffsetVector(append!([l₋₁ᴺ], l₀), -1)
##   A = [ITensor(ElT, l[n-1], s[n], dag(l[n])) for n in 1:N]
##   return MPST(A)
## end

function insert_linkinds!(A)
  # TODO: use `celllength` here
  N = length(A)
  s = siteind(A, 1)
  l = CelledVector{indtype(A)}(undef, N)
  lᴺ = Index(zero(qn(s)), "Link,l=$n,"; dir=ITensors.In)
end

function UniformMPS(s::CelledVector, f::Function)
  sᶜ¹ = s[Cell(1)]
  A = InfiniteMPS([ITensor(sⁿ) for sⁿ in sᶜ¹])
  N = length(sᶜ¹)
  for n in 1:N
    Aⁿ = A[n]
    Aⁿ[indval(s[n] => f(n))] = 1.0
    A[n] = Aⁿ
  end
  insert_linkinds!(A)
  return A
end

function InfMPS(s::CelledVector, f::Function)
  # TODO: rename cell_length
  N = length(s)
  ψL = InfiniteMPS(s.data; linksdir=ITensors.Out)
  ψR = InfiniteMPS(s.data; linksdir=ITensors.In)
  ψC = InfiniteMPS(N)
  l = linkinds(ψL)
  r = linkinds(ψR)
  for n in 1:N
    ψLₙ = ψL[n]
    ψLₙ[l[n - 1]... => 1, indval(s[n] => f(n)), l[n]... => 1] = 1.0
    ψL[n] = ψLₙ
    ψRₙ = ψR[n]
    ψRₙ[r[n - 1]... => 1, indval(s[n] => f(n)), r[n]... => 1] = 1.0
    ψR[n] = ψRₙ
    ψCₙ = ITensor(dag(l[n])..., r[n]...)
    ψCₙ[l[n]... => 1, r[n]... => 1] = 1.0
    ψC[n] = ψCₙ
  end
  ψ = InfiniteCanonicalMPS(ψL, ψC, ψR)
end

