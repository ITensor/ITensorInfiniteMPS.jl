
#
# AbstractInfiniteMPS
#

# TODO:
# Make a CelledVector type that maps elements from one cell to another?

abstract type AbstractInfiniteMPS <: AbstractMPS end

ITensors.data(ψ::AbstractInfiniteMPS) = ψ.data

(T::Type{MPST})(data::Vector{<:ITensor}, llim, rlim) where {MPST <: AbstractInfiniteMPS} =
  MPST(data, llim, rlim, false)

# TODO: use InfiniteMPS(data, -∞, ∞) when AbstractMPS
# is written more generically
(T::Type{MPST})(data::Vector{<:ITensor}) where {MPST <: AbstractInfiniteMPS} =
  MPST(data, 0, length(data)+1)

# TODO: better way to determine left and right limits
(T::Type{MPST})(ψ::MPS; reverse::Bool = false) where {MPST <: AbstractInfiniteMPS} =
  MPST(ITensors.data(ψ), ψ.llim, ψ.rlim, reverse)

# TODO: better way to determine left and right limits
(T::Type{MPST})(ψ::MPS, llim::Integer, rlim::Integer; reverse::Bool = false) where {MPST <: AbstractInfiniteMPS} =
  MPST(ITensors.data(ψ), llim, rlim; reverse = reverse)

(T::Type{MPST})(N::Integer; reverse::Bool = false) where {MPST <: AbstractInfiniteMPS} =
  MPST(MPS(N); reverse = reverse)

Base.reverse(ψ::MPST) where {MPST <: AbstractInfiniteMPS} =
  MPST(reverse(ψ.data), ψ.rlim, ψ.llim, !ψ.reverse)

# Already defined for AbstractMPS so this
# isn't needed
#data(ψ::AbstractInfiniteMPS) = ψ.data

"""
    nsites(ψ::AbstractInfiniteMPS)

Number of sites in the unit cell of an AbstractInfiniteMPS.
"""
nsites(ψ::AbstractInfiniteMPS) = length(ITensors.data(ψ))

"""
    length(ψ::AbstractInfiniteMPS)

Number of sites in the unit cell of an `AbstractInfiniteMPS` (for the sake
of generic code, this does not return `∞`).
"""
length(ψ::AbstractInfiniteMPS) = nsites(ψ)

isreversed(ψ::AbstractInfiniteMPS) = ψ.reverse

"""
    cell(ψ::AbstractInfiniteMPS, n::Integer)

Which unit cell site `n` is in.
"""
function cell(ψ::AbstractInfiniteMPS, n::Integer)
  _cell = fld1(n, nsites(ψ))
  if isreversed(ψ)
    _cell = -_cell + 2
  end
  return _cell
end

"""
    cellsite(ψ::AbstractInfiniteMPS, n::Integer)

Which site in the unit cell site `n` is in.
"""
cellsite(ψ::AbstractInfiniteMPS, n::Integer) = mod1(n, nsites(ψ))

# Get the MPS tensor on site `n`, where `n` must
# be within the first unit cell
_getindex_cell1(ψ::AbstractInfiniteMPS, n::Integer) = ITensors.data(ψ)[n]

# Set the MPS tensor on site `n`, where `n` must
# be within the first unit cell
_setindex_cell1!(ψ::AbstractInfiniteMPS, val, n::Integer) = (ITensors.data(ψ)[n] = val)

function getindex(ψ::AbstractInfiniteMPS, n::Integer)
  cellₙ = cell(ψ, n)
  siteₙ = cellsite(ψ, n)
  return translatecell(_getindex_cell1(ψ, siteₙ), cellₙ-1)
end

function setindex!(ψ::AbstractInfiniteMPS, T::ITensor, n::Int)
  cellₙ = cell(ψ, n)
  siteₙ = cellsite(ψ, n)
  _setindex_cell1!(ψ, translatecell(T, -(cellₙ-1)), siteₙ)
  return ψ
end

default_link_tags(left_or_right, n) = TagSet("Link,$left_or_right=$n")
default_link_tags(left_or_right, n, cell) = addtags(default_link_tags(left_or_right, n), celltags(cell))

# Make an AbstractInfiniteMPS from a set of site indices
function (::Type{MPST})(ElT::Type, s::Vector{<:Index};
                        linksdir = ITensors.Out,
                        space = any(hasqns, s) ? [QN() => 1] : 1,
                        cell = 1) where {MPST <: AbstractInfiniteMPS}
  s = addtags.(s, (celltags(cell),))
  N = length(s)
  s_hasqns = any(hasqns, s)
  kwargs(n) = if s_hasqns
    (tags = default_link_tags("l", n, cell), dir = linksdir)
  else
    # TODO: support non-QN constructor that accepts `dir`
    (tags = default_link_tags("l", n, cell),)
  end
  l₀ = [Index(space; kwargs(n)...) for n in 1:N]
  l₋₁ᴺ = replacetags(l₀[N], celltags(cell) => celltags(cell-1))
  l = OffsetVector(append!([l₋₁ᴺ], l₀), -1)
  A = [ITensor(ElT, dag(l[n-1]), s[n], l[n]) for n in 1:N]
  return MPST(A)
end

(::Type{MPST})(s::Vector{<:Index}; kwargs...) where {MPST <: AbstractInfiniteMPS} = MPST(Float64, s; kwargs...)

# TODO: Make this more generic, implement as
# setinds(getcell(ψ, 1), getcell(ψ, 0), getcell(ψ, 1),
#         tn -> prime(tn; whichinds = islinkind))
# Can the native AbstractMPS handle this case?
# Additionally, could implement `ψ[0:N+1] -> MPS` to make a finite
# MPS from an InfiniteMPS
function ITensors.prime(::typeof(linkinds), ψ::AbstractInfiniteMPS)
  ψextended′ = prime(linkinds, ψ[0:nsites(ψ)+1])
  return typeof(ψ)(ψextended′[2:end-1]; reverse = ψ.reverse)
end

function ITensors.dag(ψ::AbstractInfiniteMPS)
  ψdag = dag(ψ[1:nsites(ψ)])
  return typeof(ψ)(ψdag; reverse = ψ.reverse)
end

ITensors.linkinds(ψ::AbstractInfiniteMPS, n1n2) =
  linkinds(ψ, Pair(n1n2...))

ITensors.linkinds(ψ::AbstractInfiniteMPS, n1n2::Pair{<:Integer, <:Integer}) =
  commoninds(ψ[n1n2[1]], ψ[n1n2[2]])

ITensors.linkind(ψ::AbstractInfiniteMPS, n1n2::Pair{<:Integer, <:Integer}) =
  commonind(ψ[n1n2[1]], ψ[n1n2[2]])

function ITensors.siteind(ψ::AbstractInfiniteMPS, n::Integer)
  return uniqueind(ψ[n], ψ[n-1], ψ[n+1])
end

function ITensors.siteinds(ψ::AbstractInfiniteMPS, n::Integer)
  return uniqueinds(ψ[n], ψ[n-1], ψ[n+1])
end

# TODO: return a Dictionary or IndexSetNetwork?
function ITensors.siteinds(ψ::AbstractInfiniteMPS, r::AbstractRange)
  return [siteinds(ψ, n) for n in r]
end

siterange(ψ::AbstractInfiniteMPS, c::Cell) = 1:nsites(ψ) .+ (c.cell - 1)

ITensors.siteinds(ψ::AbstractInfiniteMPS, c::Cell) = siteinds(ψ, siterange(ψ, c))

Base.getindex(ψ::AbstractInfiniteMPS, r::UnitRange{Int}) =
  MPS([ψ[n] for n in r])

struct UnitRangeToFunction{T <: Real, F <: Function}
  start::T
  stop::F
end

(r::UnitRangeToFunction)(x) = r.start:r.stop(x)

Base.getindex(ψ::AbstractInfiniteMPS, r::UnitRangeToFunction) =
  finite(typeof(ψ))([ψ[n] for n in r(ψ)])

(::Colon)(n::Int, f::typeof(nsites)) = UnitRangeToFunction(n, f)

function ITensors.linkinds(f::typeof(only), ψ::AbstractInfiniteMPS)
  N = nsites(ψ)
  return CelledVector([f(commoninds(ψ[n], ψ[n + 1])) for n in 1:N])
end

function ITensors.siteinds(f::typeof(only), ψ::AbstractInfiniteMPS)
  N = nsites(ψ)
  return CelledVector([f(uniqueinds(ψ[n], ψ[n - 1], ψ[n + 1])) for n in 1:N])
end

