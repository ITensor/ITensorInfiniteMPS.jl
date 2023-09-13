
struct Cell{T}
  cell::T
end

#
# cells
#

celltagprefix() = "c="
celltags(n::Integer) = TagSet(celltagprefix() * string(n))
celltags(n1::Integer, n2::Integer) = TagSet(celltagprefix() * n1 * "|" * n2)

indextagprefix() = "n="
#
# translatecell
#

# TODO: account for shifting by a tuple for multidimensional
# indexing, for example:
# translatecell(ts"Site,c=1×2", (2, 3)) -> ts"Site,c=3×5"

# Determine the cell `n` from the tag `"c=n"`
function getcell(ts::TagSet)
  celltag = tag_starting_with(ts, celltagprefix())
  if isnothing(celltag) #dealing with link legs
    return celltag
  end
  return parse(Int, celltag[(length(celltagprefix()) + 1):end])
end

function getsite(ts::TagSet)
  celltag = tag_starting_with(ts, indextagprefix())
  return parse(Int, celltag[(length(indextagprefix()) + 1):end])
end

##Translation operators

#Default translate cell
function translatecelltags(ts::TagSet, n::Integer)
  ncell = getcell(ts)
  if isnothing(ncell)
    return ts
  end
  return replacetags(ts, celltags(ncell) => celltags(ncell + n))
end

function translatecelltags(i::Index, n::Integer)
  ts = tags(i)
  translated_ts = translatecelltags(ts, n)
  return replacetags(i, ts => translated_ts)
end

#Transfer the functional properties
#translatecell(translator, T::ITensor, n::Integer) = translator(T, n)
function translatecell(translator::Function, T::ITensor, n::Integer)
  return ITensors.setinds(T, translatecell(translator, inds(T), n))
end
translatecell(translator::Function, T::MPO, n::Integer) = translatecell.(translator, T, n)
function translatecell(translator::Function, T::Matrix{ITensor}, n::Integer)
  return translatecell.(translator, T, n)
end
translatecell(translator::Function, i::Index, n::Integer) = translator(i, n)
function translatecell(translator::Function, is::Union{<:Tuple,<:Vector}, n::Integer)
  return translatecell.(translator, is, n)
end

# Default behavior
translatecelltags(T::ITensor, n::Integer) = translatecell(translatecelltags, T, n)
translatecelltags(T::ITensors.Indices, n::Integer) = translatecell(translatecelltags, T, n)
#translatecell(T::MPO, n::Integer) = translatecell.(T, n)
#translatecell(T::Matrix{ITensor}, n::Integer) = translatecell.(T, n)

struct CelledVector{T,F} <: AbstractVector{T}
  data::Vector{T}
  translator::F
end
ITensors.data(cv::CelledVector) = cv.data
translator(cv::CelledVector) = cv.translator

Base.copy(m::CelledVector) = typeof(m)(copy(m.data), m.translator) #needed to carry the translator when copying
Base.deepcopy(m::CelledVector) = typeof(m)(deepcopy(m.data), m.translator) #needed to carry the translator when copying
Base.convert(::Type{CelledVector{T}}, v::Vector) where {T} = CelledVector{T}(v)

function CelledVector{T}(::UndefInitializer, n::Integer) where {T}
  return CelledVector(Vector{T}(undef, n))
end

function CelledVector{T}(::UndefInitializer, n::Integer, translator::Function) where {T}
  return CelledVector(Vector{T}(undef, n), translator::Function)
end
CelledVector(v::AbstractVector) = CelledVector(v, translatecelltags)
function CelledVector{T}(v::Vector{T}) where {T}
  return CelledVector(v, translatecelltags)
end
function CelledVector{T}(v::Vector{T}, translator::Function) where {T}
  return CelledVector(v, translator)
end

"""
    cell_length(cv::CelledVector)

    celllength(cv::CelledVector) # Deprecated

The length of a unit cell of a CelledVector.
"""
cell_length(cv::CelledVector) = length(ITensors.data(cv))

celllength(cv::CelledVector) = cell_length(cv)

# For compatibility with Base
Base.size(cv::CelledVector) = size(ITensors.data(cv))

"""
    cell(cv::CelledVector, n::Integer)

Which unit cell index `n` is in.
"""
function cell(cv::CelledVector, n::Integer)
  _cell = fld1(n, celllength(cv))
  return _cell
end

"""
    cellindex(cv::CelledVector, n::Integer)

Which index in the unit cell index `n` is in.
"""
cellindex(cv::CelledVector, n::Integer) = mod1(n, celllength(cv))

# Get the value at index `n`, where `n` must
# be within the first unit cell
_getindex_cell1(cv::CelledVector, n::Int) = ITensors.data(cv)[n]

# Set the value at index `n`, where `n` must
# be within the first unit cell
_setindex_cell1!(cv::CelledVector, val, n::Int) = (ITensors.data(cv)[n] = val)

# Fallback
#translatecell(x, ::Integer) = x # I think this is useless now

function getindex(cv::CelledVector, n::Int)
  cellₙ = cell(cv, n)
  siteₙ = cellindex(cv, n)
  cellₙ == 1 && return _getindex_cell1(cv, siteₙ) #Avoid unnecessary calls
  return translatecell(cv.translator, _getindex_cell1(cv, siteₙ), cellₙ - 1)
end

# Do we need this definition? Maybe uses generic Julia fallback
#getindex(cv::CelledVector, r::AbstractRange) = [cv[n] for n in r]

function Base.firstindex(cv::CelledVector, c::Cell)
  return (c.cell - 1) * celllength(cv) + 1
end

function Base.lastindex(cv::CelledVector, c::Cell)
  return c.cell * celllength(cv)
end

function Base.keys(cv::CelledVector, c::Cell)
  return firstindex(cv, c):lastindex(cv, c)
end

## function Base.eachindex(cv::CelledVector, c::Cell)
##   return firstindex(cv, c):lastindex(cv, c)
## end

getindex(cv::CelledVector, c::Cell) = cv[eachindex(cv, c)]

function setindex!(cv::CelledVector, T, n::Int)
  cellₙ = cell(cv, n)
  siteₙ = cellindex(cv, n)
  if cellₙ == 1
    _setindex_cell1!(cv, T, siteₙ)
  else
    _setindex_cell1!(cv, translatecell(cv.translator, T, -(cellₙ - 1)), siteₙ)
  end
  return cv
end

celltags(cell) = TagSet("c=$cell")
