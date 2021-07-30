
struct Cell{T}
  cell::T
end

#
# cells
#

celltagprefix() = "c="
celltags(n::Integer) = TagSet(celltagprefix() * string(n))
celltags(n1::Integer, n2::Integer) = TagSet(celltagprefix() * n1 * "|" * n2)

#
# translatecell
#

# TODO: account for shifting by a tuple, for example:
# translatecell(ts"Site,c=1|2", (2, 3)) -> ts"Site,c=3|5"
# TODO: ts"c=10|12" has too many characters
# TODO: ts"c=1|2|3" has too many characters
#

# Determine the cell `n` from the tag `"c=n"`
function getcell(ts::TagSet)
  celltag = tag_starting_with(ts, celltagprefix())
  return parse(Int, celltag[(length(celltagprefix()) + 1):end])
end

function translatecell(ts::TagSet, n::Integer)
  ncell = getcell(ts)
  return replacetags(ts, celltags(ncell) => celltags(ncell + n))
end

function translatecell(i::Index, n::Integer)
  ts = tags(i)
  translated_ts = translatecell(ts, n)
  return replacetags(i, ts => translated_ts)
end

function translatecell(is::Union{<:Tuple,<:Vector}, n::Integer)
  return translatecell.(is, n)
end

translatecell(T::ITensor, n::Integer) = ITensors.setinds(T, translatecell(inds(T), n))

struct CelledVector{T} <: AbstractVector{T}
  data::Vector{T}
end
ITensors.data(cv::CelledVector) = cv.data

function CelledVector{T}(::UndefInitializer, n::Integer) where {T}
  return CelledVector(Vector{T}(undef, n))
end

"""
    celllength(cv::CelledVector)

The length of a unit cell of a CelledVector.
"""
celllength(cv::CelledVector) = length(ITensors.data(cv))

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
translatecell(x, ::Integer) = x

function getindex(cv::CelledVector, n::Int)
  cellₙ = cell(cv, n)
  siteₙ = cellindex(cv, n)
  return translatecell(_getindex_cell1(cv, siteₙ), cellₙ - 1)
end

# Do we need this definition? Maybe uses generic Julia fallback
#getindex(cv::CelledVector, r::AbstractRange) = [cv[n] for n in r]

function Base.firstindex(cv::CelledVector, c::Cell)
  return (c.cell - 1) * celllength(cv) + 1
end

function Base.lastindex(cv::CelledVector, c::Cell)
  return c.cell * celllength(cv)
end

function Base.eachindex(cv::CelledVector, c::Cell)
  return firstindex(cv, c):lastindex(cv, c)
end

getindex(cv::CelledVector, c::Cell) = cv[eachindex(cv, c)]

function setindex!(cv::CelledVector, T, n::Int)
  cellₙ = cell(cv, n)
  siteₙ = cellindex(cv, n)
  _setindex_cell1!(cv, translatecell(T, -(cellₙ - 1)), siteₙ)
  return cv
end

celltags(cell) = TagSet("c=$cell")

#
# TODO: This version accepts a more general translation function between
# unit cells
#

#struct CelledVector{T, F, FINV} <: AbstractVector{T}
#  data::Vector{T}
#  f::F
#end
#
#CelledVector(v::AbstractVector) = CelledVector(v, identity)
#
#cell_length(cv::CelledVector) = length(cv.data)
#
## Determine which cell the index is in
#cell(cv::CelledVector, n::Integer) = fld1(n, cell_length(cv))
#
## Determine the index in the cell where the index sits
#cell_index(cv::CelledVector, n::Integer) = mod1(n, cell_length(cv))
#
## Return the cell and the cell index
#cell_and_cell_index(cv::CelledVector, n::Integer) = fldmod1(n, cell_length(cv))
#
#function getindex(cv::CelledVector, n::Integer)
#  cellₙ, cell_indexₙ = cell_and_cell_index(cv, n)
#  return cv.f(cv.data[cell_indexₙ], cellₙ)
#end
#
#function setindex!(cv::CelledVector, val, n::Integer)
#  cellₙ, cell_indexₙ = cell_and_cell_index(cv, n)
#  # XXX: is this offset generally correct?
#  # It seems to be for "linear" functions.
#  cv.data[cell_indexₙ] = cv.f(val, -cellₙ + 2)
#  return cv
#end
