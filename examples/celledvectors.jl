
import Base: getindex, setindex!

# Create a Cell type?

struct CelledVector{T, F, FINV} <: AbstractVector{T}
  data::Vector{T}
  f::F
end

CelledVector(v::AbstractVector) = CelledVector(v, identity)

cell_length(cv::CelledVector) = length(cv.data)

# Determine which cell the index is in
cell(cv::CelledVector, n::Integer) = fld1(n, cell_length(cv))

# Determine the index in the cell where the index sits
cell_index(cv::CelledVector, n::Integer) = mod1(n, cell_length(cv))

# Return the cell and the cell index
cell_and_cell_index(cv::CelledVector, n::Integer) = fldmod1(n, cell_length(cv))

function getindex(cv::CelledVector, n::Integer)
  cellₙ, cell_indexₙ = cell_and_cell_index(cv, n)
  return cv.f(cv.data[cell_indexₙ], cellₙ)
end

function setindex!(cv::CelledVector, val, n::Integer)
  cellₙ, cell_indexₙ = cell_and_cell_index(cv, n)
  # XXX: is this offset generally correct?
  # It seems to be for "linear" functions.
  cv.data[cell_indexₙ] = cv.f(val, -cellₙ + 2)
  return cv
end

