
mutable struct InfiniteMPOMatrix <: AbstractInfiniteMPS
  data::CelledVector{Matrix{ITensor}}
  llim::Int #RealInfinity
  rlim::Int #RealInfinity
  reverse::Bool
end

translator(mpo::InfiniteMPOMatrix) = mpo.data.translator
data(mpo::InfiniteMPOMatrix) = mpo.data

# TODO better printing?
function Base.show(io::IO, M::InfiniteMPOMatrix)
  print(io, "$(typeof(M))")
  (length(M) > 0) && print(io, "\n")
  for i in eachindex(M)
    if !isassigned(M, i)
      println(io, "#undef")
    else
      A = M[i]
      println(io, "Matrix tensor of size $(size(A))")
      for k in 1:size(A)[1], l in 1:size(A)[2]
        if !isassigned(A, k + (size(A)[1] - 1) * l)
          println(io, "[($k, $l)] #undef")
        elseif isempty(A[k, l])
          println(io, "[($k, $l)] empty")
        else
          println(io, "[($k, $l)] $(inds(A[k, l]))")
        end
      end
    end
  end
end

function getindex(ψ::InfiniteMPOMatrix, n::Integer)
  return ψ.data[n]
end

function InfiniteMPOMatrix(arrMat::Vector{Matrix{ITensor}})
  return InfiniteMPOMatrix(arrMat, 0, size(arrMat)[1], false)
end

function InfiniteMPOMatrix(data::Vector{Matrix{ITensor}}, translator::Function)
  return InfiniteMPOMatrix(CelledVector(data, translator), 0, size(data)[1], false)
end

function InfiniteMPOMatrix(data::CelledVector{Matrix{ITensor}}, m::Int64, n::Int64)
  return InfiniteMPOMatrix(data, m, n, false)
end

function InfiniteMPOMatrix(data::CelledVector{Matrix{ITensor}})
  return InfiniteMPOMatrix(data, 0, size(data)[1], false)
end

function ITensors.siteinds(A::InfiniteMPOMatrix)
  return CelledVector(
    [dag(only(filterinds(A[x][1, 1]; plev=0, tags="Site"))) for x in 1:nsites(A)],
    translator(A),
  )
end

function ITensors.splitblocks(H::InfiniteMPOMatrix)
  N = nsites(H)
  for j in 1:N
    for n in 1:length(H)
      H[j][n] = splitblocks(H[j][n])
    end
  end
  return H
end

function find_all_links(Hm::Matrix{ITensor})
  is = inds(Hm[1, 1]) #site inds
  lx, ly = size(Hm)
  #We extract the links from the order-3 tensors on the first column and line
  #We add dummy indices if there is no relevant indices
  ir = only(uniqueinds(Hm[1, 2], is))
  ir0 = Index(ITensors.trivial_space(ir); dir=dir(ir), tags="Link,extra")
  il0 = dag(ir0)
  left_links = typeof(ir)[]
  for x in 1:lx
    temp = uniqueinds(Hm[x, 1], is)
    if length(temp) == 0
      append!(left_links, [il0])
    elseif length(temp) == 1
      append!(left_links, temp)
    else
      error("ITensor does not seem to be of the correct order")
    end
  end
  right_links = typeof(ir)[]
  for x in 1:lx
    temp = uniqueinds(Hm[1, x], is)
    if length(temp) == 0
      append!(right_links, [ir0])
    elseif length(temp) == 1
      append!(right_links, temp)
    else
      error("ITensor does not seem to be of the correct order")
    end
  end
  return left_links, right_links
end

function convert_itensor_to_itensormatrix(tensor; kwargs...)
  if order(tensor) == 3
    return convert_itensor_3vector(tensor; kwargs...)
  elseif order(tensor) == 4
    return convert_itensor_33matrix(tensor; kwargs...)
  else
    error(
      "Conversion of ITensor into matrix of ITensor not planned for this type of tensors"
    )
  end
end

"Build the projectors on the three parts of the itensor used to split a MPO into an InfiniteMPOMatrix"
function build_three_projectors_from_index(is::Index; kwargs...)
  old_dim = dim(is)
  new_tags = get(kwargs, :tags, tags(is))
  #Build the local projectors.
  #We have to differentiate between the dense and the QN case
  #Note that as far as I know, the MPO even dense is always guaranteed to have identities at both corners
  #If it is not the case, my construction will not work
  top = onehot(dag(is) => 1)
  bottom = onehot(dag(is) => old_dim)
  if length(is.space) == 1
    new_ind = Index(is.space - 2; tags=new_tags)
    mat = zeros(new_ind.space, is.space)
    for x in 1:(new_ind.space)
      mat[x, x + 1] = 1
    end
    middle = ITensor(copy(mat), new_ind, dag(is))
  else
    new_ind = Index(is.space[2:(end - 1)]; dir=dir(is), tags=new_tags)
    middle = ITensors.BlockSparseTensor(
      Float64,
      undef,
      Block{2}[Block(x, x + 1) for x in 1:length(new_ind.space)],
      (new_ind, dag(is)),
    )
    for x in 1:length(new_ind.space)
      dim_block = new_ind.space[x][2]
      ITensors.blockview(middle, Block(x, x + 1)) .= diagm(0 => ones(dim_block))
    end
    middle = itensor(middle)
  end
  return top, middle, bottom
end

function convert_itensor_33matrix(tensor; leftdir=ITensors.In, kwargs...)
  @assert order(tensor) == 4
  left_ind = get(kwargs, :leftindex, nothing)
  #Identify the different indices
  sit = filterinds(inds(tensor); tags="Site")
  local_sit = dag(only(filterinds(sit; plev=0)))
  #A bit roundabout as filterinds does not accept dir
  if isnothing(left_ind)
    temp = uniqueinds(tensor, sit)
    if dir(temp[1]) == leftdir
      left_ind = temp[1]
      right_ind = temp[2]
    else
      left_ind = temp[2]
      right_ind = temp[1]
    end
  else
    right_ind = only(uniqueinds(tensor, sit, left_ind))
  end
  left_dim = dim(left_ind)
  right_dim = dim(right_ind)
  #Build the local projectors.
  left_tags = get(kwargs, :left_tags, tags(left_ind))
  top_left, middle_left, bottom_left = build_three_projectors_from_index(
    left_ind; tags=left_tags
  )
  right_tags = get(kwargs, :righ_tags, tags(right_ind))
  top_right, middle_right, bottom_right = build_three_projectors_from_index(
    right_ind; tags=right_tags
  )

  matrix = fill(op("Zero", local_sit), 3, 3)
  for (idx_left, proj_left) in enumerate([top_left, middle_left, bottom_left])
    for (idx_right, proj_right) in enumerate([top_right, middle_right, bottom_right])
      matrix[idx_left, idx_right] = proj_left * tensor * proj_right
    end
  end
  return matrix
end

function convert_itensor_3vector(
  tensor; leftdir=ITensors.In, first=false, last=false, kwargs...
)
  @assert order(tensor) == 3
  #Identify the different indices
  sit = filterinds(inds(tensor); tags="Site")
  local_sit = dag(only(filterinds(sit; plev=0)))
  #A bit roundabout as filterinds does not accept dir
  old_ind = only(uniqueinds(tensor, sit))
  if dir(old_ind) == leftdir || last
    new_tags = get(kwargs, :left_tags, tags(old_ind))
    top, middle, bottom = build_three_projectors_from_index(old_ind; tags=new_tags)
    vector = fill(op("Zero", local_sit), 3, 1)
  else
    new_tags = get(kwargs, :right_tags, tags(old_ind))
    top, middle, bottom = build_three_projectors_from_index(old_ind; tags=new_tags)
    vector = fill(op("Zero", local_sit), 1, 3)
  end
  for (idx, proj) in enumerate([top, middle, bottom])
    vector[idx] = proj * tensor
  end
  return vector
end

#Matrix multiplication for Matrix of ITensors
function Base.:*(A::Matrix{ITensor}, B::Matrix{ITensor})
  size(A, 2) != size(B, 1) && error("Matrix sizes are incompatible")
  C = Matrix{ITensor}(undef, size(A, 1), size(B, 2))
  for i in 1:size(A, 1)
    for j in 1:size(B, 2)
      #TODO something more resilient
      if isempty(A[i, 1]) || isempty(B[1, j])
        C[i, j] = ITensor(uniqueinds(A[i, 1], B[1, j])..., uniqueinds(B[1, j], A[i, 1])...)
      else
        C[i, j] = A[i, 1] * B[1, j]
      end
      for k in 2:size(A, 2)
        if isempty(A[i, k]) || isempty(B[k, j])
          continue
        end
        C[i, j] .+= A[i, k] * B[k, j]
      end
    end
  end
  return C
end

function Base.:*(A::Matrix{ITensor}, B::Vector{ITensor})
  size(A, 2) != length(B) && error("Matrix sizes are incompatible")
  C = Vector{ITensor}(undef, size(A, 1))
  for i in 1:size(A, 1)
    if isempty(A[i, size(A, 2)]) || isempty(B[size(A, 2)])
      C[i] = ITensor(
        uniqueinds(A[i, size(A, 2)], B[size(A, 2)])...,
        uniqueinds(B[size(A, 2)], A[i, size(A, 2)])...,
      )
      if length(inds(C[i])) == 0
        C[i] = ITensor(0)
      end
    else
      C[i] = A[i, size(A, 2)] * B[size(A, 2)]
    end
    for k in reverse(1:(size(A, 2) - 1))
      if isempty(A[i, k]) || isempty(B[k])
        continue
      end
      #This is due to some stuff missing in NDTensors
      if length(inds(C[i])) == 0
        temp = A[i, k] * B[k]
        if !isempty(temp)
          C[i] .+= temp[1]
        end
      else
        C[i] .+= A[i, k] * B[k]
      end
    end
  end
  return C
end

function Base.:*(A::Vector{ITensor}, B::Matrix{ITensor})
  length(A) != size(B, 1) && error("Matrix sizes are incompatible")
  C = Vector{ITensor}(undef, size(B, 2))
  for j in 1:size(B, 2)
    if isempty(A[1]) || isempty(B[1, j])
      C[j] = ITensor(uniqueinds(A[1], B[1, j])..., uniqueinds(B[1, j], A[1])...)
      if length(dims(C[j])) == 0
        C[j] = ITensor(0)
      end
    else
      C[j] = A[1] * B[1, j]
    end
    for k in 2:size(B, 1)
      if isempty(A[k]) || isempty(B[k, j])
        continue
      end
      if length(inds(C[j])) == 0
        temp = A[k] * B[k, j]
        if !isempty(temp)
          C[j] .+= temp[1]
        end
      else
        C[j] .+= A[k] * B[k, j]
      end
    end
  end
  return C
end

function scalar_product(A::Vector{ITensor}, B::Vector{ITensor})
  length(A) != length(B) && error("Vector sizes are incompatible")
  C = A[1] * B[1]
  for k in 2:length(B)
    if isempty(A[k]) || isempty(B[k])
      continue
    end
    C .+= A[k] * B[k]
  end
  return C
end

function Base.:+(A::InfiniteMPOMatrix, B::InfiniteMPOMatrix)
  #Asserts
  nsites(A) != nsites(B) && error("The two MPOs have different lengths, not implemented")
  N = nsites(A)
  s_A = siteinds(A)
  s_B = siteinds(B)
  for x in 1:N
    s_A[x] != s_B[x] && error("The site index are different, impossible to sum")
  end
  translator(A) != translator(B) && error("The two MPOs have different translation rules")
  #Building the sum
  sizes_A = size.(A)
  sizes_B = size.(B)
  new_MPOMatrices = Matrix{ITensor}[
    Matrix{ITensor}(
      undef, sizes_A[x][1] + sizes_B[x][1] - 2, sizes_A[x][2] + sizes_B[x][2] - 2
    ) for x in 1:N
  ] #The identities at the top corner are unchanged
  for j in 1:N
    #Filling up the A part
    for x in 1:(sizes_A[j][1] - 1), y in 1:(sizes_A[j][2] - 1)
      new_MPOMatrices[j][x, y] = A[j][x, y]
    end
    for x in 1:(sizes_A[j][1] - 1)
      new_MPOMatrices[j][x, end] = A[j][x, end]
    end
    for y in 1:(sizes_A[j][2] - 1)
      new_MPOMatrices[j][end, y] = A[j][end, y]
    end
    #new_MPOMatrices[j][end, end] = A[j][end, end] Not needed as B automatically gives it
    #Filling up the B part
    x_shift, y_shift = sizes_A[j] .- 2
    for x in 2:sizes_B[j][1], y in 2:sizes_B[j][2]
      new_MPOMatrices[j][x_shift + x, y_shift + y] = B[j][x, y]
    end
    for x in 2:sizes_B[j][1]
      new_MPOMatrices[j][x_shift + x, 1] = B[j][x, 1]
    end
    for y in 2:sizes_B[j][2]
      new_MPOMatrices[j][1, y_shift + y] = B[j][1, y]
    end
  end
  #Filling up the rest
  for j in 1:N
    s_inds = inds(A[j][1, 1])
    new_MPOMatrices[j][1, end] = A[j][1, end] + B[j][1, end] #ITensor(s_inds)
    new_MPOMatrices[j][end, 1] = A[j][end, 1] + B[j][end, 1] #ITensor(s_inds)
    x_shift, y_shift = sizes_A[j] .- 2
    for x in 2:(sizes_A[j][1] - 1)
      left_index = only(commoninds(A[j][x, x], A[j - 1][x, x]))
      for y in 2:(sizes_B[j][2] - 1)
        right_index = only(commoninds(B[j][y, y], B[j + 1][y, y]))
        new_MPOMatrices[j][x, y_shift + y] = ITensor(left_index, s_inds..., right_index)
      end
    end
    for x in 2:(sizes_B[j][1] - 1)
      left_index = only(commoninds(B[j][x, x], B[j - 1][x, x]))
      for y in 2:(sizes_A[j][2] - 1)
        right_index = only(commoninds(A[j][y, y], A[j + 1][y, y]))
        new_MPOMatrices[j][x_shift + x, y] = ITensor(left_index, s_inds..., right_index)
      end
    end
  end
  #return new_MPOMatrices
  return InfiniteMPOMatrix(new_MPOMatrices, translator(A))
end

function apply_tensor(A::Array{ITensor,N}, B::ITensor...) where {N}
  new_A = copy(A)
  for x in 1:length(new_A)
    new_A[x] = contract(new_A[x], B...)
  end
  return new_A
end
