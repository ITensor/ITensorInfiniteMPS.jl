
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

# #
# #  Hm should have the form below.  Only link indices are shown.
# #
# #    I         0                     0           0      0
# #  M-->l=n     0                     0           0      0
# #    0    l=n-->M-->l=n-1 ...        0           0      0
# #    :         :                     :           :      :
# #    0         0          ...  l=2-->M-->l=1     0      0
# #    0         0          ...        0        l=1-->M   I
# #
# #  We need to capture the corner I's and the sub-diagonal Ms, and paste them together into an ITensor.
# #  This is facilutated by making all elements of Hm into order(4) tensors by adding dummy Dw=1 indices.
# #  We ITensors.directsum() to join all the blocks into the final ITensor.
# #  This code should be 100% dense/blocks-sparse agnostic.
# #
# function cat_to_itensor(Hm::Matrix{ITensor}, inds_array)::Tuple{ITensor,Index,Index}
#   lx, ly = size(Hm)
#   @assert lx == ly
#   #
#   #  Extract the sub diagonal
#   #
#   Ms = map(n -> Hm[lx - n + 1, ly - n], 1:(lx - 1)) #Get the sub diagonal into an array.
#   N = length(Ms)
#   @assert N == lx - 1
#   #
#   # Convert edge Ms to order 4 ITensors using the dummy index.
#   #
#   il0 = inds_array[1][1]
#   T = eltype(Ms[1])
#   Ms[1] *= onehot(T, il0 => 1)
#   Ms[N] *= onehot(T, dag(il0) => 1) #We don't need distinct index IDs for this.  directsum makes all new index anyway.
#   #
#   # Start direct sums to build the single ITensor.  Order is critical to get the desired form.
#   #
#   H = Hm[1, 1] * onehot(T, il0' => 1) * onehot(T, dag(il0) => 1) #Bootstrap with the I op in the top left of Hm
#   H, il = directsum(H => il0', Ms[N] => inds_array[N][1]) #1-D directsum to put M[N] directly below the I op
#   ir = dag(il0) #setup recusion.
#   for i in (N - 1):-1:1 #2-D directsums to place new blocks below and to the right.
#     H, (il, ir) = directsum(
#       H => (il, ir), Ms[i] => inds_array[i]; tags=["Link,left", "Link,right"]
#     )
#   end
#   IN = Hm[N + 1, N + 1] * onehot(T, dag(il0) => 1) * onehot(T, il => dim(il)) #I op in the bottom left of Hm
#   H, ir = directsum(H => ir, IN => il0; tags="Link,right") #1-D directsum to the put I in the bottom row, to the right of M[1]
#
#   return H, il, ir
# end

#
#  Hm should have the form below.  Only link indices are shown.
#
#    I               M-->l=n                    :                :               :
#  l=n-->M     l=n-->M-->l=n                    :                :               :
#  l=n-1-->M l=n-1-->M-->l=n ...                ;                :               :
#    :               :                          :                :               :
#  l=1-->M     l=1->M-->l=n          ...  l=1-->M-->l=2    l=1-->M-->l=1   l=1-->M
#    M              M-->l=n          ...        M-->l=2          M-->l=1         I
#
#  We no longer assume any specific form of the matrix of tensor, and we squash all elements together
#  This is facilutated by making all elements of Hm into order(4) tensors by adding dummy Dw=1 indices.
#  We use on line then columns ITensors.directsum() to join all the blocks into the final ITensor.
#  This code should be 100% dense/blocks-sparse agnostic.
#
function cat_to_itensor(Hm::Matrix{ITensor})
  lx, ly = size(Hm)
  T = eltype(Hm[1, 1])
  left_links, right_links = find_all_links(Hm)

  #Start by fusing lines together
  Ls = []
  new_rs = []
  for x in 1:lx
    H, ir = directsum(
      [((y == 1 || y == ly) ? Hm[x, y] * onehot(T, right_links[y] => 1) : Hm[x, y]) =>
        right_links[y] for y in 1:ly]...;
      tags="Link, right",
    )
    if x == 1
      append!(new_rs, [ir])
    else
      replaceinds!(H, [ir], [new_rs[1]])
    end
    append!(Ls, [H])
  end

  H, new_l = directsum(
    [((x == 1 || x == lx) ? Ls[x] * onehot(T, left_links[x] => 1) : Ls[x]) => left_links[x] for
     x in 1:lx]...;
    tags="Link, left",
  )
  return H, new_l, new_rs[1]
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

#
# function find_all_links(Hms::InfiniteMPOMatrix)
#   is = inds(Hms[1][1, 1]) #site inds
#   ir = noncommonind(Hms[1][2, 1], is) #This op should have one link index pointing to the right.
#   #
#   #  Set up return array of 2-tuples
#   #
#   indexT = typeof(ir)
#   TupleT = NTuple{2,indexT}
#   inds_array = Vector{TupleT}[]
#   #
#   #  Make a dummy index
#   #
#   il0 = Index(ITensors.trivial_space(ir); dir=dir(dag(ir)), tags="Link,l=0")
#   #
#   #  Loop over sites and nonzero matrix elements which are linked into the next
#   #  and previous iMPOMatrix.
#   #
#   for n in 1:nsites(Hms)
#     Hm = Hms[n]
#     inds_n = TupleT[]
#     lx, ly = size(Hm)
#     @assert lx == ly
#     for iy in (ly - 1):-1:1
#       ix = iy + 1
#       il = ix < lx ? commonind(Hm[ix, iy], Hms[n - 1][ix + 1, iy + 1]) : dag(il0)
#       ir = iy > 1 ? commonind(Hm[ix, iy], Hms[n + 1][ix - 1, iy - 1]) : il0
#       push!(inds_n, (il, ir))
#     end
#     push!(inds_array, inds_n)
#   end
#   return inds_array
# end

# #
# #  Hm is the InfiniteMPOMatrix
# #  Hlrs is an array of {ITensor,Index,Index}s, one for each site in the unit cell.
# #  Hi is a CelledVector of ITensors.
# #
# function InfiniteMPO(Hm::InfiniteMPOMatrix)
#   inds_array = find_all_links(Hm)
#   Hlrs = cat_to_itensor.(Hm, inds_array) #return an array of {ITensor,Index,Index}
#   #
#   #  Unpack the array of tuples into three arrays.  And also get an array site indices.
#   #
#   Hi = CelledVector([Hlr[1] for Hlr in Hlrs], translator(Hm))
#   ils = CelledVector([Hlr[2] for Hlr in Hlrs], translator(Hm))
#   irs = CelledVector([Hlr[3] for Hlr in Hlrs], translator(Hm))
#   site_inds = [commoninds(Hm[j][1, 1], Hm[j][end, end])[1] for j in 1:nsites(Hm)]
#   #
#   #  Create new tags with proper cell and link numbers.  Also daisy chain
#   #  all the indices so right index at j = dag(left index at j+1)
#   #
#   for j in 1:nsites(Hm)
#     newTag = "Link,c=$(getcell(site_inds[j])),l=$(getsite(site_inds[j]))"
#     ir = replacetags(irs[j], tags(irs[j]), newTag) #new right index
#     Hi[j] = replaceinds(Hi[j], [irs[j]], [ir])
#     Hi[j + 1] = replaceinds(Hi[j + 1], [ils[j + 1]], [dag(ir)])
#   end
#   return InfiniteMPO(Hi)
# end

#
#  Hm is the InfiniteMPOMatrix
#  Hlrs is an array of {ITensor,Index,Index}s, one for each site in the unit cell.
#  Hi is a CelledVector of ITensors.
#
function InfiniteMPO(Hm::InfiniteMPOMatrix)
  Hlrs = cat_to_itensor.(Hm) #return an array of {ITensor,Index,Index}
  #
  #  Unpack the array of tuples into three arrays.  And also get an array site indices.
  #
  Hi = CelledVector([Hlr[1] for Hlr in Hlrs], translator(Hm))
  ils = CelledVector([Hlr[2] for Hlr in Hlrs], translator(Hm))
  irs = CelledVector([Hlr[3] for Hlr in Hlrs], translator(Hm))
  s = siteinds(Hm)
  #
  #  Create new tags with proper cell and link numbers.  Also daisy chain
  #  all the indices so right index at j = dag(left index at j+1)
  #
  for j in 1:nsites(Hm)
    newTag = "Link,c=$(getcell(s[j])),l=$(getsite(s[j]))"
    ir = replacetags(irs[j], tags(irs[j]), newTag) #new right index
    Hi[j] = replaceinds(Hi[j], [irs[j]], [ir])
    Hi[j + 1] = replaceinds(Hi[j + 1], [ils[j + 1]], [dag(ir)])
  end
  return InfiniteMPO(Hi)
end

function convert_itensor_to_itensormatrix(tensor; leftdir=ITensors.In)
  if order(tensor) == 3
    return convert_itensor_3vector(tensor; leftdir)
  elseif order(tensor) == 4
    return convert_itensor_33matrix(tensor; leftdir)
  else
    error(
      "Conversion of ITensor into matrix of ITensor not planned for this type of tensors"
    )
  end
end

function convert_itensor_33matrix(tensor; leftdir=ITensors.In)
  @assert order(tensor) == 4
  #Identify the different indices
  sit = filterinds(inds(tensor); tags="Site")
  local_sit = dag(only(filterinds(sit; plev=0)))
  #A bit roundabout as filterinds does not accept dir
  temp = uniqueinds(tensor, sit)
  if dir(temp[1]) == leftdir
    left_ind = temp[1]
    right_ind = temp[2]
  else
    left_ind = temp[2]
    right_ind = temp[1]
  end
  left_dim = dim(left_ind)
  right_dim = dim(right_ind)
  #Build the local projectors.
  top_left = onehot(dag(left_ind) => 1)
  bottom_left = onehot(dag(left_ind) => left_dim)
  top_right = onehot(dag(right_ind) => 1)
  bottom_right = onehot(dag(right_ind) => right_dim)
  new_left_index = Index(
    left_ind.space[2:(end - 1)]; dir=dir(left_ind), tags=tags(left_ind)
  )
  middle_left = ITensors.BlockSparseTensor(
    eltype(tensor),
    undef,
    Block{2}[Block(x, x + 1) for x in 1:length(new_left_index.space)],
    (new_left_index, dag(left_ind)),
  )
  for x in 1:length(new_left_index.space)
    dim_block = new_left_index.space[x][2]
    ITensors.blockview(middle_left, Block(x, x + 1)) .= diagm(0 => ones(dim_block))
  end
  middle_left = itensor(middle_left)
  new_right_index = Index(
    right_ind.space[2:(end - 1)]; dir=dir(right_ind), tags=tags(right_ind)
  )
  middle_right = ITensors.BlockSparseTensor(
    eltype(tensor),
    undef,
    Block{2}[Block(x, x + 1) for x in 1:length(new_right_index.space)],
    (new_right_index, dag(right_ind)),
  )
  for x in 1:length(new_right_index.space)
    dim_block = new_right_index.space[x][2]
    ITensors.blockview(middle_right, Block(x, x + 1)) .= diagm(0 => ones(dim_block))
  end
  middle_right = itensor(middle_right)

  matrix = fill(op("Zero", local_sit), 3, 3)
  for (idx_left, proj_left) in enumerate([top_left, middle_left, bottom_left])
    for (idx_right, proj_right) in enumerate([top_right, middle_right, bottom_right])
      matrix[idx_left, idx_right] = proj_left * tensor * proj_right
    end
  end
  return matrix
end

function convert_itensor_3vector(tensor; leftdir=ITensors.In)
  @assert order(tensor) == 3
  #Identify the different indices
  sit = filterinds(inds(tensor); tags="Site")
  local_sit = dag(only(filterinds(sit; plev=0)))
  #A bit roundabout as filterinds does not accept dir
  old_ind = only(uniqueinds(tensor, sit))
  old_dim = dim(old_ind)
  #Build the local projectors.
  top = onehot(dag(old_ind) => 1)
  bottom = onehot(dag(old_ind) => old_dim)
  new_ind = Index(old_ind.space[2:(end - 1)]; dir=dir(old_ind), tags=tags(old_ind))
  middle = ITensors.BlockSparseTensor(
    eltype(tensor),
    undef,
    Block{2}[Block(x, x + 1) for x in 1:length(new_ind.space)],
    (new_ind, dag(old_ind)),
  )
  for x in 1:length(new_ind.space)
    dim_block = new_ind.space[x][2]
    ITensors.blockview(middle, Block(x, x + 1)) .= diagm(0 => ones(dim_block))
  end
  middle = itensor(middle)
  if dir(old_ind) == leftdir
    vector = fill(op("Zero", local_sit), 3, 1)
  else
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
