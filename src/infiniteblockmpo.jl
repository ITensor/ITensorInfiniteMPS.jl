
mutable struct InfiniteBlockMPO <: AbstractInfiniteMPS
  data::CelledVector{Matrix{ITensor}}
  llim::Int #RealInfinity
  rlim::Int #RealInfinity
  reverse::Bool
end

translator(mpo::InfiniteBlockMPO) = mpo.data.translator
data(mpo::InfiniteBlockMPO) = mpo.data

# TODO better printing?
function Base.show(io::IO, M::InfiniteBlockMPO)
  print(io, "$(typeof(M))")
  (length(M) > 0) && print(io, "\n")
  for i in eachindex(M)
    if !isassigned(M, i)
      println(io, "#undef")
    else
      A = M[i]
      println(io, "Matrix tensor of size $(size(A))")
      for k in 1:size(A, 1), l in 1:size(A, 2)
        if !isassigned(A, k + (size(A, 1) - 1) * l)
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

function getindex(ψ::InfiniteBlockMPO, n::Integer)
  return ψ.data[n]
end

function InfiniteBlockMPO(arrMat::Vector{Matrix{ITensor}})
  return InfiniteBlockMPO(arrMat, 0, size(arrMat, 1), false)
end

function InfiniteBlockMPO(data::Vector{Matrix{ITensor}}, translator::Function)
  return InfiniteBlockMPO(CelledVector(data, translator), 0, size(data, 1), false)
end

function InfiniteBlockMPO(data::CelledVector{Matrix{ITensor}}, m::Int64, n::Int64)
  return InfiniteBlockMPO(data, m, n, false)
end

function InfiniteBlockMPO(data::CelledVector{Matrix{ITensor}})
  return InfiniteBlockMPO(data, 0, size(data, 1), false)
end

function ITensors.siteinds(A::InfiniteBlockMPO)
  data = [dag(only(filterinds(uniqueinds(A[1][1, 1], A[2][1, 1]); plev=0)))]
  for x in 2:(nsites(A)-1)
    append!(
      data,
      [
        dag(
          only(filterinds(uniqueinds(A[x][1, 1], A[x - 1][1, 1], A[x + 1][1, 1]); plev=0))
        ),
      ],
    )
  end
  append!(
    data,
    [dag(only(filterinds(uniqueinds(A[nsites(A)][1, 1], A[nsites(A) - 1][1, 1]); plev=0)))],
  )
  return CelledVector(data, translator(A))
end

function ITensors.splitblocks(H::InfiniteBlockMPO)
  H = copy(H)
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

"""
    local_mpo_block_projectors(is::Index; new_tags = tags(is))

  Build the projectors on the three parts of the itensor used to split a MPO into an InfiniteBlockMPO
  More precisely, create projectors on the first dimension, the 2:end-1 and the last dimension of the index
  Input: is the Index to split
  Output: the triplet of projectors (first, middle, last)
  Optional arguments: new_tags: if we want to change the tags of the index.
"""
function local_mpo_block_projectors(is::Index; tags=tags(is))
  old_dim = dim(is)
  #Build the local projectors.
  #We have to differentiate between the dense and the QN case
  #Note that as far as I know, the MPO even dense is always guaranteed to have identities at both corners
  #If it is not the case, my construction will not work
  top = onehot(dag(is) => 1)
  bottom = onehot(dag(is) => old_dim)
  if length(is.space) == 1
    new_ind = Index(is.space - 2; tags=tags)
    mat = zeros(new_ind.space, is.space)
    for x in 1:(new_ind.space)
      mat[x, x + 1] = 1
    end
    middle = ITensor(copy(mat), new_ind, dag(is))
  else
    new_ind = Index(is.space[2:(end - 1)]; dir=dir(is), tags=tags)
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

"""
    local_mpo_blocks(tensor::ITensor, left_ind::Index, right_ind::Index; left_tags = tags(inds[1]), right_tags = tags(inds[2]), ...)

  Converts a 4-legged tensor (coming from an (infinite) MPO) with two site indices and a left and a right leg into a 3 x 3 matrix of ITensor.
  We assume the normal form for MPO (before full compression) where the top left and bottom right corners are identity matrices.
  The goal is to write the tensor in the form
         1      M_12   M_13
         M_21   M_22   M_23
         M_31   M_32   1
  such that we can then easily compress it. Note that for most of our tensors, the upper triangular part will be 0.
  Input: tensor the four leg tensors and the pair of Index (left_ind, right_ind)
  Output: the 3x3 matrix of tensors
  Optional arguments: left_tags: if we want to change the tags of the left indices.
                      right_tags: if we want to change the tags of the right indices.
"""
function local_mpo_blocks(
  t::ITensor,
  left_ind::Index,
  right_ind::Index;
  left_tags=tags(left_ind),
  right_tags=tags(right_ind),
  kwargs...,
)
  @assert order(t) == 4

  left_dim = dim(left_ind)
  right_dim = dim(right_ind)
  #Build the local projectors.
  top_left, middle_left, bottom_left = local_mpo_block_projectors(left_ind; tags=left_tags)
  top_right, middle_right, bottom_right = local_mpo_block_projectors(
    right_ind; tags=right_tags
  )

  matrix = Matrix{ITensor}(undef, 3, 3)
  for (idx_left, proj_left) in enumerate([top_left, middle_left, bottom_left])
    for (idx_right, proj_right) in enumerate([top_right, middle_right, bottom_right])
      matrix[idx_left, idx_right] = proj_left * t * proj_right
    end
  end
  return matrix
end

"""
    local_mpo_blocks(t::ITensor, ind::Index; new_tags = tags(ind), position = :first, ...)

  Converts a 3-legged tensor (the extremity of a MPO) with two site indices and one leg into a 3 Vector of ITensor.
  We assume the normal form for MPO (before full compression) where the top left and bottom right corners are identity matrices in the bulk.

  Input: tensor the three leg tensors and the index connecting to the rest of the MPO
  Output: the 3x1 or 1x3 vector of tensors
  Optional arguments: new_tags: if we want to change the tags of the indices.
                      position: whether we consider the first term in the MPO or the last.
"""
function local_mpo_blocks(
  t::ITensor, ind::Index; new_tags=tags(ind), position=:first, kwargs...
)
  @assert order(t) == 3
  top, middle, bottom = local_mpo_block_projectors(ind; tags=new_tags)

  if position == :first
    vector = Matrix{ITensor}(undef, 1, 3)
  else
    vector = Matrix{ITensor}(undef, 3, 1)
  end
  for (idx, proj) in enumerate([top, middle, bottom])
    vector[idx] = proj * t
  end
  return vector
end

"""
combineblocks_linkinds_auxiliary(Hcl::InfiniteBlockMPO)

The workhorse of combineblocks_linkinds. We separated them for ease of maintenance.
Fuse the non-site legs of the infiniteBlockMPO Hcl and the corresponding left L and right R environments.
Preserve the corner structure.
Essentially the inverse of splitblocks. It becomes useful for the very dense MPOs once get after compression sometimes.
Input: Hcl the infiniteBlockMPO
Output: a copy of Hcl fused, and the two array of combiners to apply to left and right environments if needed.
"""
function combineblocks_linkinds_auxiliary(H::InfiniteBlockMPO)
  H = copy(H)
  N = nsites(H)
  for j in 1:(N-1)
    right_dim = size(H[j], 2)
    for d in 2:(right_dim-1)
      right_link = only(commoninds(H[j][1, d], H[j + 1][d, 1]))
      comb = combiner(right_link; tags=tags(right_link))
      comb_ind = combinedind(comb)
      for k in 1:size(H[j], 1)
        if isempty(H[j][k, d])
          H[j][k, d] = ITensor(Float64, uniqueinds(H[j][k, d], right_link)..., comb_ind)
        else
          H[j][k, d] = H[j][k, d] * comb
        end
      end
      for k in 1:size(H[j + 1], 2)
        if isempty(H[j + 1][d, k])
          H[j + 1][d, k] = ITensor(
            Float64, uniqueinds(H[j + 1][d, k], dag(right_link))..., dag(comb_ind)
          )
        else
          H[j + 1][d, k] = H[j + 1][d, k] * dag(comb)
        end
      end
    end
  end
  right_dim = size(H[N], 2)
  left_combs = []
  right_combs = []
  for d in 2:(right_dim-1)
    right_link = only(commoninds(H[N][1, d], H[N + 1][d, 1]))
    comb = combiner(right_link; tags=tags(right_link))
    comb_ind = combinedind(comb)
    comb2 = translatecell(translator(H), comb, -1)
    comb_ind2 = translatecell(translator(H), comb_ind, -1)
    for k in 1:size(H[N], 1)
      if isempty(H[N][k, d])
        H[N][k, d] = ITensor(Float64, uniqueinds(H[N][k, d], right_link)..., comb_ind)
      else
        H[N][k, d] = H[N][k, d] * comb
      end
    end
    for k in 1:size(H[1], 2)
      if isempty(H[1][d, k])
        H[1][d, k] = ITensor(
          Float64,
          uniqueinds(H[1][d, k], dag(translatecell(translator(H), right_link, -1)))...,
          dag(comb_ind2),
        )
      else
        H[1][d, k] = H[1][d, k] * dag(comb2)
      end
    end
    append!(left_combs, [comb2])
    append!(right_combs, [dag(comb)])
  end
  return H, left_combs, right_combs
end

"""
combineblocks_linkinds(Hcl::InfiniteBlockMPO, left_environment, right_environment)

Fuse the non-site legs of the infiniteBlockMPO Hcl and the corresponding left L and right R environments.
Preserve the corner structure.
Essentially the inverse of splitblocks. It becomes useful for the very dense MPOs once get after compression sometimes.
Input: Hcl the infiniteBlockMPO, the left environment and right environment
Output: Hcl, left_environment, right_environment the updated MPO and environments
"""
function combineblocks_linkinds(H::InfiniteBlockMPO, left_environment, right_environment)
  H, left_combs, right_combs = combineblocks_linkinds_auxiliary(H)
  left_environment = copy(left_environment)
  for j in 1:length(left_combs)
    if isempty(left_environment[j + 1])
      left_environment[j + 1] = ITensor(
        uniqueinds(left_environment[j + 1], left_combs[j])...,
        uniqueinds(left_combs[j], left_environment[j + 1])...,
      )
    else
      left_environment[j + 1] = left_environment[j + 1] * left_combs[j]
    end
  end
  right_environment = copy(right_environment)
  for j in 1:length(right_combs)
    if isempty(right_environment[j + 1])
      right_environment[j + 1] = ITensor(
        uniqueinds(right_environment[j + 1], right_combs[j])...,
        uniqueinds(right_combs[j], right_environment[j + 1])...,
      )
    else
      right_environment[j + 1] = right_environment[j + 1] * right_combs[j]
    end
  end
  return H, left_environment, right_environment
end

"""
combineblocks_linkinds(Hcl::InfiniteBlockMPO)

Fuse the non-site legs of the infiniteBlockMPO Hcl.
Preserve the corner structure.
Essentially the inverse of splitblocks. It becomes useful for the very dense MPOs once get after compression sometimes.
Input: Hcl the infiniteBlockMPO,
Output: the updated MPO
"""
function combineblocks_linkinds(H::InfiniteBlockMPO)
  H, _ = combineblocks_linkinds_auxiliary(H)
  return H
end
