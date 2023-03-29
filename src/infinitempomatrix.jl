
mutable struct InfiniteMPOMatrix <: AbstractInfiniteMPS
  data::CelledVector{Matrix{ITensor}}
  llim::Int #RealInfinity
  rlim::Int #RealInfinity
  reverse::Bool
end

translator(mpo::InfiniteMPOMatrix) = mpo.data.translator

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

function matrixITensorToITensor(
  H::Matrix{ITensor}, com_inds, left_dir, right_dir; kwargs...
)
  init_all = get(kwargs, :init_all, true)
  init_left_first = get(kwargs, :init_left_first, init_all)
  init_right_first = get(kwargs, :init_right_first, init_all)
  init_left_last = get(kwargs, :init_left_last, init_all)
  init_right_last = get(kwargs, :init_right_last, init_all)
  #TODO: fix the rev
  rev_all = get(kwargs, :rev_all, false)
  rev_left = get(kwargs, :rev_left, rev_all)
  rev_right = get(kwargs, :rev_right, rev_all)
  #TODO some fixing when rev and init are mixed up

  lx, ly = size(H)
  #Generate in order the leftbasis
  left_basis = valtype(com_inds)[]
  for j in 1:lx
    for k in 1:ly
      temp_ind = filter(x -> dir(x) == left_dir, uniqueinds(H[j, k], com_inds))
      if length(temp_ind) == 1
        append!(left_basis, temp_ind)
        break
      end
    end
  end
  if init_left_first
    left_basis = [
      Index(
        QN() => 1;
        dir=left_dir,
        tags=length(left_basis) > 0 ? tags(left_basis[1]) : "left_link",
      ),
      left_basis...,
    ]  #Dummy index for the first line
  end
  init_left_last && append!(
    left_basis,
    [
      Index(
        QN() => 1;
        dir=left_dir,
        tags=length(left_basis) > 0 ? tags(left_basis[1]) : "left_link",
      ),
    ],
  ) #Dummy index for the last line

  right_basis = valtype(com_inds)[]
  for k in 1:ly
    for j in 1:lx
      temp_ind = filter(x -> dir(x) == right_dir, uniqueinds(H[j, k], com_inds))
      if length(temp_ind) == 1
        append!(right_basis, temp_ind)
        break
      end
    end
  end
  if init_right_first
    right_basis = [
      Index(
        QN() => 1;
        dir=right_dir,
        tags=length(right_basis) > 0 ? tags(right_basis[1]) : "right_link",
      ),
      right_basis...,
    ] #Dummy index for the first column
  end
  init_right_last && append!(
    right_basis,
    [
      Index(
        QN() => 1;
        dir=right_dir,
        tags=length(right_basis) > 0 ? tags(right_basis[1]) : "right_link",
      ),
    ],
  ) #Dummy index for the last column

  left_block = Vector{Pair{QN,Int64}}()
  dic_inv_left_ind = Dict{Tuple{UInt64,Int64},Int64}()
  for index in left_basis
    for (n, qp) in enumerate(index.space)
      append!(left_block, [qp])
      dic_inv_left_ind[index.id, n] = length(left_block)
    end
  end
  new_left_index = if length(left_basis) == 1
    left_basis[1]
  else
    Index(left_block; dir=left_dir, tags=tags(left_basis[1]))
  end

  right_block = Vector{Pair{QN,Int64}}()
  dic_inv_right_ind = Dict{Tuple{UInt64,Int64},Int64}()
  for index in right_basis
    for (n, qp) in enumerate(index.space)
      append!(right_block, [qp])
      dic_inv_right_ind[index.id, n] = length(right_block)
    end
  end
  new_right_index = if length(right_basis) == 1
    right_basis[1]
  else
    Index(right_block; dir=right_dir, tags=tags(right_basis[1]))
  end
  #Determine the non-zero blocks, not efficient in memory for now TODO: improve memory use
  temp_block = Block{4}[]
  elements = []
  dummy_left = ITensor(1, Index(QN() => 1))
  dum_left_ind = only(inds(dummy_left))
  dummy_right = ITensor(1, Index(QN() => 1))
  dum_right_ind = only(inds(dummy_right))
  for x in 1:lx
    for y in 1:ly
      isempty(H[x, y]) && continue
      tli = filter(x -> dir(x) == dir(left_basis[1]), commoninds(H[x, y], left_basis))
      #This first part find which structure the local Ham has and ensure H[x, y] is properly ordered
      case = 0
      if !isempty(tli)
        li = only(tli)
        tri = filter(x -> dir(x) == dir(right_basis[1]), commoninds(H[x, y], right_basis))
        if !isempty(tri)
          ri = only(tri)
          T = permute(H[x, y], com_inds..., li, ri; allow_alias=true)
          case = 3 #This is the default case, both legs exists
        else
          #println(x, y)
          !(y == 1 || (x == lx && y == ly)) && error("Incompatible leg")
          T = permute(
            H[x, y] * dummy_right, com_inds..., li, dum_right_ind; allow_alias=true
          )
          case = 1
        end
      else
        !(x == lx || (x == 1 && y == 1)) && error("Incompatible leg")
        tri = filter(x -> dir(x) == dir(right_basis[1]), commoninds(H[x, y], right_basis))
        if !isempty(tri)
          ri = only(tri)
          T = permute(H[x, y] * dummy_left, com_inds..., dum_left_ind, ri; allow_alias=true)
          case = 2
        else
          !((x==1 && y == 1) || (x==lx && y == 1) ||(x == lx && y == ly)) && error("Incompatible leg")
          T = permute(
            H[x, y] * dummy_left * dummy_right,
            com_inds...,
            dum_left_ind,
            dum_right_ind;
            allow_alias=true,
          )
        end
      end
      for (n, b) in enumerate(eachnzblock(T))
        #TODO not completely ok for attribution to 1 and end when stuff is missing
        norm(T[b]) == 0 && continue
        if case == 0
          if x == 1 && y == 1
            append!(
              temp_block,
              [
                Block(
                  b[1],
                  b[2],
                  dic_inv_left_ind[left_basis[1].id, 1],
                  dic_inv_right_ind[right_basis[1].id, 1],
                ),
              ],
            )
          elseif x==lx && y == 1
            append!(
              temp_block,
              [
                Block(
                  b[1],
                  b[2],
                  dic_inv_left_ind[left_basis[end].id, 1],
                  dic_inv_right_ind[right_basis[1].id, 1],
                ),
              ],
            )
          elseif x == lx && y == ly
            append!(
              temp_block,
              [
                Block(
                  b[1],
                  b[2],
                  dic_inv_left_ind[left_basis[end].id, 1],
                  dic_inv_right_ind[right_basis[end].id, 1],
                ),
              ],
            )
          else
            error("Something went wrong at $((x, y))")
          end
        elseif case == 1
          append!(
            temp_block,
            [
              Block(
                b[1],
                b[2],
                dic_inv_left_ind[li.id, b[3]],
                dic_inv_right_ind[right_basis[1].id, 1],
              ),
            ],
          )
        elseif case == 2
          append!(
            temp_block,
            [
              Block(
                b[1],
                b[2],
                dic_inv_left_ind[left_basis[end].id, 1],
                dic_inv_right_ind[ri.id, b[4]],
              ),
            ],
          )
        elseif case == 3 #Default case
          append!(
            temp_block,
            [
              Block(
                b[1], b[2], dic_inv_left_ind[li.id, b[3]], dic_inv_right_ind[ri.id, b[4]]
              ),
            ],
          )
        else
          println("Not treated case")
        end
        append!(elements, [T[b]])
      end
    end
  end
  Hf = ITensors.BlockSparseTensor(
    eltype(elements[1]), undef, temp_block, (com_inds..., new_left_index, new_right_index)
  )
  for (n, b) in enumerate(temp_block)
    ITensors.blockview(Hf, b) .= elements[n]
  end
  return itensor(Hf), new_left_index, new_right_index
end

function matrixITensorToITensor(H::Vector{ITensor}, com_inds; rev=false, kwargs...)
  rev && error("not yet implemented")
  init_all = get(kwargs, :init_all, true)
  init_first = get(kwargs, :init_first, init_all)
  init_last = get(kwargs, :init_last, init_all)

  lx = length(H)
  #Generate in order the leftbasis
  left_basis = valtype(com_inds)[] #Dummy index for the first line
  for j in 1:lx
    append!(left_basis, uniqueinds(H[j], com_inds))
  end
  left_dir = dir(left_basis[1])
  if init_first
    left_basis = [Index(QN() => 1; dir=left_dir), left_basis...]
  end
  init_last && append!(left_basis, [Index(QN() => 1; dir=left_dir)]) #Dummy index for the last line

  left_block = Vector{Pair{QN,Int64}}()
  dic_inv_left_ind = Dict{Tuple{UInt64,Int64},Int64}()
  for index in left_basis
    for (n, qp) in enumerate(index.space)
      append!(left_block, [qp])
      dic_inv_left_ind[index.id, n] = length(left_block)
    end
  end
  new_left_index = Index(left_block; dir=left_dir, tags="left_link")

  #Determine the non-zero blocks, not efficient in memory for now TODO: improve memory use
  temp_block = Block{3}[]
  elements = []
  for x in 1:lx
    isempty(H[x]) && continue
    tli = commoninds(H[x], left_basis)
    #This first part find which structure the local Ham has and ensure H[x, y] is properly ordered
    case = 0
    if !isempty(tli)
      li = only(tli)
      T = permute(H[x], com_inds..., li; allow_alias=true)
      case = 1 #This is the default case, the leg exists
    else
      !(x == lx || x == 1) && error("Incompatible leg")
      T = permute(H[x], com_inds...; allow_alias=true)
    end
    for (n, b) in enumerate(eachnzblock(T))
      norm(T[b]) == 0 && continue
      if case == 0
        if x == 1
          append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[left_basis[1].id, 1])])
        elseif x == lx
          append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[left_basis[end].id, 1])])
        else
          error("Something went wrong")
        end
      elseif case == 1 #Default case
        append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[li.id, b[3]])])
      else
        println("Not treated case")
      end
      append!(elements, [T[b]])
    end
  end
  Hf = ITensors.BlockSparseTensor(
    eltype(elements[1]), undef, temp_block, (com_inds..., new_left_index)
  )
  for (n, b) in enumerate(temp_block)
    ITensors.blockview(Hf, b) .= elements[n]
  end
  return itensor(Hf), new_left_index
end

function matrixITensorToITensor(H::Matrix{ITensor}, idleft, idright; kwargs...)
  lx, ly = size(H)
  s = commoninds(H[1, 1], H[end, end])
  right_ind = uniqueinds(H[end, end], s)
  for j in reverse(1:(ly - 1))
    length(right_ind) == 1 && break
    right_ind = uniqueinds(H[end, j], s)
  end
  length(right_ind) != 1 && error("Not able to isolate the right index")
  left_ind = uniqueinds(H[end, end], s)
  for j in 2:lx
    left_ind = uniqueinds(H[j, 1], s)
    length(left_ind) == 1 && break
  end
  length(left_ind) != 1 && error("Not able to isolate the left index")
  dir_left_ind = dir(only(left_ind))
  dir_right_ind = dir(only(right_ind))
  #return matrixITensorToITensor(H, s, left_tag, right_tag; dir_left_ind, dir_right_ind)
  if length(collect(idleft)) == 1 || length(collect(idright)) == 1
    return matrixITensorToITensor(H[idleft, idright], s; kwargs...)
  end
  return matrixITensorToITensor(
    H[idleft, idright], s, dir_left_ind, dir_right_ind; kwargs...
  )
end

function matrixITensorToITensor(H::Matrix{ITensor}; kwargs...)
  lx, ly = size(H)
  s = commoninds(H[1, 1], H[end, end])
  right_ind = uniqueinds(H[end, end], s)
  for j in reverse(1:(ly - 1))
    length(right_ind) == 1 && break
    right_ind = uniqueinds(H[end, j], s)
  end
  length(right_ind) != 1 && error("Not able to isolate the right index")
  left_ind = uniqueinds(H[end, end], s)
  for j in 2:lx
    left_ind = uniqueinds(H[j, 1], s)
    length(left_ind) == 1 && break
  end
  length(left_ind) != 1 && error("Not able to isolate the left index")
  dir_left_ind = dir(only(left_ind))
  dir_right_ind = dir(only(right_ind))
  #return matrixITensorToITensor(H, s, left_tag, right_tag; dir_left_ind, dir_right_ind)
  return matrixITensorToITensor(H, s, dir_left_ind, dir_right_ind; kwargs...)
end

function InfiniteMPO(H::InfiniteMPOMatrix)
  temp = matrixITensorToITensor.(H)
  new_H = CelledVector([x[1] for x in temp], translator(H))
  lis = CelledVector([x[2] for x in temp], translator(H))
  ris = CelledVector([x[3] for x in temp], translator(H))
  #retags the right_links
  s = [commoninds(H[j][1, 1], H[j][end, end])[1] for j in 1:nsites(H)]
  for j in 1:nsites(H)
    newTag = "Link,c=$(getcell(s[j])),n=$(getsite(s[j]))"
    temp = replacetags(ris[j], tags(ris[j]), newTag)
    new_H[j] = replaceinds(new_H[j], [ris[j]], [temp])
    ris[j] = replacetags(ris[j], tags(ris[j]), newTag)
  end
  # joining the indexes
  for j in 1:nsites(H)
    temp = δ(dag(ris[j]), dag(lis[j + 1]))
    new_H[j + 1] *= temp
  end
  return InfiniteMPO(new_H)
end
