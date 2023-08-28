
#
# InfiniteMPO
#

mutable struct InfiniteMPO <: AbstractInfiniteMPS
  data::CelledVector{ITensor}
  llim::Int #RealInfinity
  rlim::Int #RealInfinity
  reverse::Bool
end

translator(mpo::InfiniteMPO) = mpo.data.translator

InfiniteMPO(data::CelledVector{ITensor}) = InfiniteMPO(data, 0, size(data, 1), false)

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
    #Formatting issues force me to write this as a loop
    input = [Hm[x, 1] * onehot(T, right_links[1] => 1) => right_links[1]]
    for y in 2:(ly - 1)
      append!(input, [Hm[x, y] => right_links[y]])
    end
    append!(input, [Hm[x, ly] * onehot(T, right_links[ly] => 1) => right_links[ly]])
    H, ir = directsum(input...; tags="Link, right")
    if x == 1
      append!(new_rs, [ir])
    else
      replaceinds!(H, [ir], [new_rs[1]])
    end
    append!(Ls, [H])
  end
  input = [Ls[1] * onehot(T, left_links[1] => 1) => left_links[1]]
  for x in 2:(lx - 1)
    append!(input, [Ls[x] => left_links[x]])
  end
  append!(input, [Ls[lx] * onehot(T, left_links[lx] => 1) => left_links[lx]])
  H, new_l = directsum(input...; tags="Link, left")
  return H, new_l, new_rs[1]
end
#
#  Hm is the InfiniteBlockMPO
#  Hlrs is an array of {ITensor,Index,Index}s, one for each site in the unit cell.
#  Hi is a CelledVector of ITensors.
#
function InfiniteMPO(Hm::InfiniteBlockMPO)
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

"""
	fuse_legs!(Hcl::InfiniteMPO, L, R)

    Fuse the non-site legs of the infiniteMPO Hcl and the corresponding left L and right R environments.
    Preserve the corner structure.
		Essentially the inverse of splitblocks. It becomes useful for the very dense MPOs once get after compression sometimes.
		Hcl is modified on site, L and R are not.
		Input: Hcl the infinite MPO
		       L   the left environment (an ITensor)
					 R   the right environment (an ITensor)
		Output: the tuple (newL, newR) the updated environments
"""
function fuse_legs!(Hcl::InfiniteMPO, L, R)
  N = nsites(Hcl)
  for j in 1:(N - 1)
    right_link = only(commoninds(Hcl[j], Hcl[j + 1]))
    top, middle, bottom = local_mpo_block_projectors(right_link)
    top = top * onehot(Index(QN() => 1) => 1)
    bottom = bottom * onehot(Index(QN() => 1) => 1)
    cb = combiner(
      only(uniqueinds(middle, dag(right_link))); dir=dir(right_link), tags=tags(right_link)
    )
    middle = cb * middle
    comb, comb_ind = directsum(
      top => uniqueinds(top, dag(right_link)),
      middle => uniqueinds(middle, dag(right_link)),
      bottom => uniqueinds(bottom, dag(right_link));
      tags=[tags(right_link)],
    )
    Hcl[j] = Hcl[j] * comb
    Hcl[j + 1] = dag(comb) * Hcl[j + 1]
  end
  right_link = only(commoninds(Hcl[N], Hcl[N + 1]))
  right_link2 = translatecell(translator(Hcl), right_link, -1)
  top, middle, bottom = local_mpo_block_projectors(right_link)
  top = top * onehot(Index(QN() => 1) => 1)
  bottom = bottom * onehot(Index(QN() => 1) => 1)
  cb = combiner(
    only(uniqueinds(middle, dag(right_link))); dir=dir(right_link), tags=tags(right_link)
  )
  middle = cb * middle
  comb, comb_ind = directsum(
    top => uniqueinds(top, dag(right_link)),
    middle => uniqueinds(middle, dag(right_link)),
    bottom => uniqueinds(bottom, dag(right_link));
    tags=[tags(right_link)],
  )
  comb2 = translatecell(translator(Hcl), comb, -1)
  comb2_ind = translatecell(translator(Hcl), comb2, -1)
  Hcl[N] = Hcl[N] * comb
  Hcl[1] = dag(comb2) * Hcl[1]
  L = L * comb2
  R = dag(comb) * R
  return L, R
end
