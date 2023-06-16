
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
    # H, ir = directsum(
    #   [((y == 1 || y == ly) ? Hm[x, y] * onehot(T, right_links[y] => 1) : Hm[x, y]) =>
    #     right_links[y] for y in 1:ly]...;
    #   tags="Link, right",
    # )
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
  # H, new_l = directsum(
  #   [((x == 1 || x == lx) ? Ls[x] * onehot(T, left_links[x] => 1) : Ls[x]) => left_links[x] for
  #    x in 1:lx]...;
  #   tags="Link, left",
  # )
  return H, new_l, new_rs[1]
end
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
