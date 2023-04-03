
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

#
#  Hm should have the form below.  Only link indices are shown.
#
#    I         0                     0           0      0
#  M-->l=n     0                     0           0      0
#    0    l=n-->M-->l=n-1 ...        0           0      0
#    :         :                     :           :      :
#    0         0          ...  l=2-->M-->l=1     0      0
#    0         0          ...        0        l=1-->M   I
#
#  We need to capture the corner I's and the sub-diagonal Ms, and paste them together into an ITensor.
#  This is facilutated by making all elements of Hm into order(4) tensors by adding dummy Dw=1 indices.
#  We ITensors.directsum() to join all the blocks into the final ITensor.
#  This code should be 100% dense/blocks-sparse agnostic.
#
function cat_to_itensor(Hm::Matrix{ITensor}, inds_array)::Tuple{ITensor,Index,Index}
  lx, ly = size(Hm)
  @assert lx == ly
  #
  #  Extract the sub diagonal
  #
  Ms = map(n -> Hm[lx - n + 1, ly - n], 1:(lx - 1)) #Get the sub diagonal into an array.
  N = length(Ms)
  @assert N == lx - 1
  #
  # Convert edge Ms to order 4 ITensors using the dummy index.
  #
  il0 = inds_array[1][1]
  T = eltype(Ms[1])
  Ms[1] *= onehot(T, il0 => 1)
  Ms[N] *= onehot(T, dag(il0) => 1) #We don't need distinct index IDs for this.  directsum makes all new index anyway.
  #
  # Start direct sums to build the single ITensor.  Order is critical to get the desired form.
  #
  H = Hm[1, 1] * onehot(T, il0' => 1) * onehot(T, dag(il0) => 1) #Bootstrap with the I op in the top left of Hm
  H, il = directsum(H => il0', Ms[N] => inds_array[N][1]) #1-D directsum to put M[N] directly below the I op
  ir = dag(il0) #setup recusion.
  for i in (N - 1):-1:1 #2-D directsums to place new blocks below and to the right.
    H, (il, ir) = directsum(
      H => (il, ir), Ms[i] => inds_array[i]; tags=["Link,left", "Link,right"]
    )
  end
  IN = Hm[N + 1, N + 1] * onehot(T, dag(il0) => 1) * onehot(T, il => dim(il)) #I op in the bottom left of Hm
  H, ir = directsum(H => ir, IN => il0; tags="Link,right") #1-D directsum to the put I in the bottom row, to the right of M[1]

  return H, il, ir
end

function find_all_links(Hms::InfiniteMPOMatrix)
  is = inds(Hms[1][1, 1]) #site inds
  ir = noncommonind(Hms[1][2, 1], is) #This op should have one link index pointing to the right.
  #
  #  Set up return array of 2-tuples
  #
  indexT = typeof(ir)
  TupleT = NTuple{2,indexT}
  inds_array = Vector{TupleT}[]
  #
  #  Make a dummy index
  #
  il0 = Index(ITensors.trivial_space(ir); dir=dir(dag(ir)), tags="Link,l=0")
  #
  #  Loop over sites and nonzero matrix elements which are linked into the next
  #  and previous iMPOMatrix.  
  #
  for n in 1:nsites(Hms)
    Hm = Hms[n]
    inds_n = TupleT[]
    lx, ly = size(Hm)
    @assert lx == ly
    for iy in (ly - 1):-1:1
      ix = iy + 1
      il = ix < lx ? commonind(Hm[ix, iy], Hms[n - 1][ix + 1, iy + 1]) : dag(il0)
      ir = iy > 1 ? commonind(Hm[ix, iy], Hms[n + 1][ix - 1, iy - 1]) : il0
      push!(inds_n, (il, ir))
    end
    push!(inds_array, inds_n)
  end
  return inds_array
end

#
#  Hm is the InfiniteMPOMatrix
#  Hlrs is an array of {ITensor,Index,Index}s, one for each site in the unit cell.
#  Hi is a CelledVector of ITensors.
#
function InfiniteMPO(Hm::InfiniteMPOMatrix)
  inds_array = find_all_links(Hm)
  Hlrs = cat_to_itensor.(Hm, inds_array) #return an array of {ITensor,Index,Index}
  #
  #  Unpack the array of tuples into three arrays.  And also get an array site indices.
  #
  Hi = CelledVector([Hlr[1] for Hlr in Hlrs], translator(Hm))
  ils = CelledVector([Hlr[2] for Hlr in Hlrs], translator(Hm))
  irs = CelledVector([Hlr[3] for Hlr in Hlrs], translator(Hm))
  site_inds = [commoninds(Hm[j][1, 1], Hm[j][end, end])[1] for j in 1:nsites(Hm)]
  #
  #  Create new tags with proper cell and link numbers.  Also daisy chain
  #  all the indices so right index at j = dag(left index at j+1)
  #
  for j in 1:nsites(Hm)
    newTag = "Link,c=$(getcell(site_inds[j])),l=$(getsite(site_inds[j]))"
    ir = replacetags(irs[j], tags(irs[j]), newTag) #new right index
    Hi[j] = replaceinds(Hi[j], [irs[j]], [ir])
    Hi[j + 1] = replaceinds(Hi[j + 1], [ils[j + 1]], [dag(ir)])
  end
  return InfiniteMPO(Hi)
end
