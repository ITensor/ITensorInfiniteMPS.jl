
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
function matrixITensorToITensor(Hm::Matrix{ITensor})::ITensor
  indexT=typeof(inds(Hm[1,1])[1])
  T=eltype(Hm[1,1])
  lx, ly = size(Hm)
  @assert lx==ly
  #
  #  Extract the sub diagonal
  #
  Ms=map(n->Hm[lx-n+1,ly-n],1:lx-1) #Get the sub diagonal into an array.
  N=length(Ms)
  @assert N==lx-1
  #
  #  We need a dummy link index, but it has to have the right direction.  
  #  Ms[1] should only have one link index which should be pointing to the right.
  #
  @assert length(inds(Ms[1],tags="Link"))==1
  ir,=inds(Ms[1],tags="Link")
  il0=Index(ITensors.trivial_space(ir);dir=dir(dag(ir)),tags="Link,l=0") #Dw=1 left dummy index.
  #
  # Convert edge Ms to order 4 ITensors using the dummy index.
  #
  Ms[1]*=onehot(T, il0 => 1)
  Ms[N]*=onehot(T, dag(il0) => 1) #We don't need distinct index IDs for this.  directsum makes all new index anyway.
  #
  #  Create list of all left and right link indices on the Ms.  In order to support dense storage we 
  #  can't use QN directions to decide left/right.  Instead we look for common tags among the neighbouring Ms.
  #  Right now the code assumes plev=0 for all Ms.
  #
  is=NTuple{2,indexT}[]
  ir=il0 #set up recursion below.  ir is from the previous M, should have the same tags as il on the next M.
  for n in 1:N
    il,=inds(Ms[n],tags=tags(ir)) #Find the left index
    ir=noncommonind(Ms[n],il,tags="Link") #Find the new right index by elimination
    push!(is,(il,ir))    
  end 
  #
  # Now direct sum everything non-empty in Hm, including those I ops in the corners.
  # Order is critical.   
  # Details of the Link index tags like "l=n" do not matter from here on.
  #
  H=Hm[1,1]*onehot(T, il0' => 1)*onehot(T, dag(il0) => 1) #Bootstrap with the I op in the top left of Hm
  H,ih=directsum(H=>il0',Ms[N]=>is[N][1]) #1-D directsum to put M[N] directly below the I op
  ih=ih,dag(il0) #setup recusion.
  for i in N-1:-1:1 #2-D directsums to place new blocks below and to the right.
      H,ih=directsum(H=>ih,Ms[i]=>is[i],tags=["Link,left","Link,right"]) 
  end
  IN=Hm[N+1,N+1]*onehot(T, dag(il0) => 1)*onehot(T, ih[1] => dim(ih[1])) #I op in the bottom left of Hm
  H,_=directsum(H=>ih[2],IN=>il0,tags="Link,right") #1-D directsum to the put I in the bottom row, to the right of M[1]
  return H
end
#
#  Hm is the InfiniteMPOMatrix
#  Hs is an array of ITensors, one for the site in the unit cell.
#  Hi is a CelledVector of ITensors.
#
function InfiniteMPO(Hm::InfiniteMPOMatrix)
  Hs = matrixITensorToITensor.(data(Hm))
  Hi = CelledVector([H for H in Hs], translator(Hm))
  lis = CelledVector([inds(H,tags="left")[1] for H in Hs], translator(Hm))
  ris = CelledVector([inds(H,tags="right")[1] for H in Hs], translator(Hm))
  site_inds = [commoninds(Hm[j][1, 1], Hm[j][end, end])[1] for j in 1:nsites(Hm)]
  #
  #  Create new tags with proper cell and link numbers.  Also daisy chain
  #  all the indices so right index at j = dag(left index at j+1)
  #
  for j in 1:nsites(Hm)
    newTag = "Link,c=$(getcell(site_inds[j])),l=$(getsite(site_inds[j]))"
    ir = replacetags(ris[j], tags(ris[j]), newTag) #new right index
    Hi[j] = replaceinds(Hi[j], [ris[j]], [ir])
    Hi[j + 1] = replaceinds(Hi[j + 1],[lis[j + 1]],[dag(ir)])
  end
  return InfiniteMPO(Hi)
end
