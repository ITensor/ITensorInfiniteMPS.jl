
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
# We need to capture the corner I's and the sub-diagonal Ms, and paste them together into on ITensor.
# In addition we need make all elements of Hm into order(4) tensors by adding dummy Dw=1 indices.
#
function matrixITensorToITensor(Hm::Matrix{ITensor})
  indexT=typeof(inds(Hm[1,1])[1])
  T=eltype(Hm[1,1])
  lx, ly = size(Hm)
  @assert lx==ly
  #
  #  Extract the sub diagonal
  #
  Ms=map(n->Hm[lx-n+1,ly-n],1:lx-1) #Get the sub diagonal into an array.
  #Ms=map(n->H[n+1,n],1:lx-1) #Get the sub diagonal into an array, reverse order.
  N=length(Ms)
  @assert N==lx-1
  #
  #  We need a dummy link index, but it has to have the right direction.  
  #  Ms[1] should only have one link index, and it should be to the right.
  #
  @assert length(inds(Ms[1],tags="Link"))==1
  ir,=inds(Ms[1],tags="Link")
  left_dir=dir(dag(ir))
  tspace=ITensors.trivial_space(ir)
  il0=Index(tspace;dir=left_dir,tags="Link,l=0") #Dw=1 left dummy index.
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
  ils,irs=indexT[],indexT[] #left the right link indices.
  ir=il0 #set up recursion below.  ir from the previous M, should have the same tags as il on the next M.
  for n in 1:N
    il,=inds(Ms[n],tags=tags(ir)) #Find the left index
    ir=noncommonind(Ms[n],il,tags="Link") #New right index by elimination
    push!(ils,il)
    push!(irs,ir)      
  end 
  #
  # Now direct sum everything non-empty in Hm, including those I ops in the corners.
  # Order is critical.   
  # Details of the Link index tags like "l=n" do not matter from here on.
  #
  H=Hm[1,1]*onehot(T, il0' => 1)*onehot(T, dag(il0) => 1) #Bootstrap with the I op in the top left of Hm
  H,ih=directsum(H=>il0',Ms[N]=>ils[N]) #1-D directsum to put M[N] directly below the I op
  ih=ih,dag(il0) #setup recusion.
  for i in N-1:-1:1 #2-D directsums to place new blocks below and to the right.
      H,ih=directsum(H=>ih,Ms[i]=>(ils[i],irs[i]),tags=["Link","Link"]) 
  end
  IN=Hm[N+1,N+1]*onehot(T, dag(il0) => 1)*onehot(T, ih[1] => dim(ih[1])) #I op in the bottom left of Hm
  H,_=directsum(H=>ih[2],IN=>il0,tags="Link") #1-D directsum to the put I in the bottom row, to the right of M[1]

  ih=inds(H,tags="Link")
  return H,ih[1],ih[2]
end

function InfiniteMPO(H::InfiniteMPOMatrix)
  temp = matrixITensorToITensor.(H)
  new_H = CelledVector([x[1] for x in temp], translator(H))
  lis = CelledVector([x[2] for x in temp], translator(H))
  ris = CelledVector([x[3] for x in temp], translator(H))
  #retags the right_links
  s = [commoninds(H[j][1, 1], H[j][end, end])[1] for j in 1:nsites(H)]
  for j in 1:nsites(H)
    newTag = "Link,c=$(getcell(s[j])),l=$(getsite(s[j]))"
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
