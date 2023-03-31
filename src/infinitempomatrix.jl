
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
#  H should have the form below.  Only link indices are shown.
#
#    I         0                     0           0      0
#  M-->l=n     0                     0           0      0
#    0    l=n-->M-->l=n-1 ...        0           0      0
#    :         :                     :           :      :
#    0         0          ...  l=2-->M-->l=1     0      0
#    0         0          ...        0        l=1-->M   I
#
# We need to capture the corner I's and the sub-diagonal Ms, and paste them together into on ITensor.
# In addition we need make all elements of H into order(4) tensors by adding dummy Dw=1 indices.
#
function matrixITensorToITensor(
  Hm::Matrix{ITensor}; kwargs...
)
  indexT=typeof(inds(Hm[1,1])[1])
  T=eltype(Hm[1,1])
  lx, ly = size(Hm)
  @assert lx==ly
  #
  #  Extract the sub diagonal
  #
  Ms=map(n->Hm[lx-n+1,ly-n],1:lx-1) #Get the sub diagonal into an array.
  #Ms=map(n->H[n+1,n],1:lx-1) #Get the sub diagonal into an array, reverse order.
  #
  #  Create list of all left and right link indices on the Ms.  In orefr to support dense storage we 
  #  can't QN directions to decide left/right.  Instead we look for common tags among the neighbourinf Ms.
  #
  N=length(Ms)
  @assert N==lx-1
  ils,irs=indexT[],indexT[]
  @assert length(inds(Ms[1],tags="Link"))==1
  ir,=inds(Ms[1],tags="Link")
  push!(irs,ir)
  for n in 2:N
    il,=inds(Ms[n],tags=tags(ir))
    ir=noncommonind(Ms[n],il,tags="Link")
    push!(ils,il)
    if !isnothing(ir) # ir==nothing on the last M
      push!(irs,ir)      
    end
  end 
  #
  # Make some dummy indices.  Details of the Link indices do not matter from here on.
  #
  left_dir=dir(ils[1])
  right_dir=dir(irs[1])
  tspace=ITensors.trivial_space(ils[1])
  i0=Index(tspace;dir=left_dir,tags="Link")
  in=Index(tspace;dir=right_dir,tags="Link")
  inp=prime(dag(in))
  ils=[i0,ils...]
  
  Ms[1]*=onehot(T, i0 => 1)
  Ms[N]*=onehot(T, in => 1)

  H=Hm[1,1]*onehot(T, inp => 1)*onehot(T, in => 1) #Bootstrap with the I op in the top left of Hm
  H,ih=directsum(H=>inp,Ms[N]=>ils[N]) #1-D directsum to put M[N] directly below the I op
  ih=ih,in
  for i in N-1:-1:1
      ilm=ils[i],irs[i]
      H,ih=directsum(H=>ih,Ms[i]=>ilm,tags=["Link","Link"]) #2-D directsums to place new blocks below and to the right.
  end
  IN=Hm[N+1,N+1]*onehot(T, dag(i0) => 1)*onehot(T, ih[1] => dim(ih[1])) #I op in the bottom left of Hm
  H,_=directsum(H=>ih[2],IN=>i0,tags="Link") #1-D directsum to the put I in the bottom row, to the right of M[1]

  ih=inds(H,tags="Link")
  return H,ih[1],ih[2]
end

#
#  Find all the right and left pointing link indices.  Index IDs don't line up from one
#  tensor to the next, so we have to rely on tags to decide what to do.
#
function find_links(H::Matrix{ITensor})
  lx, ly = size(H)
  @assert lx==ly
  
  Ms=map(n->H[lx-n+1,ly-n],1:lx-1) #Get the sub diagonal into an array.
  #Ms=map(n->H[n+1,n],1:lx-1) #Get the sub diagonal into an array.
  
  indexT=typeof(inds(Ms[1],tags="Link")[1])
  left_inds,right_inds=indexT[],indexT[]
  @assert length(inds(Ms[1],tags="Link"))==1
  ir,=inds(Ms[1],tags="Link")
  push!(right_inds,ir)
  for n in 2:length(Ms)
    il,=inds(Ms[n],tags=tags(ir))
    #@show ir noncommonind(Ms[n],ir,tags="Link")
    ir=noncommonind(Ms[n],il,tags="Link")
    push!(left_inds,il)
    if !isnothing(ir)
      push!(right_inds,ir)      
    end
  end 
  return left_inds,right_inds,Ms
end

#
#  H should have the form below.  Only link indices are shown.
#
#    I        0                   0          0      0
#  M--l=n     0                   0          0      0
#    0    l=n--M--l=n-1 ...       0          0      0
#    :        :                   :          :      :
#    0        0         ...  l=2--M--l=1     0      0
#    0        0         ...       0        l=1--M   I
#

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
