
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
  H::Matrix{ITensor}; kwargs...
)
  com_inds = commoninds(H[1, 1], H[end, end])
  left_inds,right_inds=find_links(H)
  qns=hasqns(H[1,1])
  sample_index=com_inds[1]
  tspace=ITensors.trivial_space(sample_index)
  
  left_inds=reverse(left_inds)
  right_inds=reverse(right_inds)
  @assert length(left_inds)>0
  @assert length(right_inds)>0
  
  left_dir=dir(left_inds[1])
  right_dir=dir(right_inds[1])
  # Should we check that all dirs are consistent?

  lx, ly = size(H)
  left_basis = [
    Index(tspace;dir=left_dir,tags= tags(left_inds[1]) ),
    left_inds...,
    Index(tspace;dir=left_dir,tags= tags(left_inds[1]) )
  ]

  right_basis = [
    Index(tspace;dir=right_dir,tags=tags(right_inds[1]) ),
    right_inds...,
    Index(tspace;dir=right_dir,tags=tags(right_inds[1]) )
  ]
  
  # 
  if qns
    left_block = vcat(space.(left_basis)...) #like directsum()
    right_block = vcat(space.(right_basis)...)
  else
    left_block=sum(dim.(left_basis))
    right_block=sum(dim.(right_basis))
  end
  new_left_index=Index(left_block; dir=left_dir, tags=tags(left_basis[1]))
  new_right_index=Index(right_block; dir=right_dir, tags=tags(right_basis[1]))
  
  Hf=ITensor(0.0,new_left_index,new_right_index',com_inds)
  @assert length(left_basis)==lx
  @assert length(right_basis)==ly
  xf=1 #coordinates into Hf1
  for x in 1:lx
    ilb=left_basis[x]
    Dx=dim(ilb)
    yf=1
    for y in 1:ly
       W=copy(H[x,y])
      irb=right_basis[y]
      Dy=dim(irb)
      if !isempty(W)
        if !hasind(W,ilb)
          @assert Dx==1
          W*=onehot(eltype(W), ilb => 1)
        end
        if !hasind(W,irb)
          @assert Dy==1
          W*=onehot(eltype(W), irb => 1)
        end
        prime!(W,irb)
        @assert dim(inds(W,tags=tags(ilb),plev=0)[1])==Dx
        @assert dim(inds(W,tags=tags(irb),plev=1))[1]==Dy
        W=replacetags(W,tags(ilb),tags(new_left_index),plev=0)
        W=replacetags(W,tags(irb),tags(new_right_index),plev=1)
        Hf[new_left_index=>xf:xf+Dx-1,new_right_index'=>yf:yf+Dy-1]=W
        
      end #isempty
      yf+=Dy
     end #for y
     xf+=Dx
  end #for x
  Hf=noprime(Hf,tags="Link")
  return itensor(Hf), new_left_index, new_right_index
end

#
#  Find all the right and left pointing link indices.  Index IDs don't line up from one
#  tensor to the next, so we have to rely on tags to decide what to do.
#
function find_links(H::Matrix{ITensor})
  lx, ly = size(H)
  @assert lx==ly
  
  Ms=map(n->H[lx-n+1,ly-n],1:lx-1) #Get the sub diagonal into an array.
  
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
  return left_inds,right_inds
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
