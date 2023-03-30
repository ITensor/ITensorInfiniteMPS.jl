import ITensors: dim, dims, DenseTensor, QNIndex, eachindval, eachval, getindex, setindex!
import NDTensors: getperm, permute, BlockDim, blockstart, blockend
import Base.range


struct IndexRange
    index::Index
    range::UnitRange{Int64}
    function IndexRange(i::Index,r::UnitRange)
        return new(i,r)
    end
end
irPair{T}=Pair{<:Index{T},UnitRange{Int64}} where {T}
irPairU{T}=Union{irPair{T},IndexRange} where {T}
IndexRange(ir::irPair{T}) where {T} =IndexRange(ir.first,ir.second)
IndexRange(ir::IndexRange)=IndexRange(ir.index,ir.range)

start(ir::IndexRange)=range(ir).start
range(ir::IndexRange)=ir.range
range(i::Index)=1:dim(i)
ranges(irs::Tuple) = ntuple(i -> range(irs[i]), Val(length(irs)))
indices(irs::Tuple{Vararg{IndexRange}}) = map((ir)->ir.index ,irs)
indranges(ips::Tuple{Vararg{irPairU}}) = map((ip)->IndexRange(ip) ,ips)

dim(ir::IndexRange)=dim(range(ir))
dim(r::UnitRange{Int64})=r.stop-r.start+1
dims(irs::Tuple{Vararg{IndexRange}})=map((ir)->dim(ir),irs)
redim(ip::irPair{T}) where {T} = redim(IndexRange(ip))
redim(ir::IndexRange) = redim(ir.index,dim(ir),start(ir)-1)
redim(irs::Tuple{Vararg{IndexRange}}) = map((ir)->redim(ir) ,irs)


eachval(ir::IndexRange) = range(ir)
eachval(irs::Tuple{Vararg{IndexRange}}) = CartesianIndices(ranges(irs))

eachindval(irs::Tuple{Vararg{IndexRange}}) = (indices(irs).=> Tuple(ns) for ns in eachval(irs))

 
#--------------------------------------------------------------------------------------
#
#  NDTensor level code which distinguishes between Dense and BlockSparse storage
#
function in_range(block_start::NTuple{N, Int64},block_end::NTuple{N, Int64},rs::UnitRange{Int64}...) where {N}
    ret=true
    for i in eachindex(rs)
        if block_start[i]>rs[i].stop || block_end[i]<rs[i].start
            ret=false
            #@show "out of range" block_start block_end rs
            break
        end
    end
    return ret
end

function fix_ranges(dest_block_start::NTuple{N, Int64},dest_block_end::NTuple{N, Int64},rs::UnitRange{Int64}...) where {N}
    rs1=Vector{UnitRange{Int64}}(undef,N)
    @assert length(rs)==N
    #@show b ds rs
    for i in eachindex(rs1)
        @assert dest_block_start[i]<=rs[i].stop || dest_block_end[i]>=rs[i].start #in range?
        istart=Base.max(dest_block_start[i],rs[i].start)
        istop=Base.min(dest_block_end[i],rs[i].stop)
        rs1[i]=istart-dest_block_start[i]+1:istop-dest_block_start[i]+1
    end
    #@show rs1
    return Tuple(rs1)
end
#
#  the range r will in general start at some index start(r) >1.  This function
#  counts how many QN blocks are between 1 and start(r)
#
function get_offset_block_count(in::QNIndex,r::UnitRange{Int64})
   # @show in r
    rs=r.start-1
    if rs==0
        return 0,0
    end
    @assert rs>0
    nb=0
    qns=space(in)
    for n in eachindex(qns)
        rs-=qns[n].second #dim of space
        nb+=1 #increment block counts
        if rs==0
            break
        elseif rs<0
            nb-=1
            #@show "mid block slice" 
            #@show qns[n] rs nb
            # @error("Slicing mid block is not supported yet.")
        end
    end
    #@show nb rs
    return nb,rs
end

using StaticArrays
function get_offset_block_counts(inds::NTuple{N,IndsT},rs::UnitRange{Int64}...) where {N,IndsT}
   # @show inds rs
    dbs=StaticArrays.MVector{N,UInt}(undef)
    shifts=StaticArrays.MVector{N,Int}(undef)
    for i in eachindex(inds)
        dbi,shift=get_offset_block_count(inds[i],rs[i])
        dbs[i]=dbi
        shifts[i]=shift
    end
    return Block{N}(dbs),shifts
end

function get_subtensor(T::BlockSparseTensor{ElT,N},new_inds,rs::UnitRange{Int64}...) where {ElT,N}
    Ds = Vector{DenseTensor{ElT,N}}()
    bs = Vector{Block{N}}()
    dbs,=get_offset_block_counts(inds(T),rs...)
    for (jj, b) in enumerate(eachnzblock(T))
        blockT = blockview(T, b)
        if in_range(blockstart(T,b),blockend(T,b),rs...)
            rs1=fix_ranges(blockstart(T,b),blockend(T,b),rs...)
            #@show "In of range" b dims(blockT) rs rs1 NDTensors.blockstart(T,b)
            push!(Ds,blockT[rs1...])
            bc=CartesianIndex(b)-CartesianIndex(dbs)
            b=bc #Decrement block numbers by the number of skipped blocks.
            push!(bs,b)
        end
    end
    if length(Ds)==0
        return BlockSparseTensor(new_inds)
    end
    #
    #  JR: All attempts at building the new indices here at the NDTensors level failed.
    #  The only thing I could make work was to pass the new indices down from the ITensors
    #  level and use those. 
    #
    #@show bs[1] inds(T) new_inds space(new_inds[1]) Block(bs[1][1])
    #bd=blockdim(new_inds[1],bs[1][1])
    #@show bs rs new_inds
    T_sub = BlockSparseTensor(ElT, undef, bs, new_inds)
    for ib in eachindex(Ds)
        blockT_sub = bs[ib]
        blockview(T_sub, blockT_sub) .= Ds[ib]
    end
    return T_sub
end

function blockrange(T::Tensor{<:Number,N},tb::Block{N})::NTuple{N,UnitRange{Int64}} where {N}
    bs=blockstart(T,tb)
    be=blockend(T,tb)
    return ntuple(i->bs[i]:be[i],N)
end

function fix_ranges(dest_range::NTuple{N,UnitRange{Int64}},src_range::NTuple{N,UnitRange{Int64}},rs::UnitRange{Int64}...) where {N}
    rs1=Vector{UnitRange{Int64}}(undef,N)
    @assert length(rs)==N
    #@show rs dest_range src_range
    for i in eachindex(rs1)
        @assert dest_range[i].start<=rs[i].stop || dest_range[i].stop>=rs[i].start #in range?
        @assert  src_range[i].start<=rs[i].stop ||  src_range[i].stop>=rs[i].start #in range?
        ds_start=Base.max(dest_range[i].start,src_range[i].start)
        dsrc  = src_range[i].stop-src_range[i].start
        istart=Base.max(ds_start,rs[i].start) -dest_range[i].start+1
        #istop=Base.min(ds_end,rs[i].stop)
        #@show i ds_start  dsrc istart
        rs1[i]=istart:istart+dsrc
        #@show rs1[i]
    end
    #@show rs1
    return Tuple(rs1)
end

function set_subtensor(T::BlockSparseTensor{ElT,N},A::BlockSparseTensor{ElT,N},rs::UnitRange{Int64}...) where {ElT,N}
#    @mpoc_assert nzblocks(T)==nzblocks(A)
    # dbs,shifts=get_offset_block_counts(inds(T),rs...)
    # dbs_ci=CartesianIndex(dbs)
    insert=false
    #rsa=ntuple(i->rs[i].start-dbs_ci[i]:rs[i].stop-dbs_ci[i],N)
    rsa=ntuple(i->1:dim(inds(A)[i]),N)
    #@show rs dbs_ci rsa 
    #@show inds(A) inds(T)
    for ab in eachnzblock(A)
        #@show CartesianIndex(dbs) CartesianIndex(ab)
        blockA = blockview(A, ab)
        #@show ab blockA blockstart(A,ab) blockend(A,ab) 

        if in_range(blockstart(A,ab),blockend(A,ab),rsa...)
            iA=blockstart(A,ab)
            iT=ntuple(i->iA[i]+rs[i].start-1,N)
            #iA=ntuple(i->iA[i]+dbs_ci[i],N)
            #@show "inrange"
            index_within_block,tb=blockindex(T,Tuple(iT)...)
            blockT = blockview(T, tb)
            #@show iA iT index_within_block tb blockT
            if blockT==nothing
                insertblock!(T,tb)
                blockT = blockview(T, tb)
                #@show "insert missing block" 
                #@show iA iT ab tb blockA blockT
                insert=true
                # index_within_block,tb=blockindex(T,Tuple(it)...)
                # @show index_within_block tb 
            end
            
            # rs1=fix_ranges(blockrange(T,tb),blockrange(A,ab),rs...)
            # @show rs1
            dA=[dims(blockA)...]
            dT=[blockend(T,tb)...]-[index_within_block...]+fill(1,N)
            rs1=ntuple(i->index_within_block[i]:index_within_block[i]+Base.min(dA[i],dT[i])-1,N)
            #@show rs1 dA dT 
            if length(findall(dA.>dT))==0
                blockT[rs1...]=blockA #Dense assignment for each block
                #@show blockT 
            else
                rsa1=ntuple(i->rsa[i].start:rsa[i].start+Base.min(dA[i],dT[i])-1,N)
                #@show rsa1 blockA[rsa1...]
                blockT[rs1...]=blockA[rsa1...] #partial block Dense assignment for each block
                @error "Incomplete bloc transfer."
                @assert false
                #@show blockT 
            end
        end
       # @show "-----block-------------"
    end
    if insert
        #@show "blocks inserted"
        #@assert false
    end
    #@show "---------------------------------"
end
function set_subtensor(T::DiagBlockSparseTensor{ElT,N},A::DiagBlockSparseTensor{ElT,N},rs::UnitRange{Int64}...) where {ElT,N}
    dbs,=get_offset_block_counts(inds(T),rs...)
    dbs_ci=CartesianIndex(dbs)
    rsa=ntuple(i->rs[i].start-dbs_ci[i]:rs[i].stop-dbs_ci[i],N)
    for ab in eachnzblock(A)
        if in_range(blockstart(A,ab),blockend(A,ab),rsa...)
            it=blockstart(A,ab)
            it=ntuple(i->it[i]+dbs_ci[i],N)
            index_within_block,tb=blockindex(T,Tuple(it)...)
            blockT = blockview(T, tb)
            blockA = blockview(A, ab)
            rs1=fix_ranges(blockrange(T,tb),blockrange(A,ab),rs...)
            #@show rs1 ab tb  
            
            blockT[rs1...]=blockA #Diag assignment for each block
        end
    end
end

function set_subtensor(T::DiagTensor{ElT},A::DiagTensor{ElT},rs::UnitRange{Int64}...) where {ElT}
    if !all(y->y==rs[1],rs)
        @error("set_subtensor(DiagTensor): All ranges must be the same, rs=$(rs).")
    end
    N=length(rs)
    #only assign along the diagonal.
    for i in rs[1]
        is=ntuple(i1 -> i , N)
        js=ntuple(j1 -> i-rs[1].start+1 , N)
        T[is...]=A[js...]
    end
end

setindex!(T::BlockSparseTensor{ElT,N},A::BlockSparseTensor{ElT,N},irs::Vararg{UnitRange{Int64},N}) where {ElT,N} = set_subtensor(T,A,irs...)
setindex!(T::DiagTensor{ElT,N},A::DiagTensor{ElT,N},irs::Vararg{UnitRange{Int64},N}) where {ElT,N} = set_subtensor(T,A,irs...)
setindex!(T::DiagBlockSparseTensor{ElT,N},A::DiagBlockSparseTensor{ElT,N},irs::Vararg{UnitRange{Int64},N}) where {ElT,N} = set_subtensor(T,A,irs...)


#------------------------------------------------------------------------------------
#
#  ITensor level wrappers which allows us to handle the indices in a different manner
#  depending on dense/block-sparse
#
function get_subtensor_wrapper(T::DenseTensor{ElT,N},new_inds,rs::UnitRange{Int64}...) where {ElT,N}
    return ITensor(T[rs...],new_inds)
end

function get_subtensor_wrapper(T::BlockSparseTensor{ElT,N},new_inds,rs::UnitRange{Int64}...) where {ElT,N}
    return ITensor(get_subtensor(T,new_inds,rs...))
end


function permute(indsT::T,irs::IndexRange...) where T<:(Tuple{Vararg{T, N}} where {N, T})
    ispec=indices(irs) #indices caller specified ranges for
    inot=Tuple(noncommoninds(indsT,ispec)) #indices not specified by caller
    isort=ispec...,inot... #all indices sorted so user specified ones are first.
    isort_sub=redim(irs)...,inot... #all indices for subtensor
    p=getperm(indsT, ntuple(n -> isort[n], length(isort)))
    #@show p
    return permute(isort_sub,p),permute((ranges(irs)...,ranges(inot)...),p)
end
#
#  Use NDTensors T[3:4,1:3,1:6...] syntax to extract the subtensor.
#
function get_subtensor_ND(T::ITensor,irs::IndexRange...)
    isub,rsub=permute(inds(T),irs...) #get indices and ranges for the subtensor
    return get_subtensor_wrapper(tensor(T),isub,rsub...) #virtual dispatch based on Dense or BlockSparse
end

function match_tagplev(i1::Index,i2s::Index...)
    for i in eachindex(i2s)
        #@show i  tags(i1) tags(i2s[i]) plev(i1) plev(i2s[i])
        if tags(i1)==tags(i2s[i]) && plev(i1)==plev(i2s[i])
            return i
        end
    end
    @error("match_tagplev: unable to find tag/plev for $i1 in Index set $i2s.")
    return nothing
end

function getperm_tagplev(s1, s2)
    N = length(s1)
    r = Vector{Int}(undef, N)
    return map!(i -> match_tagplev(s1[i], s2...), r, 1:length(s1))
end
#
#
#  Permute is1 to be in the order of is2. 
#  BUT: match based on tags and plevs instread of IDs
#
function permute_tagplev(is1,is2)
    length(is1) != length(is2) && throw(
        ArgumentError(
        "length of first index set, $(length(is1)) does not match length of second index set, $(length(is2))",
        ),
    ) 
    perm=getperm_tagplev(is1,is2)
    #@show perm is1 is2
    return is1[invperm(perm)]
end

#  This operation is non trivial because we need to
#   1) Establish that the non-requested (not in irs) indices are identical (same ID) between T & A
#   2) Establish that requested (in irs) indices have the same tags and 
#       prime levels (dims are different so they cannot possibly have the same IDs)
#   3) Use a combination of tags/primes (for irs indices) or IDs (for non irs indices) to permut 
#      the indices of A prior to conversion to a Tensor.
#
function set_subtensor_ND(T::ITensor, A::ITensor,irs::IndexRange...)
    ireqT=indices(irs) #indices caller requested ranges for
    inotT=Tuple(noncommoninds(inds(T),ireqT)) #indices not requested by caller
    ireqA=Tuple(noncommoninds(inds(A),inotT)) #Get the requested indices for a
    inotA=Tuple(noncommoninds(inds(A),ireqA))
    if length(ireqT)!=length(ireqA) || inotA!=inotT
        @show inotA inotT ireqT ireqA inotA!=inotT length(ireqT) length(ireqA)
        @error("subtensor assign, incompatable indices\ndestination inds=$(inds(T)),\n source inds=$(inds(A)).")
        @assert(false)
    end
    isortT=ireqT...,inotT... #all indices sorted so user specified ones are first.
    p=getperm(inds(T), ntuple(n -> isortT[n], length(isortT))) # find p such that isort[p]==inds(T)
    rsortT=permute((ranges(irs)...,ranges(inotT)...),p) #sorted ranges for T
    ireqAp=permute_tagplev(ireqA,ireqT) #match based on tags & plev, NOT IDs since dims are different.
    isortA=permute((ireqAp...,inotA...),p) #inotA is the same inotT, using inotA here for a less confusing read.
    Ap=ITensors.permute(A,isortA...;allow_alias=true)
    tensor(T)[rsortT...]=tensor(Ap)
end
function set_subtensor_ND(T::ITensor, v::Number,irs::IndexRange...)
    ireqT=indices(irs) #indices caller requested ranges for
    inotT=Tuple(noncommoninds(inds(T),ireqT)) #indices not requested by caller
    isortT=ireqT...,inotT... #all indices sorted so user specified ones are first.
    p=getperm(inds(T), ntuple(n -> isortT[n], length(isortT))) # find p such that isort[p]==inds(T)
    rsortT=permute((ranges(irs)...,ranges(inotT)...),p) #sorted ranges for T
    tensor(T)[rsortT...].=v
end

getindex(T::ITensor, irs::Vararg{IndexRange,N}) where {N} = get_subtensor_ND(T,irs...)
getindex(T::ITensor, irs::Vararg{irPairU,N}) where {N} = get_subtensor_ND(T,indranges(irs)...)
setindex!(T::ITensor, A::ITensor,irs::Vararg{IndexRange,N}) where {N} = set_subtensor_ND(T,A,irs...)
setindex!(T::ITensor, A::ITensor,irs::Vararg{irPairU,N}) where {N} = set_subtensor_ND(T,A,indranges(irs)...)
setindex!(T::ITensor, v::Number,irs::Vararg{IndexRange,N}) where {N} = set_subtensor_ND(T,v,irs...)
setindex!(T::ITensor, v::Number,irs::Vararg{irPairU,N}) where {N} = set_subtensor_ND(T,v,indranges(irs)...)
#=
#--------------------------------------------------------------------------------
#
#  Some overloads of similar so we can easily create new ITensors from a template.
#
function similar(T::DenseTensor{ElT},is::Index...) where {ElT}
    return ITensor(DenseTensor(ElT, undef, is))
end
function similar(T::BlockSparseTensor{ElT},is::Index...) where {ElT}
    return ITensor(BlockSparseTensor(ElT, undef, nzblocks(T), is))
end

function similar(T::DiagTensor{ElT},is::Index...) where {ElT}
    ds=[dims(is)...]
    if !all(y->y==ds[1],ds)
        @error("similar(DiagTensor): All indices must have the same dimensions, dims(is)=$(dims(is)).")
    end
    N=dim(is[1])
    return ITensor(Diag(ElT, N),is)
end
function similar(T::DiagBlockSparseTensor{ElT},is::Index...) where {ElT}
    ds=[dims(is)...]
    if !all(y->y==ds[1],ds)
        @error("similar(DiagTensor): All indices must have the same dimensions, dims(is)=$(dims(is)).")
    end
    return ITensor(DiagBlockSparseTensor(ElT, undef, nzblocks(T), is))
end


function similar(T::ITensor,is::Index...)
    return similar(tensor(T),is...)
end
 =#

