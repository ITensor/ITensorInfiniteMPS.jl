
#
# TODO: should ITensorMap be a special version of
# an ITensorNetwork with input and output indices specified?
#

# T is however the nodes are indexed
# TODO: how to deal with 2D, multiple networks, etc.?
# struct IndexSetNetwork{T}
#   # Use Vector{SortedVector{Pair{T, IndexSet}}}
#   data::Vector{Vector{Pair{T, IndexSet}}}
# end

# Make MPS, PEPS, etc. wrappers around `ITensorNetwork`,
# which could be subclasses of `AbstractITensorNetwork`
#
# Also `InfiniteITensorNetwork`, `AbstractInfiniteITensorNetwork`:
#
# struct InfiniteMPS <: AbstractInfiniteITensorNetwork
#   tensornetwork::InfiniteITensorNetwork
# end
#
# struct ITensorNetwork <: AbstractITensorNetwork
#   tensors::Vector{ITensor}
#   indnetwork::IndexSetNetwork # Adjacency list of IndexSets
#   input_inds::IndexSet
#   output_inds::IndexSet
# end
#
# TODO: how to deal with networks of ITensorNetwork,
# for example a network of MPS and MPO?
# ITensorNetworkNetwork that is a tree of ITensorNetwork?
#

# Represents the action of applying the
# vector of ITensors to a starting state and then mapping
# them back (from output_inds to input_inds)
# TODO: rename ITensorNetworkMap?
# TODO: maybe parametrize the type to allow storing just 1 ITensor?
# TODO: contraction order optimization!
struct ITensorMap
  tensors::Vector{ITensor}
  input_inds::IndexSet
  output_inds::IndexSet
end

input_inds(T::ITensorMap) = T.input_inds
output_inds(T::ITensorMap) = T.output_inds

function ITensorMap(tensors::Vector{<: ITensor})
  input_inds = filter(i -> plev(i) == 0, noncommoninds(tensors...))
  output_inds = dag(input_inds')
  return ITensorMap(tensors, input_inds, output_inds)
end

Base.iterate(T::ITensorMap, args...) = iterate(T.tensors, args...)

function Base.transpose(T::ITensorMap)
  return ITensorMap(reverse(T.tensors), output_inds(T), input_inds(T))
end

# This is actually a Hermitian conjugation, not priming
function Base.adjoint(T::ITensorMap)
  return ITensorMap(reverse(dag.(T.tensors)), dag(output_inds(T)), dag(input_inds(T)))
end

# TODO: make a default constructor that searches for pairs of primed and unprimed indices
# TODO: this would be useful for ITensor matrix factorizations!
# function ITensorMap(tensors::Vector{<: ITensor}, pair_match::Pair = 0 => 1)
#   # pair_match could be:
#   # pair_match = 0 => 2
#   # pair_match = "Input" => "Output"
#   # pair_match = ("Input", 0) => ("Output", 1)
#   external_inds = unioninds(siteinds(tensors)...)
#   input_match = first(pair_match)
#   output_match = first(pair_match)
#   # TODO: preprocess pair_match to be of the form `(tags, plev)`
#   ind_match(i::Index, match) = hastags(i, match[1]) && hasplev(i, match[2])
#   input_inds = filter(i -> ind_match(i, input_match), external_inds)
#   output_inds = dag(replaceprime(replacetags(input_inds, first.(pair_match)), last.(pair_match)))
#   @assert = hassameinds(output_inds, filter(i -> ind_match(i, output_match), external_inds))
#   return ITensorMap(tensors, input_inds, output_inds)
# end

# TODO: assert isempty(uniqeinds(v, T))?
# Apply the operator as T|v⟩
(T::ITensorMap * v::ITensor) = *(v, T...)
# Apply the operator as ⟨v̅|T (simple left multiplication, without conjugation)
# This applies the ITensorMap tensors in reverse
(v::ITensor * T::ITensorMap) = *(v, reverse(T)...)
(T::ITensorMap)(v::ITensor) = replaceinds(T * v, output_inds(T) => input_inds(T))

# TODO: implement Base.iterate(::Base.Iterators.Reverse{MPS})

# TODO: use something like:
# neighbors(ψ, ϕ, (1, 1))
# neighbors(ψ, ϕ, (2, 1))
# neighbors(ψ, ϕ, (1, N))
# neighbors(ψ, ϕ, (2, N))
# Transfer matrix made from two MPS: T|v⟩ -> |w⟩
function ITensorMap(ψ::MPS, ϕ::MPS; input_inds = nothing, output_inds = nothing)
  N = length(ψ)
  @assert length(ϕ) == N
  tensors::Vector{ITensor} = reverse(collect(Iterators.flatten(Iterators.zip(ψ, ϕ))))
  if isnothing(input_inds)
    input_inds = unioninds(uniqueinds(ψ[N], ψ[N-1], ϕ[N]), uniqueinds(ϕ[N], ϕ[N-1], ψ[N]))
  end
  if isnothing(output_inds)
    output_inds = unioninds(uniqueinds(ψ[1], ψ[2], ϕ[1]), uniqueinds(ϕ[1], ϕ[2], ψ[1]))
  end
  return ITensorMap(tensors, input_inds, output_inds)
end

