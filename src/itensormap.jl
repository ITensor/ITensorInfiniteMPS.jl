
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
#   itensors::Vector{ITensor}
#   indnetwork::IndexSetNetwork # Adjacency list of IndexSets
#   input_inds::IndexSet
#   output_inds::IndexSet
# end
#
# TODO: how to deal with networks of ITensorNetwork,
# for example a network of MPS and MPO?
# ITensorNetworkNetwork that is a tree of ITensorNetwork?
#

abstract type AbstractITensorMap end

input_inds(T::AbstractITensorMap) = T.input_inds
output_inds(T::AbstractITensorMap) = T.output_inds
(T::AbstractITensorMap)(v::ITensor) = replaceinds(T * v, output_inds(T) => input_inds(T))

# convert from Tuple to Vector
tuple_to_vector(t::Tuple) = collect(t)
tuple_to_vector(v::Vector) = v

# Represents the action of applying the
# vector of ITensors to a starting state and then mapping
# them back (from output_inds to input_inds)
# TODO: rename ITensorNetworkMap?
# TODO: maybe parametrize the type to allow storing just 1 ITensor?
# TODO: contraction order optimization!
struct ITensorMap <: AbstractITensorMap
  itensors::Vector{ITensor}
  scalar::Number
  input_inds::Vector{Index}
  output_inds::Vector{Index}
  function ITensorMap(itensors::Vector{ITensor}, scalar, input_inds, output_inds)
    inds_in = tuple_to_vector(input_inds)
    inds_out = tuple_to_vector(output_inds)
    #inds_eltype = promote_type(eltype(input_inds), eltype(output_inds))
    #return new{inds_eltype}(itensors, inds_in, inds_out)
    return new(itensors, scalar, inds_in, inds_out)
  end
end

function ITensorMap(itensors::Vector{ITensor}, input_inds, output_inds)
  return ITensorMap(itensors, true, input_inds, output_inds)
end

function default_input_inds(itensors::Vector{ITensor})
  return filter(i -> plev(i) == 0, noncommoninds(itensors...))
end

function ITensorMap(
  itensors::Vector{ITensor};
  input_inds=default_input_inds(itensors),
  output_inds=dag(input_inds'),
)
  return ITensorMap(itensors, input_inds, output_inds)
end

Base.iterate(T::ITensorMap, args...) = iterate(T.itensors, args...)

function Base.transpose(T::ITensorMap)
  return ITensorMap(reverse(T.itensors), output_inds(T), input_inds(T))
end

# This is actually a Hermitian conjugation, not priming
function Base.adjoint(T::ITensorMap)
  return ITensorMap(reverse(dag.(T.itensors)), dag(output_inds(T)), dag(input_inds(T)))
end

# TODO: make a default constructor that searches for pairs of primed and unprimed indices
# TODO: this would be useful for ITensor matrix factorizations!
# function ITensorMap(itensors::Vector{ITensor}, pair_match::Pair = 0 => 1)
#   # pair_match could be:
#   # pair_match = 0 => 2
#   # pair_match = "Input" => "Output"
#   # pair_match = ("Input", 0) => ("Output", 1)
#   external_inds = unioninds(siteinds(itensors)...)
#   input_match = first(pair_match)
#   output_match = first(pair_match)
#   # TODO: preprocess pair_match to be of the form `(tags, plev)`
#   ind_match(i::Index, match) = hastags(i, match[1]) && hasplev(i, match[2])
#   input_inds = filter(i -> ind_match(i, input_match), external_inds)
#   output_inds = dag(replaceprime(replacetags(input_inds, first.(pair_match)), last.(pair_match)))
#   @assert = hassameinds(output_inds, filter(i -> ind_match(i, output_match), external_inds))
#   return ITensorMap(itensors, input_inds, output_inds)
# end

function set_scalar(T::ITensorMap, scalar::Number)
  return ITensorMap(T.itensors, scalar, input_inds(T), output_inds(T))
end

# Lazily scale by a scalar
(T::ITensorMap * c::Number) = set_scalar(T, T.scalar * c)
(c::Number * T::ITensorMap) = set_scalar(T, c * T.scalar)

-(T::ITensorMap) = set_scalar(T, -T.scalar)

# TODO: assert isempty(uniqeinds(v, T))?
# Apply the operator as T|v⟩
(T::ITensorMap * v::ITensor) = T.scalar * contract(pushfirst!(copy(T.itensors), v)) #*(v, T...)
# Apply the operator as ⟨v̅|T (simple left multiplication, without conjugation)
# This applies the ITensorMap tensors in reverse (maybe this is not always the best contraction
# ordering)
(v::ITensor * T::ITensorMap) = T.scalar * contract(pushfirst!(reverse(T.itensors, v))) #*(v, reverse(T)...)

# TODO: implement Base.iterate(::Base.Iterators.Reverse{MPS})

# TODO: use something like:
# neighbors(ψ, ϕ, (1, 1))
# neighbors(ψ, ϕ, (2, 1))
# neighbors(ψ, ϕ, (1, N))
# neighbors(ψ, ϕ, (2, N))
# Transfer matrix made from two MPS: T|v⟩ -> |w⟩
function ITensorMap(ψ::MPS, ϕ::MPS; input_inds=nothing, output_inds=nothing)
  N = length(ψ)
  @assert length(ϕ) == N
  itensors::Vector{ITensor} = reverse(collect(Iterators.flatten(Iterators.zip(ψ, ϕ))))
  if isnothing(input_inds)
    input_inds = unioninds(
      uniqueinds(ψ[N], ψ[N - 1], ϕ[N]), uniqueinds(ϕ[N], ϕ[N - 1], ψ[N])
    )
  end
  if isnothing(output_inds)
    output_inds = unioninds(uniqueinds(ψ[1], ψ[2], ϕ[1]), uniqueinds(ϕ[1], ϕ[2], ψ[1]))
  end
  return ITensorMap(itensors, input_inds, output_inds)
end

# Represents a sum of ITensor maps
struct ITensorMapSum <: AbstractITensorMap
  itensormaps::Vector{ITensorMap}
  input_inds::Vector{Index}
  output_inds::Vector{Index}
  function ITensorMapSum(itensormaps::Vector{ITensorMap})
    # TODO: check that all input_inds and output_inds are the same
    return new(itensormaps, input_inds(first(itensormaps)), output_inds(first(itensormaps)))
  end
end

(M1::ITensorMap + M2::ITensorMap) = ITensorMapSum([M1, M2])
(M1::ITensorMapSum + M2::ITensorMap) = ITensorMapSum(push!(copy(M1.itensormaps), M2))
(M1::ITensorMap + M2::ITensorMapSum) = M2 + M1
(M1::ITensorMap - M2::ITensorMap) = M1 + (-M2)
(M1::ITensorMapSum - M2::ITensorMap) = M1 + (-M2)
(M1::ITensorMap - M2::ITensorMapSum) = M1 + (-M2)
(M::ITensorMapSum * v::ITensor) = sum([m * v for m in M.itensormaps])

(M::ITensorMapSum * c::Number) = ITensorMapSum([m * c for m in M.itensormaps])
(c::Number * M::ITensorMapSum) = M * c
-(M::ITensorMapSum) = -1 * M
