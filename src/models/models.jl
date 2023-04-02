struct Model{model} end
Model(model::Symbol) = Model{model}()
Model(model::String) = Model{Symbol(model)}()

macro Model_str(s)
  return :(Model{$(Expr(:quote, Symbol(s)))})
end

struct Observable{obs} end
Observable(obs::Symbol) = Observable{obs}()
Observable(obs::String) = Observable{Symbol(obs)}()

macro Observable_str(s)
  return :(Observable{$(Expr(:quote, Symbol(s)))})
end

∫(f, a, b) = quadgk(f, a, b)[1]

# Create an infinite sum of Hamiltonian terms
function InfiniteSum{T}(model::Model, s::Vector; kwargs...) where {T}
  return InfiniteSum{T}(model, infsiteinds(s); kwargs...)
end

function InfiniteSum{MPO}(model::Model, s::CelledVector; kwargs...)
  return InfiniteSum{MPO}(opsum_infinite(model, cell_length(s); kwargs...), s)
end

function InfiniteSum{ITensor}(model::Model, s::CelledVector; kwargs...)
  N = length(s)
  itensors = [ITensor(model, s, n; kwargs...) for n in 1:N]
  return InfiniteSum{ITensor}(itensors, translator(s))
end

# Get the first site with nontrivial support of the OpSum
first_site(opsum::OpSum) = minimum(ITensors.sites(opsum))
last_site(opsum::OpSum) = maximum(ITensors.sites(opsum))

function set_site(o::Op, s::Int)
  return Op(ITensors.which_op(o), s; ITensors.params(o)...)
end

function shift_site(o::Op, shift::Int)
  return set_site(o, ITensors.site(o) + shift)
end

function shift_sites(term::Scaled{C,Prod{Op}}, shift::Int) where {C}
  shifted_term = ITensors.coefficient(term)
  for o in ITensors.terms(term)
    shifted_term *= shift_site(o, shift)
  end
  return shifted_term
end

# Shift the sites of the terms of the OpSum by shift.
# By default, it shifts 
function shift_sites(opsum::OpSum, shift::Int)
  shifted_opsum = OpSum()
  for o in ITensors.terms(opsum)
    shifted_opsum += shift_sites(o, shift)
  end
  return shifted_opsum
end

function InfiniteSum{MPO}(opsum::OpSum, s::CelledVector)
  n = cell_length(s)
  nrange = 0 # Maximum operator support
  opsums = [OpSum() for _ in 1:n]
  for o in ITensors.terms(opsum)
    js = sort(ITensors.sites(o))
    j1 = first(js)
    nrange = max(nrange, last(js) - j1 + 1)
    opsums[j1] += o
  end
  shifted_opsums = [shift_sites(opsum, -first_site(opsum) + 1) for opsum in opsums]
  mpos = [
    splitblocks(linkinds, MPO(shifted_opsums[j], [s[k] for k in j:(j + nrange - 1)])) for
    j in 1:n
  ]
  return InfiniteSum{MPO}(mpos, translator(s))
end
# Helper function to make an MPO
import ITensors: op
op(::OpName"Zero", ::SiteType, s::Index) = ITensor(s', dag(s))

function InfiniteMPO(model::Model, s::CelledVector; kwargs...)
  return InfiniteMPO(model, s, translator(s); kwargs...)
end

function InfiniteMPO(model::Model, s::CelledVector, translator::Function; kwargs...)
  return InfiniteMPO(InfiniteMPOMatrix(model, s, translator; kwargs...))
end

function InfiniteMPOMatrix(model::Model, s::CelledVector; kwargs...)
  return InfiniteMPOMatrix(model, s, translator(s); kwargs...)
end

function InfiniteMPOMatrix(model::Model, s::CelledVector, translator::Function; kwargs...)
  N = length(s)
  temp_H = InfiniteSum{MPO}(model, s; kwargs...)
  range_H = nrange(temp_H)[1]
  ls = CelledVector([Index(1, "Link,c=1,n=$n") for n in 1:N], translator)
  mpos = [Matrix{ITensor}(undef, 1, 1) for i in 1:N]
  for j in 1:N
    Hmat = fill(op("Zero", s[j]), range_H + 1, range_H + 1)
    identity = op("Id", s[j])
    Hmat[1, 1] = identity
    Hmat[end, end] = identity
    for n in 0:(range_H - 1)
      idx = findfirst(x -> x == j, findsites(temp_H[j - n]; ncell=N))
      if isnothing(idx)
        Hmat[range_H + 1 - n, range_H - n] = identity
      else
        Hmat[range_H + 1 - n, range_H - n] = temp_H[j - n][idx]#replacetags(linkinds, temp_H[j - n][idx], "Link, l=$n", tags(ls[j-1]))
      end
    end
    mpos[j] = Hmat
    #mpos[j] += dense(Hmat) * setelt(ls[j-1] => total_dim) * setelt(ls[j] => total_dim)
  end
  #return mpos
  return InfiniteMPOMatrix(mpos, translator)
end

function ITensors.MPO(model::Model, s::Vector{<:Index}; kwargs...)
  opsum = opsum_finite(model, length(s); kwargs...)
  return splitblocks(linkinds, MPO(opsum, s))
end

translatecell(translator, opsum::OpSum, n::Integer) = translator(opsum, n)

function infinite_terms(model::Model; kwargs...)
  # An `OpSum` storing all of the terms in the
  # first unit cell.
  # TODO: Allow specifying the unit cell size
  # explicitly.
  return infinite_terms(unit_cell_terms(model; kwargs...))
end

function infinite_terms(opsum::OpSum; kwargs...)
  # `Vector{OpSum}`, vector of length of number
  # of sites in the unit cell where each element
  # contains the terms with support starting on that
  # site of the unit cell, i.e. `opsum_cell[i]`
  # stores all terms starting on site `i`.
  opsum_cell_dict = groupreduce(minimum ∘ ITensors.sites, +, opsum)
  nsites = maximum(keys(opsum_cell_dict))
  # Assumes each site in the unit cell has a term
  for j in 1:nsites
    if !haskey(opsum_cell_dict, j)
      error(
        "The input unit cell terms for the $nsites-site unit cell doesn't have a term starting on site $j. Skipping sites in the unit cell is currently not supported. A workaround is to define a term in the unit cell with a coefficient of 0, for example `opsum += 0, \"I\", $j`.",
      )
    end
  end
  opsum_cell = [opsum_cell_dict[j] for j in 1:nsites]
  function _shift_cell(opsum::OpSum, cell::Int)
    return shift_sites(opsum, nsites * cell)
  end
  return CelledVector(opsum_cell, _shift_cell)
end

function opsum_infinite(model::Model, nsites::Int; kwargs...)
  _infinite_terms = infinite_terms(model::Model; kwargs...)
  # Desired unit cell size must be commensurate
  # with the primitive unit cell of the model.
  if !iszero(nsites % length(_infinite_terms))
    error(
      "Desired unit cell size $nsites must be commensurate with the primitive unit cell size of the model $model, which is $(length(_infinite_terms))",
    )
  end
  opsum = OpSum()
  for j in 1:nsites
    opsum += _infinite_terms[j]
  end
  return opsum
end

function filter_terms(f, opsum::OpSum; by=identity)
  filtered_opsum = OpSum()
  for t in ITensors.terms(opsum)
    if f(by(t))
      filtered_opsum += t
    end
  end
  return filtered_opsum
end

function finite_terms(model::Model, n::Int; kwargs...)
  _infite_terms = infinite_terms(model; kwargs...)
  _finite_terms = OpSum[]
  for j in 1:n
    term_j = _infite_terms[j]
    filtered_term_j = filter_terms(s -> all(≤(n), s), term_j; by=ITensors.sites)
    push!(_finite_terms, filtered_term_j)
  end
  return _finite_terms
end

# For finite Hamiltonian with open boundary conditions
# Obtain from infinite Hamiltonian, dropping terms
# that extend outside of the system.
function opsum_finite(model::Model, n::Int; kwargs...)
  opsum = OpSum()
  for term in finite_terms(model, n; kwargs...)
    opsum += term
  end
  return opsum
end

# The ITensor of a single term `n` of the model.
function ITensors.ITensor(model::Model, s::Vector{<:Index}, n::Int; kwargs...)
  opsum = infinite_terms(model; kwargs...)[n]
  opsum = shift_sites(opsum, -first_site(opsum) + 1)
  return contract(MPO(opsum, [s...]))
end

function ITensors.ITensor(model::Model, s::Vector{<:Index}; kwargs...)
  return ITensor(model, s, 1; kwargs...)
end

function ITensors.ITensor(model::Model, n::Int, s::Index...; kwargs...)
  return ITensor(model, [s...], n; kwargs...)
end

function ITensors.ITensor(model::Model, s::Index...; kwargs...)
  return ITensor(model, 1, s...; kwargs...)
end

function ITensors.ITensor(model::Model, s::CelledVector, n::Int64; kwargs...)
  opsum = infinite_terms(model; kwargs...)[n]
  opsum = shift_sites(opsum, -first_site(opsum) + 1)
  site_range = n:(n + last_site(opsum) - 1)
  return contract(MPO(opsum, [s[j] for j in site_range]))

  # Deprecated version
  # return contract(MPO(model, s, n; kwargs...))
end
