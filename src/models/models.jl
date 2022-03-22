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
ITensorInfiniteMPS.nrange(model::Model) = 2; #required to keep everything compatible with the current implementation for 2 band models

# Create an infinite sum of Hamiltonian terms
function InfiniteSum{T}(model::Model, s::Vector; kwargs...) where {T}
  return InfiniteSum{T}(model, infsiteinds(s); kwargs...)
end

function InfiniteSum{MPO}(model::Model, s::CelledVector; kwargs...)
  N = length(s)
  mpos = [MPO(model, s, n; kwargs...) for n in 1:N] #slightly improved version. Note: the current implementation does not really allow for staggered potentials for example
  return InfiniteSum{MPO}(mpos, translater(s))
end

function InfiniteSum{ITensor}(model::Model, s::CelledVector; kwargs...)
  N = length(s)
  itensors = [ITensor(model, s, n; kwargs...) for n in 1:N] #slightly improved version. Note: the current implementation does not really allow for staggered potentials for example
  return InfiniteSum{ITensor}(itensors, translater(s))
end

# MPO building version
function ITensors.MPO(model::Model, s::CelledVector, n::Int64; kwargs...)
  n1, n2 = 1, 2
  opsum = OpSum(model, n1, n2; kwargs...)
  return MPO(opsum, [s[x] for x in n:(n + nrange(model) - 1)]) #modification to allow for more than two sites per term in the Hamiltonians
end

# Version accepting IndexSet
function ITensors.ITensor(model::Model, s::CelledVector, n::Int64; kwargs...)
  return prod(MPO(model, s, n; kwargs...)) #modification to allow for more than two sites per term in the Hamiltonians
end

# Helper function to make an MPO
import ITensors: op
op(::OpName"Zero", ::SiteType, s::Index) = ITensor(s', dag(s))

function InfiniteMPOMatrix(model::Model, s::CelledVector; kwargs...)
  return InfiniteMPOMatrix(model, s, translater(s); kwargs...)
end

function InfiniteMPOMatrix(model::Model, s::CelledVector, translater::Function; kwargs...)
  N = length(s)
  temp_H = InfiniteSum{MPO}(model, s; kwargs...)
  range_H = nrange(temp_H)[1]
  ls = CelledVector([Index(1, "Link,c=1,n=$n") for n in 1:N], translater)
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
  return InfiniteMPOMatrix(mpos, translater)
end
