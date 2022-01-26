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
function InfiniteITensorSum(model::Model, s::Vector; kwargs...)
  return InfiniteITensorSum(model, infsiteinds(s); kwargs...)
end

function InfiniteITensorSum(model::Model, s::CelledVector; kwargs...)
  N = length(s)
  H = InfiniteITensorSum(N)
  tensors = [ITensor(model, s, n; kwargs...) for n in 1:N] #slightly improved version. Note: the current implementation does not really allow for staggered potentials for example
  return InfiniteITensorSum(tensors)
end

# Version accepting IndexSet
function ITensors.ITensor(model::Model, s::CelledVector, n::Int64; kwargs...)
  n1, n2 = 1, 2
  opsum = OpSum(model, n1, n2; kwargs...)
  return prod(MPO(opsum, [s[x] for x in n:(n + nrange(model) - 1)])) #modification to allow for more than two sites per term in the Hamiltonians
end