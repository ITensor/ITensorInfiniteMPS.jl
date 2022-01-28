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
function InfiniteITensorSum(model::Model, s::Vector; kwargs...)
  return InfiniteITensorSum(model, infsiteinds(s); kwargs...)
end

function InfiniteITensorSum(model::Model, s::CelledVector; kwargs...)
  N = length(s)
  H = InfiniteITensorSum(N)
  tensors = [ITensor(model, s[n], s[n + 1]; n = n, kwargs...) for n in 1:N] #support for staggered potentials and dimerized hoppings.
  return InfiniteITensorSum(tensors)
end

# Version accepting IndexSet
function ITensors.ITensor(model::Model, s1, s2; kwargs...)
  return ITensor(model, Index(s1), Index(s2); kwargs...)
end

function ITensors.ITensor(model::Model, s1::Index, s2::Index; kwargs...)
  n1, n2 = 1, 2                                                             
  opsum = OpSum(model, n1, n2; kwargs...)
  return prod(MPO(opsum, [s1, s2]))
end
