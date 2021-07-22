using ITensors
using ITensorInfiniteMPS

# More general siteind that allows specifying
# the space
function _siteind(site_tag, n::Int; space)
  return addtags(Index(space, "Site,n=$n"), site_tag)
end

_siteinds(site_tag, N::Int; space) = __siteinds(site_tag, N, space)

function __siteinds(site_tag, N::Int, space::Vector)
  return [_siteind(site_tag, n; space=space[n]) for n in 1:N]
end

function __siteinds(site_tag, N::Int, space)
  return [_siteind(site_tag, n; space=space) for n in 1:N]
end

struct Model{model} end
Model(model::Symbol) = Model{model}()

using ITensorInfiniteMPS: celltags

# Create an infinite sum of Hamiltonian terms
function ITensorInfiniteMPS.InfiniteITensorSum(model::Model, s::Vector; kwargs...)
  return InfiniteITensorSum(model, CelledVector(addtags(s, celltags(1))); kwargs...)
end

function ITensorInfiniteMPS.InfiniteITensorSum(model::Model, s::CelledVector; kwargs...)
  N = length(s)
  H = InfiniteITensorSum(N)
  tensors = [ITensor(model, s[n], s[n+1]; kwargs...) for n in 1:N]
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

# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ
function ITensors.MPO(::Model{:ising}, s; J, h)
  N = length(s)
  a = OpSum()
  for n in 1:N-1
    a .+= -J, "X", n, "X", n+1
  end
  for n in 1:N
    a .+= -h, "Z", n
  end
  return MPO(a, s)
end

# H = -J X₁X₂ - h Z₁
# XXX: use `op` instead of `ITensor`
function ITensors.ITensor(::Model{:ising}, s1::Index, s2::Index; J, h)
  # -J * op("X", s1) * op("X", s2) - h * op("Z", s1) * op("Id", s2)
  opsum = OpSum()
  n = 1
  opsum += -J, "X", n, "X", n + 1
  opsum += -h, "Z", n
  return prod(MPO(opsum, [s1, s2]))
end

# t = 1.0
# U = 1.0
# V = 0.5
function ITensors.OpSum(::Model{:hubbard}, n1, n2; t, U, V)
  opsum = OpSum()
  opsum += -t, "Cdagup", n1, "Cup", n2
  opsum += -t, "Cdagup", n2, "Cup", n1
  opsum += -t, "Cdagdn", n1, "Cdn", n2
  opsum += -t, "Cdagdn", n2, "Cdn", n1
  if U ≠ 0
    opsum += U, "Nupdn", n1
  end
  if V ≠ 0
    opsum += V, "Ntot", n1, "Ntot", n2
  end
  return opsum
end

