using ITensors
using ITensorInfiniteMPS

struct Model{model} end
Model(model::Symbol) = Model{model}()

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
  return -J * op("X", s1) * op("X", s2) - h * op("Z", s1) * op("Id", s2)
end

# Version accepting IndexSet
function ITensors.ITensor(model::Model{:ising}, s1, s2; kwargs...)
  return ITensor(model, Index(s1), Index(s2); kwargs...)
end

function ITensorInfiniteMPS.InfiniteITensorSum(model::Model, s; J, h)
  N = length(s)
  H = InfiniteITensorSum(N)
  s∞ = CelledVector(addtags(s, "c=1"))
  tensors = [ITensor(model, s∞[n], s∞[n+1]; J = J, h = h) for n in 1:N]
  return InfiniteITensorSum(tensors)
end

