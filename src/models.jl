struct Model{model} end
Model(model::Symbol) = Model{model}()
Model(model::String) = Model{Symbol(model)}()

macro Model_str(s)
  return :(Model{$(Expr(:quote, Symbol(s)))})
end

# Create an infinite sum of Hamiltonian terms
function ITensorInfiniteMPS.InfiniteITensorSum(model::Model, s::Vector; kwargs...)
  return InfiniteITensorSum(model, infsiteinds(s); kwargs...)
end

function ITensorInfiniteMPS.InfiniteITensorSum(model::Model, s::CelledVector; kwargs...)
  N = length(s)
  H = InfiniteITensorSum(N)
  tensors = [ITensor(model, s[n], s[n + 1]; kwargs...) for n in 1:N]
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
  for n in 1:(N - 1)
    a .+= -J, "X", n, "X", n + 1
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

# t = 1.0
# U = 1.0
# V = 0.5
function ITensors.MPO(::Model{:hubbard}, s; t, U, V)
  N = length(s)
  opsum = OpSum()
  for n in 1:(N - 1)
    n1, n2 = n, n + 1
    opsum .+= -t, "Cdagup", n1, "Cup", n2
    opsum .+= -t, "Cdagup", n2, "Cup", n1
    opsum .+= -t, "Cdagdn", n1, "Cdn", n2
    opsum .+= -t, "Cdagdn", n2, "Cdn", n1
    if V ≠ 0
      opsum .+= V, "Ntot", n1, "Ntot", n2
    end
  end
  if U ≠ 0
    for n in 1:N
      opsum .+= U, "Nupdn", n
    end
  end
  return MPO(opsum, s)
end

"""
    energy_exact(::Model{:heisenberg}, n)

Compute the analytic isotropic heisenberg chain ground energy for length `n`.
Assumes the heisenberg model is defined with spin
operators not pauli matrices (overall factor of 2 smaller). Taken from [1].

[1] Nickel, Bernie. "Scaling corrections to the ground state energy
of the spin-½ isotropic anti-ferromagnetic Heisenberg chain." Journal of
Physics Communications 1.5 (2017): 055021
"""
function energy_exact(::Model{:heisenberg}, n)
  E∞ = (0.5 - 2 * log(2)) * n
  Eᶠⁱⁿⁱᵗᵉ = π^2 / (6n)
  correction = 1 + 0.375 / log(n)^3
  return (E∞ - Eᶠⁱⁿⁱᵗᵉ * correction) / 2
end

function ITensors.OpSum(::Model{:heisenberg}, n1, n2)
  opsum = OpSum()
  opsum += 0.5, "S+", n1, "S-", n2
  opsum += 0.5, "S-", n1, "S+", n2
  opsum += "Sz", n1, "Sz", n2
  return opsum
end

# H = Σⱼ (½ S⁺ⱼS⁻ⱼ₊₁ + ½ S⁻ⱼS⁺ⱼ₊₁ + SᶻⱼSᶻⱼ₊₁)
function ITensors.MPO(::Model{:heisenberg}, s)
  N = length(s)
  os = OpSum()
  for j in 1:(N - 1)
    os .+= 0.5, "S+", j, "S-", j + 1
    os .+= 0.5, "S-", j, "S+", j + 1
    os .+= "Sz", j, "Sz", j + 1
  end
  return MPO(os, s)
end
