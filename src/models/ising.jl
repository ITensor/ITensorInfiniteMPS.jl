# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ
function ITensors.MPO(::Model"ising", s; J, h)
  N = length(s)
  a = OpSum()
  for n in 1:(N - 1)
    a .+= -J, "X", n, "X", n + 1
  end
  for n in 1:N
    a .+= -h, "Z", n
  end
  return splitblocks(linkinds, MPO(a, s))
end

# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ
function ITensors.OpSum(::Model"ising", n1, n2; J, h)
  opsum = OpSum()
  if J != 0
    opsum += -J, "X", n1, "X", n2
  end
  if h != 0
    opsum += -h / 2, "Z", n1
    opsum += -h / 2, "Z", n2
  end
  return opsum
end

# H = -J X₁X₂ - h Z₁
# XXX: use `op` instead of `ITensor`
function ITensors.ITensor(::Model"ising", s1::Index, s2::Index; J, h)
  # -J * op("X", s1) * op("X", s2) - h * op("Z", s1) * op("Id", s2)
  opsum = OpSum()
  n = 1
  opsum += -J, "X", n, "X", n + 1
  opsum += -h, "Z", n
  return prod(MPO(opsum, [s1, s2]))
end

# P. Pfeuty, The one-dimensional Ising model with a transverse field, Annals of Physics 57, p. 79 (1970)
function reference(::Model"ising", ::Observable"energy"; h, J=1.0)
  (J == h) && return -4 / π
  f(k; λ) = sqrt(1 + λ^2 + 2λ * cos(k))
  return -h / (2π * J) * ∫(k -> f(k; λ=(J / h)), -π, π)
end

# https://journals.aps.org/pr/abstract/10.1103/PhysRev.65.117
# https://en.wikipedia.org/wiki/Ising_model#Onsager's_exact_solution
function reference(::Model"ising_classical", ::Observable"free_energy"; β, J=1.0)
  βJ = β * J
  function f(θ₁, θ₂)
    return log(cosh(2βJ)^2 - sinh(2βJ) * cos(θ₁) - sinh(2βJ) * cos(θ₂))
  end
  I = ∫(θ₂ -> ∫(θ₁ -> f(θ₁, θ₂), 0, 2π), 0, 2π)
  return -(log(2) + I / (8π^2)) / β
end

function reference(::Model"ising_classical", ::Observable"critical_inverse_temperature")
  # log1p(x) = log(1 + x)
  return log1p(sqrt(2)) / 2
end
