# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ
function unit_cell_terms(::Model"ising"; J=1.0, h=1.0)
  opsum = OpSum()
  opsum += -J, "X", 1, "X", 2
  opsum += -h, "Z", 1
  return opsum
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
