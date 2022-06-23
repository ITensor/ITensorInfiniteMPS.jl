# A vector of the terms associated with each site of the
# unit cell.
# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ - J₂ Σⱼ XⱼZⱼ₊₁Xⱼ₊₂
function unit_cell_terms(::Model"ising_extended"; J=1.0, h=1.0, J₂=0.0)
  opsum = OpSum()
  opsum += -J, "X", 1, "X", 2
  opsum += -J₂, "X", 1, "Z", 2, "X", 3
  opsum += -h, "Z", 1
  return [opsum]
end

function reference(::Model"ising_extended", ::Observable"energy"; J=1.0, h=1.0, J₂=0.0)
  f(k) = sqrt((J * cos(k) + J₂ * cos(2k) - h)^2 + (J * sin(k) + J₂ * sin(2k))^2)
  return -1 / 2π * ITensorInfiniteMPS.∫(k -> f(k), -π, π)
end
