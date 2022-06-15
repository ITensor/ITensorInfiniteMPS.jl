# For infinite Hamiltonian
# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ - J₂ Σⱼ XⱼZⱼ₊₁Xⱼ₊₂
function opsum_infinite(::Model"ising_extended", n; J=1.0, h=1.0, J₂=0.0)
  opsum = OpSum()
  for j in 1:n
    opsum += -J, "X", j, "X", j + 1
    opsum += -J₂, "X", j, "Z", j + 1, "X", j + 2
    opsum -= h, "Z", j
  end
  return opsum
end

# For finite Hamiltonian with open boundary conditions
# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ - J₂ Σⱼ XⱼZⱼ₊₁Xⱼ₊₂
function opsum_finite(::Model"ising_extended", n; J=1.0, h=1.0, J₂=0.0)
  # TODO: define using opsum_infinite, filtering sites
  # that extend over the boundary.
  opsum = OpSum()
  for j in 1:n
    if j ≤ (n - 1)
      opsum += -J, "X", j, "X", j + 1
    end
    if j ≤ (n - 2)
      opsum += -J₂, "X", j, "Z", j + 1, "X", j + 2
    end
    opsum -= h, "Z", j
  end
  return opsum
end

function reference(::Model"ising_extended", ::Observable"energy"; J=1.0, h=1.0, J₂=0.0)
  f(k) = sqrt((J * cos(k) + J₂ * cos(2k) - h)^2 + (J * sin(k) + J₂ * sin(2k))^2)
  return -1 / 2π * ITensorInfiniteMPS.∫(k -> f(k), -π, π)
end

#
# Deprecated
#

# nrange(::Model"ising_extended") = 3

# function ITensors.ITensor(::Model"ising_extended", s1::Index, s2::Index, s3::Index; J=1.0, h=1.0, J₂=0.0)
#   opsum = OpSum(Model"ising_extended"(), 1, 2; J=J, h=h, J₂=J₂)
#   return prod(MPO(opsum, [s1, s2, s3]))
# end
