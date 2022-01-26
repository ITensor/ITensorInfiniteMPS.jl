nrange(::Model"ising_extended") = 3
# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ - J₂ Σⱼ XⱼZⱼ₊₁Xⱼ₊₂
function ITensors.MPO(::Model"ising_extended", s; J=1.0, h=1.0, J₂=0.0)
  N = length(s)
  a = OpSum()
  if J != 0
    for n in 1:(N - 1)
      a += -J, "X", n, "X", n + 1
    end
  end
  if J₂ != 0
    for n in 1:(N - 2)
      a += -J₂, "X", n, "Z", n + 1, "X", n + 2
    end
  end
  if h != 0
    for n in 1:N
      a += -h, "Z", n
    end
  end
  return MPO(a, s)
end

# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ - J₂ Σⱼ XⱼZⱼ₊₁Xⱼ₊₂
function ITensors.OpSum(::Model"ising_extended", n1, n2; J=1.0, h=1.0, J₂=0.0)
  opsum = OpSum()
  if J != 0
    opsum += -J / 2, "X", n1, "X", n2
    opsum += -J / 2, "X", n2, "X", n2 + 1
  end
  if J₂ != 0
    opsum += -J₂, "X", n1, "Z", n2, "X", n2 + 1
  end
  opsum += -h / 3, "Z", n1
  opsum += -h / 3, "Z", n2
  opsum += -h / 3, "Z", n2 + 1
  return opsum
end

function reference(::Model"ising_extended", ::Observable"energy"; J=1.0, h=1.0, J₂=0.0)
  f(k) = sqrt((J * cos(k) + J₂ * cos(2k) - h)^2 + (J * sin(k) + J₂ * sin(2k))^2)
  return -1 / 2π * ITensorInfiniteMPS.∫(k -> f(k), -π, π)
end
