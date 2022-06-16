# H = JΣⱼ (½ S⁺ⱼS⁻ⱼ₊₁ + ½ S⁻ⱼS⁺ⱼ₊₁) + J₂Σⱼ (½ S⁺ⱼZⱼ₊₁S⁻ⱼ₊₂ + ½ S⁻ⱼZⱼ₊₁S⁺ⱼ₊₂) - h Σⱼ Sⱼᶻ
function ITensors.OpSum(::Model"xx_extended", n1, n2; J=1.0, J₂=1.0, h=0.0)
  opsum = OpSum()
  if J != 0.0
    opsum += 0.25 * J, "S+", n1, "S-", n2
    opsum += 0.25 * J, "S-", n1, "S+", n2
    opsum += 0.25 * J, "S+", n2, "S-", n2 + 1
    opsum += 0.25 * J, "S-", n2, "S+", n2 + 1
  end
  if J₂ != 0
    opsum += 0.5 * J₂, "S+", n1, "Sz", n2, "S-", n2 + 1
    opsum += 0.5 * J₂, "S-", n1, "Sz", n2, "S+", n2 + 1
  end
  if h != 0
    opsum += -h / 3, "Sz", n1
    opsum += -h / 3, "Sz", n2
    opsum += -h / 3, "Sz", n2 + 1
  end
  return opsum
end

# H = JΣⱼ (½ S⁺ⱼS⁻ⱼ₊₁ + ½ S⁻ⱼS⁺ⱼ₊₁) + J₂Σⱼ (½ S⁺ⱼS⁻ⱼ₊₂ + ½ S⁻ⱼS⁺ⱼ₊₂) - h Σⱼ Sⱼᶻ
function ITensors.MPO(::Model"xx_extended", s; J=1.0, J₂=1.0, h=0.0)
  N = length(s)
  os = OpSum()
  if h != 0
    for j in 1:N
      os += -h, "Sz", j
    end
  end
  if J != 0
    for j in 1:(N - 1)
      os += 0.5 * J, "S+", j, "S-", j + 1
      os += 0.5 * J, "S-", j, "S+", j + 1
    end
  end
  if J₂ != 0
    for j in 1:(N - 2)
      os += 0.5 * J₂, "S+", j, "Sz", j + 1, "S-", j + 2
      os += 0.5 * J₂, "S-", j, "Sz", j + 1, "S+", j + 2
    end
  end
  return splitblocks(linkinds, MPO(os, s))
end

nrange(::Model"xx_extended") = 3

function ITensorInfiniteMPS.reference(
  ::Model"xx_extended", ::Observable"energy"; N=1000, J=1.0, J₂=1.0, h=0.0, filling=0.5
)
  mat = zeros(N, N)
  for j in 1:(N - 1)
    mat[j, j + 1] = -J
    mat[j + 1, j] = -J
  end
  for j in 1:(N - 2)
    mat[j, j + 2] = -J₂
    mat[j + 2, j] = -J₂
  end
  temp = sort(eigvals(mat))
  return sum(temp[1:round(Int64, filling * N)]) / N / 2 - h * (1 / 2 + filling)
end

function ITensorInfiniteMPS.reference(
  ::Model"xx_extended", ::Observable"energy"; N=1000, J=1.0, J₂=1.0, h=0.0, filling=0.5
)
  λ(x) = -J / 2 * cos(x * 2 * pi / N) - J₂ / 2 * cos(x * 4 * pi / N)
  temp = sort(λ.(0:(N - 1)))
  return sum(temp[1:round(Int64, filling * N)]) / N - h * (2 * filling - 1)
end
