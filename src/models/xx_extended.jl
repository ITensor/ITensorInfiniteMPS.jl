# H = JΣⱼ (½ S⁺ⱼS⁻ⱼ₊₁ + ½ S⁻ⱼS⁺ⱼ₊₁) + J₂Σⱼ (½ S⁺ⱼZⱼ₊₁S⁻ⱼ₊₂ + ½ S⁻ⱼZⱼ₊₁S⁺ⱼ₊₂) - h Σⱼ Sⱼᶻ
function unit_cell_terms(::Model"xx_extended"; J=1.0, J₂=1.0, h=0.0)
  opsum = OpSum()
  opsum += 0.25 * J, "S+", 1, "S-", 2
  opsum += 0.25 * J, "S-", 1, "S+", 2
  opsum += 0.25 * J, "S+", 2, "S-", 2 + 1
  opsum += 0.25 * J, "S-", 2, "S+", 2 + 1
  opsum += 0.5 * J₂, "S+", 1, "Sz", 2, "S-", 3
  opsum += 0.5 * J₂, "S-", 1, "Sz", 2, "S+", 3
  opsum += -h / 3, "Sz", 1
  opsum += -h / 3, "Sz", 2
  opsum += -h / 3, "Sz", 3
  return opsum
end

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
