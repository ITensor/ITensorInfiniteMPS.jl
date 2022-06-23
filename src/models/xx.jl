# H = Σⱼ XⱼXⱼ₊₁ + YⱼYⱼ₊₁
function unit_cell_terms(::Model"xx")
  os = OpSum()
  os += 1, "X", 1, "X", 2
  os += 1, "Y", 1, "Y", 2
  return os
end

function reference(::Model"xx", ::Observable"energy"; N=∞)
  isinf(N) && return -4 / π
  # Exact eigenvalues of uniform symmetric tridiagonal matrix
  λ(k) = cos(k * π / (N + 1))
  return 4 * sum(k -> min(λ(k), 0.0), 1:N) / N
end
