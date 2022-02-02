# H = Σⱼ XⱼXⱼ₊₁ + YⱼYⱼ₊₁
function ITensors.MPO(::Model"xx", s)
  N = length(s)
  os = OpSum()
  for n in 1:(N - 1)
    os .+= 1, "X", n, "X", n + 1
    os .+= 1, "Y", n, "Y", n + 1
  end
  return MPO(os, s)
end

# H = X₁X₂ + Y₁Y₂
function ITensors.OpSum(::Model"xx", n1, n2)
  opsum = OpSum()
  for j in 1:(N - 1)
    opsum .+= 0.5, "S+", j, "S-", j + 1
    opsum .+= 0.5, "S-", j, "S+", j + 1
  end
  return opsum
end

# H = X₁X₂ + Y₁Y₂
# XXX: use `op` instead of `ITensor`
function ITensors.ITensor(::Model"xx", s1::Index, s2::Index)
  # op("X", s1) * op("X", s2) + op("Y", s1) * op("Y", s2)
  opsum = OpSum()
  n = 1
  opsum += "X", n, "X", n + 1
  opsum += "Y", n, "Y", n + 1
  return prod(MPO(opsum, [s1, s2]))
end

function reference(::Model"xx", ::Observable"energy"; N=∞)
  isinf(N) && return -4 / π
  # Exact eigenvalues of uniform symmetric tridiagonal matrix
  λ(k) = cos(k * π / (N + 1))
  return 4 * sum(k -> min(λ(k), 0.0), 1:N) / N
end
