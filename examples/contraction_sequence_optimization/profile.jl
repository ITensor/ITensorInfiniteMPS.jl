using ITensors
using ITensorInfiniteMPS
using ITensorInfiniteMPS.ContractionSequenceOptimization
using ProfileView

function main(N, d = 2; random_order = false, profile = false)
  #@show N
  #@show d

  i = Index(d, "i")
  A = randomITensor(i', dag(i))

  function A⃗(N; random_order = false)
    v = [prime(A, n) for n in reverse(0:N-1)]
    if random_order
      return shuffle(v)
    end
    return v
  end
  f(A, N) = for _ in 1:N depth_first_constructive(A) end

  AN = A⃗(N; random_order = random_order)
  #display(inds.(AN))
  #println()

  #println("Depth-first sequence optimization time")
  #@show depth_first_constructive(AN)
  time = @belapsed depth_first_constructive($AN)
  println(N, "  ", time)

  if profile
    println()
    @profview f(AN, 1e6)
  end

  return
end

#
# Results
#
# d = 2, matrix multiplications
#
# Using bit sets for the index labels
#
# 2   1.7902e-6
# 3   2.678e-6
# 4   4.825571428571428e-6
# 5   1.4088e-5
# 6   8.8627e-5
# 7   0.000769776
# 8   0.00761083
# 9   0.099911281
# 10  1.293794184
#
# Micro optimizations for N=3 case
# Currently, most of the time is spent constructing the tree
# structure Any[3, [1, 2]]
#
# N  Time
# 3  128.497 ns
#
# 62cd05c58dffcd278bb87b7df10e616357a0e2f4
#
# d = 2, matrix multiplications
#
# XXX conclusion: don't use a cache for N ≤ 8
#
# Starting from optimal ordering
#
# N  No caching    Caching      Contraction time
# 2  37.923 ns     38.069 ns    854.769 ns
# 3  1.835 μs      3.812 μs     1.663 μs         # 399.065 ns with special optimization path
# 4  5.878 μs      13.506 μs    2.462 μs
# 5  30.853 μs     67.228 μs    4.128 μs
# 6  221.736 μs    424.022 μs   5.276 μs
# 7  1.962 ms      3.264 ms     6.156 μs
# 8  29.596 ms     31.897 ms    7.008 μs
# 9  392.239 ms    375.781 ms   8.094 μs
# 10 5.210 s       4.807 s      8.163 μs
#
# Random start ordering
#
# N  No caching    Caching
# 2  38.701 ns     38.693 ns
# 3  2.311 μs      4.630 μs    # 399.065 ns with special optimization path
# 4  6.179 μs      14.231 μs
# 5  30.074 μs     67.212 μs
# 6  237.478 μs    445.363 μs
# 7  2.060 ms      3.354 ms
# 8  33.332 ms     35.449 ms
# 9  475.974 ms    431.469 ms
# 10 6.759 s       5.740 s
#

