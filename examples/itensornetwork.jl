using AbstractTrees
using ITensors
using ITensorsVisualization
using ITensorsInfiniteMPS
using IterTools # For subsets
using ProfileView
using Random # For seed!
using StatsBase # For sample

# Testing for improved algorithms
using ITensorsInfiniteMPS.ContractionSequenceOptimization

function main(N; enable_caching = false)
  Random.seed!(1234)

  Sz0 = ("Sz", 0) => 1
  Sz1 = ("Sz", 1) => 1
  #space = Sz0 ⊕ Sz1
  space = 2

  # Make a random network
  max_nedges = max(1, N * (N-1) ÷ 2)
  # Fill 2/3 of the edges
  nedges = max(1, max_nedges * 2 ÷ 3)
  # 1/3 of tensors have physical indices
  nsites = N ÷ 3
  edges = StatsBase.sample(collect(subsets(1:N, 2)), nedges; replace = false)
  sites = StatsBase.sample(collect(1:N), nsites; replace = false)

  indsnetwork = ITensorsInfiniteMPS.IndexSetNetwork(N)
  for e in edges
    le = IndexSet(Index(space, "l=$(e[1])↔$(e[2])"))
    pair_e = Pair(e...)
    indsnetwork[pair_e] = le
    indsnetwork[reverse(pair_e)] = dag(le)
  end

  for n in sites
    indsnetwork[n => n] = IndexSet(Index(space, "s=$(n)"))
  end

  T = Vector{ITensor}(undef, N)
  for n in 1:N
    T[n] = randomITensor(only.(ITensorsInfiniteMPS.eachlinkinds(indsnetwork, n))...)
  end

  # Can just use indsnetwork, this recreates indsnetwork
  TN = ⊗(T...)

  #@show N

  #sequence, cost = @time ITensorsInfiniteMPS.optimal_contraction_sequence(TN)
  #@show cost
  #@show sequence
  #@show Tree(sequence)
  #@profview ITensorsInfiniteMPS.optimal_contraction_sequence(TN)

  stats = @timed depth_first_constructive(T; enable_caching = enable_caching)
  sequence2, cost2 = stats.value
  time = stats.time
  #@show cost2
  #@show sequence2
  #@show Tree(sequence2)
  println(N, "  ", time, "  ", cost2, "  ", sequence2)
  #@profview depth_first_constructive(T)
end

#
# 62cd05c58dffcd278bb87b7df10e616357a0e2f4
#
# XXX: enable_caching = false for N < 7
#
# enable_caching = false
#
# 2  1.027e-6      0      Any[1 2]
# 3  6.3711e-5     8      Any[3, Any[1, 2]]
# 4  5.3855e-5     24     Any[4, Any[1, Any[2, 3]]]
# 5  8.1188e-5     48     Any[5, Any[1, Any[4, Any[2, 3]]]]
# 6  0.000532094   384    Any[2, Any[Any[1, Any[3, 6]], Any[4, 5]]]
# 7  0.015745355   1744   Any[3, Any[7, Any[1, Any[5, Any[6, Any[2, 4]]]]]]
# 8  0.208512772   4928   Any[7, Any[Any[2, 3], Any[8, Any[5, Any[4, Any[1, 6]]]]]]
# 9  12.773462454  82432  Any[2, Any[6, Any[8, Any[3, Any[7, Any[5, Any[9, Any[1, 4]]]]]]]]
#
# enable_caching = true
#
# 2  7.63e-7      0      Any[1 2]
# 3  7.3079e-5    8      Any[3, Any[1, 2]]
# 4  6.2208e-5    24     Any[4, Any[1, Any[2, 3]]]
# 5  0.000106835  48     Any[5, Any[1, Any[4, Any[2, 3]]]]
# 6  0.000711473  384    Any[2, Any[Any[1, Any[3, 6]], Any[4, 5]]]
# 7  0.006858842  1744   Any[3, Any[7, Any[1, Any[5, Any[6, Any[2, 4]]]]]]
# 8  0.068384588  4928   Any[7, Any[Any[2, 3], Any[8, Any[5, Any[4, Any[1, 6]]]]]]
# 9  4.583904197  82432  Any[2, Any[6, Any[8, Any[3, Any[7, Any[5, Any[9, Any[1, 4]]]]]]]]
#
#
# bf5885d2d7234d9a59a38397d5acdb3eef9333a7
#
# N  time(s)     cost  sequence
# 2  5.5955e-5   2     Any[1, 2]
# 3  7.2351e-5   8     Any[3, Any[1, 2]]
# 4  9.5568e-5   24    Any[4, Any[1, Any[2, 3]]]
# 5  0.000440426 48    Any[5, Any[1, Any[4, Any[2, 3]]]]
# 6  0.002089596 384   Any[2, Any[Any[1, Any[3, 6]], Any[4, 5]]]
# 7  0.008770299 1744  Any[3, Any[7, Any[1, Any[5, Any[6, Any[2, 4]]]]]]
# 8  0.069674617 4928  Any[7, Any[Any[2, 3], Any[8, Any[5, Any[4, Any[1, 6]]]]]]
# 9  4.641397023 82432 Any[2, Any[6, Any[8, Any[3, Any[7, Any[5, Any[9, Any[1, 4]]]]]]]]
#
#
#
# de8fc59563ad546b1f0b6be1af130a260e3f67d5
#
# Results
# N  time(s)      cost   sequence
# 7  0.015418301  1744   [2 => 4, 6 => 8, 5 => 9, 1 => 10, 7 => 11, 3 => 12]
# 8  0.164726236  4928   [1 => 6, 2 => 3, 4 => 9, 5 => 11, 8 => 12, 10 => 13, 7 => 14]
# 9  13.03959649  82432  [1 => 4, 9 => 10, 5 => 11, 7 => 12, 3 => 13, 8 => 14, 6 => 15, 2 => 16]


#
# commit 3a31f201b71fa660b7709a733cd79e8c31c08dae
#
# seed = 1234
#
# Results
# N  cost  sequence
# 2  2     Any[1, 2]
# 3  8     Any[3, Any[1, 2]]
# 4  24    Any[4, Any[1, Any[2, 3]]]
# 5  48    Any[5, Any[1, Any[4, Any[2, 3]]]]
# 6  384   Any[2, Any[1, Any[Any[3, 6], Any[4, 5]]]]
# 7  1744  Any[3, Any[7, Any[1, Any[5, Any[2, Any[4, 6]]]]]]
# 8  4928  Any[7, Any[Any[8, Any[1, Any[4, Any[5, 6]]]], Any[2, 3]]]
#
# Benchmark results
# N  time
# 2  0.000034
# 3  0.000054
# 4  0.000104
# 5  0.000870
# 6  0.017992
# 7  0.475907
# 8  20.809482
#

