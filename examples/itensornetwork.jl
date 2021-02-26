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

Random.seed!(1234)

N = 8 # Tensors in the network

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

@show N

#sequence, cost = @time ITensorsInfiniteMPS.optimal_contraction_sequence(TN)
#@show cost
#@show sequence
#@show Tree(sequence)
#@profview ITensorsInfiniteMPS.optimal_contraction_sequence(TN)

sequence2, cost2 = @time depth_first_constructive(T)
@show cost2
@show sequence2
@show Tree(sequence2)
#@profview depth_first_constructive(T)

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

