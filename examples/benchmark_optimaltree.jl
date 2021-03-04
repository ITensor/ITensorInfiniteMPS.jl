using ITensors
using ITensorsInfiniteMPS
using ITensorsInfiniteMPS.ContractionSequenceOptimization
using TensorOperations
using GraphPlot
using LightGraphs
using ProfileView
using Statistics

hascommoninds(inds1::Vector{Int}, inds2::Vector{Int}) =
  any(i₁ -> any(==(i₁), inds2), inds1)

function indslist_to_edgelist(indslist::Vector{Vector{Int}})
  edgelist = Tuple{Int, Int}[]
  for nodeᵢ in 1:length(indslist)
    for nodeⱼ in 1:length(indslist)
      if hascommoninds(indslist[nodeᵢ], indslist[nodeⱼ])
        push!(edgelist, (nodeᵢ, nodeⱼ))
      end
    end
  end
  return edgelist
end

function adjlist_to_edgelist(adjlist::Vector{Vector{Int}})
  edgelist = Tuple{Int, Int}[]
  for nodeᵢ in 1:length(adjlist)
    neighbors = adjlist[nodeᵢ]
    for nodeⱼ in neighbors
      push!(edgelist, (nodeᵢ, nodeⱼ))
    end
  end
  return edgelist
end

function itensor_network(network, ind_dims)
  nnodes = length(network)
  allinds = sort(union(network...))
  inds = Dict(ind => Index(ind_dims[ind], "i$ind") for ind in allinds)
  indsnetwork = map(t -> getindex.((inds,), t), network)
  return map(inds -> randomITensor(inds...), indsnetwork)
end

# https://github.com/Jutho/TensorOperations.jl/issues/63
# https://github.com/Jutho/TensorOperations.jl/pull/65
#network =
#  [[11, 17], [2, 20], [15, 28, 38], [15, 21, 34],
#   [12, 21, 40], [10, 30], [16, 22, 32],
#   [13, 22, 38], [12, 20], [2, 27, 33],
#   [5, 16, 37], [3, 26, 40], [6, 24],
#   [14, 24, 36], [9, 18, 37], [1, 19, 31],
#   [8, 26], [11, 23], [6, 17, 39], [14, 19, 36],
#   [13, 30], [4, 25, 33], [7, 23, 39], [4, 28],
#   [5, 27, 32], [9, 25, 35], [8, 29], [10, 18, 31],
#   [7, 29, 35], [1, 3, 34]]
#nnodes = length(network)

# Simple matrix multiplication chain
matmul_network(N::Int) = [[n, n+1] for n in 1:N]

function main(which_network; profile = false, fscale = maximum)
  @show which_network
  if which_network == "fullerene"
    network =
      [[11, 17], [2, 20], [15, 28, 38], [15, 21, 34],
       [12, 21, 40], [10, 30], [16, 22, 32],
       [13, 22, 38], [12, 20], [2, 27, 33],
       [5, 16, 37], [3, 26, 40], [6, 24],
       [14, 24, 36], [9, 18, 37], [1, 19, 31],
       [8, 26], [11, 23], [6, 17, 39], [14, 19, 36],
       [13, 30], [4, 25, 33], [7, 23, 39], [4, 28],
       [5, 27, 32], [9, 25, 35], [8, 29], [10, 18, 31],
       [7, 29, 35], [1, 3, 34]]
    nnodes = length(network)
  else
    network = matmul_network(which_network)
    nnodes::Int = which_network
  end
  @show nnodes

  # Take the network and turn it into an edgelist to plot
  #gplothtml(SimpleGraph(Edge.(indslist_to_edgelist(network))))

  # The dimension of index n
  #_dim(n) = 100
  _dim(n) = n == length(allinds) ? 2 : n+1
  #_dim(n) = TensorOperations.Power{:χ}(1,1)

  allinds = sort(union(network...))
  ind_dims = Dict(allinds[n] => _dim(n) for n in 1:length(allinds))

  local time_tensoroperations
  try
    time_tensoroperations = @belapsed TensorOperations.optimaltree($network, $ind_dims)
    @show time_tensoroperations
  catch
  end

  tensornetwork = itensor_network(network, ind_dims)

    # Depth first
    time_itensor = @belapsed depth_first_constructive($tensornetwork)
    println()
    println("Depth first")
    @show time_itensor
    if @isdefined time_tensoroperations
      @show time_tensoroperations / time_itensor
    end

  DimT = UInt64

  if nnodes ≤ 16 && length(allinds) ≤ 16
    time_itensor = @belapsed breadth_first_constructive(UInt16, UInt16, $DimT, $tensornetwork; fscale = $fscale)
    println()
    println("UInt16, UInt16")
    @show time_itensor
    if @isdefined time_tensoroperations
      @show time_tensoroperations / time_itensor
    end
  end

  if nnodes ≤ 32 && length(allinds) ≤ 32
    time_itensor = @belapsed breadth_first_constructive(UInt32, UInt32, $DimT, $tensornetwork; fscale = $fscale)
    println()
    println("UInt32, UInt32")
    @show time_itensor
    if @isdefined time_tensoroperations
      @show time_tensoroperations / time_itensor
    end
  end

  if nnodes ≤ 64 && length(allinds) ≤ 64
    time_itensor = @belapsed breadth_first_constructive(UInt64, UInt64, $DimT, $tensornetwork; fscale = $fscale)
    println()
    println("UInt64, UInt64")
    @show time_itensor
    if @isdefined time_tensoroperations
      @show time_tensoroperations / time_itensor
    end
  end

  if nnodes ≤ 128 && length(allinds) ≤ 128
    time_itensor = @belapsed breadth_first_constructive(UInt128, UInt128, $DimT, $tensornetwork; fscale = $fscale)
    println()
    println("UInt128, UInt128")
    @show time_itensor
    if @isdefined time_tensoroperations
      @show time_tensoroperations / time_itensor
    end
  end

  #stats_itensor = @timed breadth_first_constructive(BitSet, BitSet, DimT, tensornetwork)
  #println()
  #println("BitSet, BitSet")
  #@show stats_itensor.time
  #@show stats_itensor.value
  #if @isdefined stats_tensoroperations
  #  @show stats_tensoroperations.time / stats_itensor.time
  #end

  profile_breadth_first_constructive(::Type{TensorT},
                                     ::Type{IndexSetT}, A, N; fscale) where {TensorT, IndexSetT} =
    for _ in 1:N breadth_first_constructive(TensorT, IndexSetT, A; fscale = fscale) end

  if profile
    @profview profile_breadth_first_constructive(UInt128, UInt128, tensornetwork, round(Int, 3/stats_itensor.time, RoundUp); fscale = fscale)
  end
end

#
# Results:
#
# matmul
#
# N  Time (sec)
# 5  0.000160
# 10 0.002146
# 15 0.037941
# 20 0.212614
# 25 1.160097
# 30 10.905362
# 35 55.789789
#

