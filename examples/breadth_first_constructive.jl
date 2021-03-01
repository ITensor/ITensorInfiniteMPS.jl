using ITensors
using ITensorsInfiniteMPS
using ITensorsInfiniteMPS.ContractionSequenceOptimization
using ProfileView
using Random

function main(N, d = 2; random_order = false)
  Random.seed!(1234)

  @show N
  @show d

  function A⃗(N; random_order = false)
    #i = Index(d, "i")
    #A = randomITensor(i', dag(i))
    #v = [prime(A, n) for n in reverse(0:N-1)]

    χ = [d:d+N-1..., d]
    @show χ
    @show length(χ)
    i = [Index(χ[n], "i$n") for n in 1:N+1]
    v = [randomITensor(i[n], dag(i[n+1])) for n in 1:N]

    if random_order
      return shuffle(v)
    end
    return v
  end

  AN = A⃗(N; random_order = random_order)
  display(inds.(AN))
  println()

  println()
  println("Breadth-first constructive")
  enable_caching = false
  @show @time breadth_first_constructive(AN)

  println()
  println("Depth-first constructive")
  enable_caching = false
  @show @time depth_first_constructive(AN; enable_caching = false)

  println()
  display(inds.(AN))

  return
end

