using ITensors
using ITensorInfiniteMPS
using Test
using Random

@testset "unitcell_qns" begin
  function initstate(n)
    if n % 2 == 0
      if (n / 2) % 2 == 1
        return "Dn"
      else
        return "Up"
      end
    else
      if (n + 1) / 2 <= 19
        if ((n + 1) / 2) % 2 == 1
          return "Up"
        else
          return "Dn"
        end
      else
        return "Emp"
      end
    end
  end
  conserve_qns = true
  N = 62
  s = infsiteinds(n -> isodd(n) ? "tJ" : "S=1/2", N; conserve_qns, initstate)
  ψ = InfMPS(s, initstate)
  @show N
  @show s

  function initstate(n)
    if mod(n, 4) == 1
      return 2
    else
      return 1
    end
  end

  for N in 1:6
    s = infsiteinds("S=1/2", N; conserve_szparity=true, initstate)
    ψ = InfMPS(s, initstate)
    @show N
    @show s
  end
end
