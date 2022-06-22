# H = Σⱼ (½ S⁺ⱼS⁻ⱼ₊₁ + ½ S⁻ⱼS⁺ⱼ₊₁ + SᶻⱼSᶻⱼ₊₁)
function unit_cell_terms(::Model"heisenberg")
  opsum = OpSum()
  opsum += 0.5, "S+", 1, "S-", 2
  opsum += 0.5, "S-", 1, "S+", 2
  opsum += "Sz", 1, "Sz", 2
  return [opsum]
end

"""
    reference(::Model"heisenberg", ::Observable"energy"; N)

Compute the analytic isotropic heisenberg chain ground energy per site for length `N`.
Assumes the heisenberg model is defined with spin
operators not pauli matrices (overall factor of 2 smaller). Taken from [1].

[1] Nickel, Bernie. "Scaling corrections to the ground state energy
of the spin-½ isotropic anti-ferromagnetic Heisenberg chain." Journal of
Physics Communications 1.5 (2017): 055021
"""
function reference(::Model"heisenberg", ::Observable"energy"; N=∞)
  isinf(N) && return (0.5 - 2 * log(2)) / 2
  E∞ = (0.5 - 2 * log(2)) * N
  Eᶠⁱⁿⁱᵗᵉ = π^2 / (6N)
  correction = 1 + 0.375 / log(N)^3
  return (E∞ - Eᶠⁱⁿⁱᵗᵉ * correction) / (2N)
end

# function ITensors.OpSum(::Model"heisenberg", n1, n2)
#   opsum = OpSum()
#   opsum += 0.5, "S+", n1, "S-", n2
#   opsum += 0.5, "S-", n1, "S+", n2
#   opsum += "Sz", n1, "Sz", n2
#   return opsum
# end
# 
# function ITensors.MPO(::Model"heisenberg", s)
#   N = length(s)
#   os = OpSum()
#   for j in 1:(N - 1)
#     os .+= 0.5, "S+", j, "S-", j + 1
#     os .+= 0.5, "S-", j, "S+", j + 1
#     os .+= "Sz", j, "Sz", j + 1
#   end
#   return splitblocks(linkinds, MPO(os, s))
# end
