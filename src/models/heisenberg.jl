# H = Σⱼ (½ S⁺ⱼS⁻ⱼ₊₁ + ½ S⁻ⱼS⁺ⱼ₊₁ + SᶻⱼSᶻⱼ₊₁)
function unit_cell_terms(::Model"heisenberg")
  opsum = OpSum()
  opsum += 0.5, "S+", 1, "S-", 2
  opsum += 0.5, "S-", 1, "S+", 2
  opsum += "Sz", 1, "Sz", 2
  return opsum
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

function unit_cell_terms(::Model"heisenberg2D"; width, yperiodic)
  opsum = OpSum()
  for i in 1:width
    # Vertical
    if (i < width) || (yperiodic && width > 2)
      opsum -= 0.5, "S+", i, "S-", mod(i, width) + 1
      opsum -= 0.5, "S-", i, "S+", mod(i, width) + 1
      opsum += "Sz", i, "Sz", mod(i, width) + 1
    end
    # Horizontal
    opsum -= 0.5, "S+", i, "S-", i + width
    opsum -= 0.5, "S-", i, "S+", i + width
    opsum += "Sz", i, "Sz", i + width
  end
  return opsum
end

"""
    reference(::Model"heisenberg2D", ::Observable"energy"; width, yperiodic)

Report the reference isotropic 2D square heisenberg ground energy per site for length `N`.
Taken from [1,2]. Note that periodic results have an errorbar of ~1e-4 - 1e-3

[1] Ramos and Xavier. "N-leg spin-S Heisenberg ladders:
A density-matrix renormalization group study"
Phys. Rev. B 89, 094424 - Published 27 March 2014
[2] Frischmuth, Ammon, and Troyer. "Susceptibility and low-temperature
thermodynamics of spin-½ Heisenberg ladders"
Phys. Rev. B 54, R3714(R) - Published 1 August 1996
"""
function reference(::Model"heisenberg2D", ::Observable"energy"; width, yperiodic)
  if width > 6
    error("Ladders of width greater than 6 are not in reference data")
  end

  if yperiodic
    (width == ∞) && return -0.66931
    energies = [-0.4432, -0.5780, -0.6006, -0.6187, -0.6278, -0.635]
    return energies[width]
  else
    (width == ∞) && return -0.6768
    energies = [-0.4431471, -0.578043140180, -0.600537, -0.618566, -0.62776, -0.6346]
    return energies[width]
  end
end
