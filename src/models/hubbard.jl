function ITensors.OpSum(::Model"hubbard", n1, n2; t, U, V)
  opsum = OpSum()
  opsum += -t, "Cdagup", n1, "Cup", n2
  opsum += -t, "Cdagup", n2, "Cup", n1
  opsum += -t, "Cdagdn", n1, "Cdn", n2
  opsum += -t, "Cdagdn", n2, "Cdn", n1
  if U ≠ 0
    opsum += U, "Nupdn", n1
  end
  if V ≠ 0
    opsum += V, "Ntot", n1, "Ntot", n2
  end
  return opsum
end

function ITensors.MPO(::Model"hubbard", s; t, U, V)
  N = length(s)
  opsum = OpSum()
  for n in 1:(N - 1)
    n1, n2 = n, n + 1
    opsum .+= -t, "Cdagup", n1, "Cup", n2
    opsum .+= -t, "Cdagup", n2, "Cup", n1
    opsum .+= -t, "Cdagdn", n1, "Cdn", n2
    opsum .+= -t, "Cdagdn", n2, "Cdn", n1
    if V ≠ 0
      opsum .+= V, "Ntot", n1, "Ntot", n2
    end
  end
  if U ≠ 0
    for n in 1:N
      opsum .+= U, "Nupdn", n
    end
  end
  return MPO(opsum, s)
end

"""
@article{PhysRevB.6.930,
  title = {Magnetic Susceptibility at Zero Temperature for the One-Dimensional Hubbard Model},
  author = {Shiba, H.},
  journal = {Phys. Rev. B},
  volume = {6},
  issue = {3},
  pages = {930--938},
  numpages = {0},
  year = {1972},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.6.930},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.6.930}
}
"""
function reference(::Model"hubbard", ::Observable"energy"; U, Npoints=50)
  f(x) = 1 / π * sum(n -> (-1)^(n + 1) * (2n) / (x^2 + (2n)^2), 1:10000)
  function m(i, j)
    return (8π) / U * (2π) / Npoints * f(
      4 / U * (sin(-π + (2(i - 1) * π) / Npoints) - sin(-π + (2(j - 1) * π) / Npoints))
    )
  end
  matrix = [m(i, j) for i in 1:Npoints, j in 1:Npoints]
  for i in 1:Npoints
    matrix[i, :] *= cos(-π + (i - 1) * (2π) / Npoints)
  end
  ones_vec = ones(Npoints)
  d = Diagonal(2π .* ones_vec)
  v, _ = linsolve(d - matrix, ones_vec)
  g(i) = -2 * (2π) / Npoints * cos(-π + (i - 1) / Npoints * 2π) * v[i]
  return sum(g, 1:Npoints)
end
