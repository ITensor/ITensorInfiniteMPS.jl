function unit_cell_terms(::Model"hubbard"; t, U, V)
  opsum = OpSum()
  opsum += -t, "Cdagup", 1, "Cup", 2
  opsum += -t, "Cdagup", 2, "Cup", 1
  opsum += -t, "Cdagdn", 1, "Cdn", 2
  opsum += -t, "Cdagdn", 2, "Cdn", 1
  opsum += U, "Nupdn", 1
  opsum += V, "Ntot", 1, "Ntot", 2
  return [opsum]
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
