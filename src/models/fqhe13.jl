function unit_cell_terms(
  ::Model"fqhe_2b_pot"; Ly::Float64, Vs::Array{Float64,1}, prec::Float64
)
  rough_N = round(Int64, 2 * Ly)
  coeff = build_two_body_coefficient_pseudopotential(; N_phi=rough_N, Ly=Ly, Vs=Vs)
  opt = optimize_coefficients(coeff; prec=prec)
  opt = filter_optimized_Hamiltonian_by_first_site(opt)
  #sorted_opt = sort_by_configuration(opt);
  #println(opt)
  return [generate_Hamiltonian(opt)]
end

#Please contact Loic Herviou before using this part of the code for production
# loic.herviou@epfl.ch
###############################
###Pseudpotentials
###############################
"""
    two_body_factorized_pseudomomentum_auxiliary(x, Ly, m)
Auxiliary function that computes the value of the factorized form of the pseudocoefficient
Inputs: x momentum `2*pi*n/Ly`, Ly the length in the periodic direction and m the moment of the pseudo coefficient
Outputs: a floating value
"""
function two_body_factorized_pseudomomentum_auxiliary(x, Ly, m::Int64)
  return 2^(0.75) * sqrt(2 * pi / Ly) * exp(-x^2) * Hermite_polynomial(2 * x; n=m) /
         sqrtfactorial(m) / pi^0.25
end

"""
    build_two_body_coefficient_pseudopotential_factorized_cylinder(Lx, Ly, N_phi, maximalMoment::Int64; prec=1e-12, fermions = true)
Build the dictionary of the factorized form of the two body pseudopotentials on the cylinder
Inputs: Lx and Ly are the dimensions on the torus (Ly the orbital basis), N_phi the number of orbitals, and m the maximal moment (starting at 0)
Outputs: a dictionary of coefficients with keys the shift of the center of mass, values are either Float64
"""
function build_two_body_coefficient_pseudopotential_factorized_cylinder(
  Lx, Ly, N_phi::Int64, maximalMoment::Int64; prec=1e-12, fermions=true
)
  if fermions
    sg_ph = -1
    admissible_m = 1:2:maximalMoment
  else
    sg_ph = 1
    admissible_m = 0:2:maximalMoment
  end
  coefficients = Dict{Float64,Array{Float64,1}}()
  for k in 0.5:0.5:(N_phi / 2 - 0.5)
    coefficients[k] = zeros(Float64, length(admissible_m))
    for (idx_m, m) in enumerate(admissible_m)
      coefficients[k][idx_m] +=
        (
          two_body_factorized_pseudomomentum_auxiliary(2 * pi * k / Ly, Ly, m) +
          sg_ph * two_body_factorized_pseudomomentum_auxiliary(-2 * pi * k / Ly, Ly, m)
        ) / 2
    end
  end
  return coefficients
end

function build_two_body_coefficient_pseudopotential_cylinder(coeff, Vs, N_phi)
  full_coeff = Dict()
  for j in 0.5:0.5:(N_phi - 1.5)
    for l in mod1(j, 1):1:min(j, N_phi - 1 - j)
      sg_1 = 1
      n_1 = mod(round(Int64, j + l), N_phi)
      n_2 = mod(round(Int64, j - l), N_phi)
      if n_1 > n_2
        sg_1 = -1
        temp = n_2
        n_2 = n_1
        n_1 = temp
      elseif n_1 == n_2
        continue
      end
      for k in mod1(j, 1):1:min(j, N_phi - 1 - j)
        sg_2 = 1
        m_1 = mod(round(Int64, j + k), N_phi)
        m_2 = mod(round(Int64, j - k), N_phi)
        if m_1 > m_2
          sg_2 = -1
          temp = m_2
          m_2 = m_1
          m_1 = temp
        elseif m_1 == m_2
          continue
        end
        if !haskey(full_coeff, (m_1, m_2, n_2, n_1))
          full_coeff[[m_1, m_2, n_2, n_1]] =
            sg_1 * sg_2 * sum([coeff[l][x] * Vs[x] * coeff[k][x] for x in 1:length(Vs)])
        else
          full_coeff[[m_1, m_2, n_2, n_1]] +=
            sg_1 * sg_2 * sum([coeff[l][x] * Vs[x] * coeff[k][x] for x in 1:length(Vs)])
        end
      end
    end
  end
  return full_coeff
end

function build_two_body_coefficient_pseudopotential(;
  r::Float64=1.0,
  Lx::Float64=-1.0,
  Ly::Float64=-1.0,
  N_phi::Int64=10,
  Vs::Array{Float64,1}=[0],
  prec=1e-12,
)
  if Lx != -1
    println("Generating pseudopotential coefficients from Lx")
    Ly = 2 * pi * N_phi / Lx
    r = Lx / Ly
  elseif Ly != -1
    println("Generating pseudopotential coefficients from Ly")
    Lx = 2 * pi * N_phi / Ly
    r = Lx / Ly
  else
    println("Generating pseudopotential coefficients from r")
    Lx = sqrt(2 * pi * N_phi * r)
    Ly = sqrt(2 * pi * N_phi / r)
  end
  println(
    string(
      "Parameters are N_phi=",
      N_phi,
      ", r=",
      round(r; digits=3),
      ", Lx =",
      round(Lx; digits=3),
      " and Ly =",
      round(Ly; digits=3),
    ),
  )
  maximalMoment = 2 * length(Vs) - 1
  temp_coeff = build_two_body_coefficient_pseudopotential_factorized_cylinder(
    Lx, Ly, N_phi, maximalMoment; prec=prec, fermions=true
  )
  return build_two_body_coefficient_pseudopotential_cylinder(temp_coeff, Vs, N_phi)
end

######################
## Generic
######################
function filter_optimized_Hamiltonian_by_first_site(coeff::Dict; n=1)
  res = Dict()
  for (k, v) in coeff
    if k[2] == n
      res[k] = v
    end
  end
  return res
end

function check_max_range_optimized_Hamiltonian(coeff::Dict)
  temp = 0
  for k in keys(coeff)
    temp = max(temp, k[end] - k[2])
  end
  return temp
end

function sort_by_configuration(coeff::Dict)
  res = Dict()
  for (k, v) in coeff
    if !haskey(res, k[1:2:end])
      res[k[1:2:end]] = Dict()
    end
    res[k[1:2:end]][k] = v
  end
  return res
end
"""
    sqrtfactorial(m)
Return `sqrt(m!)`
Inputs: `m` a number
"""
function sqrtfactorial(m)
  res = 1
  for j in 1:m
    res *= sqrt(j)
  end
  return res
end

"""
    Laguerre_polynomial(x; n=0)
Return the Laguerre polynomial of arbitrary degre
Inputs: `x` the input and `n` the degree of the Laguerre polynomial
Output: the value of the Laguerre polynomial
"""
function Laguerre_polynomial(x; n=0)
  if n == 0
    return 1
  elseif n == 1
    return 1 - x
  elseif n == 2
    return x^2 / 2 - 2x + 1
  end
  Lm2 = 1
  Lm1 = 1 - x
  Lm = x^2 / 2 - 2x + 1
  for j in 3:n
    Lm2 = Lm1
    Lm1 = Lm
    Lm = ((2j - 1 - x) * Lm1 - (j - 1) * Lm2) / j
  end
  return Lm
end

"""
    Hermite_polynomial(x; n=0)
Return the Hermite polynomial of arbitrary degre
Inputs: `x` the input and `n` the degree of the Hermite polynomial (probabilist notation)
Output: the value of the Laguerre polynomial
"""
function Hermite_polynomial(x; n=0)
  if n == 0
    return 1
  elseif n == 1
    return x
  elseif n == 2
    return x^2 - 1
  end
  Lm2 = 1
  Lm1 = x
  Lm = x^2 - 1
  for j in 3:n
    Lm2 = Lm1
    Lm1 = Lm
    Lm = x * Lm1 - (j - 1) * Lm2
  end
  return Lm
end

#########################
#Building the Hamiltonian
#########################
function get_perm!(lis, name)
  for j in 1:(length(lis) - 1)
    if lis[j] > lis[j + 1]
      c = lis[j]
      lis[j] = lis[j + 1]
      lis[j + 1] = c
      c = name[j]
      name[j] = name[j + 1]
      name[j + 1] = c
      sg = get_perm!(lis, name)
      return -sg
    end
  end
  return 1
end

function filter_op!(lis, name)
  x = 1
  while x <= length(lis) - 1
    if lis[x] == lis[x + 1]
      if name[x] == "Cdag" && name[x + 1] == "C"
        popat!(lis, x + 1)
        popat!(name, x + 1)
        name[x] = "N"
      else
        print("Wrong order in filter_op")
      end
    end
    x += 1
  end
end

function optimize_coefficients(coeff::Dict; prec=1e-12)
  optimized_dic = Dict()
  for (ke, v) in coeff
    if abs(v) < prec
      continue
    end
    if length(ke) == 4
      name = ["Cdag", "Cdag", "C", "C"]
    elseif length(ke) == 6
      name = ["Cdag", "Cdag", "Cdag", "C", "C", "C"]
    elseif length(ke) == 8
      name = ["Cdag", "Cdag", "Cdag", "Cdag", "C", "C", "C", "C"]
    else
      println("Optimization not implemented")
      continue
    end
    k = Base.copy(ke)
    sg = get_perm!(k, name)
    filter_op!(k, name)
    new_k = [isodd(n) ? name[n ÷ 2 + 1] : k[n ÷ 2] + 1 for n in 1:(2 * length(name))]
    optimized_dic[new_k] = sg * v
  end
  return optimized_dic
end

function generate_Hamiltonian(mpo::OpSum, coeff::Dict; global_factor=1, prec=1e-12)
  for (k, v) in coeff
    if abs(v) > prec
      add!(mpo, global_factor * v, k...)
    end
  end
  return mpo
end

function generate_Hamiltonian(coeff::Dict; global_factor=1, prec=1e-12)
  mpo = OpSum()
  return generate_Hamiltonian(mpo, coeff; global_factor=global_factor, prec=prec)
end

## function ITensors.OpSum(
##   ::Model"fqhe_2b_pot", n1, n2; Ly::Float64, Vs::Array{Float64,1}, prec::Float64
## )
##   rough_N = round(Int64, 2 * Ly)
##   coeff = build_two_body_coefficient_pseudopotential(; N_phi=rough_N, Ly=Ly, Vs=Vs)
##   opt = optimize_coefficients(coeff; prec=prec)
##   opt = filter_optimized_Hamiltonian_by_first_site(opt)
##   #sorted_opt = sort_by_configuration(opt);
##   #println(opt)
##   return generate_Hamiltonian(opt)
## end
## 
## function ITensors.MPO(::Model"fqhe_2b_pot", s::CelledVector, n::Int64; Ly, Vs, prec)
##   rough_N = round(Int64, 2 * Ly)
##   coeff = build_two_body_coefficient_pseudopotential(; N_phi=rough_N, Ly=Ly, Vs=Vs)
##   opt = optimize_coefficients(coeff; prec=prec)
##   opt = filter_optimized_Hamiltonian_by_first_site(opt)
##   range_model = check_max_range_optimized_Hamiltonian(opt)
##   while range_model >= rough_N
##     rough_N = 2 * (rough_N + 1)
##     coeff = build_two_body_coefficient_pseudopotential(; N_phi=rough_N, Ly=Ly, Vs=Vs)
##     opt = optimize_coefficients(coeff; prec=prec)
##     opt = filter_optimized_Hamiltonian_by_first_site(opt)
##     range_model = check_max_range_optimized_Hamiltonian(opt)
##   end
##   opsum = generate_Hamiltonian(opt)
## 
##   return splitblocks(linkinds, MPO(opsum, [s[x] for x in n:(n + range_model)]))
## end
