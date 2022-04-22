# H = -J Σⱼ XⱼXⱼ₊₁ - h Σⱼ Zⱼ
function ITensors.OpSum(
  ::Model"fqhe_2b_pot", n1, n2; Ly::Float64, Vs::Array{Float64,1}, prec::Float64
)
  rough_N = round(Int64, 2 * Ly)
  coeff = build_two_body_coefficient_pseudopotential(; N_phi=rough_N, Ly=Ly, Vs=Vs)
  opt = optimize_coefficients(coeff; prec=prec)
  opt = filter_optimized_Hamiltonian_by_first_site(opt)
  #sorted_opt = sort_by_configuration(opt);
  #println(opt)
  return generate_Hamiltonian(opt)
end

function ITensors.MPO(::Model"fqhe_2b_pot", s::CelledVector, n::Int64; Ly, Vs, prec)
  rough_N = round(Int64, 2 * Ly)
  coeff = build_two_body_coefficient_pseudopotential(; N_phi=rough_N, Ly=Ly, Vs=Vs)
  opt = optimize_coefficients(coeff; prec=prec)
  opt = filter_optimized_Hamiltonian_by_first_site(opt)
  range_model = check_max_range_optimized_Hamiltonian(opt)
  while range_model >= rough_N
    rough_N = 2 * (rough_N + 1)
    coeff = build_two_body_coefficient_pseudopotential(; N_phi=rough_N, Ly=Ly, Vs=Vs)
    opt = optimize_coefficients(coeff; prec=prec)
    opt = filter_optimized_Hamiltonian_by_first_site(opt)
    range_model = check_max_range_optimized_Hamiltonian(opt)
  end
  opsum = generate_Hamiltonian(opt)

  return MPO(opsum, [s[x] for x in n:(n + range_model)])
end
