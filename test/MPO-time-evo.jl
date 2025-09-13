# Reference: Efficient higher-order matrix product operators for time evolution
# Maarten Van Damme, Jutho Haegeman, Ian McCulloch, Laurens Vanderstraeten
# SciPost Phys. 17, 135 (2024) · published 15 November 2024 
# https://scipost.org/SciPostPhys.17.5.135
# 
# 
# Given an MPO W of a Hamiltonian H in form 
#       | Id | C | D  |
#  W =  |  0 | A | B  |
#       |  0 | 0 | Id |

# then we find that the time evolution opertator U(t) 
# U(t) = exp(-ı t H)
#   | Id + τD + τ^2/2 D^2 + τ^3/6 D^3 | C + τ^2/2 {CD} + τ^3/6 {CDD}                 | CC + τ/3 {CCD}
# ≈ | τB + τ^2/2 {B D} + τ^3/6 {BDD}  | A + τ/2({BC} + {AD}) + τ^2/6( {CBD} + {ADD}) | {AC} + τ/3 ({ACD} + {CCB})
#   | τ^2/2 BB + τ^3/6 {BBD}          | τ/2 {AB} + τ^2/6 ({ABD} + {BBC})             | AA + τ/3 ({ABC} +{AAD})

# NB:
# {AB}  = AB + BA
# {ABB} = {BBA} = ABB + BAB + BBA
# {ABC} = ABC + ACB + BAC + BCA + CAB + CBA
# measure τ in imaginary units for actualy time evolution

braket(A::ITensor, B::ITensor) = replaceprime(prime(A) * B + prime(B) * A, 2 => 1)
function braket(A::ITensor, B::ITensor, C::ITensor)
  return replaceprime(
    A'' * B' * C + A'' * C' * B + B'' * A' * C + B'' * C' * A + C'' * A' * B + C'' * B' * A,
    3 => 1,
  )
end
function braket2(A::ITensor, B::ITensor)
  return replaceprime(A'' * B' * B + B'' * A' * B + B'' * B' * A, 3 => 1)
end

function timeEvo_MPO_2ndOrder(
  s::Union{CelledVector{Index{Int64}},CelledVector{Index{Vector{Pair{QN,Int64}}}}},
  AStrings::Vector{String},
  JAs::Vector{<:Number},
  BStrings::Vector{String},
  JBs::Vector{<:Number},
  CStrings::Vector{String},
  JCs::Vector{<:Number},
  DString::String,
  JD::Number,
  t::Number;
)
  return MPO(
    timeEvo_ITensors_2ndOrder(
      s.data, AStrings, JAs, BStrings, JBs, CStrings, JCs, DString, JD, t;
    ),
  )
end

function timeEvo_ITensor_2ndOrder(
  s::Union{CelledVector{Index{Int64}},CelledVector{Index{Vector{Pair{QN,Int64}}}}},
  # linkindices,
  AStrings::Vector{String},
  JAs::Vector{<:Number},
  BStrings::Vector{String},
  JBs::Vector{<:Number},
  CStrings::Vector{String},
  JCs::Vector{<:Number},
  DString::String,
  JD::Number,
  dt::Number;
)
  linkindices = get_linkindices_timeEvo_MPO(s, BStrings, CStrings)

  # note that
  τ = -1.0im * dt

  return CelledVector(
    map(
      x -> bulk_timeEvo_ITensor_2ndOrder(
        x[1], s, linkindices, AStrings, JAs, BStrings, JBs, CStrings, JCs, DString, JD, τ
      ),
      enumerate(s),
    ),
    translator(s),
  )
end

function bulk_timeEvo_ITensor_2ndOrder(
  n,
  sites,
  linkindices,
  AStrings::Vector{String},
  JAs::Vector{<:Number},
  BStrings::Vector{String},
  JBs::Vector{<:Number},
  CStrings::Vector{String},
  JCs::Vector{<:Number},
  DString::String,
  JD::Number,
  τ::Number;
)
  s_n = sites[n]
  # left link index ll with daggered QN conserving direction (if applicable)
  ll = (linkindices[n - 1])
  # right link index rl
  rl = dag(linkindices[n])
  # Id = op(sites, "Id", n)

  NrOfTerms = length(AStrings)

  # A is possible exponential decay so test for "0"
  As = map(x -> x[1] * op(s_n, x[2]), zip(JAs, AStrings))
  Bs = map(x -> x[1] * op(s_n, x[2]), zip(JBs, BStrings)) # JB * op(sites, BString, n)
  Cs = map(x -> x[1] * op(s_n, x[2]), zip(JCs, CStrings)) # JC * op(sites, CString, n)
  D = JD * op(s_n, DString)
  # D = JD * Op(DString, n)

  # Init ITensor inside MPO
  # U_t[n] = ITensor(ComplexF64, ll, dag(s), s', rl)

  # first element
  firstelement = if iszero(D)
    setelt(ll[1]) * (setelt(rl[1])) * op(s_n, "Id")
  else
    setelt(ll[1]) *
    setelt(rl[1]) *
    (
      op(s_n, "Id") +
      τ * D +
      (τ^2 / 2) * replaceprime(D' * D, 2, 1) +
      (τ^3 / 6) * replaceprime(D'' * D' * D, 3, 1)
    )
  end

  # first row
  Cterm = mapreduce(
    x ->
      setelt(ll[1]) *
      setelt(rl[1 + x[1]]) *
      (x[2] + (τ / 2) * braket(x[2], D) + (τ^2 / 6) * braket2(x[2], D)),
    +,
    enumerate(Cs),
  )

  # CHECK FOR NILL-POTENT OPERATORS or operators proportional to zero
  # avoid setting entries explicitly zero because that counters the purpose of sparse matrices and 
  # reduces the runtime efficiency
  # (this applies only to Sparse Matrices in the case of QN conservation)
  # Note that the empty ITensor is not equal to the ITensor with only zero entries!
  Cterm_checked = iszero(Cterm) ? emptyITensor(ll, rl, s_n', dag(s_n)) : Cterm

  Csquared_term = mapreduce(
    x ->
      setelt(ll[1]) *
      setelt(rl[1 + x[1] + NrOfTerms]) *
      (replaceprime(x[2]' * x[2], 2, 1) + (τ / 3) * braket2(D, x[2])),
    +,
    enumerate(Cs),
  )
  Csquared_term_checked =
    iszero(Csquared_term) ? emptyITensor(ll, rl, s_n', dag(s_n)) : Csquared_term

  # first column (exept first row)
  Bterm = mapreduce(
    x ->
      setelt(ll[1 + x[1]]) *
      setelt(rl[1]) *
      (τ * x[2] + (τ^2 / 2) * braket(x[2], D) + (τ^3 / 6) * braket2(x[2], D)),
    +,
    enumerate(Bs),
  )
  Bterm_checked = iszero(Bterm) ? emptyITensor(ll, rl, s_n', dag(s_n)) : Bterm

  Bsquared_term = mapreduce(
    x ->
      setelt(ll[1 + x[1] + NrOfTerms]) *
      setelt(rl[1]) *
      ((τ^2 / 2) * replaceprime(x[2]' * x[2], 2, 1) + (τ^3 / 6) * braket2(D, x[2])),
    +,
    enumerate(Bs),
  )
  Bsquared_term_checked =
    iszero(Bsquared_term) ? emptyITensor(ll, rl, s_n', dag(s_n)) : Bsquared_term

  # diagonal
  diagterm = mapreduce(
    x ->
      setelt(ll[1 + x[1]]) *
      setelt(rl[1 + x[1]]) *
      (
        x[2][1] +
        (τ / 2) * (braket(x[2][2], x[2][3]) + braket(x[2][1], D)) +
        (τ^2 / 6) * (braket(x[2][3], x[2][2], D) + braket2(x[2][1], D))
      ),
    +,
    enumerate(zip(As, Bs, Cs)),
  )
  diagterm_checked = iszero(diagterm) ? emptyITensor(ll, rl, s_n', dag(s_n)) : diagterm

  diagsquaredterm = mapreduce(
    x ->
      setelt(ll[1 + x[1] + NrOfTerms]) *
      setelt(rl[1 + x[1] + NrOfTerms]) *
      (
        replaceprime(x[2][1]' * x[2][1], 2, 1) +
        (τ / 3) * (braket(x[2][1], x[2][2], x[2][3]) + braket2(D, x[2][1]))
      ),
    +,
    enumerate(zip(As, Bs, Cs)),
  )
  diagsquaredterm_checked =
    iszero(diagsquaredterm) ? emptyITensor(ll, rl, s_n', dag(s_n)) : diagsquaredterm

  # the "rest" of mixed terms
  mixedterm = mapreduce(
    x ->
      setelt(ll[1 + x[1]]) *
      setelt(rl[1 + x[1] + NrOfTerms]) *
      (
        braket(x[2][1], x[2][3]) +
        (τ / 3) * (braket(x[2][1], x[2][3], D) + braket2(x[2][2], x[2][3]))
      ),
    +,
    enumerate(zip(As, Bs, Cs)),
  )
  mixedterm_checked = iszero(mixedterm) ? emptyITensor(ll, rl, s_n', dag(s_n)) : mixedterm

  mixedsquareterm = mapreduce(
    x ->
      setelt(ll[1 + x[1] + NrOfTerms]) *
      (setelt(rl[1 + x[1]])) *
      (
        (τ / 2) * braket(x[2][1], x[2][2]) +
        (τ^2 / 6) * (braket(x[2][1], x[2][2], D) + braket2(x[2][3], x[2][2]))
      ),
    +,
    enumerate(zip(As, Bs, Cs)),
  )
  mixedsquareterm_checked =
    iszero(mixedsquareterm) ? emptyITensor(ll, rl, s_n', dag(s_n)) : mixedsquareterm

  return sum((
    firstelement,
    Cterm_checked,
    Csquared_term_checked,
    Bterm_checked,
    Bsquared_term_checked,
    diagterm_checked,
    diagsquaredterm_checked,
    mixedterm_checked,
    mixedsquareterm_checked,
  ))
end

function Left_timeEvo_ITensor_2ndOrder(
  sites,
  linkindices,
  CStrings::Vector{String},
  JCs::Vector{<:Number},
  DString::String,
  JD::Number,
  τ::Number;
)
  NrOfTerms = length(CStrings)
  # right link index rl
  rl = linkindices[1]
  n = 1

  Cs = map(x -> x[1] * op(sites, x[2], 1), zip(JCs, CStrings)) # JC * op(sites, CString, n)
  D = JD * op(sites, DString, 1)

  # first element
  firstelement = if iszero(D)
    setelt(rl[1]) * op(sites, "Id", n)
  else
    setelt(rl[1]) * (
    op(sites, "Id", n) +
    τ * D +
    (τ^2 / 2) * replaceprime(D' * D, 2, 1) +
    (τ^3 / 6) * replaceprime(D'' * D' * D, 3, 1)
  )
  end

  # first row
  Cterm = mapreduce(
    x ->
      setelt(rl[1 + x[1]]) *
      (x[2] + (τ / 2) * braket(x[2], D) + (τ^2 / 6) * braket2(x[2], D)),
    +,
    enumerate(Cs),
  )
  Csquared_term = mapreduce(
    x ->
      setelt(rl[1 + x[1] + NrOfTerms]) *
      (replaceprime(x[2]' * x[2], 2, 1) + (τ / 3) * braket2(D, x[2])),
    +,
    enumerate(Cs),
  )

  return sum(
  # filter(
  # !(iszero),
  [firstelement, Cterm, Csquared_term]
  # )
)
end

function Right_timeEvo_ITensor_2ndOrder(
  sites,
  linkindices,
  BStrings::Vector{String},
  JBs::Vector{<:Number},
  DString::String,
  JD::Number,
  τ::Number;
)
  L = length(sites)
  n = L
  # left link index ll with daggered QN conserving direction (if applicable)
  ll = dag(linkindices[L - 1])

  NrOfTerms = length(BStrings)

  # A is possible exponential decay so test for "0"
  # As = map(x -> x[1] * op(sites, x[2], n), zip(JAs, AStrings))
  Bs = map(x -> x[1] * op(sites, x[2], n), zip(JBs, BStrings)) # JB * op(sites, BString, n)
  # Cs = map(x -> x[1] * op(sites, x[2], n), zip(JCs, CStrings)) # JC * op(sites, CString, n)
  D = JD * op(sites, DString, n)

  # first element
  firstelement = if iszero(D)
    setelt(ll[1]) * op(sites, "Id", n)
  else
    setelt(ll[1]) * (
    op(sites, "Id", n) +
    τ * D +
    (τ^2 / 2) * replaceprime(D' * D, 2, 1) +
    (τ^3 / 6) * replaceprime(D'' * D' * D, 3, 1)
  )
  end

  # first column (exept first row)
  Bterm = mapreduce(
    x ->
      setelt(ll[1 + x[1]]) *
      (τ * x[2] + (τ^2 / 2) * braket(x[2], D) + (τ^3 / 6) * braket2(x[2], D)),
    +,
    enumerate(Bs),
  )
  Bsquared_term = mapreduce(
    x ->
      setelt(ll[1 + x[1] + NrOfTerms]) *
      ((τ^2 / 2) * replaceprime(x[2]' * x[2], 2, 1) + (τ^3 / 6) * braket2(D, x[2])),
    +,
    enumerate(Bs),
  )

  return sum(
  # filter(
  # !(iszero),
  [firstelement, Bterm, Bsquared_term]
  # )
)
end

function get_linkindices_timeEvo_MPO(
  sites::Vector{Index{Vector{Pair{QN,Int64}}}},
  BStrings::Vector{String},
  CStrings::Vector{String},
)
  N = length(sites)
  NOps = length(BStrings)
  if NOps != length(CStrings)
    throw(ArgumentError("Input BStrings and CString do not share same length!"))
  end

  # save QN flux of each operator, note that multiple operators may have same flux,
  # thus beloning to the same block and increasing the local dimension of this block
  # reuse the vector to save memory
  QNFlux_vector = Vector{QN}(undef, length(BStrings))

  linkindices = Vector{Index{Vector{Pair{QN,Int64}}}}(undef, N)

  for n in 1:N
    # save the flux and the corresponding dimension in a dynamically sized vector as it is ordered in historical order
    # same as the corresponding operators
    QN_local_index_dim = Vector{Pair{QN,Int64}}(undef, 1)
    nameQN = String(qn(sites[n][1]).data[1].name)
    QNmodulus = qn(sites[n][1]).data[1].modulus
    # loop over all interaction operators
    for (indexOP, (BString, CString)) in enumerate(zip(BStrings, CStrings))
      # get the flux of interaction
      QNFlux_vector[indexOP] = flux(op(sites[n], BString))
      checkflux = flux(op(sites[n], CString))
      if !(checkflux == -QNFlux_vector[indexOP])
        error(
          "Operators B and C are not conserving the total QN in the system as their flux is not opposite",
        )
      end
      # local dimension of flux is at least 1
      if indexOP == 1
        QN_local_index_dim[indexOP] = Pair(QNFlux_vector[indexOP], 1)
      elseif indexOP > 1 && QNFlux_vector[indexOP - 1] == QNFlux_vector[indexOP]
        # increase tuple of number and dimension by one in dimension
        QN_local_index_dim[end] = Pair(
          QN_local_index_dim[end][1], QN_local_index_dim[end][2] + 1
        )
      elseif indexOP > 1
        # if new flux is encountered, save in vector
        push!(QN_local_index_dim, Pair(QNFlux_vector[indexOP], 1))
      end
    end # loop over interaction operators

    # second part of the entries are generated by the double of the fluxes from before
    # generated by the square of the interaction operators...
    # Note this is because we are using the second order approximation!

    QN_local_index_dim_final = Vector{Pair{QN,Int64}}(undef, 2 * length(QN_local_index_dim))

    for (index, element) in enumerate(QN_local_index_dim)
      dim_of_flux = element[2]
      flux_val = val(element[1], nameQN)
      QN_local_index_dim_final[index] = element
      QN_local_index_dim_final[index + length(QN_local_index_dim)] = Pair(
        QN(nameQN, 2 * flux_val, QNmodulus), dim_of_flux
      )
    end

    # construct the local link index from all flux blocks and their dimension
    linkindices[n] = Index(
      [QN(nameQN, 0, QNmodulus) => 1, QN_local_index_dim_final...], "Link,c=1,l=$(n)"
    )
  end

  return linkindices
end

function get_linkindices_timeEvo_MPO(
  sites::Vector{Index{Int64}}, BStrings::Vector{String}, CStrings::Vector{String}
)

  # L = 
  NOps = length(BStrings)
  if NOps != length(CStrings)
    throw(ArgumentError("Input BStrings and CString do not share same length!"))
  end

  return [Index(1 + 2 * length(BStrings), "Link,l=$(n)") for n in 1:(length(sites) - 1)]
end

function get_linkindices_timeEvo_MPO(
  sites::CelledVector{Index{Int64}}, BStrings::Vector{String}, CStrings::Vector{String}
)

  # L = 
  NOps = length(BStrings)
  if NOps != length(CStrings)
    throw(ArgumentError("Input BStrings and CString do not share same length!"))
  end

  return CelledVector(
    [Index(1 + 2 * length(BStrings), "Link,c=1,l=$(n)") for n in 1:length(sites)],
    translator(s),
  )
end

function get_linkindices_timeEvo_MPO(
  sites::CelledVector{Index{Vector{Pair{QN,Int64}}}},
  BStrings::Vector{String},
  CStrings::Vector{String},
)
  linkindices = get_linkindices_timeEvo_MPO(sites.data, BStrings, CStrings)
  return CelledVector(linkindices, translator(sites))
end

function timeEvo_ITensors_2ndOrder(
  sites,
  AStrings::Vector{String},
  JAs::Vector{<:Number},
  BStrings::Vector{String},
  JBs::Vector{<:Number},
  CStrings::Vector{String},
  JCs::Vector{<:Number},
  DString::String,
  JD::Number,
  t::Number;
)
  # Reference: Efficient higher-order matrix product operators for time evolution
  # Maarten Van Damme, Jutho Haegeman, Ian McCulloch, Laurens Vanderstraeten
  # SciPost Phys. 17, 135 (2024) · published 15 November 2024 
  # https://scipost.org/SciPostPhys.17.5.135
  # Given an MPO W of a Hamiltonian H in form 
  #       | Id | C | D  |
  #  W =  |  0 | A | B  |
  #       |  0 | 0 | Id |

  # then we find that the time evolution opertator U(t) 
  # U(t) = exp(-ı t H)
  #   | Id + τD + τ^2/2 D^2 + τ^3/6 D^3 | C + τ^2/2 {CD} + τ^3/6 {CDD}                 | CC + τ/3 {CCD}
  # ≈ | τB + τ^2/2 {B D} + τ^3/6 {BDD}  | A + τ/2({BC} + {AD}) + τ^2/6( {CBD} + {ADD}) | {AC} + τ/3 ({ACD} + {CCB})
  #   | τ^2/2 BB + τ^3/6 {BBD}          | τ/2 {AB} + τ^2/6 ({ABD} + {BBC})             | AA + τ/3 ({ABC} +{AAD})

  # NB:
  # {AB}  = AB + BA
  # {ABB} = {BBA} = ABB + BAB + BBA
  # {ABC} = ABC + ACB + BAC + BCA + CAB + CBA
  # measure τ in imaginary units for actualy time evolution
  τ = -1.0im * t
  N = length(sites)

  if length(AStrings) != length(BStrings) || length(BStrings) != length(CStrings)
    error(
      "Input length of operator vectors is not the same!\n
      Found length(AStrings)=$(length(AStrings)), length(BStrings)=$(length(BStrings)), length(CStrings)=$(length(CStrings))",
    )
  end
  if length(AStrings) != length(JAs) ||
    length(BStrings) != length(JBs) ||
    length(CStrings) != length(JCs)
    error("Input length of couppling vectors is not the same as the operator vector!")
  end

  linkindices = get_linkindices_timeEvo_MPO(sites, BStrings, CStrings)

  ## LEFT boundary
  Ut_1 = Left_timeEvo_ITensor_2ndOrder(sites, linkindices, CStrings, JCs, DString, JD, τ;)

  ## BULK
  # loop over BULK real space
  bulk = map(
    n -> bulk_timeEvo_ITensor_2ndOrder(
      n, sites, linkindices, AStrings, JAs, BStrings, JBs, CStrings, JCs, DString, JD, τ;
    ),
    2:(N - 1),
  )

  ## RIGHT boundary  
  Ut_N = Right_timeEvo_ITensor_2ndOrder(sites, linkindices, BStrings, JBs, DString, JD, τ;)

  return [Ut_1, bulk..., Ut_N]
end
