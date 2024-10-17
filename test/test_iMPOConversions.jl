using ITensors, ITensorMPS
using ITensorInfiniteMPS
using Test

#With the new definition of InfiniteBlockMPO, the MPO is better behaved, and hence we need to be a bit more careful
function expect_over_unit_cell(ψ::InfiniteCanonicalMPS, h::InfiniteSum{MPO})
  s = siteinds(ψ)
  Ncell = nsites(h)

  energy = expect(ψ, h[1])
  for j in 2:Ncell
    hf = MPO(Ncell)
    for x in 1:(j-1)
      hf[x] = op("Id", s[x])
    end
    for x in j:min(j+length(h[j])-1, Ncell)
      hf[x] = h[j][x + 1 - j]
    end
    if j + length(h[j]) - 1 > Ncell
      right_link = commonind(h[j][Ncell + 1 - j], h[j][Ncell + 2 - j])
      #dim_right = dim(right_link)
      hf[end] *= onehot(dag(right_link) => 1)
    elseif Ncell > j + length(h[j]) - 1
      for x in (j+length(h[j])):Ncell
        hf[x] = op("Id", s[x])
      end
    end
    energy += expect(ψ, hf)
  end
  return energy
end

#
# InfiniteMPO has dangling links at the end of the chain.  We contract these on the outside
#   with l,r terminating vectors, to make a finite lattice MPO.
#
function terminate(h::InfiniteMPO)::MPO
  Ncell = nsites(h)
  # left termination vector
  il0 = commonind(h[1], h[0])
  l = ITensor(0.0, il0)
  l[il0 => dim(il0)] = 1.0 #assuming lower reg form in h
  # right termination vector
  iln = commonind(h[Ncell], h[Ncell + 1])
  r = ITensor(0.0, iln)
  r[iln => 1] = 1.0 #assuming lower reg form in h
  # build up a finite MPO
  hf = MPO(Ncell)
  hf[1] = dag(l) * h[1] #left terminate
  hf[Ncell] = h[Ncell] * dag(r) #right terminate
  for n in 2:(Ncell-1)
    hf[n] = h[n] #fill in the bulk.
  end
  return hf
end
#
# Terminate and then call expect
# for inf ψ and finite h, which is already supported in src/infinitecanonicalmps.jl
#
function ITensors.expect(ψ::InfiniteCanonicalMPS, h::InfiniteMPO)
  return expect(ψ, terminate(h)) #defer to src/infinitecanonicalmps.jl
end

function generate_edges(h::InfiniteBlockMPO)
  Ncell = nsites(h)
  # left termination vector
  Ls = Matrix{ITensor}(undef, 1, size(h[1], 1))
  Ls[1] = ITensor(0)
  for x in 2:(size(h[0], 2)-1)
    il0 = commonind(h[0][1, x], h[1][x, 1])
    Ls[x] = ITensor(0, il0)
  end
  Ls[end] = ITensor(1)

  Rs = Vector{ITensor}(undef, size(h[Ncell], 2))
  Rs[1] = ITensor(1)
  for x in 2:(size(h[Ncell + 1], 1)-1)
    ir0 = commonind(h[Ncell + 1][x, 1], h[Ncell][1, x])
    Rs[x] = ITensor(0, ir0)
  end
  Rs[end] = ITensor(0)
  return Ls, Rs
end

function ITensors.expect(ψ::InfiniteCanonicalMPS, h::InfiniteBlockMPO)
  Ncell = nsites(h)
  L, R = generate_edges(h)
  l = commoninds(ψ.AL[0], ψ.AL[1])
  L = map(a -> contract(a, δ(l, dag(prime(l)))), L)
  r = commoninds(ψ.AR[Ncell + 1], ψ.AR[Ncell])
  R = map(a -> contract(a, δ(r, dag(prime(r)))), R)
  L = map(a -> contract(a, ψ.C[0], dag(prime(ψ.C[0]))), L)
  for j in 1:nsites(ψ)
    temp = map(a -> contract(a, ψ.AR[j], dag(prime(ψ.AR[j]))), h[j])
    L = L * temp
  end
  return (L * R)[][]
end

#H = ΣⱼΣn (½ S⁺ⱼS⁻ⱼ₊n + ½ S⁻ⱼS⁺ⱼ₊n + SᶻⱼSᶻⱼ₊n)
function ITensorInfiniteMPS.unit_cell_terms(::Model"heisenbergNNN"; NNN::Int64)
  opsum = OpSum()
  for n in 1:NNN
    J = 1.0 / n
    opsum += J * 0.5, "S+", 1, "S-", 1 + n
    opsum += J * 0.5, "S-", 1, "S+", 1 + n
    opsum += J, "Sz", 1, "Sz", 1 + n
  end
  return opsum
end

function ITensorInfiniteMPS.unit_cell_terms(::Model"hubbardNNN"; NNN::Int64)
  U::Float64 = 0.25
  t::Float64 = 1.0
  V::Float64 = 0.5
  opsum = OpSum()
  opsum += (U, "Nupdn", 1)
  for n in 1:NNN
    tj, Vj = t / n, V / n
    opsum += -tj, "Cdagup", 1, "Cup", 1 + n
    opsum += -tj, "Cdagup", 1 + n, "Cup", 1
    opsum += -tj, "Cdagdn", 1, "Cdn", 1 + n
    opsum += -tj, "Cdagdn", 1 + n, "Cdn", 1
    opsum += Vj, "Ntot", 1, "Ntot", 1 + n
  end
  return opsum
end

function fermion_momentum_translator(i::Index, n::Integer; N=6)
  #@show n
  ts = tags(i)
  translated_ts = ITensorInfiniteMPS.translatecelltags(ts, n)
  new_i = replacetags(i, ts => translated_ts)
  for j in 1:length(new_i.space)
    ch = new_i.space[j][1][1].val
    mom = new_i.space[j][1][2].val
    new_i.space[j] = Pair(QN(("Nf", ch), ("NfMom", mom + n * N * ch)), new_i.space[j][2])
  end
  return new_i
end

@testset verbose = true "InfiniteBlockMPO -> InfiniteMPO" begin
  ferro(n) = "↑"
  antiferro(n) = isodd(n) ? "↑" : "↓"

  models = [(Model"heisenbergNNN"(), "S=1/2"), (Model"hubbardNNN"(), "Electron")]
  @testset "H=$model, Ncell=$Ncell, NNN=$NNN, Antiferro=$Af, qns=$qns" for (model, site) in
                                                                           models,
    qns in [false, true], Ncell in 2:6, NNN in 1:(Ncell-1),
    Af in [true, false]

    if isodd(Ncell) && Af #skip test since Af state does fit inside odd cells.
      continue
    end
    initstate(n) = Af ? antiferro(n) : ferro(n)
    model_kwargs = (NNN=NNN,)
    s = infsiteinds(site, Ncell; initstate, conserve_qns=qns)
    ψ = InfMPS(s, initstate)
    Hi = InfiniteMPO(model, s; model_kwargs...)
    L = ITensor(only(commoninds(Hi[0], Hi[1])))
    R = ITensor(only(commoninds(Hi[Ncell + 1], Hi[Ncell])))
    Hi, L, R = ITensorInfiniteMPS.combineblocks_linkinds(Hi, L, R)
    temp = order(L * Hi[1])
    @test temp ≈ 3
    temp = order(Hi[Ncell] * R)
    @test temp ≈ 3
    Hm = InfiniteBlockMPO(model, s; model_kwargs...)
    L = Array{ITensor}(undef, 1, size(Hm[1], 1))
    L[1] = ITensor()
    L[end] = ITensor(1.0)
    R = Vector{ITensor}(undef, size(Hm[Ncell], 2))
    R[1] = ITensor(1)
    R[end] = ITensor()
    for x in 2:(length(L)-1)
      L[x] = ITensor(only(commoninds(Hm[0][x, x], Hm[1][x, x])))
      R[x] = ITensor(only(commoninds(Hm[Ncell + 1][x, x], Hm[Ncell][x, x])))
    end
    Hm, L, R = ITensorInfiniteMPS.combineblocks_linkinds(Hm, L, R)
    temp = order.(L * Hm[1])
    ref = order.(L) .+ 2
    @test temp ≈ ref
    temp = order.(Hm[Ncell] * R)
    ref = order.(R) .+ 2
    @test temp ≈ ref
    Hs = InfiniteSum{MPO}(model, s; model_kwargs...)
    Es = expect_over_unit_cell(ψ, Hs)
    Ei = expect(ψ, Hi)
    Em = expect(ψ, Hm)
    #@show Es Ei
    @test Es ≈ Ei atol = 1e-14
    @test Em ≈ Ei atol = 1e-14
  end

  @testset "FQHE Hamitonian" begin
    N = 6
    model = Model"fqhe_2b_pot"()
    model_params = (Vs=[1.0, 0.0, 1.0, 0.0, 0.1], Ly=3.0, prec=1e-8)
    trf = fermion_momentum_translator
    function initstate(n)
      mod1(n, 3) == 1 && return 2
      return 1
    end
    p = 1
    q = 3
    conserve_momentum = true
    s = infsiteinds("FermionK", N; translator=trf, initstate, conserve_momentum, p, q)
    ψ = InfMPS(s, initstate)
    Hs = InfiniteSum{MPO}(model, s; model_params...)
    Hi = InfiniteMPO(model, s, trf; model_params...)
    Es = expect_over_unit_cell(ψ, Hs)
    Ei = expect(ψ, Hi)
    #@show Es Ei
    @test Es ≈ Ei atol = 1e-14
  end
end
nothing
