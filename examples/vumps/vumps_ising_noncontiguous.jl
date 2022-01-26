using ITensors
using ITensorInfiniteMPS

##############################################################################
# VUMPS parameters
#

maxdim = 64 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 100 # Maximum number of iterations of the VUMPS algorithm at each bond dimension
vumps_tol = 1e-6
outer_iters = 4 # Number of times to increase the bond dimension

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#
N = 2# Number of sites in the unit cell
J = 1.0
h = 1.;

function space_shifted(q̃sz)
  return [QN("SzParity", 1 - q̃sz, 2) => 1, QN("SzParity", 0 - q̃sz, 2) => 1]
end

space_ = fill(space_shifted(1), N);
s = infsiteinds("S=1/2", N; space=space_)
initstate(n) = "↑"
ψ = InfMPS(s, initstate);

model = Model("ising");
H = InfiniteITensorSum(model, s, J=J, h = h);
#to test the case where the range is larger than the unit cell size
for x in 1:N
    H.data[x] = H.data[x] * delta(prime(s[x+3]), dag(s[x+3]))
end

@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...));

vumps_kwargs = (tol=vumps_tol, maxiter=max_vumps_iters)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)
ψ_0 = vumps(H, ψ; vumps_kwargs...)

for j in 1:outer_iters
    println("\nIncrease bond dimension")
    ψ_1 = subspace_expansion(ψ_0, H; subspace_expansion_kwargs...)
    println("Run VUMPS with new bond dimension")
    ψ_0 = vumps(H, ψ_1; vumps_kwargs...)
end


function ITensors.expect(ψ::InfiniteCanonicalMPS, o, n)
  return (noprime(ψ.AL[n] * ψ.C[n] * op(o, s[n])) * dag(ψ.AL[n] * ψ.C[n]))[]
end


function ITensors.expect(ψ::InfiniteCanonicalMPS, h::ITensor)
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  s = siteinds(only, ψ)
  δˢ(n) = δ(dag(s[n]), prime(s[n]))
  δˡ(n) = δ(l[n], prime(dag(l[n])))
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  ψ′ = prime(dag(ψ))

  ns = sort(findsites(ψ, h))
  nrange = ns[end] - ns[1] + 1
  idx = 2
  temp_O =  δˡ(ns[1] - 1) * ψ.AL[ns[1]] * h * ψ′.AL[ns[1]]
  for n in ns[1]+1:ns[1]+nrange-1
    if n == ns[idx]
      temp_O =  temp_O * ψ.AL[n] *  ψ′.AL[n]
      idx +=1
    else
      temp_O =  temp_O * ψ.AL[n] *  δˢ(n) * ψ′.AL[n]
    end
  end
  temp_O = temp_O * ψ.C[ns[end]] *   δʳ(ns[end]) * ψ′.C[ns[end]]
  return temp_O[]
end

function ITensors.expect(ψ::InfiniteCanonicalMPS, h::InfiniteITensorSum)
  return [expect(ψ, h[(j, j+1)]) for j in 1:nsites(ψ)]
end

Sz = [expect(ψ_0, "Sz", n) for n in 1:N]
energy_infinite = expect(ψ_0, H)



Nfinite = 100
sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
Hfinite = MPO(model, sfinite; J=J, h = h)
ψfinite = randomMPS(sfinite, initstate; linkdims=10)
@show flux(ψfinite)
sweeps = Sweeps(15)
setmaxdim!(sweeps, maxdim)
setcutoff!(sweeps, cutoff)
energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps)

nfinite = Nfinite ÷ 2
Sz_finite = expect(ψfinite, "Sz")[nfinite:nfinite+N-1]

@show (energy_infinite, energy_finite_total / Nfinite, reference(model, Observable("energy"), J=J, h=h, J₂=J₂))

@show (Sz, Sz_finite)
