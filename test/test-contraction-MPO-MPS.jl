using ITensors, ITensorMPS
using ITensorInfiniteMPS

includet("../examples/vumps/src/entropy.jl")
includet("../examples/vumps/src/vumps_subspace_expansion.jl")
includet("MPO-time-evo.jl")

n_sites = 2
initstate(n) = "↑"

s = infsiteinds("S=1/2", n_sites; initstate, conserve_szparity=false)
# sqn = infsiteinds("S=1/2", n_sites; initstate, conserve_szparity=true)

ψ = InfMPS(s, initstate)
# ψqn = InfMPS(sqn, initstate)

Szs_infinite = [expect(ψ, "Z", n) for n in 1:length(s)]

H = InfiniteSum{MPO}(Model("ising"), s; J=1.0, h=1.0)
H[1]

dt = 0.05

AStrings = ["Id"]
BStrings = ["X"]
CStrings = ["X"]
DString = "Z"

JAs = [0.6]
JBs = [1.0]
JCs = [-1.0]
JD = 0.0

T_Z = InfiniteMPO(CelledVector([op(s[1], "Z"), op(s[2], "Z")], translator(s)))

# T_Z * ψ

linkindices = get_linkindices_timeEvo_MPO(
  infsiteinds("S=1/2", n_sites; initstate, conserve_szparity=true), BStrings, CStrings
)

U_t = timeEvo_ITensor_2ndOrder(
  s, AStrings, JAs, BStrings, JBs, CStrings, JCs, DString, JD, dt
)
U_tqn = timeEvo_ITensor_2ndOrder(
  sqn, AStrings, JAs, BStrings, JBs, CStrings, JCs, DString, JD, dt
)

dims(U_t[1])
dims(ψ.AL[1])

dims(U_t[2])
dims(ψ.AR[2])

apply(U_t[2], ψ.AR[2])
apply(U_t[1], ψ.AL[1])

dims(U_tqn[1])
dims(ψqn.AL[1])

dims(U_tqn[2])
dims(ψqn.AR[2])

apply(U_tqn[2], ψqn.AR[2])
apply(U_tqn[1], ψqn.AL[1])

apply(U_t[1], ψ.AL[1]) * ψ.C[1] * apply(U_t[2], ψ.AR[2])

apply(U_tqn[1], ψqn.AL[1]) * ψqn.C[1] * apply(U_tqn[2], ψqn.AR[2])

ψ.AL[1] * ψ.C[1] * ψ.AR[2]

H_test_exp = InfiniteExpHTFI(s, λ, J, hz)

#################
# Time evo
#################

dt = 0.05
tf = 5.0

timeline = dt:dt:tf

ψ_tdvp = tdvp_subspace_expansion(
  H_test_exp,
  InfMPS(ComplexF64, s, initstate);
  time_step=-im * dt,
  outer_iters=length(timeline),
  subspace_expansion_kwargs,
  vumps_kwargs=(
    tol=1e-10, maxiter=1, solver_tol=solver_tol, multisite_update_alg=multisite_update_alg
  ),
)

includet("MPO-time-evo.jl")

As = ["Id"]
JAs = [λ]
Bs = ["X"]
JBs = [1.0]
Cs = ["X"]
JCs = [-J]
D = "Z"

U_dt = timeEvo_ITensor_2ndOrder(s, As, JAs, Bs, JBs, Cs, JCs, D, hz, dt)

ψ_trotter = InfMPS(ComplexF64, s, initstate)

# for t_now in enumerate(timeline)
ACs = [ψ_trotter.C[1] * ψ_trotter.AL[1], ψ_trotter.AR[2] * dag(ψ_trotter.C[2])]
linkAC_l = [only(inds(ACs[1], "Link,c=0")), only(inds(ACs[2], "Link,c=1,l=1"))]
linkAC_r = [only(inds(ACs[1], "Link,c=1")), only(inds(ACs[2], "Link,c=1,l=2"))]

linkU_l = [only(inds(U_dt[1], "Link,c=0")), only(inds(U_dt[2], "Link,c=1,l=1"))]
linkU_r = [only(inds(U_dt[1], "Link,c=1")), only(inds(U_dt[2], "Link,c=1,l=2"))]

combiners_l = [
  combiner(linkAC_l[1], linkU_l[1]; tags="Link,c=0,l=2"),# dir=only(unique(dir.([linkAC_l[1], linkU_l[1]])))),
  combiner(linkAC_l[2], linkU_l[2]; tags="Link,c=1,l=1"),# dir=only(unique(dir.([linkAC_l[2], linkU_l[2]])))),
]

combiners_r = [
  # combiner(linkAC_r[1], linkU_r[1], tags="Link,c=1,l=1"),
  dag(combiners_l[2]),
  combiner(linkAC_r[2], linkU_r[2]; tags="Link,c=1,l=2"),# dir=only(unique(dir.([linkAC_r[2], linkU_r[2]])))),
]

U_dt_ψC = map(
  x -> combiners_l[x[1]] * apply(x[2][1], x[2][2]) * combiners_r[x[1]],
  enumerate(zip(U_dt, ACs)),
)

links_new_l = [only(inds(U_dt_ψC[1], "Link,c=0")), only(inds(U_dt_ψC[2], "Link,c=1,l=1"))]
links_new_r = [only(inds(U_dt_ψC[1], "Link,c=1")), only(inds(U_dt_ψC[2], "Link,c=1,l=2"))]

UR1, SR1, VR1 = svd(
  U_dt_ψC[1],
  links_new_l[1];
  cutoff,
  maxdim,
  righttags=tags(links_new_l[1]),
  lefttags=tags(links_new_r[1]),
)
UR2, SR2, VR2 = svd(
  U_dt_ψC[2],
  links_new_l[2];
  cutoff,
  maxdim,
  righttags=tags(links_new_l[2]),
  lefttags=tags(links_new_r[2]),
)
UL1, SL1, VL1 = svd(
  U_dt_ψC[1],
  (links_new_l[1], only(inds(U_dt_ψC[1], "Site")));
  cutoff,
  maxdim,
  righttags=tags(links_new_l[1]),
  lefttags=tags(links_new_r[1]),
)
UL2, SL2, VL2 = svd(
  U_dt_ψC[2],
  (links_new_l[2], only(inds(U_dt_ψC[2], "Site")));
  cutoff,
  maxdim,
  righttags=tags(links_new_l[2]),
  lefttags=tags(links_new_r[2]),
)

inds(SR1)
inds(SR2)
inds(SL1)
inds(SL2)

VR1
VR2

UL1
UL2

Cs = []
new_state = CelledVector([UL1, UL2], translator(s))

# end

# ψ_tdvp[:] = tdvp(H_test_exp, ψ_tdvp; time_step=dt, vumps_kwargs...)
