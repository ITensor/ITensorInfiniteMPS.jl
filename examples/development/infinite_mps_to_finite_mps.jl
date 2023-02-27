using ITensors
using ITensorInfiniteMPS

include(
  joinpath(
    pkgdir(ITensorInfiniteMPS), "examples", "vumps", "src", "vumps_subspace_expansion.jl"
  ),
)

##########################################################
#Get a state 

maxdim = 20 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 10 # Maximum number of iterations of the VUMPS/TDVP algorithm at a fixed bond dimension
tol = 1e-5 # Precision error tolerance for outer loop of VUMPS or TDVP
outer_iters = 5 # Number of times to increase the bond dimension
time_step = -Inf # -Inf corresponds to VUMPS, finite time_step corresponds to TDVP
solver_tol = (x -> x / 100) # Tolerance for the local solver (eigsolve in VUMPS and exponentiate in TDVP)
multisite_update_alg = "parallel" # Choose between ["sequential", "parallel"]. Only parallel works with TDVP.
conserve_qns = true # Whether or not to conserve spin parity
nsite = 2 # Number of sites in the unit cell
localham_type = ITensor # Can choose `ITensor` or `MPO`

# Parameters of the transverse field Ising model
#model_params = (J=1.0, h=0.9)
model_params = (t=1.0, U=10.0, V=0.0)

model = Model("hubbard")

initstate(n) = isodd(n) ? "Up" : "0"
s = infsiteinds("Electron", nsite; initstate, conserve_qns=conserve_qns)
ψ = InfMPS(s, initstate)

# Form the Hamiltonian
H = InfiniteSum{localham_type}(model, s; model_params...)

# Check translational invariance
@show norm(contract(ψ.AL[1:nsite]..., ψ.C[nsite]) - contract(ψ.C[0], ψ.AR[1:nsite]...))

vumps_kwargs = (
  tol=tol,
  maxiter=max_vumps_iters,
  solver_tol=solver_tol,
  multisite_update_alg=multisite_update_alg,
)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

ψ = vumps_subspace_expansion(H, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs)

##########################################################
#Cut of the Infinite MPS by using deltas and orthogonalize
#TODO: this does not work with Itensors standard correlation_matrix

function toMPS(ψ::InfiniteCanonicalMPS, start, stop)
    N = stop - start + 1
    ψc = (ψ.AL)[start:stop]
    s = inds(ψc[1])[1]
    ss = inds(ψ.C[N])[2]
    s0 = Index(s.space, tags = tags(s), dir = dir(s))
    s0 = removetags(s0, "Link"); s0 = addtags(s0, "Dummy")
    
    sn = Index(ss.space, tags = tags(ss), dir = dir(ss))
    sn = removetags(sn, "Link"); sn = addtags(sn, "Dummy")

    M = MPS(N+2)
    M[1] = ITensor(LinearAlgebra.diagm(fill(1, dim(s0))), s0, dag(s))
    for i in 2:(N)
        M[i] = ψc[i-1]
    end
    M[N+1] = ψc[N]*ψ.C[N]
    M[N+2] = ITensor(LinearAlgebra.diagm(fill(1, dim(ss))), dag(ss), sn)
    return M
end

##########################################################
#Correlation matrix for infinite MPS

function correlation_matrix(ψ::InfiniteCanonicalMPS, op1, op2, dim)
    C = zeros(ComplexF64, dim, dim)
    for i in 1:dim
        for j in (i+1):dim
            h = op(op1, siteinds(ψ)[i]) * op(op2, siteinds(ψ)[j])
            ϕ = ψ.AL[i]
            for k in (i+1):(j)
                ϕ *= ψ.AL[k]
            end
            ϕ *= ψ.C[j]
            C[i,j] = (noprime(ϕ * h) * dag(ϕ))[]
        end
    end
    return C + C'
end

##########################################################
#Correlation matrix for finite MPS using explicit contraction of operator

function correlation_matrix_gates(ψ::MPS, op1, op2, start, stop)
    C = zeros(ComplexF64, stop-start+1, stop-start+1)
    for i in start:stop
        orthogonalize!(ψ, i)
        for j in (i+1):stop
            h = op(op1, siteinds(ψ)[i]) * op(op2, siteinds(ψ)[j])
            ϕ = ψ[i]
            for k in (i+1):(j)
                ϕ *= ψ[k]
            end
            C[i-start+1,j-start+1] = (noprime(ϕ * h) * dag(ϕ))[]
        end
    end
    return C + C'
end

start = 1; stop = 5 #where to slice the infiniteMPS

psi = toMPS(ψ, start, stop)

A = correlation_matrix_gates(psi, "Sz", "Sz", start+1, stop+1)
B = correlation_matrix(ψ, "Sz", "Sz", stop-start+1)

println("with infinite = ")
display(A)
display("with finite = ")
display(B)
display("difference = ")
display(A-B)