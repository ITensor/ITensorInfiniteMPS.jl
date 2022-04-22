using ITensors
using ITensorInfiniteMPS
using Test
using Random

#Currently, VUMPS cannot give the right result as the subspace expansion is too small
#This is meant to test the generalized translation rules
@time @testset "vumpsmpo_fqhe" begin
  Random.seed!(1234)

  function fermions_space_shift(pos; p=1, q=1, conserve_momentum=true, momentum_shift=1)
    if !conserve_momentum
      return [QN("Nf", -p) => 1, QN("Nf", q - p) => 1]
    else
      return [
        QN(("Nf", -p), ("NfMom", -p * pos + momentum_shift)) => 1,
        QN(("Nf", q - p), ("NfMom", (q - p) * pos + momentum_shift)) => 1,
      ]
    end
  end

  function initstate(n)
    if mod(n, 3) == 1
      return 2
    else
      return 1
    end
  end

  # VUMPS arguments
  cutoff = 1e-8
  maxdim = 3
  tol = 1e-8
  maxiter = 50
  outer_iters = 1

  #### Test 1/3
  model = Model"fqhe_2b_pot"()
  model_kwargs = (Ly=3.0, Vs=[1.0], prec=1e-6) #parent Hamiltonian of Laughlin 13
  p = 1
  q = 3
  conserve_momentum = true
  momentum_shift = 1

  #ψ1 = InfMPS(s, n->[1, 2, 2, 1, 1, 1][n]);

  @testset "VUMPS/TDVP with: multisite_update_alg = $multisite_update_alg, conserve_qns = $conserve_qns, nsites = $nsites" for multisite_update_alg in
                                                                                                                               [
      "sequential"
    ],
    conserve_qns in [true],
    nsites in [6],
    time_step in [-Inf]

    vumps_kwargs = (
      multisite_update_alg=multisite_update_alg,
      tol=tol,
      maxiter=maxiter,
      outputlevel=0,
      time_step=time_step,
    )
    subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

    function fermion_momentum_translator(i::Index, n::Integer; N=nsites)
      ts = tags(i)
      translated_ts = ITensorInfiniteMPS.translatecelltags(ts, n)
      new_i = replacetags(i, ts => translated_ts)
      for j in 1:length(new_i.space)
        ch = new_i.space[j][1][1].val
        mom = new_i.space[j][1][2].val
        new_i.space[j] = Pair(
          QN(("Nf", ch), ("NfMom", mom + n * N * ch)), new_i.space[j][2]
        )
      end
      return new_i
    end

    fermionic_space = [
      fermions_space_shift(
        x; p=p, q=q, conserve_momentum=conserve_momentum, momentum_shift=momentum_shift
      ) for x in 1:nsites
    ]
    s = infsiteinds(
      "Fermion", nsites; space=fermionic_space, translator=fermion_momentum_translator
    )
    ψ = InfMPS(s, initstate)

    Hmpo = InfiniteMPOMatrix(model, s; model_kwargs...)
    # Alternate steps of running VUMPS and increasing the bond dimension
    ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
    for _ in 1:outer_iters
      println("Subspace expansion")
      ψ = @time subspace_expansion(ψ, Hmpo; subspace_expansion_kwargs...)
      println("TDVP")
      ψ = @time tdvp(Hmpo, ψ; vumps_kwargs...)
    end
  end
end
