using ITensors
using ITensorInfiniteMPS
using Test

#
# InfiniteMPO has dangling links at the end of the chain.  We contract these on the outside
#   with l,r terminating vectors, to make a finite lattice MPO.
#
function terminate(h::InfiniteMPO)::MPO
    Ncell=nsites(h)
    # left termination vector
    il1=commonind(h[1],h[2])
    il0,=noncommoninds(h[1],il1,tags="Link")
    l=ITensor(0.0,il0)
    l[il0=>dim(il0)]=1.0 #assuming lower reg form in h
    # right termination vector
    iln=commonind(h[Ncell-1],h[Ncell])
    ilnp,=noncommoninds(h[Ncell],iln,tags="Link")
    r=ITensor(0.0,ilnp)
    r[ilnp=>1]=1.0 #assuming lower reg form in h
    # build up a finite MPO
    hf=MPO(Ncell)
    hf[1]=dag(l)*h[1] #left terminate
    hf[Ncell]=h[Ncell]*dag(r) #right terminate
    for n in 2:Ncell-1
        hf[n]=h[n] #fill in the bulk.
    end
    return hf
end
#
# Terminate and then call expect
# for inf ψ and finite h, which is already supported in src/infinitecanonicalmps.jl
#
function ITensors.expect(ψ::InfiniteCanonicalMPS, h::InfiniteMPO)
    return expect(ψ,terminate(h)) #defer to src/infinitecanonicalmps.jl
end

#H = ΣⱼΣn (½ S⁺ⱼS⁻ⱼ₊n + ½ S⁻ⱼS⁺ⱼ₊n + SᶻⱼSᶻⱼ₊n)
function ITensorInfiniteMPS.unit_cell_terms(::Model"heisenbergNNN";NNN::Int64)
    opsum = OpSum()
    for n=1:NNN
        J=1.0/n
        opsum += J*0.5, "S+", 1, "S-", 1+n
        opsum += J*0.5, "S-", 1, "S+", 1+n
        opsum += J,     "Sz", 1, "Sz", 1+n
    end
    return opsum
end

function ITensorInfiniteMPS.unit_cell_terms(::Model"hubbardNNN";NNN::Int64)
    U::Float64=0.25
    t::Float64=1.0
    V::Float64=0.5
    opsum = OpSum()
    opsum += (U, "Nupdn", 1)
    for n=1:NNN
        tj,Vj=t/n,V/n
        opsum += -tj, "Cdagup", 1, "Cup", 1 + n
        opsum += -tj, "Cdagup", 1 + n, "Cup", 1
        opsum += -tj, "Cdagdn", 1, "Cdn", 1 + n
        opsum += -tj, "Cdagdn", 1 + n, "Cdn", 1
        opsum +=  Vj, "Ntot"  , 1, "Ntot", 1 + n
    end
    return opsum
end
#
# InfiniteMPO has dangling links at the end of the chain.  We contract these outside
# with l,r terminating vectors, to make a finite lattice MPO.  And then call expect
# for inf ψ and finite h, whihc is alread supported in src/infinitecanonicalmps.jl
#
function ITensors.expect(ψ::InfiniteCanonicalMPS, h::InfiniteMPO)
    Ncell=nsites(h)
    # left termination vector
    il1=commonind(h[1],h[2])
    il0,=noncommoninds(h[1],il1,tags="Link")
    l=ITensor(0.0,il0)
    l[il0=>dim(il0)]=1.0 #assuming lower reg form in h
    # right termination vector
    iln=commonind(h[Ncell-1],h[Ncell])
    ilnp,=noncommoninds(h[Ncell],iln,tags="Link")
    r=ITensor(0.0,ilnp)
    r[ilnp=>1]=1.0 #assuming lower reg form in h
    # build up a finite MPO
    hf=MPO(Ncell)
    hf[1]=dag(l)*h[1] #left terminate
    hf[Ncell]=h[Ncell]*dag(r) #right terminate
    for n in 2:Ncell-1
        hf[n]=h[n] #fill in the bulk.
    end
    return expect(ψ,hf) #defer to src/infinitecanonicalmps.jl
end

@testset verbose=true "InfiniteMPOMatrix -> InfiniteMPO" begin
    ferro(n) = "↑"
    antiferro(n) = isodd(n) ? "↑" : "↓"

    models=[
        (Model"heisenbergNNN"(),"S=1/2"),
        (Model"hubbardNNN"(),"Electron")
        ]
    @testset "H=$model, Ncell=$Ncell, NNN=$NNN, Antiferro=$Af, qns=$qns" for (model,site) in models, qns=[false,true], Ncell in 2:6, NNN in 1:Ncell-1, Af in [true,false]
#     @testset "H=$model, Ncell=$Ncell, NNN=$NNN, Antiferro=$Af" for (model,site) in models, qns in [true], Ncell in 3:3, NNN in 2:2, Af in [false]
            if isodd(Ncell) && Af #skip test since Af state does fit inside odd cells.
            continue
        end
        initstate(n) = Af ? antiferro(n) : ferro(n) 
        model_kwargs=(NNN=NNN,)
        s = infsiteinds(site, Ncell; initstate,conserve_qns=qns);
        ψ = InfMPS(s, initstate) 
        Hi=InfiniteMPO(model, s;model_kwargs...)
        Hs=InfiniteSum{MPO}(model, s;model_kwargs...)
        Es=expect(ψ,Hs)
        Ei=expect(ψ,Hi)
        #@show Es Ei
        @test sum(Es[1:Ncell-NNN]) ≈ Ei atol=1e-14
    end

end
nothing