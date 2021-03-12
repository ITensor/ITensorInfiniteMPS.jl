
vumps(ψ::InfiniteMPS; kwargs...) = vumps(orthogonalize(ψ, :); kwargs...)

function vumps(H::InfiniteMPO, ψ::InfiniteCanonicalMPS)
  L = left_environment(H, ψ)
end

