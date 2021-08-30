# TODO: make a TransferMatrix type?

TransferMatrix(ψ::InfiniteMPS) = TransferMatrix(ψ, Cell(1))

function TransferMatrix(ψ::InfiniteMPS, c::Cell)
  N = nsites(ψ)
  ψᴴ = prime(linkinds, dag(ψ))
  ψᶜ = ψ[c]
  ψᶜᴴ = ψᴴ[c]
  r = unioninds(linkinds(ψ, N => N + 1), linkinds(ψᴴ, N => N + 1))
  l = unioninds(linkinds(ψ, 1 => 0), linkinds(ψᴴ, 1 => 0))
  return ITensorMap(ψᶜ, ψᶜᴴ; input_inds=r, output_inds=l)
end
