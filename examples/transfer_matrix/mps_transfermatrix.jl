using ITensors
using ITensorInfiniteMPS

using LinearMaps
using LinearAlgebra

import Base: *
import ITensorInfiniteMPS: input_inds, output_inds

struct ITensorNetworkMap{T} <: LinearMap{T}
  A::ITensorMap
end
ITensorNetworkMap(A::ITensorMap) = ITensorNetworkMap{eltype(A)}(A)
function ITensorNetworkMap(tn::Vector{ITensor}; kwargs...)
  return ITensorNetworkMap(ITensorMap(tn; kwargs...))
end

Base.size(A::ITensorNetworkMap) = size(A.A)

function input_inds(A::ITensorNetworkMap)
  return input_inds(A.A)
end
function output_inds(A::ITensorNetworkMap)
  return output_inds(A.A)
end

function input_inds(A::LinearMaps.LinearCombination)
  in_inds = input_inds(A.maps[1])
  @assert all(M -> hassameinds(input_inds(M), in_inds), A.maps)
  return in_inds
end
function output_inds(A::LinearMaps.LinearCombination)
  out_inds = output_inds(A.maps[1])
  @assert all(M -> hassameinds(output_inds(M), out_inds), A.maps)
  return out_inds
end

function input_inds(A::LinearMaps.CompositeMap)
  # TODO: it is actually an ITensorNetworkMap
  return input_inds(A.maps[1])
end
function output_inds(A::LinearMaps.CompositeMap)
  # TODO: it is actually an ITensorNetworkMap
  return output_inds(A.maps[end])
end

LinearAlgebra.adjoint(A::ITensorNetworkMap) = ITensorNetworkMap(adjoint(A.A))
LinearAlgebra.transpose(A::ITensorNetworkMap) = ITensorNetworkMap(transpose(A.A))

callable(x, y) = x(y)

function apply(f, A::ITensorMap, v::ITensor)
  return f(A, v)
end

# Application on ITensor
apply(f, A::ITensorNetworkMap, v::ITensor) = f(A.A, v)
(A::ITensorNetworkMap)(v::ITensor) = apply(callable, A, v)
(A::ITensorNetworkMap * v::ITensor) = apply(*, A, v)

apply(f, A::LinearMaps.ScaledMap, v::ITensor) = (A.λ * f(A.lmap, v))
(A::LinearMaps.ScaledMap)(v::ITensor) = apply(callable, A, v)
(A::LinearMaps.ScaledMap * v::ITensor) = apply(*, A, v)

apply(f, A::LinearMaps.UniformScalingMap, v::ITensor) = (A.λ * v)
(A::LinearMaps.UniformScalingMap)(v::ITensor) = apply(callable, A, v)
(A::LinearMaps.UniformScalingMap * v::ITensor) = apply(*, A, v)

function apply(f, A::LinearMaps.LinearCombination, v::ITensor)
  N = length(A.maps)
  Av = f(A.maps[1], v)
  for n in 2:N
    Av += f(A.maps[n], v)
  end
  return Av
end
(A::LinearMaps.LinearCombination)(v::ITensor) = apply(callable, A, v)
(A::LinearMaps.LinearCombination * v::ITensor) = apply(*, A, v)

function _replaceinds(::typeof(callable), A::LinearMaps.CompositeMap, v::ITensor)
  return replaceinds(v, output_inds(A.maps[end]) => input_inds(A.maps[1]))
end
function _replaceinds(::typeof(*), A::LinearMaps.CompositeMap, v::ITensor)
  return v
end

function apply(f, A::LinearMaps.CompositeMap, v::ITensor)
  N = length(A.maps)
  Av = v
  for n in 1:N
    Av = A.maps[n] * Av
  end
  Av = _replaceinds(f, A, Av)
  return Av
end
(A::LinearMaps.CompositeMap)(v::ITensor) = apply(callable, A, v)
(A::LinearMaps.CompositeMap * v::ITensor) = apply(*, A, v)

function apply(f, A::LinearMaps.BlockMap, v::Vector{ITensor})
  nrows = length(A.rows)
  ncols = A.rows[1]
  @assert all(==(ncols), A.rows)
  M = reshape(collect(A.maps), nrows, ncols)
  Av = fill(ITensor(), nrows)
  for i in 1:nrows, j in 1:ncols
    Av[i] += f(M[i, j], v[j])
  end
  return Av
end
(A::LinearMaps.BlockMap)(v::Vector{ITensor}) = apply(callable, A, v)
(A::LinearMaps.BlockMap * v::Vector{ITensor}) = apply(*, A, v)

function infmps(N; χ⃗, d)
  n⃗ = 1:N
  e⃗ = [n => mod1(n + 1, N) for n in 1:N]
  linkindex(χ⃗, e) = Index(χ⃗[e], "l=$(e[1])↔$(e[2])")
  l⃗ = Dict([e .=> linkindex(χ⃗, e) for e in e⃗])
  s⃗ = [Index(d, "s=$n") for n in n⃗]
  neigbhors(n, N) = [mod1(n - 1, N) => n, n => mod1(n + 1, N)]
  return [ITensor(getindex.(Ref(l⃗), neigbhors(n, N))..., s⃗[n]) for n in n⃗]
end

ITensors.dag(tn::Vector{ITensor}) = dag.(tn)
function ITensors.prime(::typeof(linkinds), tn::Vector{ITensor})
  tn_p = copy(tn)
  N = length(tn)
  for i in 1:N, j in (i + 1):N
    l = commoninds(tn[i], tn[j])
    tn_p[i] = prime(tn_p[i]; inds=l)
    tn_p[j] = prime(tn_p[j]; inds=l)
  end
  return tn_p
end

interleave(x::Vector, y::Vector) = permutedims([x y])[:]

function ITensors.linkind(tn::Vector{ITensor}, e)
  return commonind(tn[e[1]], tn[e[2]])
end

function transfer_matrix(ψ::Vector{ITensor})
  N = length(ψ)
  ψ′ = prime(linkinds, dag(ψ))
  tn = interleave(reverse(ψ), reverse(ψ′))
  right_inds = [linkind(ψ, N => 1), linkind(ψ′, N => 1)]
  left_inds = [linkind(ψ, N => 1), linkind(ψ′, N => 1)]
  T = ITensorNetworkMap(tn; input_inds=right_inds, output_inds=left_inds)
  return T
end

function transfer_matrices(ψ::Vector{ITensor})
  N = length(ψ)
  ψ′ = prime(linkinds, dag(ψ))
  # Build from individual transfer matrices
  T⃗ = Vector{ITensorNetworkMap}(undef, N)
  for n in 1:N
    n⁺¹ = mod1(n + 1, N)
    n⁻¹ = mod1(n - 1, N)
    right_inds = [linkind(ψ, n => n⁺¹), linkind(ψ′, n => n⁺¹)]
    left_inds = [linkind(ψ, n⁻¹ => n), linkind(ψ′, n⁻¹ => n)]
    T⃗[n] = ITensorNetworkMap([ψ[n], ψ′[n]]; input_inds=right_inds, output_inds=left_inds)
  end
  return T⃗
end

N = 3 # Number of sites in the unit cell
e⃗ = [n => mod1(n + 1, N) for n in 1:N]
χ⃗ = Dict()
χ⃗[1 => 2] = 3
χ⃗[2 => 3] = 4
χ⃗[3 => 1] = 5
d = 2
ψ = infmps(N; χ⃗=χ⃗, d=d)
randn!.(ψ)
T = transfer_matrix(ψ)
v = randomITensor(input_inds(T))
Tv_expected = ITensors.contract([v, reverse(ψ)..., prime(linkinds, dag(reverse(ψ)))...])
T′v_expected = ITensors.contract([v, ψ..., prime(linkinds, dag(ψ))...])
@show T(v) ≈ Tv_expected
@show (2T)(v) ≈ 2Tv_expected
@show (2T + I)(v) ≈ 2Tv_expected + v
@show (2T + 3I)(v) ≈ 2Tv_expected + 3v
@show (T + T)(v) ≈ 2Tv_expected
@show T'(v) ≈ T′v_expected
@show (T + T')(v) ≈ Tv_expected + T′v_expected
@show (T + T)'(v) ≈ 2T′v_expected
@show [T T; T T]([v; v]) .≈ [2Tv_expected; 2Tv_expected]

T⃗ = transfer_matrices(ψ)

@show (T⃗[1] * T⃗[2] * T⃗[3])(v) ≈ Tv_expected
@show (T⃗[1] * T⃗[2] * T⃗[3] + 3I)(v) ≈ Tv_expected + 3v
@show (T⃗[1] * T⃗[2] * T⃗[3])'(v) ≈ T′v_expected
@show (I * T⃗[1] * T⃗[2] * T⃗[3])(v) ≈ Tv_expected
@show (2T⃗[1] * T⃗[2] * T⃗[3])(v) ≈ 2Tv_expected
@show (T⃗[1] * T⃗[2] * T⃗[3] + 2I)(v) ≈ Tv_expected + 2v
@show ((T⃗[1] * T⃗[2] + T⃗[1] * T⃗[2]) * T⃗[3])(v) ≈ 2Tv_expected
@show ((T⃗[1] * T⃗[2] + T⃗[1] * T⃗[2]) * T⃗[3])'(v) ≈ 2T′v_expected
@show ((T⃗[1] * T⃗[2] + T⃗[1] * T⃗[2]) * T⃗[3] + 3I)(v) ≈ 2Tv_expected + 3v

using KrylovKit
dk, vk = eigsolve(T, v)
for n in 1:length(dk)
  @show norm((T - dk[n]I)(vk[n]))
end
