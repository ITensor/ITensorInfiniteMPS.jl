
# power method
function power_method(T::ITensorMap, v::ITensor;
                      maxiter::Integer = 1000, tol::Real = 1e-12)
  # First iteration
  v /= norm(v)
  Tv = T(v)
  λ = dag(v) * Tv
  err = norm(Tv - λ*v)
  Tv ./= norm(Tv)
  v = Tv

  iter = 1
  while iter < maxiter && err > tol
    Tv = T(v)
    λ = dag(v) * Tv
    err = norm(Tv - λ*v)
    Tv ./= norm(Tv)
    v = Tv
    iter += 1
  end
  return λ, v, (numiter = iter, residual = err)
end

