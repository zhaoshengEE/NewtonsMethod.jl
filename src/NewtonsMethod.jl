module NewtonsMethod

using LinearAlgebra
using ForwardDiff

function newtonroot(f, f′; x₀, tol = 1E-7, maxiter = 1000)
    x_old = x₀
    normdiff = Inf
    n = 1

    while normdiff > tol && n <= maxiter
        x_new = x_old  - f(x_old)/f′(x_old)
        normdiff = norm(x_new - x_old)
        x_old = x_new
        n += 1
    end

    x_root = x_old
    return x_root
end

function newtonroot(f; x₀, tol = 1E-7, maxiter = 1000)
    f′ = x -> ForwardDiff.derivative(f, x)
    x_root = newtonroot(f, f′; x₀, tol = 1E-7, maxiter = 1000)
    return x_root
end

export newtonroot

end
