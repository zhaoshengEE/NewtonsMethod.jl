using NewtonsMethod
using Test

@testset "NewtonsMethod.jl" begin

# several @test for the root of a known function, given the f and analytical f' derivatives
f(x) = x - 10
f′(x)= 1
@test newtonroot(f,f′;x₀ = 0.0) == 10.0

f(x) = x-cos(x)
f′(x)= 1+sin(x)
@test newtonroot(f,f′;x₀ = 0.0) == 0.7390851332151607

f(x) = (x+5)^3
f′(x)= 3*(x+5)^2
@test newtonroot(f,f′;x₀ = 0.0) == -4.999999866018182

# tests of those roots using the automatic differentiation version of the function
f(x) = (x-1)^3
@test newtonroot(f;x₀=0.0) == 0.9999998643434097

f(x) = (x+1)^2
@test newtonroot(f;x₀=0.0) == -0.9999999403953552

f(x) = x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1
@test newtonroot(f;x₀=0.0) == -3.24522908806064


# test of finding those roots with a BigFloat and not just a Float64
f(x) = x - BigFloat(sqrt(5))
@test newtonroot(f;x₀ = 0.0) == BigFloat(sqrt(5))

f(x) =x-BigFloat(pi)
@test newtonroot(f;x₀ = 0.0) == BigFloat(pi)

# test of non-convergence for a function without a root (e.g. f(x)=2+x^2)
f(x) = x^2 + 2
@test newtonroot(f;x₀ = 1.0) == nothing

# test to ensure that the maxiter is working (e.g. what happens if you call maxiter = 5)
f(x) = (x-1)^3
f′(x)= 3*(x-1)^2
@test newtonroot(f,f′;x₀ = 0.0,maxiter = 5) == nothing

f(x) = (x-6)^6
@test newtonroot(f;x₀ = 0.0, maxiter = 10) == nothing

# test to ensure that tol is working
f(x) = (x-1)^3
f′(x)= 3*(x-1)^2
@test newtonroot(f,f′;x₀ = 0.0,tol = 1E-20) == 0.9999999999999999

f(x) = (x-6)^6
@test newtonroot(f;x₀ = 0.0,tol = 1E-3) == 5.995101279575065

end
