using NamedDims
using Test

@testset "NamedDims.jl" begin
    # Write your own tests here.

    include("name_core.jl")
    include("wrapper_array.jl")
    include("functions.jl")
    include("functions_dims.jl")
    include("functions_math.jl")
end
