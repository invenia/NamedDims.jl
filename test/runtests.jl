using NamedDims
using Test

@testset "NamedDims.jl" begin
    # Write your own tests here.

    include("name_core.jl")
    include("wrapper_array.jl")
    #include("base_functions.jl")
end
