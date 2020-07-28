using NamedDims
using BenchmarkTools
using SparseArrays
using Test

include("test_helpers.jl")

const testfiles = (
    "name_core.jl",
    "wrapper_array.jl",
    "name_operations.jl",
    "functions.jl",
    "functions_dims.jl",
    "functions_math.jl",
    "cat.jl",
    "functions_linearalgebra.jl",
    "broadcasting.jl",
    "fft.jl",
    "tracker_compat.jl",
    "cov_compat.jl",
)

@testset "NamedDims.jl" begin
    @testset "$file" for file in testfiles
        include(file)
    end
end
