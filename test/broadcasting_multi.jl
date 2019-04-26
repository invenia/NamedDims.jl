using NamedDims
using Tracker
using Test

parent_typeof(nda) = typeof(parent(nda))

@testset "Competing Wrappers" begin
    nda = NamedDimsArray(ones(4), :foo)
    ta = TrackedArray(5*ones(4))
    ndt = NamedDimsArray(TrackedArray(5*ones(4)), :foo)

    arrays = (nda, ta, ndt)
    @testset "$a .- $b" for (a, b) in Iterators.product(arrays, arrays)
        a === b && continue
        @test typeof(nda .- ta) <: NamedDimsArray
        @test nda .- ta |> parent_typeof <: TrackedArray
    end
end
