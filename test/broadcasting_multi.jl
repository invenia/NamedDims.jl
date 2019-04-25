using NamedDims
using Tracker
using Test

parent_typeof(nda) = typeof(parent(nda))

@testset "Competing Wrappers" begin
    nda = NamedDimsArray(ones(4), :foo)
    ta = TrackedArray(5*ones(4))
    ndt = NamedDimsArray(TrackedArray(5*ones(4)), :foo)

    @show typeof(nda .- ta)
    @test typeof(nda .- ta) <: NamedDimsArray
    @test nda .- ta |> parent_typeof <: TrackedArray


end
