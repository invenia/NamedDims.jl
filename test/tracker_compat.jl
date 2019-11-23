using NamedDims
using NamedDims: names
using Test
using Tracker
using Tracker: data, TrackedReal

@testset "Tracker Compat" begin

    tw = TrackedArray(2ones(3,5))
    tb = TrackedArray(5ones(3))

    nx = NamedDimsArray{(:features, :obs)}(ones(5,10))
    ny = NamedDimsArray{(:targets, :obs)}(ones(3,10))

    y = tw*nx .+ tb

    @test data(y) == 15ones(3,10)
    @test dimnames(y) == (:_, :obs)
    @test dimnames(data(y)) == (:_, :obs)


    loss = sum((y .- ny).^2)
    @test loss isa TrackedReal

    Tracker.back!(loss)
    @test size(Tracker.grad(tw)) == (3,5)
    @test all(!isequal(0.0), Tracker.grad(tw))
    @test size(Tracker.grad(tb)) == (3,)
    @test all(!isequal(0.0), Tracker.grad(tb))
end
