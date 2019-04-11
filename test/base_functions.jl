using NamedDims
using Test

@testset "sum" begin
    nda = NamedDimsArray([10 20; 30 40], (:x, :y))

    @test sum(nda) == 100 == sum(identity, nda)

    @test sum(nda; dims=:x) == [40 60]
    @test sum(nda; dims=1) == [40 60]
end
