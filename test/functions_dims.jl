using NamedDims
using Test

@testset "dropdims" begin
    nda = NamedDimsArray{(:a, :b, :c, :d)}(ones(10,1,1,20))

    @test_throws ArgumentError dropdims(nda; dims=1)
    @test_throws ArgumentError dropdims(nda; dims=:a)

    @test dropdims(nda; dims=:b) == ones(10,1,20) == dropdims(nda; dims=2)
    @test names(dropdims(nda; dims=:b)) == (:a, :c, :d) == names(dropdims(nda; dims=2))

    @test dropdims(nda; dims=(:b,:c)) == ones(10,20) == dropdims(nda; dims=(2,3))
    @test names(dropdims(nda; dims=(:b,:c))) == (:a, :d) == names(dropdims(nda; dims=(2,3)))
end
