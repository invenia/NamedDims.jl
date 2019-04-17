using NamedDims
using NamedDims: names
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

@testset "adjoint" begin
    @testset "Vector adjoint" begin
        ndv = NamedDimsArray{(:foo,)}([10,20,30])
        @test ndv' == [10 20 30]
        @test names(ndv') == (:_, :foo)

        # Make sure vector double adjoint gets you back to the start.
        @test (ndv')' == [10, 20, 30]
        @test names((ndv')') == (:foo,)
    end

    @testset "Matrix adjoint" begin
        ndm = NamedDimsArray{(:foo,:bar)}([10 20 30; 11 22 33])
        @test ndm' == [10 11; 20 22; 30 33]
        @test names(ndm') == (:bar, :foo)

        # Make sure implementation of matrix double adjoint is correct
        # since it is easy for the implementation of vector double adjoint broke it
        @test (ndm')' == [10 20 30; 11 22 33]
        @test names((ndm')') == (:foo, :bar)
    end
end
