using LinearAlgebra
using NamedDims
using NamedDims: dimnames
using Test

# LinearAlgebra
@testset "lu" begin
    nda = NamedDimsArray{(:foo, :bar)}([1.0 2; 3 4])
    x = lu(nda)
    @test dimnames(x.L) == (:foo, :_)
    @test dimnames(x.U) == (:_, :bar)
    @test dimnames(x.p) == (:foo,)
    @test dimnames(x.P) == (:foo, :foo)

    # Idenity opperations should give back original dimnames
    @test dimnames(x.P * nda) == (:foo, :bar)
    @test dimnames(x.L * x.U) == (:foo, :bar)
    @test dimnames(nda[x.p, :]) == (:foo, :bar)
end

@testset "lq" begin
    nda = NamedDimsArray{(:foo, :bar)}([1.0 2; 3 4])
    x = lq(nda)
    @test dimnames(x.L) == (:foo, :_)
    @test dimnames(x.Q) == (:_, :bar)

    # Idenity opperations should give back original dimnames
    @test dimnames(x.L * x.Q) == (:foo, :bar)
end

@testset "svd" begin
    nda = NamedDimsArray{(:foo, :bar)}([1.0 2; 3 4])
    x = svd(nda)
    # Test based on visualization on wikipedia
    # https://en.wikipedia.org/wiki/File:Singular_value_decomposition_visualisation.svg
    @test dimnames(x.U) == (:foo, :_)
    @test dimnames(x.S) == (:_,)
    @test dimnames(x.V) == (:bar, :_)
    @test dimnames(x.Vt) == (:_, :bar)

    # Identity operation should give back original nam,es
    @test dimnames(x.U * Diagonal(x.S) * x.Vt) == (:foo, :bar)
end

@testset "qr" begin
    for pivot in (true, false)
        for data in ([1.0 2; 3 4], [big"1.0" 2; 3 4], [1.0 2 3; 4 5 6])
            nda = NamedDimsArray{(:foo, :bar)}(data)
            x = qr(nda, Val(pivot));
            @test dimnames(x.Q) == (:foo, :_)
            @test dimnames(x.R) == (:_, :bar)

            # Identity operation should give back original dimnames
            @test dimnames(x.Q * x.R) == (:foo, :bar)

            pivot && @testset "pivoted" begin
                @test parent(x) isa QRPivoted
                @test dimnames(x.p) == (:bar,)
                @test dimnames(x.P) == (:bar, :bar)

                # Identity operation should give back original dimnames
                @test dimnames(nda * x.P') == (:foo, :bar)
                @test dimnames(nda[:, x.p]) == (:foo, :bar)
                @test dimnames(x.Q * x.R * x.P') == (:foo, :bar)
            end
        end
    end
end
