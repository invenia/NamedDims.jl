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
    @test dimnames(x.U) == (:foo, :_)
    @test dimnames(x.S) == (:_,)
    @test dimnames(x.V) == (:_, :bar)
    @test dimnames(x.Vt) == (:bar, :_)

    # Identity operation should give back original nam,es
    @test dimnames(x.U * Diagonal(x.S) * x.V) == (:foo, :bar)
end

@testset "qr" begin
    for pivot in (true,false)
        for data in ([1.0 2; 3 4], [big"1.0" 2; 3 4])
            nda = NamedDimsArray{(:foo, :bar)}(data)
            x = qr(nda, Val(pivot));
            @test dimnames(x.Q) == (:foo, :_)
            @test dimnames(x.R) == (:_, :bar)

            # Identity operation should give back original dimnames
            @test dimnames(x.Q * x.R) == (:foo, :bar)

            pivot && @testset "pivoted" begin
                @test x isa QRPivoted
                @test dimnames(x.p) == (:foo,)
                @test dimnames(x.P) == (:foo, :foo)

                # Identity operation should give back original dimnames
                @test dimnames(x.P * nda) == (:foo, :bar)
                @test dimnames(nda[x.p, :]) == (:foo, :bar)
            end
        end
    end
end
