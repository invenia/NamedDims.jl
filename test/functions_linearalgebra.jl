
# LinearAlgebra
@testset "lu" begin
    nda = NamedDimsArray{(:foo, :bar)}([1.0 2; 3 4])
    x = lu(nda)
    @test names(x.L) == (:foo, :_)
    @test names(x.U) == (:_, :bar)
    @test names(x.p) == (:foo,)
    @test names(x.P) == (:foo, :foo)

    # Idenity opperations should give back original names
    @test names(x.P * nda) == (:foo, :bar)
    @test names(x.L * x.U) == (:foo, :bar)
    @test names(nda[x.p, :]) == (:foo, :bar)
end

@testset "lq" begin
    nda = NamedDimsArray{(:foo, :bar)}([1.0 2; 3 4])
    x = lq(nda)
    @test names(x.L) == (:foo, :_)
    @test names(x.Q) == (:_, :bar)

    # Idenity opperations should give back original names
    @test names(x.L * x.Q) == (:foo, :bar)
end

@testset "svd" begin
    nda = NamedDimsArray{(:foo, :bar)}([1.0 2; 3 4])
    x = svd(nda)
    @test names(x.U) == (:foo, :_)
    @test names(x.S) == (:_,)
    @test names(x.V) == (:_, :bar)
    @test names(x.Vt) == (:bar, :_)

    # Identity operation should give back original nam,es
    @test names(x.U * Diagonal(x.S) * x.V) == (:foo, :bar)
end
