# Has become "refine_names"
@testset "Name-asserting constructor" begin
    orig_full = NamedDimsArray(ones(3, 4, 5), (:a, :b, :c))
    @test dimnames(NamedDimsArray(orig_full, (:a, :b, :c))) == (:a, :b, :c)
    @test dimnames(NamedDimsArray(orig_full, (:a, :b, :_))) == (:a, :b, :c)
    @test_throws DimensionMismatch NamedDimsArray(orig_full, (:a, :b, :wrong))
    @test_throws DimensionMismatch NamedDimsArray(orig_full, (:c, :a, :b))

    orig_partial = NamedDimsArray(ones(3, 4, 5), (:a, :_, :c))
    @test dimnames(NamedDimsArray(orig_partial, (:a, :b, :c))) == (:a, :b, :c)
    @test dimnames(NamedDimsArray(orig_partial, (:a, :_, :c))) == (:a, :_, :c)
end
