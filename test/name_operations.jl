@testset "unname" begin
    for orig in ([1 2; 3 4], spzeros(2, 2))
        @test unname(NamedDimsArray(orig, (:x, :y))) === orig
        @test unname(orig) === orig
    end
end

@testset "get the names" begin
    @test dimnames(NamedDimsArray([10 20; 30 40], (:x, :y))) === (:x, :y)
end


@testset "refine_names" begin
    @testset "Named Array into a NamedDimsArray" begin
        nda = refine_names(ones(3, 4, 5), (:a, :b, :c))
        @test dimnames(nda) == (:a, :b, :c)
        @test nda isa NamedDimsArray
    end

    @testset "Functioning on a fully named NamedDimsArray" begin
        orig_full = NamedDimsArray(ones(3, 4, 5), (:a, :b, :c))
        @test dimnames(refine_names(orig_full, (:a, :b, :c))) == (:a, :b, :c)
        @test dimnames(refine_names(orig_full, (:a, :b, :_))) == (:a, :b, :c)
        @test_throws DimensionMismatch refine_names(orig_full, (:a, :b, :wrong))
        @test_throws DimensionMismatch refine_names(orig_full, (:c, :a, :b))
    end

    @testset "Functioning on a partially named  NamedDimsArray" begin
        orig_partial = NamedDimsArray(ones(3, 4, 5), (:a, :_, :c))
        @test dimnames(refine_names(orig_partial, (:a, :b, :c))) == (:a, :b, :c)
        @test dimnames(refine_names(orig_partial, (:a, :_, :c))) == (:a, :_, :c)
    end
end


@testset "rename" begin
    nda = NamedDimsArray{(:a, :b, :c, :d)}(ones(10, 1, 1, 20))
    new_nda = rename(nda, (:w, :x, :y, :z))

    @test dimnames(new_nda) === (:w, :x, :y, :z)
    @test parent(new_nda) === parent(nda)
end
