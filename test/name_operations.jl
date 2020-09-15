@testset "unname" begin
    for orig in ([1 2; 3 4], spzeros(2, 2))
        @test unname(NamedDimsArray(orig, (:x, :y))) === orig
        @test unname(orig) === orig
    end
end


@testset "dimnames" begin   
    nda = NamedDimsArray([10 20; 30 40], (:x, :y))

    @test dimnames(nda) === (:x, :y)
    @test dimnames(nda, 2) === :y
    @test dimnames(nda, 3) === :_

    @test dimnames([10 20; 30 40]) === (:_, :_)
    @test dimnames([10 20; 30 40], 2) === :_
    @test dimnames([10 20; 30 40], 3) === :_
    
    v = NamedDimsArray(1:2, :a)
    @test dimnames(v, 2) == dimnames(permutedims(v), 1) # That's why :_ for d > ndims

    @test_throws Exception dimnames(nda, 0)
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

    pair = :a => :w
    @test dimnames(rename(nda, pair)) === (:w, :b, :c, :d)
    @test parent(rename(nda, pair)) === parent(nda)

    pairs = (:a => :w, :b => :x, :c => :y, :d => :z)
    @test dimnames(rename(nda, pairs...)) === (:w, :x, :y, :z)
    @test parent(rename(nda, pairs...)) === parent(nda)

    # parallel (rather than consecutive) computation of pairs, as in Base.replace
    ppairs = (:a => :b, :b => :x)
    @test dimnames(rename(nda, ppairs...)) === (:b, :x, :c, :d) # rather than (:x, :x, :c, :d)
    @test parent(rename(nda, ppairs...)) === parent(nda)
end

@testset "rename allocations" begin
    @test_modern 0 == @ballocated (()->NamedDims._rename((:a, :b), :a => :x, :b => :y))()
end
