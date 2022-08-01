@testset "chainrules.jl" begin
    @testset "constructor" begin
        test_rrule(NamedDimsArray, rand(2, 3), (:a, :b); check_inferred=VERSION >= v"1.6")
        test_rrule(NamedDimsArray{(:a, :b)}, (rand(2, 3)); check_inferred=VERSION >= v"1.6")
    end

    @testset "ProjectTo" begin
        nda = NamedDimsArray{(:a, :b)}(rand(3, 3))

        @testset "NoTangent()" begin
            @test NoTangent() == ProjectTo(nda)(NoTangent())
        end

        @testset "(:c, :d) -> (:a, :b) == error" begin
            ndb = NamedDimsArray{(:c, :d)}(rand(3, 3))
            @test_throws DimensionMismatch ProjectTo(nda)(ndb)
        end

        @testset "(:_, :_) -> (:a, :b) == (:a, :b)" begin
            x = rand(3, 3)
            projected = @inferred ProjectTo(nda)(x)
            @test dimnames(projected) == dimnames(nda)
        end

        @testset "(:a, :_) -> (:a, :b) == (:a, :b)" begin
            nd1 = NamedDimsArray{(:a, :_)}(rand(3, 3))
            projected = @inferred ProjectTo(nda)(nd1)
            @test dimnames(projected) == dimnames(nda)
        end

        @testset "(:a, :b) -> (:a, :_) == (:a, :b)" begin
            nd1 = NamedDimsArray{(:a, :_)}(rand(3, 3))
            projected = @inferred ProjectTo(nd1)(nda) # switched order compared to above
            @test dimnames(projected) == dimnames(nda)
        end

        @testset "(:_, :b) -> (:a, :_) == (:a, :b)" begin
            nd1 = NamedDimsArray{(:a, :_)}(rand(3, 3))
            nd2 = NamedDimsArray{(:_, :b)}(rand(3, 3))
            projected = @inferred ProjectTo(nd1)(nd2)
            @test dimnames(projected) == (:a, :b)
        end
    end
end
