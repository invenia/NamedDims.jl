@testset "cat" begin
    a = [10 20; 31 40]
    nda = NamedDimsArray(a, (:x, :y))

    @testset "basic functionality" begin
        for d in 1:3
            @test cat(a; dims=d) ==
                  parent(cat(nda; dims=d))
            @test cat(a, a; dims=d) ==
                  parent(cat(nda, a; dims=d)) ==
                  parent(cat(a, nda; dims=d)) ==
                  parent(cat(nda, nda; dims=d))
            @test cat(a, a, a; dims=d) ==
                  parent(cat(a, nda, nda; dims=d)) == 
                  parent(cat(nda, a, nda; dims=d)) == 
                  parent(cat(nda, nda, nda; dims=d))
        end

        @test dimnames(cat(nda; dims=3)) == (dimnames(nda)..., :_)
        @test dimnames(cat(nda, nda; dims=3)) == (dimnames(nda)..., :_)
        @test dimnames(cat(nda, nda, nda; dims=3)) == (dimnames(nda)..., :_)
        @test dimnames(cat(a, nda, nda; dims=3)) == (dimnames(nda)..., :_)
        @test dimnames(cat(nda, a, nda; dims=3)) == (dimnames(nda)..., :_)
    end

    @testset "dimensions requirements" begin
        for d in 1:3
            @test_throws DimensionMismatch cat(nda, nda'; dims=d)
            @test_throws DimensionMismatch cat(nda, nda, nda'; dims=d)
            @test_throws DimensionMismatch cat(a, nda, nda'; dims=d)
            @test_skip @test_throws DimensionMismatch cat(a, a, nda, nda'; dims=d)
        end

        for d in 1:2
            @test dimnames(cat(nda, nda; dims=d)) == dimnames(nda)
            @test dimnames(cat(nda, a; dims=d)) == dimnames(nda)
            @test dimnames(cat(a, nda; dims=d)) == dimnames(nda)
        end
        @test dimnames(cat(nda, nda; dims=3)) == (dimnames(nda)..., :_)
        @test dimnames(cat(nda, a; dims=3)) == (dimnames(nda)..., :_)
        @test dimnames(cat(a, nda; dims=3)) == (dimnames(nda)..., :_)
    end

    @testset "dims argument is named" begin
        @test cat(nda, nda; dims=1) == cat(nda, nda; dims=:x)
        @test cat(nda, nda; dims=2) == cat(nda, nda; dims=:y)

        @test dimnames(cat(nda, nda; dims=:z)) == (:x, :y, :z)
    end
end

for (f, d) in zip((vcat, hcat), (1, 2))
    a = [10 20; 31 40]
    nda = NamedDimsArray(a, (:x, :y))

    @testset "$f" begin
        @testset "basic functionality" begin
            @test f(nda, nda) == cat(nda, nda; dims=d)
            @test f(a, nda) == cat(a, nda; dims=d)
            @test f(nda, a) == cat(nda, a; dims=d)

            @test f(nda, nda, nda) == cat(nda, nda, nda; dims=d)
            @test f(a, nda, nda) == cat(a, nda, nda; dims=d)
            @test f(nda, a, nda) == cat(nda, a, nda; dims=d)
            @test f(nda, nda, a) == cat(nda, nda, a; dims=d)
        end

        @testset "dimension requirements" begin
            @test_throws DimensionMismatch f(nda, nda')
            @test_throws DimensionMismatch f(nda, nda, nda')

            @test dimnames(f(nda, nda)) == dimnames(cat(nda, nda; dims=d))
            @test dimnames(f(nda, a)) == dimnames(cat(nda, a; dims=d))
            @test dimnames(f(a, nda)) == dimnames(cat(a, nda; dims=d))
        end
    end
end
