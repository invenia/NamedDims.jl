using NamedDims
using SparseArrays
using Test

@testset "get the parent array that was wrapped" begin
    orig = [1 2; 3 4]
    @test parent(NamedDimsArray(orig, (:x, :y))) === orig
end


@testset "get the named array that was wrapped" begin
    @test dim_names(NamedDimsArray([10 20; 30 40], (:x, :y))) === (:x, :y)
end


@testset "getindex" begin
    nda = NamedDimsArray([10 20; 30 40], (:x, :y))

    @test nda[x=1, y=1] == nda[y=1, x=1] == nda[1, 1] == 10
    @test nda[y=end,x=end] == nda[end, end] == 40

    # Missing dims become slices
    @test nda[y=1] == nda[y=1, x=:] == nda[:, 1] == [10; 30]
end

@testset "setindex!" begin
    @testset "by name" begin
        nda = NamedDimsArray([10 20; 30 40], (:x, :y))

        nda[x=1, y=1] = 100
        @test nda == [100 20; 30 40]

        # WORKWORK OUT WHY this is an error
        #nda[x=1] .= 1000
        #@test nda == [1000 1000; 30 40]
    end

    @testset "by position" begin
        nda = NamedDimsArray([10 20; 30 40], (:x, :y))

        nda[1, 1] = 100
        @test nda == [100 20; 30 40]

        nda[1,:] .= 1000
        @test nda == [1000 1000; 30 40]
    end
end

@testset "IndexStyle" begin
    nda = NamedDimsArray([10 20; 30 40], (:x, :y))
    @test IndexStyle(typeof(nda)) == IndexLinear()

    sparse_nda = NamedDimsArray(spzeros(4,2), (:x, :y))
    @test IndexStyle(typeof(sparse_nda)) == IndexCartesian()
end


@testset "length/size/axes" begin
    nda = NamedDimsArray([10 20; 30 40; 50 60], (:x, :y))

    @test length(nda) == 6

    @test axes(nda) == (1:3, 1:2)
    @test axes(nda,:x) == (1:3) == axes(nda,1)

    @test size(nda) == (3,2)
    @test size(nda,:x) == 3 == size(nda,1)
end
