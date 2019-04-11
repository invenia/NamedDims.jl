using NamedDims
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
