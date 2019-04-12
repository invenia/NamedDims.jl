using NamedDims
using NamedDims: order_named_inds
using Test


@testset "name2dim" begin
    @testset "get map only" begin
        @test name2dim((:x, :y)) == (x=1, y=2)

        manynames = Tuple(Symbol.('A':'z'))
        namemap = name2dim(manynames)
        @test keys(namemap) == manynames
        @test values(namemap) == Tuple(1:length(manynames))
    end
    @testset "small case" begin
        @test name2dim((:x, :y), :x)==1
        @test name2dim((:x, :y), :y)==2
        @test name2dim((:x, :y), :z)==0  # not found
    end
    @testset "large case that" begin
        @test name2dim((:x, :y, :a, :b, :c, :d), :x)==1
        @test name2dim((:x, :y, :a, :b, :c, :d), :a)==3
        @test name2dim((:x, :y, :a, :b, :c, :d), :d)==6
        @test name2dim((:x, :y, :a, :b, :c, :d), :z)==0 # not found
    end
end


@testset "order_named_inds" begin
    @test order_named_inds((:x,)) == (:,)
    @test order_named_inds((:x,); x=2) == (2,)

    @test order_named_inds((:x, :y,)) == (:,:)
    @test order_named_inds((:x, :y); x=2) == (2, :)
    @test order_named_inds((:x, :y); y=2, ) == (:, 2)
    @test order_named_inds((:x, :y); y=20, x=30) == (30, 20)
    @test order_named_inds((:x, :y); x=30, y=20) == (30, 20)
end