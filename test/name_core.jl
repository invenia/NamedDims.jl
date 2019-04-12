using NamedDims
using NamedDims: order_named_inds, determine_remaining_dim
using Test


@testset "dim" begin
    @testset "get map only" begin
        @test dim((:x, :y)) == (x=1, y=2)

        manynames = Tuple(Symbol.('A':'z'))
        namemap = dim(manynames)
        @test keys(namemap) == manynames
        @test values(namemap) == Tuple(1:length(manynames))
    end
    @testset "small case" begin
        @test dim((:x, :y), :x)==1
        @test dim((:x, :y), :y)==2
        @test dim((:x, :y), :z)==0  # not found
    end
    @testset "large case that" begin
        @test dim((:x, :y, :a, :b, :c, :d), :x)==1
        @test dim((:x, :y, :a, :b, :c, :d), :a)==3
        @test dim((:x, :y, :a, :b, :c, :d), :d)==6
        @test dim((:x, :y, :a, :b, :c, :d), :z)==0 # not found
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

@testset "determine_remaining_dim" begin
    @test determine_remaining_dim((:a, :b, :c), (10,20,30)) == tuple()
    @test determine_remaining_dim((:a, :b, :c), (10,:,30)) == (:b,)
    @test determine_remaining_dim((:a, :b, :c), (1:1, [true], [20])) == (:a, :b, :c)
end
