using NamedDims
using NamedDims:
    names,
    combine_names,
    order_named_inds,
    remaining_dimnames_from_indexing,
    remaining_dimnames_after_dropping
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
    @testset "large case" begin
        @test dim((:x, :y, :a, :b, :c, :d), :x)==1
        @test dim((:x, :y, :a, :b, :c, :d), :a)==3
        @test dim((:x, :y, :a, :b, :c, :d), :d)==6
        @test dim((:x, :y, :a, :b, :c, :d), :z)==0 # not found
    end
end

@testset "combine_names" begin
    @test combine_names((:a,), (:a,)) == (:a,)
    @test combine_names((:a,:b), (:a,:b)) == (:a,:b)
    @test combine_names((:a,:_), (:a,:b)) == (:a,:b)
    @test combine_names((:a,:_), (:a,:_)) == (:a,:_)

    @test combine_names((:a,:b,:c), (:_,:_,:_)) == (:a,:b,:c)
    @test combine_names((:a,:_,:c), (:_,:b,:_)) == (:a,:b,:c)
    @test combine_names((:_,:_,:_), (:_,:_,:_)) == (:_,:_,:_)

    @test_throws DimensionMismatch combine_names((:a,), (:b,))
    @test_throws DimensionMismatch combine_names((:a,), (:a, :b,))
    @test_throws DimensionMismatch combine_names((:a,:b,:c), (:_,:_,:d))
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

@testset "remaining_dimnames_from_indexing" begin
    @test remaining_dimnames_from_indexing((:a, :b, :c), (10,20,30)) == tuple()
    @test remaining_dimnames_from_indexing((:a, :b, :c), (10,:,30)) == (:b,)
    @test remaining_dimnames_from_indexing((:a, :b, :c), (1:1, [true], [20])) == (:a, :b, :c)
end


@testset "remaining_dimnames_after_dropping" begin
    @test remaining_dimnames_after_dropping((:a, :b, :c), 1) == (:b, :c)
    @test remaining_dimnames_after_dropping((:a, :b, :c), 3) == (:a, :b)
    @test remaining_dimnames_after_dropping((:a, :b, :c), (1,3)) == (:b,)
    @test remaining_dimnames_after_dropping((:a, :b, :c), (3,1)) == (:b,)
end
