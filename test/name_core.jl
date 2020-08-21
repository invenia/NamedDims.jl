using NamedDims
using NamedDims:
    names,
    unify_names,
    unify_names_longest,
    unify_names_shortest,
    dim_noerror,
    expand_dimnames,
    tuple_issubset,
    tuple_cat,
    names_are_unifiable,
    order_named_inds,
    permute_dimnames,
    remaining_dimnames_from_indexing,
    remaining_dimnames_after_dropping
using Test


@testset "dim" begin
    @testset "small case" begin
        @test dim((:x, :y), :x) == 1
        @test dim((:x, :y), :y) == 2
        @test_throws ArgumentError dim((:x, :y), :z)  # not found
    end
    @testset "large case" begin
        @test dim((:x, :y, :a, :b, :c, :d), :x) == 1
        @test dim((:x, :y, :a, :b, :c, :d), :a) == 3
        @test dim((:x, :y, :a, :b, :c, :d), :d) == 6
        @test_throws ArgumentError dim((:x, :y, :a, :b, :c, :d), :z) # not found
    end
end
@testset "allocations: dim" begin
    @test 0 == @ballocated (()->dim((:a, :b), :b))()
    @test 0 == @ballocated (()->dim_noerror((:a, :b, :c), :c))()
    if VERSION >= v"1.1"
        @test 0 == @ballocated (()->dim((:a,:b), (:a,:b)))()
    else
        @test_broken 0 == @ballocated (()->dim((:a,:b), (:a,:b)))()
    end
end


@testset "unify_names_*" begin
    @test_throws DimensionMismatch unify_names((:a,), (:a, :b,))

    @test unify_names_longest((:a,), (:a, :b,)) == (:a, :b)
    @test unify_names_longest((:a,), (:a, :_)) == (:a, :_)
    @test unify_names_longest((:a, :b), (:a, :_, :c)) == (:a, :b, :c)

    @test unify_names_shortest((:a,), (:a, :b,)) == (:a,)
    @test unify_names_shortest((:a,), (:a, :_)) == (:a,)
    @test unify_names_shortest((:a, :b), (:a, :_, :c)) == (:a, :b)

    @test_throws DimensionMismatch unify_names_longest((:a, :b, :c), (:b, :a))

    for unify in (unify_names, unify_names_longest, unify_names_shortest)
        @test unify((:a,), (:a,)) == (:a,)
        @test unify((:a, :b), (:a, :b)) == (:a, :b)
        @test unify((:a, :_), (:a, :b)) == (:a, :b)
        @test unify((:a, :_), (:a, :_)) == (:a, :_)

        @test unify((:a, :b, :c), (:_, :_, :_)) == (:a, :b, :c)
        @test unify((:a, :_, :c), (:_, :b, :_)) == (:a, :b, :c)
        @test unify((:_, :_, :_), (:_, :_, :_)) == (:_, :_, :_)

        @test_throws DimensionMismatch unify((:a,), (:b,))
        @test_throws DimensionMismatch unify((:a,:b), (:b, :a))
        @test_throws DimensionMismatch unify((:a, :b, :c), (:_, :_, :d))
    end

    # vararg version
    @test unify_names((:a, :_), (:a, :b,), (:_, :b)) == (:a, :b)
    @test unify_names((:a, :b,)) == (:a, :b)
    @test_throws DimensionMismatch unify_names((:a, :_), (:a, :b,), (:_, :c))
end
@testset "allocations: unify_names_*" begin
    for unify in (unify_names, unify_names_longest, unify_names_shortest)
        if VERSION >= v"1.1"
            @test 0 == @ballocated (()->$unify((:a, :b), (:a, :_)))()
        else
            @test_broken 0 == @ballocated (()->$unify((:a, :b), (:a, :_)))()
        end
    end
    if VERSION >= v"1.1"
        @test 0 == @ballocated (()->unify_names_longest((:a, :b), (:a, :_, :c)))()
        @test 0 == @ballocated (()->unify_names_shortest((:a, :b), (:a, :_, :c)))()
        @test 0 == @ballocated (()->names_are_unifiable((:a, :b), (:a, :_)))()
        @test 0 == @ballocated (()->names_are_unifiable((:a, :b), (:a, :c)))()
    else
        @test_broken 0 == @ballocated (()->unify_names_longest((:a, :b), (:a, :_, :c)))()
        @test_broken 0 == @ballocated (()->unify_names_shortest((:a, :b), (:a, :_, :c)))()
        @test_broken 0 == @ballocated (()->names_are_unifiable((:a, :b), (:a, :_)))()
        @test_broken 0 == @ballocated (()->names_are_unifiable((:a, :b), (:a, :c)))()
    end
    @test 0 == @ballocated (()->names_are_unifiable((:a, :b), (:a, :b)))()
end

@testset "expand_dimnames" begin
    @testset "single symbol" begin
        @test expand_dimnames((), :x) == (:x,)
        @test expand_dimnames((:x,), ()) == (:x,)

        @test expand_dimnames((:x, :y), :x) == (:x, :y)
        @test expand_dimnames((:x, :y), :z) == (:x, :y, :z)
    end

    @testset "multiple symbols" begin
        @test expand_dimnames((), (:x, :y)) == (:x, :y)
        @test expand_dimnames((:x, :y), (:z, :w)) == (:x, :y, :z, :w)
        @test expand_dimnames((:x, :y), (:x, :w)) == (:x, :y, :w)
    end

    @testset "numbers et al" begin
        @test expand_dimnames((:x, :y), 1) == (:x, :y)
        @test expand_dimnames((:x, :y), 3) == (:x, :y, :_)
        @test expand_dimnames((:x, :y), (1, 3)) == (:x, :y, :_)
        @test expand_dimnames((:x, :y), :) == (:x, :y)
        @test expand_dimnames((:x, :y), ()) == (:x, :y)
    end
end

@testset "allocations: expand_dims" begin
    @testset "names" begin
        @test 0 == @ballocated (()->NamedDims.expand_dimnames((:x, :y), :x))()
        @test 0 == @ballocated (()->NamedDims.expand_dimnames((:x, :y), :z))()
        @test 0 == @ballocated (()->NamedDims.expand_dimnames((:x, :y), (:x, :z)))()
    end

    @testset "numbers et al" begin
        @test 0 == @ballocated (()->NamedDims.expand_dimnames((:x, :y), 1))()
        @test 0 == @ballocated (()->NamedDims.expand_dimnames((:x, :y), 5))()
        @test 0 == @ballocated (()->NamedDims.expand_dimnames((:x, :y), (1, 5)))()
        @test 0 == @ballocated (()->NamedDims.expand_dimnames((:x, :y), :))()
        @test 0 == @ballocated (()->NamedDims.expand_dimnames((:x, :y), ()))()
    end
end

@testset "order_named_inds" begin
    @test order_named_inds(Val((:x,))) == (:,)
    @test order_named_inds(Val((:x,)); x=2) == (2,)

    @test order_named_inds(Val((:x, :y))) == (:, :)
    @test order_named_inds(Val((:x, :y)); x=2) == (2, :)
    @test order_named_inds(Val((:x, :y)); y=2, ) == (:, 2)
    @test order_named_inds(Val((:x, :y)); y=20, x=30) == (30, 20)
    @test order_named_inds(Val((:x, :y)); x=30, y=20) == (30, 20)
end
@testset "allocations: order_named_inds" begin
    @test 0 == @ballocated (()->order_named_inds(Val((:a, :b, :c)); b=1, c=2))()
    @test 0 == @ballocated (()->order_named_inds(Val((:a, :b, :c)), (b=1, c=2)))()
end


@testset "remaining_dimnames_from_indexing" begin
    @test remaining_dimnames_from_indexing((:a, :b, :c), (10,20,30)) === ()
    @test remaining_dimnames_from_indexing((:a, :b, :c), (10, :,30)) == (:b,)
    @test remaining_dimnames_from_indexing((:a, :b, :c), (1:1, [true], [20])) == (:a, :b, :c)

    @test remaining_dimnames_from_indexing((:a, :b, :c), (:, [CartesianIndex()], :, :)) == (:a, :_, :b, :c)
    @test remaining_dimnames_from_indexing((:a, :b, :c), (1, [CartesianIndex()], 2, :)) == (:_, :c)
    @test remaining_dimnames_from_indexing((:a, :b, :c), (CartesianIndex(1,1), :)) == (:c,)

    # Cases which drop dimension names
    logical = Base.to_indices(ones(2,3), (rand(2,3).>0.5,))
    @test remaining_dimnames_from_indexing((:a, :b), logical) === ()
    cart_vec = findall(x -> x>0.5, rand(2,3))
    @test remaining_dimnames_from_indexing((:a, :b), (cart_vec,)) === ()
end
@testset "allocations: remaining_dimnames_from_indexing" begin
    @test 0 == @ballocated (()->remaining_dimnames_from_indexing((:a, :b, :c), (:,390,:)))()
end


@testset "remaining_dimnames_after_dropping" begin
    @test remaining_dimnames_after_dropping((:a, :b, :c), 1) == (:b, :c)
    @test remaining_dimnames_after_dropping((:a, :b, :c), 3) == (:a, :b)
    @test remaining_dimnames_after_dropping((:a, :b, :c), (1,3)) == (:b,)
    @test remaining_dimnames_after_dropping((:a, :b, :c), (3,1)) == (:b,)
end
@testset "allocations: remaining_dimnames_after_dropping" begin
    @test 0 == @ballocated remaining_dimnames_after_dropping((:a,:b,:c,:d,:e), 4)
    @test 0 == @ballocated (()->remaining_dimnames_after_dropping((:a,:b,:c,:d,:e), (1,3)))()
end


@testset "permute_dimnames" begin
    @test permute_dimnames((:a, :b, :c), (1, 2, 3)) == (:a ,:b, :c)
    @test permute_dimnames((:a, :b, :c), (3, 2, 1)) == (:c ,:b, :a)

    # permute_dimnames allows non-bijective "permutations"
    @test permute_dimnames((:a, :b, :c), (3, 3, 1)) == (:c ,:c, :a)
    @test permute_dimnames((:a, :b, :c), (3, 1))== (:c, :a)

    @test_throws BoundsError permute_dimnames((:a, :b, :c), (30, 30, 30))
    @test_throws BoundsError permute_dimnames((:a, :b), (1, 0))
end
@testset "allocations: permute_dimnames" begin
    if VERSION >= v"1.1"
        @test 0 == @ballocated permute_dimnames((:a,:b,:c), (1,3,2))
    end
end


@testset "tuple_issubset" begin
    @test tuple_issubset((:a, :c), (:a, :b, :c)) == true
    @test tuple_issubset((:a, :b, :c), (:a, :c)) == false
end
@testset "allocations: tuple_issubset" begin
    @test 0 == @ballocated tuple_issubset((:a, :c), (:a, :b, :c))
    @test 0 == @ballocated tuple_issubset((:a, :b, :c), (:a, :c))
end


@testset "tuple_cat" begin
    @test tuple_cat((1, 2), (3, 4, 5), (6,)) == (1, 2, 3, 4, 5, 6)
    @test tuple_cat((1, 2)) == (1, 2)
end
@testset "allocations: tuple_cat" begin
    @test 0 == @ballocated tuple_cat((1, 2), (3, 4, 5), (6,))
end
