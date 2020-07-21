using NamedDims
using NamedDims: names
using SparseArrays
using Test


@testset "get the parent array that was wrapped" begin
    for orig in ([1 2; 3 4], spzeros(2, 2))
        @test parent(NamedDimsArray(orig, (:x, :y))) === orig

        @test unname(NamedDimsArray(orig, (:x, :y))) === orig
        @test unname(orig) === orig
    end
end


@testset "get the named array that was wrapped" begin
    @test dimnames(NamedDimsArray([10 20; 30 40], (:x, :y))) === (:x, :y)
end


@testset "Name-asserting constructor" begin
    orig_full = NamedDimsArray(ones(3, 4, 5), (:a, :b, :c))
    @test dimnames(NamedDimsArray(orig_full, (:a, :b, :c))) == (:a, :b, :c)
    @test dimnames(NamedDimsArray(orig_full, (:a, :b, :_))) == (:a, :b, :c)
    @test_throws DimensionMismatch NamedDimsArray(orig_full, (:a, :b, :wrong))
    @test_throws DimensionMismatch NamedDimsArray(orig_full, (:c, :a, :b))

    orig_partial = NamedDimsArray(ones(3, 4, 5), (:a, :_, :c))
    @test dimnames(NamedDimsArray(orig_partial, (:a, :b, :c))) == (:a, :b, :c)
    @test dimnames(NamedDimsArray(orig_partial, (:a, :_, :c))) == (:a, :_, :c)

    @testset "deprecated: refine_names" begin
        @test dimnames(refine_names(orig_full, (:a, :b, :_))) == (:a, :b, :c)
        @test_throws DimensionMismatch NamedDimsArray(orig_full, (:a, :b, :wrong))
    end
end

@testset "getindex" begin
    nda = NamedDimsArray([10 20; 30 40], (:x, :y))

    @test nda[x=1, y=1] == nda[y=1, x=1] == nda[1, 1] == nda[CartesianIndex(1, 1)] == 10
    @test nda[y=end, x=end] == nda[end, end] == 40

    # Unspecified dims become slices
    @test nda[y=1] == nda[y=1, x=:] == nda[:, 1] == [10; 30]

    @test nda[CartesianIndex(1), 1] == nda[1, 1]

    @testset "Views with all dims specified are not scalars" begin
        @test view(nda, x=1, y=1) isa SubArray{Int, 0}
        @test view(nda, 1,1) == view(nda.data, 1,1)
    end

    @testset "No arguments" begin
        ndm = NamedDimsArray([10 20; 30 40], (:x, :y))
        ndv = NamedDimsArray([1, 2, 3], (:x,))
        @test_throws BoundsError ndm[]
        @test_throws BoundsError ndv[]

        nds = NamedDimsArray([4], (:x,))  # 1d 1el array (vector with one element)
        nds2 = NamedDimsArray{()}(fill(4)); # 0d 1el array (scalar)
        @test nds[] == 4
        @test nds2[] == 4
    end

    @testset "name preservation" begin
        @test dimnames(nda[y=1]) == (:x, )
        @test dimnames(nda[y=1:1]) == (:x, :y)
    end

    # https://github.com/invenia/NamedDims.jl/issues/8
    @testset "with multiple-wildcards" begin
        nda_mw = NamedDimsArray{(:_, :_, :c)}(ones(10, 20, 30));
        @test nda_mw[c=2] == ones(10, 20)
        @test dimnames(nda_mw[c=2]) == (:_, :_)
    end

    @testset "newaxis" begin
        newaxis = [CartesianIndex()];

        @test dimnames(nda[:, newaxis, :]) == (:x, :_, :y)
        @test size(nda[:, newaxis, :]) == (2, 1, 2)

        @test dimnames(nda[1, newaxis, 1]) == (:_, )
        @test size(nda[1, newaxis, 1]) == (1, )

        @test dimnames(nda[CartesianIndex(1, 1), newaxis]) == (:_, )
        @test size(nda[CartesianIndex(1, 1), newaxis]) == (1, )
        @test nda[CartesianIndex(1, 1), newaxis][1] == nda[1, 1]
    end

    @testset "BitArray / Array{Bool}" begin
        nda = NamedDimsArray(rand(1:9, 10,11,12), (:x, :y, :z))
        ndv = NamedDimsArray(rand(1:9, 10), :x)

        # one of several dimensions:
        @test dimnames(nda[rand(10) .> 0.3, 1, :]) == (:x, :z)          # BitArray
        @test dimnames(nda[collect(rand(10) .> 0.3), 1, :]) == (:x, :z) # Array{Bool}
        @test dimnames(nda[ndv .> 3, 1, :]) == (:x, :z)                 # NamedDimsArray{...}
        @test dimnames(nda[NamedDimsArray(collect(rand(10) .> 0.3), :x), 1, :]) == (:x, :z)

        # only dim of a vector:
        @test dimnames(ndv[rand(10) .> 0.3]) == (:x,)
        @test dimnames(ndv[collect(rand(10) .> 0.3)]) == (:x,)
        @test dimnames(ndv[ndv .> 3]) == (:x,)
        @test dimnames(ndv[NamedDimsArray(collect(ndv .> 3), :x)]) == (:x,)

        # https://github.com/invenia/NamedDims.jl/issues/70
        # flattening of ndims>1 should drop names, as new dim is none of original ones:
        @test nda[rand(10,11,12) .> 0.3] isa Vector
        @test nda[collect(rand(10,11,12) .> 0.3)] isa Vector
        @test nda[nda .> 3] isa Vector
        @test nda[NamedDimsArray(collect(nda .> 3), dimnames(nda))] isa Vector
    end

    @testset "arrays of indices" begin
        nda = NamedDimsArray(rand(1:9, 3,3,3), (:x, :y, :z))
        ndv = NamedDimsArray(rand(1:9, 10), :x)

        # integers
        @test dimnames(nda[:, [1,3], :]) == (:x, :y, :z)
        @test dimnames(nda[:, [1 3; 3 1], :]) == (:x, :_, :_, :z)

        @test dimnames(ndv[[1, 3]]) == (:x,)
        @test dimnames(ndv[[1 3; 3 1]]) == (:_, :_)

        # Vector{CartesianIndex{N}}: for N>1 this makes a new dim, like nda[nda .> 3]
        @test nda[findall(iseven, nda)] isa Vector
        @test dimnames(ndv[findall(iseven, ndv)]) == (:x,)
    end
end


@testset "views" begin
    nda = NamedDimsArray([10 20; 30 40], (:x, :y))

    @test @view(nda[y=1]) == @view(nda[y=1, x=:]) == @view(nda[:, 1]) == [10; 30]

    @testset "name preservation" begin
        @test dimnames(nda[y=1]) == (:x, )
        @test dimnames(nda[y=1:1]) == (:x, :y)
    end
end


@testset "setindex!" begin
    @testset "by name" begin
        nda = NamedDimsArray([10 20; 30 40], (:x, :y))

        nda[x=1, y=1] = 100
        @test nda == [100 20; 30 40]

        nda[x=1] .= 1000
        @test nda == [1000 1000; 30 40]
    end

    @testset "no arguments" begin
        nds = NamedDimsArray([4], (:x,))  # 1d 1el array (vector with one element)
        nds2 = NamedDimsArray{()}(fill(4)); # 0d 1el array (scalar)
        @test (nds[] = 2) == 2
        @test (nds2[] = 2) == 2
    end

    @testset "by position" begin
        nda = NamedDimsArray([10 20; 30 40], (:x, :y))

        nda[1, 1] = 100
        @test nda == [100 20; 30 40]

        nda[1, :] .= 1000
        @test nda == [1000 1000; 30 40]
    end
end


@testset "IndexStyle" begin
    nda = NamedDimsArray([10 20; 30 40], (:x, :y))
    @test IndexStyle(typeof(nda)) == IndexLinear()

    sparse_nda = NamedDimsArray(spzeros(4, 2), (:x, :y))
    @test IndexStyle(typeof(sparse_nda)) == IndexCartesian()
end


@testset "length/size/axes" begin
    nda = NamedDimsArray([10 20; 30 40; 50 60], (:x, :y))

    @test length(nda) == 6

    @test axes(nda) == (1:3, 1:2)
    @test axes(nda, :x) == (1:3) == axes(nda, 1)

    @test size(nda) == (3, 2)
    @test size(nda, :x) == 3 == size(nda, 1)
end


@testset "similar" begin
    nda = NamedDimsArray{(:a, :b, :c, :d)}(ones(10, 20, 30, 40))

    @testset "content" begin
        ndb = similar(nda)
        @test parent(ndb) !== parent(nda)
        @test eltype(ndb) == Float64
        @test size(ndb) == (10, 20 , 30, 40)
        @test dimnames(ndb) == (:a, :b, :c, :d)
    end
    @testset "eltype" begin
        ndb = similar(nda, Char)
        @test parent(ndb) !== parent(nda)
        @test eltype(ndb) == Char
        @test size(ndb) == (10, 20, 30, 40)
        @test dimnames(ndb) == (:a, :b, :c, :d)
    end
    @testset "size" begin
        ndb = similar(nda, Float64, (15, 25, 35, 45))
        @test parent(ndb) !== parent(nda)
        @test eltype(ndb) == Float64
        @test size(ndb) == (15, 25, 35, 45)
        @test dimnames(ndb) == (:a, :b, :c, :d)
    end
    @testset "dim names" begin
        ndb = similar(nda, Float64, (:w, :x, :y, :z))
        @test parent(ndb) !== parent(nda)
        @test eltype(ndb) == Float64
        @test size(ndb) == (10, 20, 30, 40)
        @test dimnames(ndb) == (:w, :x, :y, :z)
    end

    @testset "dimensions" begin
        ndb = similar(nda, Float64, (w=11, x=22))
        @test parent(ndb) !== parent(nda)
        @test eltype(ndb) == Float64
        @test size(ndb) == (11, 22)
        @test dimnames(ndb) == (:w, :x)
    end
end

@testset "Strided Array Interface" begin
    x = ones(3, 5)
    nda = NamedDimsArray{(:a, :b)}(x)
    @test strides(nda) == (1, 3) == strides(x)
    @test stride(nda, :b) == 3 == stride(nda, 2) == stride(x, 2)
end

const cnda = NamedDimsArray([10 20; 30 40], (:x, :y))
@testset "allocations: wrapper" begin
    @test 0 == @ballocated parent(cnda)
    @test 0 == @ballocated dimnames(cnda)

    # These tests use `@allocated` as for some reason `@ballocated` reports 1 alloc
    if VERSION >= v"1.4" # see https://github.com/invenia/NamedDims.jl/issues/115
        @test_broken 0 == @allocated NamedDimsArray(cnda, (:x, :y))
        @test_broken 0 == @allocated NamedDimsArray(cnda, (:x, :_))
    elseif VERSION >= v"1.1"
        @test 0 == @allocated NamedDimsArray(cnda, (:x, :y))
        @test 0 == @allocated NamedDimsArray(cnda, (:x, :_))
    else
        @test 0 == @allocated NamedDimsArray(cnda, (:x, :y))
        @test_broken 0 == @allocated NamedDimsArray(cnda, (:x, :_))
    end

    # indexing
    @test 0 == @ballocated cnda[1,1]
    @test 0 == @ballocated cnda[1,1] = 55

    @test 0 == @ballocated cnda[x=1, y=1]
    @test @ballocated(cnda[x=1]) == @ballocated(cnda[1, :])
    @test 0 == @ballocated cnda[x=1, y=1] = 66
end
