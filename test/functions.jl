using NamedDims
using NamedDims: names
using Test
using Statistics

@testset "Base" begin
    a = [10 20; 31 40]
    nda = NamedDimsArray(a, (:x, :y))

    @testset "$f" for f in (sum, prod, maximum, minimum, extrema)
        @test f(nda) == f(a)
        @test f(nda; dims=:x) == f(nda; dims=1) == f(a; dims=1)

        @test dimnames(f(nda; dims=:x)) == (:x, :y) == dimnames(f(nda; dims=1))
    end

    @testset "$f" for f in (cumsum, cumprod, sort)
        @test f(nda; dims=:x) == f(nda; dims=1) == f(a; dims=1)

        @test dimnames(f(nda; dims=:x)) == (:x, :y) == dimnames(f(nda; dims=1))

        @test f([1, 4, 3]) == f(NamedDimsArray([1, 4, 3], :vec))
        @test_throws UndefKeywordError f(nda)
        @test_throws UndefKeywordError f(a)
    end

    @testset "sort!" begin
        a = [1 9; 7 3]
        nda = NamedDimsArray(a, (:x, :y))

        # Vector case
        veca = [1, 9, 7, 3]
        sort!(NamedDimsArray(veca, :vec); order=Base.Reverse)
        @test issorted(veca; order=Base.Reverse)

        # Higher-dim case: `dims` keyword in `sort!` requires Julia v1.1+
        if VERSION > v"1.1-"
            sort!(nda, dims=:y)
            @test issorted(a[2, :])
            @test_throws UndefKeywordError sort!(nda)

            sort!(nda; dims=:x, order=Base.Reverse)
            @test issorted(a[:, 1]; order=Base.Reverse)
        end
    end

    @testset "$f!" for (f,f!) in zip((sum, prod, maximum, minimum), (sum!, prod!, maximum!, minimum!))
        a = [10 20; 31 40]
        nda = NamedDimsArray(a, (:x, :y)) # size (2,2)

        nda1 = sum(nda, dims=1)           # size (1,2)
        nda2 = sum(nda, dims=2)           # size (2,1)
        @testset "ndims==2" begin
            @test f!(nda1, nda) == f!(nda1, a) == f(a, dims=1)
            @test f!(nda2, nda) == f!(nda2, a) == f(a, dims=2)

            @test dimnames(f!(nda1, nda)) == (:x, :y) == dimnames(f!(nda1, a))
            @test dimnames(f!(nda2, nda)) == (:x, :y) == dimnames(f!(nda2, a))

            @test_throws DimensionMismatch f!(nda1, transpose(nda)) # names wrong way around
        end
        @testset "ndims==1 too" begin
            ndx = NamedDimsArray([3,4], :x)
            ndy = NamedDimsArray([5,6], :y)
            nd_ = NamedDimsArray([7,8], :_)

            @test f!(ndx, nda) == f!([0,0], nda) == dropdims(f(a, dims=2), dims=2)
            @test f!(nd_, nda) == f!(ndx, a)

            @test f!(ndy', nda) == f!([0 0], nda) == f(a, dims=1)

            @test_throws DimensionMismatch f!(ndy, nda) # name y on wrong dimension
        end
    end

    @testset "eachslice" begin
        if VERSION > v"1.1-"
            slices = [[111 121; 211 221], [112 122; 212 222]]
            a = cat(slices...; dims=3)
            nda = NamedDimsArray(a, (:a, :b, :c))

            @test (
                sum(eachslice(nda; dims=:c)) ==
                sum(eachslice(nda; dims=3)) ==
                sum(eachslice(a; dims=3)) ==
                slices[1] + slices[2]
            )
            @test_throws ArgumentError eachslice(nda; dims=(1, 2))
            @test_throws ArgumentError eachslice(a; dims=(1, 2))

            @test_throws UndefKeywordError eachslice(nda)
            @test_throws UndefKeywordError eachslice(a)

            @test (
                dimnames(first(eachslice(nda; dims=:b))) ==
                dimnames(first(eachslice(nda; dims=2))) ==
                (:a, :c)
            )
        end
    end

    @testset "mapslices" begin
        a = [10 20; 31 40]
        nda = NamedDimsArray(a, (:x, :y))

        @test (
            mapslices(join, nda; dims=:x) ==
            mapslices(join, nda; dims=1) ==
            ["1031" "2040"]
        )
        @test (
            mapslices(join, nda; dims=:y) ==
            mapslices(join, nda; dims=2) ==
            reshape(["1020", "3140"], Val(2))
        )
        @test (
            mapslices(join, nda; dims=(:x, :y)) ==
            mapslices(join, nda; dims=(1, 2)) ==
            reshape(["10312040"], (1, 1))
        )
        @test_throws UndefKeywordError mapslices(join, nda)
        @test_throws UndefKeywordError mapslices(join, a)

        @test (
            dimnames(mapslices(join, nda; dims=:y)) ==
            dimnames(mapslices(join, nda; dims=2)) ==
            (:x, :y)
        )
    end

    @testset "mapreduce" begin
        a = [10 20; 31 40]
        nda = NamedDimsArray(a, (:x, :y))

        @test mapreduce(isodd, |, nda) == true == mapreduce(isodd, |, a)
        @test (
            mapreduce(isodd, |, nda; dims=:x) ==
            mapreduce(isodd, |, nda; dims=1) ==
            [true false]
        )
        @test (
            mapreduce(isodd, |, nda; dims=:y) ==
            mapreduce(isodd, |, nda; dims=2) ==
            [false true]'
        )
        @test (
            dimnames(mapreduce(isodd, |, nda; dims=:y)) ==
            dimnames(mapreduce(isodd, |, nda; dims=2)) ==
            (:x, :y)
        )
    end

    @testset "zero" begin
        a = [10 20; 31 40]
        nda = NamedDimsArray(a, (:x, :y))

        @test zero(nda) == [0 0; 0 0] == zero(a)
        @test dimnames(zero(nda)) == (:x, :y)
    end

    @testset "count" begin
        a = [true false; true true]
        nda = NamedDimsArray(a, (:x, :y))

        @test count(nda) == count(a) == 3
        @test_throws Exception count(nda; dims=:x)
        @test_throws Exception count(a; dims=1)
    end

    @testset "push!, pop!, etc" begin
        ndv = NamedDimsArray([10, 20, 30], (:i,))

        @test length(push!(ndv, 40)) == 4
        @test dimnames(pushfirst!(ndv, 0)) == (:i,)
        @test ndv == 0:10:40

        @test pop!(ndv) == 40
        @test popfirst!(ndv) == 0
        @test ndv == [10, 20, 30]
    end

    @testset "append!, empty!" begin
        ndv = NamedDimsArray([10, 20, 30], (:i,))
        ndv45 = NamedDimsArray([40, 50], (:i,))
        ndv0 = NamedDimsArray([0, 0], (:zero,))

        @test length(append!(ndv, ndv45)) == 5
        @test dimnames(append!(ndv, [60,70])) == (:i,)

        @test_throws DimensionMismatch append!(ndv, ndv0)
        @test ndv == 10:10:70 # error was thrown before altering

        @test dimnames(empty!(ndv)) == (:i,)
        @test length(ndv) == 0
    end

    @testset "map, map!" begin
        nda = NamedDimsArray([11 12; 21 22], (:x, :y))

        @test dimnames(map(+, nda, nda, nda)) == (:x, :y)
        @test dimnames(map(+, nda, parent(nda), nda)) == (:x, :y)
        @test dimnames(map(+, parent(nda), nda)) == (:x, :y)

        # this method only called based on first two arguments:
        @test dimnames(map(+, parent(nda), parent(nda), nda)) == (:_, :_)

        # one-arg forms work without adding anything... except on 1.0...
        @test dimnames(map(sqrt, nda)) == (:x, :y)
        @test foreach(sqrt, nda) === nothing

        # map! may return a different wrapper of the same data, like sum!
        semi = NamedDimsArray(rand(2,2), (:x, :_))
        @test dimnames(map!(sqrt, rand(2,2), nda)) == (:x, :y)
        @test dimnames(map!(sqrt, semi, nda)) == (:x, :y)

        zed = similar(nda, Float64)
        @test map!(sqrt, zed, nda) == sqrt.(nda)
        @test zed[1,1] == sqrt(nda[1,1])

        # mismatching names
        @test_throws DimensionMismatch map(+, nda, transpose(nda))
        @test_throws DimensionMismatch map(+, nda, parent(nda), nda, transpose(nda))
        @test_throws DimensionMismatch map!(sqrt, semi, transpose(nda))

        @test foreach(+, semi, nda) === nothing
        @test_throws DimensionMismatch foreach(+, semi, transpose(nda))
    end

    @testset "filter" begin
        nda = NamedDimsArray([11 12; 21 22], (:x, :y))
        ndv = NamedDimsArray(1:7, (:z,))

        @test dimnames(filter(isodd, ndv)) == (:z,)
        @test dimnames(filter(isodd, nda)) == (:_,)
    end

    @testset "collect(generator)" begin
        nda = NamedDimsArray([11 12; 21 22], (:x, :y))
        ndv = NamedDimsArray([10, 20, 30], (:z,))

        @test dimnames([sqrt(x) for x in nda]) == (:x, :y)

        @test dimnames([x^i for (i,x) in enumerate(ndv)]) == (:z,)
        @test dimnames([x^i for (i,x) in enumerate(nda)]) == (:x, :y)

        # Iterators.product -- has all names
        @test dimnames([x+y for x in nda, y in ndv]) == (:x, :y, :z)
        @test dimnames([x+y for x in nda, y in 1:5]) == (:x, :y, :_)
        @test dimnames([x+y for x in 1:5, y in ndv]) == (:_, :z)
        four = [x*y/z^p for p in 1:2, x in ndv, y in 1:2, z in nda]
        @test dimnames(four) == (:_, :z, :_, :x, :y)

        # Iterators.flatten -- no obvious name to use
        @test dimnames([x+y for x in nda for y in ndv]) == (:_,)

        if VERSION >= v"1.1"
            # can't see inside eachslice generators, until:
            # https://github.com/JuliaLang/julia/pull/32310
            @test dimnames([sum(c) for c in eachcol(nda)]) == (:_,)
        end
    end

    @testset "equality" begin
        nda = NamedDimsArray([10 20; 30 40], (:x, :y))
        nda2 = NamedDimsArray([10 20; 30 40], (:x, :_))
        nda3 = NamedDimsArray([10 20; 30 40], (:x, :z))
        nda4 = NamedDimsArray([11 22; 33 44], (:x, :y))
        ndv = NamedDimsArray([10, 20, 30], (:x,))

        @testset "$eq" for eq in (Base.:(==), isequal, isapprox)
            @test eq(nda, nda)
            @test eq(nda, nda2)
            @test eq(nda, nda3) == false
            @test eq(nda, nda4) == false
            @test eq(nda, ndv) == false
        end
        @test isapprox(nda, nda4; atol=2π)
    end

end  # Base

@testset "Statistics" begin
    a = [10 20; 30 40]
    nda = NamedDimsArray(a, (:x, :y))
    @testset "$f" for f in (mean, std, var, median)
        @test f(nda) == f(a)
        @test f(nda; dims=:x) == f(nda; dims=1) == f(a; dims=1)

        @test dimnames(f(nda; dims=:x)) == (:x, :y) == dimnames(f(nda; dims=1))
    end
end
