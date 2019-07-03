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

        @test names(f(nda; dims=:x)) == (:x, :y) == names(f(nda; dims=1))
    end

    @testset "$f" for f in (cumsum, cumprod, sort)
        @test f(nda; dims=:x) == f(nda; dims=1) == f(a; dims=1)

        @test names(f(nda; dims=:x)) == (:x, :y) == names(f(nda; dims=1))

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
                names(first(eachslice(nda; dims=:b))) ==
                names(first(eachslice(nda; dims=2))) ==
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
            names(mapslices(join, nda; dims=:y)) ==
            names(mapslices(join, nda; dims=2)) ==
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
            names(mapreduce(isodd, |, nda; dims=:y)) ==
            names(mapreduce(isodd, |, nda; dims=2)) ==
            (:x, :y)
        )
    end

    @testset "zero" begin
        a = [10 20; 31 40]
        nda = NamedDimsArray(a, (:x, :y))

        @test zero(nda) == [0 0; 0 0] == zero(a)
        @test names(zero(nda)) == (:x, :y)
    end

    @testset "count" begin
        a = [true false; true true]
        nda = NamedDimsArray(a, (:x, :y))

        @test count(nda) == count(a) == 3
        @test_throws ErrorException count(nda; dims=:x)
        @test_throws ErrorException count(a; dims=1)
    end
end  # Base

@testset "Statistics" begin
    a = [10 20; 30 40]
    nda = NamedDimsArray(a, (:x, :y))
    @testset "$f" for f in (mean, std, var, median)
        @test f(nda) == f(a)
        @test f(nda; dims=:x) == f(nda; dims=1) == f(a; dims=1)

        @test names(f(nda; dims=:x)) == (:x, :y) == names(f(nda; dims=1))
    end
end
