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
    a2 = [1 9; 7 3]
    nda2 = NamedDimsArray(a2, (:x, :y))
    sort!(nda2, dims=:y)
    @test issorted(a2[2, :])
    @test_throws UndefKeywordError sort!(nda2)
end

@testset "mapslices" begin
    @test mapslices(join, nda; dims=:x) == ["1031" "2040"] == mapslices(join, nda; dims=1)
    @test mapslices(join, nda; dims=:y) == reshape(["1020", "3140"], Val(2)) == mapslices(join, nda; dims=2)
    @test mapslices(join, nda; dims=(:x, :y)) == reshape(["10312040"], (1, 1)) == mapslices(join, nda; dims=(1, 2))
    @test_throws UndefKeywordError mapslices(join, nda)
    @test_throws UndefKeywordError mapslices(join, a)

    @test names(mapslices(join, nda; dims=:y)) == (:x, :y) == names(mapslices(join, nda; dims=2))
end

@testset "mapreduce" begin
    @test mapreduce(isodd, |, nda; dims=:x) == [true false] == mapreduce(isodd, |, nda; dims=1)
    @test mapreduce(isodd, |, nda; dims=:y) == [false true]' == mapreduce(isodd, |, nda; dims=2)
    @test mapreduce(isodd, |, nda) == true == mapreduce(isodd, |, a)

    @test names(mapreduce(isodd, |, nda; dims=:y)) == (:x, :y) == names(mapreduce(isodd, |, nda; dims=2))
end

@testset "zero" begin
    @test zero(nda) == [0 0; 0 0] == zero(a)
    @test names(zero(nda)) == (:x, :y)
end

@testset "count" begin
    a = [true false; true true]
    nda = NamedDimsArray(a, (:x, :y))
    @test count(nda) == count(a) == 3
    @test_throws MethodError count(nda; dims=:x)
    @test_throws MethodError count(a; dims=1)
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
