using NamedDims
using NamedDims: names
using Test
using Statistics

@testset "sum" begin
    a = [10 20; 30 40]
    nda = NamedDimsArray(a, (:x, :y))

    @test sum(nda) == 100 == sum(a)
    @test sum(nda; dims=:x) == [40 60] == sum(nda; dims=1) == sum(a; dims=1)

    @test names(sum(nda; dims=:x)) == (:x, :y) == names(sum(nda; dims=1))
end

@testset "mean" begin
    a = [10 20; 30 40]
    nda = NamedDimsArray(a, (:x, :y))

    @test mean(nda) == 25 == mean(a)
    @test mean(nda; dims=:x) == [20 30] == mean(nda; dims=1) == mean(a; dims=1)

    @test names(mean(nda; dims=:x)) == (:x, :y) == names(mean(nda; dims=1))
end

@testset "mapslices" begin
    a = [10 20; 30 40]
    nda = NamedDimsArray(a, (:x, :y))

    @test mapslices(join, nda; dims=:x) == ["1030" "2040"] == mapslices(join, nda; dims=1)
    @test mapslices(join, nda; dims=:y) == reshape(["1020", "3040"], Val(2)) == mapslices(join, nda; dims=2)
    @test_throws UndefKeywordError mapslices(join, nda)
    @test_throws UndefKeywordError mapslices(join, a)

    @test names(mapslices(join, nda; dims=:y)) == (:x, :y) == names(mapslices(join, nda; dims=2))
end

@testset "mapreduce" begin
    a = [10 20; 31 40]
    nda = NamedDimsArray(a, (:x, :y))
    @test mapreduce(isodd, |, nda; dims=:x) == [true false] == mapreduce(isodd, |, nda; dims=1)
    @test mapreduce(isodd, |, nda; dims=:y) == [false true]' == mapreduce(isodd, |, nda; dims=2)
    @test mapreduce(isodd, |, nda) == true == mapreduce(isodd, |, a)

    @test names(mapreduce(isodd, |, nda; dims=:y)) == (:x, :y) == names(mapreduce(isodd, |, nda; dims=2))
end

@testset "zero" begin
    a = [10 20; 31 40]
    nda = NamedDimsArray(a, (:x, :y))
    @test zero(nda) == [0 0; 0 0] == zero(a)
    @test names(zero(nda)) == (:x, :y)
end
