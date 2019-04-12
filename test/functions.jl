using NamedDims
using Test
using Statistics

@testset "sum" begin
    nda = NamedDimsArray([10 20; 30 40], (:x, :y))

    @test sum(nda) == 100
    @test sum(nda; dims=:x) == [40 60]
    @test sum(nda; dims=1) == [40 60]
end

@testset "mean" begin
    nda = NamedDimsArray([10 20; 30 40], (:x, :y))

    @test mean(nda) == 25
    @test mean(nda; dims=:x) == [20 30]
    @test mean(nda; dims=1) == [20 30]
end

@testset "mapslices" begin
    nda = NamedDimsArray([10 20; 30 40], (:x, :y))

    @test mapslices(join, nda; dims=:x) == ["1030" "2040"] == mapslices(join, nda; dims=1)
    @test mapslices(join, nda; dims=:y) == reshape(["1020", "3040"], Val(2)) == mapslices(join, nda; dims=2)
end

@testset "mapreduce" begin
    nda = NamedDimsArray([10 20; 31 40], (:x, :y))
    @test mapreduce(isodd, |, nda; dims=:x) == [true false] == mapreduce(isodd, |, nda; dims=1)
    @test mapreduce(isodd, |, nda; dims=:y) == [false true]' == mapreduce(isodd, |, nda; dims=2)
end
