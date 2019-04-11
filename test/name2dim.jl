using NamedDims
using Test


@testset "name2dim" begin
    @testset "small case, that hits unrolled method" begin
        @test name2dim(Tuple{:x, :y}, :x)==1
        @test name2dim(Tuple{:x, :y}, :y)==2
        @test name2dim(Tuple{:x, :y}, :z)==0  # not found
    end
    @testset "large case that hits generic fallback" begin
        @test name2dim(Tuple{:x, :y, :a, :b, :c, :d}, :x)==1
        @test name2dim(Tuple{:x, :y, :a, :b, :c, :d}, :a)==3
        @test name2dim(Tuple{:x, :y, :a, :b, :c, :d}, :d)==6
        @test name2dim(Tuple{:x, :y, :a, :b, :c, :d}, :z)==0 # not found
    end
end
