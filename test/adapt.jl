@testset "Adapt" begin
    nda = NamedDimsArray([1.0 2.0; 3.0 4.0], (:x, :y))
    res = adapt(Array{Float32}, nda)
    @test res isa NamedDimsArray
    @test dimnames(res) == (:x, :y)
    @test parent(res) isa Matrix{Float32}
    @test res == nda

    ndv = NamedDimsArray([1, 2, 3], :z)
    @test dimnames(adapt(Array{Float32}, ndv)) == (:z,)
end
