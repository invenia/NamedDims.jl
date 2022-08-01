@testset "chainrules.jl" begin
    @testset "constructor" begin
        test_rrule(NamedDimsArray, rand(2, 3), (:a, :b); check_inferred=VERSION >= v"1.6")
        test_rrule(NamedDimsArray{(:a, :b)}, (rand(2, 3)); check_inferred=VERSION >= v"1.6")
    end

    @testset "ProjectTo" begin
        nda = NamedDimsArray{(:a, :b)}(rand(3, 3))
        proj = ProjectTo(nda)

        x = rand(3, 3)
        @inferred proj(x)
        projected = proj(x)
        @test projected isa NamedDimsArray{(:a, :b)}
        @test projected.data == x
    end
end
