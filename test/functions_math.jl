using NamedDims
using Test

@testset "+" begin
    nda = NamedDimsArray{(:a,)}(ones(3))

    @testset "correct" begin
        @test +(nda) == ones(3)
        @test names(+(nda)) == (:a,)

        @test +(nda, nda) == 2*ones(3)
        @test names(+(nda, nda)) == (:a,)

        @test +(nda, nda, nda) == 3*ones(3)
        @test names(+(nda, nda, nda)) == (:a,)
    end

    @testset "Dimension disagreement" begin
        @test_throws DimensionMismatch +(
            NamedDimsArray{(:a,:b,:c,:d)}(zeros(3,3,3,3)),
            NamedDimsArray{(:w,:x,:y,:z)}(ones(3,3,3,3))
        )

        @test_throws DimensionMismatch +(
            NamedDimsArray{(:time,)}(zeros(3,)),
            NamedDimsArray{(:time, :value)}(ones(3,3))
        )
    end

    @testset "Mixed array types" begin
        lhs_sum = +(
            NamedDimsArray{(:a,:b,:c,:d)}(zeros(3,3,3,3)),
            ones(3,3,3,3)
        )
        @test lhs_sum == ones(3,3,3,3)
        @test names(lhs_sum) == (:a,:b,:c,:d)


        rhs_sum = +(
            zeros(3,3,3,3),
            NamedDimsArray{(:w,:x,:y,:z)}(ones(3,3,3,3))
        )
        @test rhs_sum == ones(3,3,3,3)
        @test names(rhs_sum) == (:w,:x,:y,:z)



    end
end
