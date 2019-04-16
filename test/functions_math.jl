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

    @testset "partially named dims" begin
        ndx = NamedDimsArray{(:x,:_)}(ones(3,5))
        ndy = NamedDimsArray{(:_,:y)}(ones(3,5))

        lhs = ndx + ndy
        rhs = ndy + ndx
        @test names(lhs) == (:x, :y) == names(rhs)
        @test lhs == 2*ones(3,5) == rhs 
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


        casts = (NamedDimsArray{(:foo, :bar)}, identity)
        for (T1,T2,T3,T4) in Iterators.product(casts, casts, casts, casts)
            all(isequal(identity), (T1,T2,T3,T4)) && continue
            total = T1(ones(3,6)) + T2(2*ones(3,6)) + T3(3*ones(3,6)) + T4(4*ones(3,6))
            @test total == 10*ones(3,6)
            @test names(total) == (:foo, :bar)
        end
    end
end



@testset "*" begin
    nda = NamedDimsArray{(:a,:b)}(ones(2,3))
    ndb = NamedDimsArray{(:b,:c)}(ones(3,2))

    @testset "correct" begin
        @test nda * ndb == 3*ones(2,2)
        @test names(nda * ndb) == (:a,:c)

        @test ones(4,3) * ndb == 3*ones(4,2)
        @test names(ones(4,3) * ndb) == (:_,:c)

        @test nda * ones(3,7) == 3*ones(2,7)
        @test names(nda * ones(3,7)) == (:a,:_)
    end

    @testset "Dimension disagreement" begin
        @test_throws DimensionMismatch ndb * nda
    end
end
