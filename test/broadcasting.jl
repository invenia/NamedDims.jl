using NamedDims
using NamedDims: names
using Test

@testset "Binary broadcasting operations (.+)" begin
    nda = NamedDimsArray{(:a,)}(ones(3))

    @testset "standard case" begin
        @test nda .+ nda == 2ones(3)
        @test names(nda .+ nda) == (:a,)

        @test nda .+ nda .+ nda == 3ones(3)
        @test names(nda .+ nda .+ nda) == (:a,)
    end

    @testset "partially named dims" begin
        ndx = NamedDimsArray{(:x, :_)}(ones(3, 5))
        ndy = NamedDimsArray{(:_, :y)}(ones(3, 5))

        lhs = ndx .+ ndy
        rhs = ndy .+ ndx
        @test names(lhs) == (:x, :y) == names(rhs)
        @test lhs == 2ones(3, 5) == rhs
    end

    @testset "Dimension disagreement" begin
        @test_throws DimensionMismatch .+(
            NamedDimsArray{(:a, :b, :c, :d)}(zeros(3, 3, 3, 3)),
            NamedDimsArray{(:w, :x, :y, :z)}(ones(3, 3, 3, 3))
        )
    end

    @testset "named and unnamed" begin
        lhs_sum = .+(
            NamedDimsArray{(:a,:b,:c,:d)}(zeros(3, 3, 3, 3)),
            ones(3, 3, 3, 3)
        )
        @test lhs_sum == ones(3, 3, 3, 3)
        @test names(lhs_sum) == (:a, :b, :c, :d)


        rhs_sum = .+(
            zeros(3, 3, 3, 3),
            NamedDimsArray{(:w, :x, :y, :z)}(ones(3, 3, 3, 3))
        )
        @test rhs_sum == ones(3, 3, 3, 3)
        @test names(rhs_sum) == (:w, :x, :y, :z)
    end

    @testset "broadcasting" begin
        v = NamedDimsArray{(:time,)}(zeros(3,))
        m = NamedDimsArray{(:time, :value)}(ones(3, 3))
        s = 0

        @test v .+ m == ones(3, 3) == m .+ v
        @test s .+ m == ones(3, 3) == m .+ s
        @test s .+ v .+ m == ones(3, 3) == m .+ s .+ v

        @test names(v .+ m) == (:time, :value) == names(m .+ v)
        @test names(s .+ m) == (:time, :value) == names(m .+ s)
        @test names(s .+ v .+ m) == (:time, :value) == names(m .+ s .+ v)
    end

    @testset "Mixed array types" begin
        casts = (
            NamedDimsArray{(:foo, :bar)},  # Named Matrix
            x->NamedDimsArray{(:foo,)}(x[:, 1]),  # Named Vector
            x->NamedDimsArray{(:foo, :bar)}(x[:, 1:1]),  # Named Single Column Matrix
            identity, # Matrix
            x->x[:, 1], # Vector
            x->x[:, 1:1], # Single Column Matrix
            first, # Scalar
         )
        for (T1, T2, T3) in Iterators.product(casts, casts, casts)
            all(isequal(identity), (T1, T2, T3)) && continue
            !any(isequal(NamedDimsArray{(:foo, :bar)}), (T1, T2, T3)) && continue

            total = T1(ones(3, 6)) .+ T2(2ones(3, 6)) .+ T3(3ones(3, 6))
            @test total == 6ones(3, 6)
            @test names(total) == (:foo, :bar)
        end
    end
end
