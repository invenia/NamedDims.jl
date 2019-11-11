using NamedDims
using NamedDims: names
using Test
using Tracker

@testset "Binary broadcasting operations (.+)" begin
    nda = NamedDimsArray{(:a,)}(ones(3))

    @testset "standard case" begin
        @test nda .+ nda == 2ones(3)
        @test names(nda .+ nda) == (:a,)

        @test nda .+ nda .+ nda == 3ones(3)
        @test names(nda .+ nda .+ nda) == (:a,)

        # in-place
        @test names(nda .= 0 .* nda .+ 7) == (:a,)
        @test unname(nda .= 0 .* nda .+ 7) == 7*ones(3)
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

    @testset "Regression test again #8b" begin
        # https://github.com/invenia/NamedDims.jl/issues/8#issuecomment-490124369
        nda = NamedDimsArray{(:x,:y,:z)}(ones(10,20,30))
        @test nda .+ ones(1,20) == 2ones(10,20,30)
        @test names(nda .+ ones(1,20)) == (:x, :y, :z)
    end

    @testset "in-place assignment .=" begin
        ab = NamedDimsArray(rand(2,2), (:a, :b))
        a_ = NamedDimsArray(rand(2,2), (:a, :_))
        ba = NamedDimsArray(rand(2,2), (:b, :a))
        ac = NamedDimsArray(rand(2,2), (:a, :c))
        z = zeros(2,2)

        # https://github.com/invenia/NamedDims.jl/issues/71
        @test_throws DimensionMismatch z .= ab .+ ba
        @test_throws DimensionMismatch z .= ab .+ ac
        @test_throws DimensionMismatch a_ .= ab .+ ac
        @test_throws DimensionMismatch ab .= a_ .+ ac
        @test_throws DimensionMismatch ac .= ab .+ ba

        # check that dest is written into:
        @test names(z .= ab .+ ba') == (:a, :b)
        @test z == (ab.data .+ ba.data')
        @test z isa Array  # has not itself magically gained names

        @test names(z .= ab .+ a_) == (:a, :b)
        @test names(a_ .= ba' .+ ab) == (:a, :b)
    end

end

@testset "Competing Wrappers" begin
    nda = NamedDimsArray(ones(4), :foo)
    ta = TrackedArray(5*ones(4))
    ndt = NamedDimsArray(TrackedArray(5*ones(4)), :foo)

    arrays = (nda, ta, ndt)
    @testset "$a .- $b" for (a, b) in Iterators.product(arrays, arrays)
        a === b && continue
        @test typeof(nda .- ta) <: NamedDimsArray
        @test typeof(parent(nda .- ta)) <: TrackedArray
    end
end
