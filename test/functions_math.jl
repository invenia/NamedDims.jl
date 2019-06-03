using LinearAlgebra
using NamedDims
using NamedDims: matrix_prod_names, names
using Test

@testset "+" begin
    nda = NamedDimsArray{(:a,)}(ones(3))

    @testset "standard case" begin
        @test +(nda) == ones(3)
        @test names(+(nda)) == (:a,)

        @test +(nda, nda) == 2ones(3)
        @test names(+(nda, nda)) == (:a,)

        @test +(nda, nda, nda) == 3ones(3)
        @test names(+(nda, nda, nda)) == (:a,)
    end

    @testset "partially named dims" begin
        ndx = NamedDimsArray{(:x, :_)}(ones(3, 5))
        ndy = NamedDimsArray{(:_, :y)}(ones(3, 5))

        lhs = ndx + ndy
        rhs = ndy + ndx
        @test names(lhs) == (:x, :y) == names(rhs)
        @test lhs == 2ones(3, 5) == rhs
    end

    @testset "Dimension disagreement" begin
        @test_throws DimensionMismatch +(
            NamedDimsArray{(:a, :b, :c, :d)}(zeros(3, 3, 3, 3)),
            NamedDimsArray{(:w, :x, :y, :z)}(ones(3, 3, 3, 3))
        )

        @test_throws DimensionMismatch +(
            NamedDimsArray{(:time,)}(zeros(3,)),
            NamedDimsArray{(:time, :value)}(ones(3, 3))
        )
    end

    @testset "Mixed array types" begin
        lhs_sum = +(
            NamedDimsArray{(:a, :b, :c, :d)}(zeros(3, 3, 3, 3)),
            ones(3, 3, 3, 3)
        )
        @test lhs_sum == ones(3, 3, 3, 3)
        @test names(lhs_sum) == (:a, :b, :c, :d)


        rhs_sum = +(
            zeros(3, 3, 3, 3),
            NamedDimsArray{(:w, :x, :y, :z)}(ones(3, 3, 3, 3))
        )
        @test rhs_sum == ones(3, 3, 3, 3)
        @test names(rhs_sum) == (:w, :x, :y, :z)


        casts = (NamedDimsArray{(:foo, :bar)}, identity)
        for (T1, T2, T3, T4) in Iterators.product(casts, casts, casts, casts)
            all(isequal(identity), (T1, T2, T3, T4)) && continue
            total = T1(ones(3, 6)) + T2(2ones(3, 6)) + T3(3ones(3, 6)) + T4(4ones(3, 6))
            @test total == 10ones(3, 6)
            @test names(total) == (:foo, :bar)
        end
    end
end


@testset "-" begin
    # This is actually covered by the tests for + above, since that uses the same code
    # just one extra as a sensability check
    nda = NamedDimsArray{(:a, :b)}(ones(3, 100))
    @test nda - nda == zeros(3, 100)
    @test names(nda - nda) == (:a, :b)
end


@testset "scalar product" begin
    nda = NamedDimsArray{(:a, :b, :c, :d, :e)}(ones(10, 20, 30, 40, 50))
    @test 10nda == 10ones(10, 20, 30, 40, 50)
    @test names(10nda) == (:a, :b, :c, :d, :e)
end


@testset "matmul" begin
    @testset "matrix_prod_names" begin
        @test matrix_prod_names((:foo, :bar), (:bar, :buzz)) == (:foo, :buzz)
        @test matrix_prod_names((:foo, :bar), (:_, :buzz)) == (:foo, :buzz)
        @test matrix_prod_names((:foo, :_), (:bar, :buzz)) == (:foo, :buzz)
        @test matrix_prod_names((:foo, :_), (:_, :buzz)) == (:foo, :buzz)
        @test_throws DimensionMismatch matrix_prod_names((:foo, :bar), (:nope, :buzz))

        @test matrix_prod_names((:foo,), (:bar, :buzz)) == (:foo, :buzz)
        @test matrix_prod_names((:foo,), (:_, :buzz)) == (:foo, :buzz)
        # No error case with name mismatch here, as a Vector has "virtual" wildcard second dimension

        @test matrix_prod_names((:foo, :bar), (:bar,)) == (:foo,)
        @test matrix_prod_names((:foo, :bar), (:_, )) == (:foo,)
        @test matrix_prod_names((:foo, :_), (:bar,)) == (:foo,)
        @test matrix_prod_names((:foo, :_), (:_,)) == (:foo,)
        @test_throws DimensionMismatch matrix_prod_names((:foo, :bar), (:nope,))
    end

    @testset "Matrix-Matrix" begin
        nda = NamedDimsArray{(:a, :b)}(ones(2, 3))
        ndb = NamedDimsArray{(:b, :c)}(ones(3, 2))

        @testset "standard case" begin
            @test nda * ndb == 3ones(2, 2)
            @test names(nda * ndb) == (:a, :c)

            @test ones(4, 3) * ndb == 3ones(4, 2)
            @test names(ones(4, 3) * ndb) == (:_, :c)

            @test nda * ones(3, 7) == 3ones(2, 7)
            @test names(nda * ones(3,7)) == (:a, :_)
        end

        @testset "Dimension disagreement" begin
            @test_throws DimensionMismatch ndb * nda
        end
    end

    @testset "Matrix-Vector" begin
        ndm = NamedDimsArray{(:a, :b)}(ones(1, 1))
        ndv = NamedDimsArray{(:b, )}(ones(1))

        @test ndm * ndv == ones(1)
        @test names(ndm * ndv) == (:a,)
    end

    @testset "Vector-Matrix" begin
        ndm = NamedDimsArray{(:a, :b)}(ones(1, 1))
        ndv = NamedDimsArray{(:a, )}(ones(1))

        @test ndv * ndm == ones(1, 1)
        @test names(ndv * ndm) == (:a, :b)
    end
end

@testset "Mutmul with special types" begin
    nda = NamedDimsArray{(:a, :b)}(ones(5,5))
    @testset "$T" for T in (Diagonal, Symmetric, Tridiagonal, SymTridiagonal, BitArray,)
        x = T(ones(5,5))
        @test names(x * nda) == (:_, :b)
        @test names(nda * x) == (:a, :_)
    end
end


@testset "mldivide" begin
    outputs = NamedDimsArray{(:variates, :observations)}(randn(3, 10))
    inputs = NamedDimsArray{(:features, :observations)}(rand(4, 10))

    # outputs = mapping * inputs
    mapping = outputs / inputs

    @test names(mapping) == (:variates, :features)
end
