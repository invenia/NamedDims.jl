using LinearAlgebra
using FFTW
using NamedDims
using NamedDims: matrix_prod_names, names, symmetric_names
using Test
using Statistics

@testset "+" begin
    nda = NamedDimsArray{(:a,)}(ones(3))

    @testset "standard case" begin
        @test +(nda) == ones(3)
        @test dimnames(+(nda)) == (:a,)

        @test +(nda, nda) == 2ones(3)
        @test dimnames(+(nda, nda)) == (:a,)

        @test +(nda, nda, nda) == 3ones(3)
        @test dimnames(+(nda, nda, nda)) == (:a,)
    end

    @testset "partially named dims" begin
        ndx = NamedDimsArray{(:x, :_)}(ones(3, 5))
        ndy = NamedDimsArray{(:_, :y)}(ones(3, 5))

        lhs = ndx + ndy
        rhs = ndy + ndx
        @test dimnames(lhs) == (:x, :y) == dimnames(rhs)
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
        @test dimnames(lhs_sum) == (:a, :b, :c, :d)


        rhs_sum = +(
            zeros(3, 3, 3, 3),
            NamedDimsArray{(:w, :x, :y, :z)}(ones(3, 3, 3, 3))
        )
        @test rhs_sum == ones(3, 3, 3, 3)
        @test dimnames(rhs_sum) == (:w, :x, :y, :z)


        casts = (NamedDimsArray{(:foo, :bar)}, identity)
        for (T1, T2, T3, T4) in Iterators.product(casts, casts, casts, casts)
            all(isequal(identity), (T1, T2, T3, T4)) && continue
            total = T1(ones(3, 6)) + T2(2ones(3, 6)) + T3(3ones(3, 6)) + T4(4ones(3, 6))
            @test total == 10ones(3, 6)
            @test dimnames(total) == (:foo, :bar)
        end
    end
end


@testset "-" begin
    # This is actually covered by the tests for + above, since that uses the same code
    # just one extra as a sensability check
    nda = NamedDimsArray{(:a, :b)}(ones(3, 100))
    @test nda - nda == zeros(3, 100)
    @test dimnames(nda - nda) == (:a, :b)
end


@testset "scalar product" begin
    nda = NamedDimsArray{(:a, :b, :c, :d, :e)}(ones(10, 20, 30, 40, 50))
    @test 10nda == 10ones(10, 20, 30, 40, 50)
    @test dimnames(10nda) == (:a, :b, :c, :d, :e)
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
            @test dimnames(nda * ndb) == (:a, :c)

            @test ones(4, 3) * ndb == 3ones(4, 2)
            @test dimnames(ones(4, 3) * ndb) == (:_, :c)

            @test nda * ones(3, 7) == 3ones(2, 7)
            @test dimnames(nda * ones(3,7)) == (:a, :_)
        end

        @testset "Dimension disagreement" begin
            @test_throws DimensionMismatch ndb * nda
        end
    end

    @testset "Matrix-Vector" begin
        ndm = NamedDimsArray{(:a, :b)}(ones(1, 1))
        ndv = NamedDimsArray{(:b, )}(ones(1))

        @test ndm * ndv == ones(1)
        @test dimnames(ndm * ndv) == (:a,)
    end

    @testset "Vector-Matrix" begin
        ndm = NamedDimsArray{(:a, :b)}(ones(1, 1))
        ndv = NamedDimsArray{(:a, )}(ones(1))

        @test ndv * ndm == ones(1, 1)
        @test dimnames(ndv * ndm) == (:a, :b)
    end

    @testset "Vector-Vector" begin
        v = [1, 2, 3]
        ndv = NamedDimsArray{(:vec,)}(v)
        @test_throws MethodError ndv * ndv
        @test ndv' * ndv == 14
        @test ndv' * ndv == adjoint(ndv) * v == transpose(ndv) * v
        @test ndv' * ndv == adjoint(v) * ndv == transpose(v) * ndv
        @test ndv * ndv' == [1 2 3; 2 4 6; 3 6 9]

        ndv2 = NamedDimsArray{(:b,)}([3, 2, 1])
        @test_throws DimensionMismatch ndv' * ndv2
    end

    @testset "NDAs from CoVectors" begin
        v = [1, 2, 3]
        ndv = NamedDimsArray{(:vec,)}(v)
        @test NamedDimsArray{dimnames(v')}(v') * ndv == 14
        @test NamedDimsArray{dimnames(v')}(transpose(v)) * ndv == 14
    end
end
@testset "allocations: matmul names" begin
    @test 0 == @ballocated (() -> matrix_prod_names((:foo, :bar), (:bar,)))()
    @test 0 == @ballocated (() -> symmetric_names((:foo, :bar), 1))()
end


@testset "Mutmul with special types" begin

    special_types = (Adjoint, Diagonal, Symmetric, Tridiagonal, SymTridiagonal, BitArray,)

    @testset "matrix" begin
        ndm = NamedDimsArray{(:a, :b)}(ones(5,5))
        @testset "$T" for T in special_types
            x = T(ones(5,5))
            @test dimnames(x * ndm) == (:_, :b)
            @test dimnames(ndm * x) == (:a, :_)

            @test typeof(x * ndm) <: NamedDimsArray
            @test typeof(ndm * x) <: NamedDimsArray
        end
    end

    @testset "vector" begin
        ndv = NamedDimsArray{(:vec,)}(ones(5))
        @testset "$T" for T in special_types
            x = T(ones(5,5))
            @test dimnames(x * ndv) == (:_,)
            @test dimnames(ndv' * x) == (:_, :_)

            @test typeof(x * ndv) <: NamedDimsArray
            @test typeof(ndv' * x) <: NamedDimsArray
        end
    end

end


@testset "inv" begin
    nda = NamedDimsArray{(:a, :b)}([1.0 2; 3 4])
    @test dimnames(inv(nda)) == (:b, :a)
    @test nda * inv(nda) ≈ NamedDimsArray{(:a, :a)}([1.0 0; 0 1])
    @test inv(nda) * nda ≈ NamedDimsArray{(:b, :b)}([1.0 0; 0 1])
end


@testset "cov/cor" begin
    @testset "symmetric_names" begin
        @test symmetric_names((:a, :b), 1) == (:b, :b)
        @test symmetric_names((:a, :b), 2) == (:a, :a)
        @test symmetric_names((:a, :b), 5) == (:_, :_)

        @test_throws MethodError symmetric_names((:a, :b, :c), 2)
    end
    @testset "$f" for f in (cov, cor)
        @testset "matrix input, matrix result" begin
            A = rand(3, 5)
            nda = NamedDimsArray{(:a, :b)}(A)
            @test f(nda; dims=:a) == f(A, dims=1)
            @test dimnames(f(nda; dims=:a)) == (:b, :b)
            @test dimnames(f(nda, dims=:b)) == (:a, :a)
            # `Statistic.cov/cor(A, dims=p)` for `p > 2` is allowed but returns NaNs.
            @test dimnames(f(nda, dims=3)) == (:_, :_)
        end
        @testset "vector input, scalar result" begin
            a = rand(4)
            nda = NamedDimsArray{(:a,)}(a)
            @test f(nda) isa Number
            @test f(nda) == f(a)
        end
        @testset "high dimensional input" begin
            @test_throws MethodError f(NamedDimsArray(rand(3, 4, 5), (:a, :b, :c)))
        end
    end
    @testset "cov corrected=$bool" for bool in (true, false)
        # test that kwargs get passed on correctly
        A = rand(2, 4)
        nda = NamedDimsArray{(:a, :b)}(A)
        @test cov(nda; corrected=bool) == cov(A; corrected=bool)
        @test cov(nda; corrected=bool, dims=:b)  == cov(A; corrected=bool, dims=2)
    end
end


@testset "FFT" begin
    ndv = NamedDimsArray(zeros(ComplexF64, 16), :μ)
    ndv[μ=2] = 1

    @testset "vector" begin
        @test names(fft(ndv)) == (:μ∿,)
        @test names(ifft(ndv)) == (:μ∿,)
        @test names(bfft(ndv)) == (:μ∿,)
        @test names(ifft(fft(ndv))) == (:μ,)
    end

    @testset "vector plan" begin
        pv1 = plan_fft(ndv)
        pv2 = plan_fft(randn(ComplexF64, 16))
        pv3 = plan_ifft(ndv)
        pv4 = plan_bfft(ndv)

        @test names(pv1 * ndv) == (:μ∿,)
        @test names(pv2 * ndv) == (:μ∿,)
        @test names(pv3 * ndv) == (:μ∿,)
        @test names(pv4 * ndv) == (:μ∿,)

        @test names(pv3 * (pv2 * ndv)) == (:μ,)
        @test names(pv1 * (pv4 * ndv)) == (:μ,)
    end

    nda = NamedDimsArray(zeros(Float32, 4,4,4), (:a, :b′, :c))
    nda[1,2,3] = nda[2,3,4] = 1

    @testset "three dims" begin
        @test names(fft(nda)) == (:a∿, :b′∿, :c∿)
        @test names(ifft(nda, 1)) == (:a∿, :b′, :c)
        @test names(bfft(nda, :b′)) == (:a, :b′∿, :c)
        @test_broken names(fft(nda, 1:2)) == (:a∿, :b′∿, :c)
        @test names(fft(nda, (:a, :c))) == (:a∿, :b′, :c∿)
    end

    @testset "three plan" begin
        p1 = plan_fft(nda, :a)
        p2 = plan_ifft(zeros(Float32, 4,4,4), 2:2)
        p3 = plan_bfft(nda, :c)
        p4 = plan_fft(nda, (:a, :c))

        names(p1 * nda) == (:a∿, :b′, :c)
        names(p2 * nda) == (:a, :b′∿, :c)
        names(p3 * nda) == (:a, :b′, :c∿)
        names(p4 * nda) == (:a∿, :b′, :c∿)
        @test names(p2 * (p1 * nda)) == (:a∿, :b′∿, :c)
        @test names(p3 * (p4 * nda)) == (:a∿, :b′, :c)
    end
end
