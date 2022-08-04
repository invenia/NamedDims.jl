using Test: approx_full
using LinearAlgebra
using NamedDims
using NamedDims: dimnames
using Test

if !isdefined(@__MODULE__, :ColumnNorm)
    # Make work on old julia versions
    ColumnNorm() = Val(true)
    NoPivot() = Val(false)
end

_test_data(::Val{:rectangle}) = [1.0 2 3; 4 5 6];
function _test_data(::Val{:pdmat})
    return [8.0 7.0 6.0 5.0; 7.0 8.0 6.0 6.0; 6.0 6.0 6.0 5.0; 5.0 6.0 5.0 5.0]
end
_test_names(::Val{:rectangle}) = (:foo, :bar)
_test_names(::Val{:pdmat}) = (:foo, :foo)

function baseline_tests(fact, identity; test_data_type = :rectangle)
    # A set of generic tests to ensure that our components don't accidentally reverse the
    # `:foo` and `:bar` labels for any components
    @testset "Baseline" begin
        names = _test_names(Val{test_data_type}())
        data = _test_data(Val{test_data_type}())
        nda = NamedDimsArray{names}(data)

        base_fact = fact(data)
        named_fact = fact(nda)

        # Check that the size is the same
        @test size(named_fact) == size(base_fact)

        # Check that the returned destructuring via iteration is the same
        @test parent.([named_fact...]) == [base_fact...]

        @test propertynames(named_fact) == propertynames(base_fact)

        # Test that all properties work as expected
        @testset "Property $P" for P in propertynames(named_fact)
            _base = getproperty(base_fact, P)
            _named = getproperty(named_fact, P)

            @test size(_base) == size(_named)

            # If our property is a NamedDimsArray make sure that the names make sense
            _named isa NamedDimsArray && @testset "Test name for dim $d" for d in 1:ndims(_named)
                # Don't think it make sense for an factorization to produce properties with
                # dimension sizes outside 1, 2 or 3
                @test d in (1, 2, 3)

                if size(_named, d) == 1
                    # Neither name makes sense here
                    @test dimnames(_named, d) == :_
                elseif size(_named, d) == 2
                    # Name must either be :foo or :_
                    @test dimnames(_named, d) in (:foo, :_)
                elseif size(_named, d) == 3
                    # Name must either be :bar or :_
                    @test dimnames(_named, d) in (:bar, :_)
                elseif size(_named, d) == 4
                    # Name can only be foo, as this is the pdmat case
                    @test dimnames(_named, d) in (:foo,)
                end
            end
        end

        @testset "Reconstructions" begin
            @test identity(base_fact) ≈ data
            @test identity(named_fact) ≈ nda
        end
    end
end

# LinearAlgebra
@testset "lu" begin
    baseline_tests(lu, F -> getindex(F.L * F.U, F.p, :))

    # Explicit `dimnames` tests for readability
    nda = NamedDimsArray{(:foo, :bar)}([1.0 2 3; 4 5 6])
    x = lu(nda)
    @test dimnames(x.L) == (:foo, :_)
    @test dimnames(x.U) == (:_, :bar)
    @test dimnames(x.p) == (:foo,)
    @test dimnames(x.P) == (:foo, :foo)

    # Opperations that should give back original dimnames
    @test dimnames(x.P * nda) == (:foo, :bar)
    @test dimnames(x.L * x.U) == (:foo, :bar)
    @test dimnames(nda[x.p, :]) == (:foo, :bar)
end

@testset "lq" begin
    baseline_tests(lq, S -> S.L * S.Q)

    # Explicit `dimnames` tests for readability
    nda = NamedDimsArray{(:foo, :bar)}([1.0 2 3; 4 5 6])
    x = lq(nda)
    @test size(x) == size(parent(x))
    @test dimnames(x.L) == (:foo, :_)
    @test dimnames(x.Q) == (:_, :bar)

    # Idenity opperations should give back original dimnames
    @test dimnames(x.L * x.Q) == (:foo, :bar)
end

@testset "svd" begin
    baseline_tests(svd, F -> F.U * Diagonal(F.S) * F.Vt)

    # Explicit `dimnames` tests for readability
    nda = NamedDimsArray{(:foo, :bar)}([1.0 2 3; 4 5 6])
    x = svd(nda)
    @test size(x) == size(parent(x))
    # Test based on visualization on wikipedia
    # https://en.wikipedia.org/wiki/File:Singular_value_decomposition_visualisation.svg
    @test dimnames(x.U) == (:foo, :_)
    @test dimnames(x.S) == (:_,)
    @test dimnames(x.V) == (:bar, :_)
    @test dimnames(x.Vt) == (:_, :bar)

    # Identity operation should give back original names
    @test dimnames(x.U * Diagonal(x.S) * x.Vt) == (:foo, :bar)
end

@testset "qr" begin
    baseline_tests(qr, F -> F.Q * F.R)
    baseline_tests(A -> qr(A, ColumnNorm()), F -> F.Q * F.R * F.P')

    # Explicit `dimnames` tests for readability
    for pivot in (ColumnNorm(), NoPivot())
        for data in ([1.0 2 3; 4 5 6], [big"1.0" 2; 3 4])
            nda = NamedDimsArray{(:foo, :bar)}(data)
            x = qr(nda, pivot)
            @test size(x) == size(parent(x))
            @test dimnames(x.Q) == (:foo, :_)
            @test dimnames(x.R) == (:_, :bar)

            # Identity operation should give back original dimnames
            @test dimnames(x.Q * x.R) == (:foo, :bar)

            pivot === ColumnNorm() && @testset "pivoted" begin
                @test parent(x) isa QRPivoted
                @test dimnames(x.p) == (:bar,)
                @test dimnames(x.P) == (:bar, :bar)

                # Identity operation should give back original dimnames
                @test dimnames(nda * x.P') == (:foo, :bar)
                @test dimnames(nda[:, x.p]) == (:foo, :bar)
                @test dimnames(x.Q * x.R * x.P') == (:foo, :bar)
            end
        end
    end
end

@testset "cholesky" begin
    baseline_tests(cholesky, S -> S.L * S.L'; test_data_type=:pdmat)
    baseline_tests(cholesky, S -> S.U' * S.U; test_data_type=:pdmat)

    # Explicit `dimnames` tests for readability
    nda = NamedDimsArray{(:foo, :foo)}(_test_data(Val{:pdmat}()))
    nda_mismatch = NamedDimsArray{(:foo, :bar)}(_test_data(Val{:pdmat}()))
    x = cholesky(nda)
    @test size(x) == size(parent(x))
    @test dimnames(x.L) == (:foo, :foo)
    @test dimnames(x.U) == (:foo, :foo)

    @test_throws DimensionMismatch cholesky(nda_mismatch)
end

@testset "#164 factorization eltype not same as input eltype" begin
    # https://github.com/invenia/NamedDims.jl/issues/164
    nda = NamedDimsArray{(:foo, :bar)}([1 2 3; 4 5 6; 7 8 9])  # Int eltype
    @test qr(nda) isa NamedDims.NamedFactorization{(:foo, :bar),Float64}
end

@testset "LinearAlgebra.:ldiv " begin
    r1 = [2 3 5; 7 11 13; 17 19 23]
    r2 = r1[:, 1:2]
    b = [29, 31, 37]
    b_nda = NamedDimsArray{(:foo,)}(b)

    for A in (r1, r2)
        (m, n) = size(A)
        issquare = m == n
        fn = issquare ? (identity, triu, tril, Diagonal) : (identity,)
        for f in fn
            for B in (b_nda, b)
                nda = NamedDimsArray{(:foo, :bar)}(f(A))
                x = nda \ B
                @test parent(x) ≈ f(A) \ parent(B)
                # NOTE: Diagonal loses NamedDimness so specialcase
                f != Diagonal && @test dimnames(x) == (:bar,)
            end
        end
    end

    @test_throws DimensionMismatch (\)(
        NamedDimsArray{(:A, :B)}(r1), NamedDimsArray{(:NotA,)}(b)
    )
end
