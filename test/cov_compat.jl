using NamedDims
using CovarianceEstimation
using Test

@testset "CovarianceEstimation" begin
    estimators = (
        LinearShrinkage(DiagonalCommonVariance()),
        SimpleCovariance(),
        AnalyticalNonlinearShrinkage(),
    )

    A = rand(20, 20)  # AnalyticalNonlinearShrinkage requires at least 12 samples
    nda = NamedDimsArray{(:a, :b)}(A)

    @testset "$(typeof(e))" for e in estimators
        @test cov(e, nda) == cov(e, A)
    end
end
