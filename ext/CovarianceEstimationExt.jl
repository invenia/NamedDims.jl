module CovarianceEstimationExt

using NamedDims
using CovarianceEstimation


for E in (:LinearShrinkage, :SimpleCovariance, :AnalyticalNonlinearShrinkage)
    @eval function CovarianceEstimation.cov(
        estimator::$E, a::NamedDimsArray{L,T,2}; dims=1, kwargs...
    ) where {L,T}
        numerical_dims = dim(a, dims)
        # cov returns a Symmetric matrix which needs to be rewrapped in a NamedDimsArray
        data = cov(estimator, parent(a); dims=numerical_dims, kwargs...)
        names = NamedDims.symmetric_names(L, numerical_dims)
        return NamedDimsArray{names}(data)
    end
end

end
