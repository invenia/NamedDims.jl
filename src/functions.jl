
# 1 Arg
for (mod, funs) in (
    (:Base, (
        :sum, :prod, :count, :maximum, :minimum, :extrema, :cumsum, :cumprod,
        :sort, :sort!)
    ),
    (:Statistics, (:mean, :std, :var, :median, :cov, :cor)),
)
    for fun in funs
        @eval function $mod.$fun(a::NamedDimsArray; dims=:, kwargs...)
            new_dims = name2dim(a, dims)
            return $mod.$fun(parent(a); dims=new_dims, kwargs...)
        end
    end
end

# 1 arg before
for (mod, funs) in (
    (:Base, (:mapslices,)),
)
    for fun in funs
        @eval function $mod.$fun(f, a::NamedDimsArray; dims=:, kwargs...)
            new_dims = name2dim(a, dims)
            return $mod.$fun(f, parent(a); dims=new_dims, kwargs...)
        end
    end
end

# 2 arg before
for (mod, funs) in (
    (:Base, (:mapreduce,)),
)
    for fun in funs
        @eval function $mod.$fun(f1, f2, a::NamedDimsArray; dims=:, kwargs...)
            new_dims = name2dim(a, dims)
            return $mod.$fun(f1, f2, parent(a); dims=new_dims, kwargs...)
        end
    end
end
