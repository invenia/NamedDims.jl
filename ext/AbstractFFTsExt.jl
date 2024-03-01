module AbstractFFTsExt

using AbstractFFTs
using NamedDims
using NamedDims: wave_name, _rename


################################################
# FFT

for fun in (:fft, :ifft, :bfft, :rfft, :irfft, :brfft)
    plan_fun = Symbol(:plan_, fun)

    if fun in (:irfft, :brfft)  # These take one more argument, a size
        arg, str = (:(d::Integer),), "d, "
    else
        arg, str = (), ""
    end

    @eval begin

        """
            $($fun)(A, $($str):time => :freq, :x => :kx)

        Acting on a `NamedDimsArray`, this specifies to take the transform along the dimensions
        named `:time, :x`, and return an array with names `:freq` and `:kx` in their places.

            $($fun)(A, $($str):x) # => :x∿

        If new names are not given, then the default is `:x => :x∿` and `:x∿ => :x`,
        applied to all dimensions, or to those specified as usual, e.g. `$($fun)(A, $($str)(1,2))`
        or `$($fun)(A, $($str):time)`. The symbol "∿" can be typed by `\\sinewave<tab>`.
        """
        function AbstractFFTs.$fun(A::NamedDimsArray{L}, $(arg...)) where {L}
            data = AbstractFFTs.$fun(parent(A), $(arg...))
            return NamedDimsArray(data, wave_name(L))
        end

        function AbstractFFTs.$fun(A::NamedDimsArray{L,T,N}, $(arg...), dims) where {L,T,N}
            numerical_dims = dim(A, dims)
            data = AbstractFFTs.$fun(parent(A), $(arg...), numerical_dims)
            newL = wave_name(L, numerical_dims)
            return NamedDimsArray(data, newL)
        end

        function AbstractFFTs.$fun(A::NamedDimsArray{L,T,N}, $(arg...), p::Pair{Symbol,Symbol}, ps::Pair{Symbol,Symbol}...) where {L,T,N}
            numerical_dims = dim(A, (first(p), first.(ps)...))
            data = AbstractFFTs.$fun(parent(A), $(arg...), numerical_dims)
            newL = _rename(L, p, ps...)
            return NamedDimsArray(data, newL)
        end

        """
            F = $($plan_fun)(A, $($str):time)
            A∿ = F * A
            A ≈ F \\ A∿ ≈ inv(F) * A∿

        FFT plans for `NamedDimsArray`s, identical to `A∿ = $($fun)(A, $($str):time)`.
        Note you cannot specify the final name, it always transforms `:time => :time∿`.
        And that the plan `F` stores which dimension number to act on, not which name.
        """
        function AbstractFFTs.$plan_fun(A::NamedDimsArray, $(arg...), dims = ntuple(identity, ndims(A)); kw...)
            dims isa Pair && throw(ArgumentError("$($plan_fun) does not store final names, got Pair $dims"))
            numerical_dims = Tuple(dim(A, dims))
            AbstractFFTs.$plan_fun(parent(A), $(arg...), numerical_dims; kw...)
        end
    end

end

for shift in (:fftshift, :ifftshift)
    @eval begin

        function AbstractFFTs.$shift(A::NamedDimsArray)
            data = AbstractFFTs.$shift(parent(A))
            NamedDimsArray(data, dimnames(A))
        end

        function AbstractFFTs.$shift(A::NamedDimsArray, dims)
            numerical_dims = dim(A, dims)
            data = AbstractFFTs.$shift(parent(A), numerical_dims)
            NamedDimsArray(data, dimnames(A))
        end

    end
end

# The dimensions on which plans act are not part of the type, unfortunately
for plan_type in (:Plan, :ScaledPlan)
    @eval function Base.:*(plan::AbstractFFTs.$plan_type, A::NamedDimsArray{L,T,N}) where {L,T,N}
        data = plan * parent(A)
        if Base.sym_in(:region, propertynames(plan)) # true for plan_fft from FFTW
            dims = plan.region  # dims can be 1, (1,3) or 1:3
        elseif Base.sym_in(:p, propertynames(plan))
            dims = plan.p.region
        else
            return data
        end
        newL = ntuple(d -> d in dims ? wave_name(L[d]) : L[d], N)::NTuple{N,Symbol}
        # newL = wave_name(L, Tuple(dims)) # this, using compile_time_return_hack, is much slower
        return NamedDimsArray(data, newL)
    end
end

end
