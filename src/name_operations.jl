# This file is for functions that are new functions sepecifically for workin with
# NamedDimsArrays, rather than overloads of existing functions

"""
    rename(nda::NamedDimsArray, names)

Returns a new `NameDimsArray` with the given dimension `names`.
`rename` outright replaces the names; while still wrapping the same backing array.
Unlike the constructor, it does not require that new names are compatible
with the old names (though you do still need to match the number of dimensions).
"""
rename(nda::NamedDimsArray, names) = NamedDimsArray(parent(nda), names)

"""
    refine_names(x, names)

Refine the names of the dimensions of `x` to match `names`.
This is like [`rename`](ref), but it only affects unnamed dimensions.
I.e. dimensions of a `NamedDimsArray` called `:_`, or any dimensions of an
`AbstractArray` in general.
"""
@inline function refine_names(orig::NamedDimsArray{L}, names::Tuple) where {L}
    new_names = unify_names(names, L)
    return NamedDimsArray{new_names}(parent(orig))
end
@inline refine_names(orig::AbstractArray, names::Tuple) = NamedDimsArray{names}(orig)

"""
    unname(A::NamedDimsArray) -> AbstractArray
    unname(A::AbstractArray) -> AbstractArray

Return the input array `A` without any dimension names.

For `NamedDimsArray`s this returns the parent array, equivalent to calling `parent`, but for
any other `AbstractArray` simply returns the input.
"""
unname(x::NamedDimsArray) = parent(x)
unname(x::AbstractArray) = x

"""
    dimnames(A) -> Tuple
    dimnames(A, d) -> Symbol

Return the names of all the dimensions of the array `A`, 
or just the one for the `d`-th dimension. 

Gives wildcards `:_` if this is not a `NamedDimsArray`.
And like `size(A, d)`, it allows `d > ndims(A)`.
"""
dimnames(::Type{<:NamedDimsArray{L}}) where {L} = L
dimnames(::Type{<:AbstractArray{T, N}}) where {T, N} = ntuple(_->:_, N)
dimnames(x::T) where T<:AbstractArray = dimnames(T)

function dimnames(AT::Type{<:AbstractArray{T,N}}, d::Integer) where {T,N}
    if d in 1:N
        return dimnames(AT)[d]
    elseif d>N
        return :_
    else
        error("dimnames: dimension out of range")
    end
end
dimnames(x::T, d::Integer) where T<:AbstractArray = dimnames(T, d)
