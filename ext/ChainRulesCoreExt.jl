module ChainRulesCoreExt

using ChainRulesCore
using NamedDims: NamedDimsArray, dimnames, unify_names


_NamedDimsArray_pullback(ȳ::AbstractArray) = (NoTangent(), ȳ, NoTangent())
_NamedDimsArray_pullback(ȳ::Tangent) = (NoTangent(), ȳ.data, NoTangent())
_NamedDimsArray_pullback(ȳ::AbstractThunk) = _NamedDimsArray_pullback(unthunk(ȳ))
function ChainRulesCore.rrule(::Type{<:NamedDimsArray}, values, names)
    return NamedDimsArray(values, names), _NamedDimsArray_pullback
end

function ChainRulesCore.rrule(T::Type{<:NamedDimsArray}, values)
    NamedDimsArray_values_pullback(ȳ) = _NamedDimsArray_pullback(ȳ)[1:2]
    return T(values), NamedDimsArray_values_pullback
end

function ChainRulesCore.ProjectTo(x::NamedDimsArray)
    return ProjectTo{NamedDimsArray{dimnames(x)}}(; data=ProjectTo(parent(x)))
end

(project::ProjectTo{NDA})(dx::AbstractZero) where {NDA<:NamedDimsArray} = dx
function (project::ProjectTo{NDA})(dx) where {NDA<:NamedDimsArray}
    names = unify_names(dimnames(NDA), dimnames(dx))
    return NamedDimsArray{names}(project.data(parent(dx)))
end

end
