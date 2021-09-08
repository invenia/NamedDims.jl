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
