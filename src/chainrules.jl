_NamedDimsArray_pullback(ȳ::AbstractArray) = (NoTangent(), ȳ, NoTangent())
_NamedDimsArray_pullback(ȳ::Tangent) = (NoTangent(), ȳ.data, NoTangent())
_NamedDimsArray_pullback(ȳ::AbstractThunk) = _NamedDimsArray_pullback(unthunk(ȳ))

function ChainRulesCore.rrule(::Type{NamedDimsArray}, values, names)
    return NamedDimsArray(values, names), _NamedDimsArray_pullback
end
