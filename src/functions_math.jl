# This supports no broadcasting math on NamedDimArrays

function Base.:+(a::NamedDimsArray{A}, b::NamedDimsArray{B}) where {A,B}
    if A !== B
        throw(DimensionMismatch(
            "Attempted to add arrays with diffent dimension names. $A ≠ $B."
        ))
    end

    data = +(parent(a), parent(b))
    return NamedDimsArray{A}(data)
end

function Base.:+(a::NamedDimsArray{L}, b::AbstractArray) where {L}
    data = +(parent(a), b)
    return NamedDimsArray{L}(data)
end
function Base.:+(a::AbstractArray, b::NamedDimsArray{L}) where {L}
    data = +(a, parent(b))
    return NamedDimsArray{L}(data)
end
