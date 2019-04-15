# This supports no broadcasting math on NamedDimArrays

function Base.:+(a::NamedDimsArray{A}, b::NamedDimsArray{B}) where {A,B}
    if A !== B
        throw(DimensionMismatch(
            "Attempted to add arrays with different dimension names. $A ≠ $B."
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


function Base.:*(a::NamedDimsArray{A,T,2}, b::NamedDimsArray{B,S,2}) where {A,B,T,S}
    if last(A) != :_ && first(B) != :_ && last(A) != first(B)
        throw(DimensionMismatch(
            "Attempted to take the matrix product of arrays with different inner dimension names. $A vs $B."
        ))
    end
    data = *(parent(a), parent(b))
    L = (first(A), last(B))
    return NamedDimsArray{L}(data)
end

function Base.:*(a::NamedDimsArray{L,T,2}, b::AbstractMatrix) where {L,T}
    return *(a, NamedDimsArray{names(b)}(b))
end
function Base.:*(a::AbstractMatrix, b::NamedDimsArray{L,T,2}) where {L,T}
    return *(NamedDimsArray{names(a)}(a), b)
end
