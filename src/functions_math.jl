# This supports no broadcasting math on NamedDimArrays
function Base.:+(a::NamedDimsArray{A}, b::NamedDimsArray{B}) where {A,B}
    L = combine_names(A, B)
    data = +(parent(a), parent(b))
    return NamedDimsArray{L}(data)
end
function Base.:+(a::NamedDimsArray{L}, b::AbstractArray) where {L}
    data = +(parent(a), b)
    return NamedDimsArray{L}(data)
end
function Base.:+(a::AbstractArray, b::NamedDimsArray{L}) where {L}
    data = +(a, parent(b))
    return NamedDimsArray{L}(data)
end


matrix_prod_error(A, B) = throw(DimensionMismatch(
    "Cannot take matrix product of arrays with different inner dimension names. $A vs $B"
))
# Matrix*Vector => Vector
matrix_prod_names(::Type{Tuple{A1,A2}}, ::Type{Tuple{B}}) where {A1,A2,B} = throw(matrix_prod_error(A2, B))
matrix_prod_names(::Type{Tuple{A,B}}, ::Type{Tuple{B}}) where {A,B} = (A,)
matrix_prod_names(::Type{Tuple{A,B}}, ::Type{Tuple{:_}}) where {A,B} = (A,)
matrix_prod_names(::Type{Tuple{A,:_}}, ::Type{Tuple{B}}) where {A,B} = (A,)
matrix_prod_names(::Type{Tuple{A,:_}}, ::Type{Tuple{:_}}) where {A,} = (A,)

#Vector*Matrix => Matrix  (Note in this case the first dim of matrix must have size 1)
matrix_prod_names(::Type{Tuple{A}}, ::Type{Tuple{B1,B2}}) where {A,B1,B2} = (A,B2)

#Matrix*Matrix => Matrix
matrix_prod_names(::Type{Tuple{A1,A2}}, ::Type{Tuple{B1,B2}}) where {A1,A2,B1,B2} = throw(matrix_prod_error(A2, B1))
matrix_prod_names(::Type{Tuple{A1,A2}}, ::Type{Tuple{A2,B2}}) where {A1,A2,B2} = (A1,B2)
matrix_prod_names(::Type{Tuple{A1,A2}}, ::Type{Tuple{:_,B2}}) where {A1,A2,B2} = (A1,B2)
matrix_prod_names(::Type{Tuple{A1,:_}}, ::Type{Tuple{B1,B2}}) where {A1,B1,B2} = (A1,B2)
matrix_prod_names(::Type{Tuple{A1,:_}}, ::Type{Tuple{:_,B2}}) where {A1,B2} = (A1,B2)


matrix_prod_names(A::Tuple, B::Tuple) = matrix_prod_names(Tuple{A...}, Tuple{B...})

for (NA, NB) in ((1,2), (2,1), (2,2))  #Vector * Vector, is not allowed
    @eval function Base.:*(a::NamedDimsArray{A,T,$NA}, b::NamedDimsArray{B,S,$NB}) where {A,B,T,S}
        L = matrix_prod_names(A,B)
        data = *(parent(a), parent(b))
        return NamedDimsArray{L}(data)
    end

    @eval function Base.:*(a::NamedDimsArray{L,T,$NA}, b::AbstractArray{S,$NB}) where {L,T,S}
        return *(a, NamedDimsArray{names(b)}(b))
    end
    @eval function Base.:*(a::AbstractArray{T,$NA}, b::NamedDimsArray{L,S,$NB}) where {L,T,S}
        return *(NamedDimsArray{names(a)}(a), b)
    end
end
