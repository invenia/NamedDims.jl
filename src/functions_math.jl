# This supports nonbroadcasting math on NamedDimArrays

# Matrix product
valid_matmul_dims(a::Tuple{Symbol}, b::Tuple{Vararg{Symbol}}) = true
function valid_matmul_dims(a::Tuple{Symbol, Symbol}, b::Tuple{Vararg{Symbol}})
    a_dim = a[end]
    b_dim = b[1]

    return a_dim === b_dim || a_dim === :_ || b_dim === :_
end

matmul_names((a1, a2)::Tuple{Symbol,Symbol}, (b,)::Tuple{Symbol}) = (a1,)
matmul_names((a1, a2)::Tuple{Symbol,Symbol}, (b1, b2)::Tuple{Symbol,Symbol}) = (a1, b2)
matmul_names((a1,)::Tuple{Symbol,},(b1,b2)::Tuple{Symbol,Symbol}) = (a1, b2)

function throw_matrix_dim_error(a, b)
    msg = "Cannot take matrix product of arrays with different inner dimension names. $a vs $b"
    return throw(DimensionMismatch(msg))
end

function matrix_prod_names(a, b)
    # 0 Allocations. See `@btime (()-> matrix_prod_names((:foo, :bar),(:bar,)))()`
    valid_matmul_dims(a, b) || throw_matrix_dim_error(a, b)
    res = matmul_names(a, b)
    return compile_time_return_hack(res)
end

for (NA, NB) in ((1,2), (2,1), (2,2))  #Vector * Vector, is not allowed
    @eval function Base.:*(a::NamedDimsArray{A,T,$NA}, b::NamedDimsArray{B,S,$NB}) where {A,B,T,S}
        L = matrix_prod_names(A,B)
        data = *(parent(a), parent(b))
        return NamedDimsArray{L}(data)
    end
end

# vector^T * vector
function Base.:*(a::NamedDimsArray{A,T,2,<:CoVector}, b::NamedDimsArray{B,S,1}) where {A,B,T,S}
    last(A) === first(B) || throw_matrix_dim_error(last(A), first(B))
    return *(parent(a), parent(b))
end

function Base.:*(a::NamedDimsArray{L,T,2,<:CoVector}, b::AbstractVector) where {L,T}
    return *(parent(a), b)
end

# Using `CovVector` results in Method ambiguities; have to define more specific methods.
for A in (Adjoint{<:Any, <:AbstractVector}, Transpose{<:Real, <:AbstractVector{<:Real}})
    @eval function Base.:*(a::$A, b::NamedDimsArray{L,T,1,<:AbstractVector{T}}) where {L,T}
        return *(a, parent(b))
    end
end

"""
    @declare_matmul(MatrixT, VectorT=nothing)

This macro helps define matrix multiplication for the types
with 2D type parameterization `MatrixT` and 1D `VectorT`.
It defines the various overloads for `Base.:*` that are required.
It should be used at the top level of a module.
"""
macro declare_matmul(MatrixT, VectorT=nothing)
    dim_combos = VectorT === nothing ? ((2,2),) : ((1,2), (2,1), (2,2))
    codes = map(dim_combos) do (NA, NB)
        TA_named = :(NamedDims.NamedDimsArray{<:Any, <:Any, $NA})
        TB_named = :(NamedDims.NamedDimsArray{<:Any, <:Any, $NB})
        TA_other = (VectorT, MatrixT)[NA]
        TB_other = (VectorT, MatrixT)[NB]

        quote
            function Base.:*(a::$TA_named, b::$TB_other)
                return *(a, NamedDims.NamedDimsArray{NamedDims.names(b)}(b))
            end
            function Base.:*(a::$TA_other, b::$TB_named)
                return *(NamedDims.NamedDimsArray{NamedDims.names(a)}(a), b)
            end
        end
    end
    return esc(Expr(:block, codes...))
end

@declare_matmul(AbstractMatrix, AbstractVector)
@declare_matmul(Diagonal,)

function Base.inv(nda::NamedDimsArray{L, T, 2}) where {L,T}
    data = inv(parent(nda))
    names = reverse(L)
    return NamedDimsArray{names}(data)
end

# Statistics
for fun in (:cor, :cov)
    @eval function Statistics.$fun(a::NamedDimsArray{L, T, 2}; dims=1, kwargs...) where {L, T}
        numerical_dims = dim(a, dims)
        data = Statistics.$fun(parent(a); dims=numerical_dims, kwargs...)
        names = symmetric_names(L, numerical_dims)
        return NamedDimsArray{names}(data)
    end
end

function symmetric_names(L::Tuple{Symbol,Symbol}, dims::Integer)
    # 0 Allocations. See `@btime (()-> symmetric_names((:foo, :bar), 1))()`
    names = if dims == 1
        (L[2], L[2])
    elseif dims == 2
        (L[1], L[1])
    else
        (:_, :_)
    end
    return compile_time_return_hack(names)
end

"""
    *(s::Symbol, A::NamedDimsArray, B::NamedDimsArray)

Matrix multiplication by summing over the specified index `s`,
by evaluating something like `transpose(A) * B` if required.
Works only on matrices & vectors, not higher-dimensional arrays.
For two vectors it gives a scalar result.

Allows matrices with the same name twice.
When the elements are not just numbers (such as with arrays of matrices),
it uses `permutedims` to keep recursive `transpose` from acting on elements.
And it never changes the order of multiplication,
elements of `A` always left-multiply elements of `B`.

```
julia> A = NamedDimsArray(rand(2,3), (:i, :j));

julia> B = NamedDimsArray(rand(4,3), (:k, :j));

julia> *(:j, A, B) |> summary
"2×4 NamedDimsArray{(:i, :k),Float64,2,Array{Float64,2}}"

julia> *ⱼ(x,y) = *(:j,x,y)   # define infix function
*ⱼ (generic function with 1 method)

julia> B *ⱼ A |> summary
"4×2 NamedDimsArray{(:k, :i),Float64,2,Array{Float64,2}}"
```
"""
function Base.:*(s::Symbol, x::NamedDimsArray{Lx,Tx,1}, y::NamedDimsArray{Ly,Ty,1}) where {Lx,Tx,Ly,Ty}
    s == Lx[1] == Ly[1] || throw_contract_dim_error(s, x, y)
    if Tx <: Number
        return transpose(x) * y
    else
        return first(permutedims(x) * y)
    end
end

function Base.:*(s::Symbol, x::NamedDimsArray{Lx,Tx,2}, y::NamedDimsArray{Ly,Ty,1}) where {Lx,Tx,Ly,Ty}
    if s == Lx[2] == Ly[1]
        return x * y
    elseif s == Lx[1] == Ly[1]
        return shallow_transpose(x) * y
    else
        throw_contract_dim_error(s, x, y)
    end
end

function Base.:*(s::Symbol, x::NamedDimsArray{Lx,Tx,1}, y::NamedDimsArray{Ly,Ty,2}) where {Lx,Tx,Ly,Ty}
    dropdims(*(s, shallow_transpose(x), y), dims=1)
end

function Base.:*(s::Symbol, x::NamedDimsArray{Lx,Tx,2}, y::NamedDimsArray{Ly,Ty,2}) where {Lx,Tx,Ly,Ty}
    if s == Lx[2] == Ly[1]
        return x * y
    elseif s == Lx[1] == Ly[1]
        return shallow_transpose(x) * y
    elseif s == Lx[2] == Ly[2]
        return x * shallow_transpose(y)
    elseif s == Lx[1] == Ly[2]
        return shallow_transpose(x) * shallow_transpose(y)
    else
        throw_contract_dim_error(s, x, y)
    end
end

function throw_contract_dim_error(s::Symbol, x, y)
    msg = "Cannot contract index :$s between arrays with indices $(names(x)) and $(names(x))"
    return throw(DimensionMismatch(msg))
end

shallow_transpose(x::AbstractArray{<:Number}) = transpose(x)
shallow_transpose(x::AbstractArray) = permutedims(x)
