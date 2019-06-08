# This supports nonbroadcasting math on NamedDimArrays

# Up/downstairs: indicated by sentinel characters
export up, down
up(s) = Symbol('♂',s)   # these parse as letters, and can be typed as \male, \female
down(s) = Symbol('♀',s) # other options... ⬆️⬇️ can't be typed & overlap next char

up(a::NamedDimsArray{L}) where {L} = NamedDimsArray(parent(a), map(up, L))
down(a::NamedDimsArray{L}) where {L} = NamedDimsArray(parent(a), map(down, L))

function unpack_updown(a::Symbol)
    a1 = first(string(a))
    a1==='♂' && return 1, string(a)[4:end] # nextind(string(up(:a)),1) == 4
    a1==='♀' && return -1, string(a)[4:end]
    return 0, string(a)
end

function valid_updown(a::Symbol, b::Symbol)
    (a === :_ || b === :_) && return true

    a_up, a_str = unpack_updown(a)
    b_up, b_str = unpack_updown(b)

    a_str == b_str || return false    # the names must always match
    a_up==0 || b_up==0 && return true # if either is indifferent, then they match
    return a_up != b_up
end

# Matrix product
function valid_matmul_dims(a::Tuple{Vararg{Symbol}}, b::Tuple{Vararg{Symbol}})
    a_dim = a[end]
    b_dim = b[1]
    valid_updown(a_dim, b_dim)
end

matmul_names((a1, a2)::Tuple{Symbol,Symbol}, (b,)::Tuple{Symbol}) = (a1,)
matmul_names((a1, a2)::Tuple{Symbol,Symbol}, (b1, b2)::Tuple{Symbol,Symbol}) = (a1, b2)
matmul_names((a1,)::Tuple{Symbol,},(b1,b2)::Tuple{Symbol,Symbol}) = (a1, b2)

function matrix_prod_names(a, b)
    # 0 Allocations. See `@btime (()-> matrix_prod_names((:foo, :bar),(:bar,)))()`
    valid_matmul_dims(a, b) || throw(DimensionMismatch(
            "Cannot take matrix product of arrays with different inner dimension names. $a vs $b"
        ))
    res = matmul_names(a, b)
    compile_time_return_hack(res)
end


for (NA, NB) in ((1,2), (2,1), (2,2))  #Vector * Vector, is not allowed
    @eval function Base.:*(a::NamedDimsArray{A,T,$NA}, b::NamedDimsArray{B,S,$NB}) where {A,B,T,S}
        L = matrix_prod_names(A,B)
        data = *(parent(a), parent(b))
        return NamedDimsArray{L}(data)
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

function LinearAlgebra.dot(a::NamedDimsArray{La, Ta, 1}, b::NamedDimsArray{Lb, Tb, 1}) where {La,Lb,Ta,Tb}
    valid_matmul_dims(La, Lb) || throw(DimensionMismatch(
            "Cannot take dot of vectors with incompatible dimension names. $La vs $Lb"
        ))
    dot(parent(a), parent(b))
end
