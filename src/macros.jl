export @named, @unname

using MacroTools

"""
    A′ = @named A[i,j,k]

Macro for adding dimension names, or asserting that they agree.
The same as calling `NamedDimsArray(A, (:i, :j, :k))`.

    @named begin
        A′ = A[i,j,k]
        B′ = B[k]
        C[i,_,k]
        f(A′, C) .+ B′
    end

When applied to a block of code, `A′` and `B′` will be defined, and it wil be asserted
that `C` is a 3-tensor whose dimension names (if any) are compatible with `(:i, :_, :k)`.
Other expressions like `f` are run as usual.

    @named *ⱼ = *(j)

This defines a function `*ⱼ` which multiplies two `NamedDimsArray`s always along index `j`.
The function may have any name, but decorations of `*` such as `*′` or `*ⱼ` give an
infix operator which may be used `A′ *ⱼ D`.
Like `sum(A′, dims=:k)` this will error on ordinary arrays.
"""
macro named(ex)
    named_macro(ex)
end

function named_macro(input_ex)
    outex = MacroTools.prewalk(input_ex) do ex

        if @capture(ex, A_[ijk__])
            return :( NamedDims.NamedDimsArray{($(QuoteNode.(ijk)...),)}($A) )

        elseif @capture(ex, s_ = *(i_) )
            return :( $s(xs...) = *($(QuoteNode(i)), xs...) )

        end
        return ex
    end
    esc(outex)
end

"""
    @unname A[i,j,k]

Macro for removing dimension names. If `A == @named A[i,j,k]` then this returns `parent(A)`,
but if `A` has these names in another order, then it returns `permutedims(parent(A), (:i,:j,:k))`.
"""
macro unname(ex)
    unname_macro(ex)
end

function unname_macro(input_ex)
    outex = MacroTools.prewalk(input_ex) do ex
        if @capture(ex, A_[ijk__])
            stup = :( ($(QuoteNode.(ijk)...),) )
            return :( NamedDims.names($A) == $stup ? parent($A) : parent(permutedims($A, $stup)) )
        end

        return ex
    end
end
