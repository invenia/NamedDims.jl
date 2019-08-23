export @named, @unname

using MacroTools

"""
    @named begin
        A′ = A[i,j,k]
        B′ = B[k]
        C[i,_,k]
        f(A′, C) .+ B′
    end

Convenience macro for adding dimension names, or asserting that they agree.
The same as calling `A′ = NamedDimsArray(A, (:i, :j, :k))`. Here it wil be asserted
that `C` is a 3-tensor whose dimension names (if any) are compatible with `(:i, :_, :k)`.
Other expressions like `f` are run as usual.

    @named begin
        *′ = *(i)
        *ᵢⱼ = *(i,j)
        /ⱼ = /(j)
    end

This defines a function `*′` which multiplies two `NamedDimsArray`s always along index `i`.
(You will need the package `OMEinsum` for 3-index and higher arrays, or to sum over multiple indices.)
The function may have any name, but decorations of `*` such as `*′` or `*ᵢⱼ` give an
infix operator which may be used `A′ *′ D`.
Similarly `/ⱼ` is like `/` but transposes as needed to act on shared index `j`.
(Soon!)
"""
macro named(ex)
    named_macro(ex)
end

function named_macro(input_ex)
    outex = MacroTools.prewalk(input_ex) do ex

        if @capture(ex, A_[ijk__])
            return :( NamedDims.NamedDimsArray{($(QuoteNode.(ijk)...),)}($A) )

        elseif @capture(ex, s_ = *(i_) )
            return :( $s(xs...) = Base.:*($(QuoteNode(i)), xs...) )
        elseif @capture(ex, s_ = *(ijk__) )
            return :( $s(xs...) = Base.:*(($(QuoteNode.(ijk)...),), xs...) )

        elseif @capture(ex, s_ = /(i_) )
            return :( $s(x,y) = Base.:/($(QuoteNode(i)), x,y) )

        end
        return ex
    end
    esc(outex)
end

"""
    @unname A[i,j,k]

Macro for removing dimension names. If `A == @named A[i,j,k]` then this returns `parent(A)`,
but if `A` has these names in another order, then it returns `permutedims(parent(A), (:i,:j,:k))`.
@named
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
