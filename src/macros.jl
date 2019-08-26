export @namedef

using MacroTools

"""
    @namedef begin
        A => A′{i,j}
        B′{i,j} => B
        C′{i,j} => C′′{x,y}
        contract => *′{i}
    end
    A′ = @namedef A => {i,j}

Macro for adding and removing dimension names.
`A′ = NamedDimsArray{(:i, :j)}(A)` has the same data as `A` but its type contains `i,j`.
It is asserted that `B′` has names `i,j`, and this is unwrapped to array `B` in this order.
`C′′` is a re-named version of `C′`.
And `*′` is an infix contraction function along index `j`.
"""
macro namedef(ex)
    names_macro(ex)
end

function names_macro(input_ex)
    outex = quote end
    if input_ex.head == :block
        for ex in input_ex.args
            ex isa LineNumberNode && continue

            # Special words contract etc must come first
            if @capture(ex, contract => g_{i_})
                push!(outex.args, :( $g(xs...) = Base.:*($(QuoteNode(i)), xs...) ))

            # Then conversion of arrays etc
            elseif @capture(ex, A_ => B_{ijk__})
                stup = :( ($(QuoteNode.(ijk)...),) )
                push!(outex.args, :( $B = NamedDims.NamedDimsArray($A, $stup) ))

            elseif @capture(ex, C_{ijk__} => D_)
                stup = :( ($(QuoteNode.(ijk)...),) )
                # push!(outex.args, :( $D = NamedDims.unname($C, $stup) ))
                push!(outex.args, :( $D = NamedDims.unname(Base.permutedims($C, $stup)) ))

            elseif @capture(ex, E_{ijk__} => F_{xyz__})
                stup = :( ($(QuoteNode.(ijk)...),) )
                stup2 = :( ($(QuoteNode.(xyz)...),) )
                push!(outex.args, :( $F = NamedDims.NamedDimsArray(NamedDims.unname(Base.permutedims($C, $stup)), $stup2) ))

            else
                error("@namedef doesn't know what to do with $ex")
            end
        end
    else
        if @capture(input_ex, A_ => {ijk__})
            @gensym B
            return names_macro(quote
                $A => $B{$(ijk...)}
            end)
        else
            return names_macro(quote
                $input_ex
            end)
        end
    end
    esc(outex)
end
