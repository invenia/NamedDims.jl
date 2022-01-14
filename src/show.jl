
# 2-arg show, mainly for repr():
function Base.show(io::IO, A::NamedDimsArray{L,T,N}) where {L,T,N}
    if get(io, :typeinfo, Any) <: NamedDimsArray
        show(io, parent(A))
    else
        print(io, "NamedDimsArray(")
        show(io, parent(A))
        print(io, ", ", N == 1 ? QuoteNode(L[1]) : L, ")")
    end
end

# This is called by summary(), for main REPL printing:
function Base.showarg(io::IO, A::NamedDimsArray{L,T,N}, outer) where {L,T,N}
    print(io, "NamedDimsArray(")
    Base.showarg(io, parent(A), false)
    print(io, ", ", ColourString(N == 1 ? QuoteNode(L[1]) : L), ")")
    return nothing
end

function Base.print_matrix(io::IO, A::NamedDimsArray)
    s1 = ColourString("↓ ", dimnames(A, 1), "  ")
    if ndims(A) == 2
        println(io, " "^Base.Unicode.textwidth(s1), ColourString("→ ", dimnames(A, 2)))
    end
    ioc = IOContext(io, :displaysize => displaysize(io) .- (1, 0))
    Base.print_matrix(ioc, parent(A), s1)
    return nothing
end

# This exists because including ascii colour codes in the string `s1` passed
# to `Base.print_matrix` will cause it to become confused about the spacing.
struct ColourString <: AbstractString
    string
    ColourString(xs...) = new(string(xs...))
end
Base.alignment(io::IO, x::ColourString) = alignment(io, x.string)
Base.length(x::ColourString) = length(x.string)
Base.ncodeunits(x::ColourString) = ncodeunits(x.string)
Base.textwidth(x::ColourString) = textwidth(x.string)
Base.print(io::IO, x::ColourString) = printstyled(io, x.string; color=:magenta)

if VERSION > v"1.6.0-DEV.1561" # 809f27c53df7a54388a687a847e9494e0d29bd4f
    function Base._show_nd_label(io::IO, A::NamedDimsArray, idxs)
        print(io, "[:, :, ")
        for i in 1:length(idxs)
            print(io, ColourString(dimnames(A, i + 2), "=", idxs[i]))
            i == length(idxs) ? println(io, "] =") : print(io, ", ")
        end
    end
end
