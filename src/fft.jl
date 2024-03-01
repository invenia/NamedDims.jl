wave_name(s::Symbol) = wave_name(Val(s))

@generated function wave_name(::Val{sym}) where {sym}
    str = string(sym)
    if sym == :_
        return QuoteNode(:_)
    elseif endswith(str, '∿')
        chars = collect(str)[1:end-1]
        return QuoteNode(Symbol(chars...))
    else
        return QuoteNode(Symbol(str, '∿'))
    end
end
# @btime wave_name(:k) # :k∿ , zero allocations

wave_name(tup::Tuple) = map(wave_name, tup) |> compile_time_return_hack
# @btime NamedDims.wave_name((:k1, :k2∿)) # zero

function wave_name(tup::Tuple, dims)
    out = ntuple(i -> i in dims ? wave_name(tup[i]) : tup[i], length(tup))
    return compile_time_return_hack(out)
end
# @btime NamedDims.wave_name((:k1, :k2, :k3), 2)

wave_name(tup::Tuple, dims::Tuple) = wave_name(wave_name(tup, first(dims)), Base.tail(dims))
wave_name(tup::Tuple, dims::Tuple{}) = tup
# @btime NamedDims.wave_name((:k1, :k2, :k3), (1,3))
