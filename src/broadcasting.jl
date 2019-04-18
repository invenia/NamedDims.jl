# See: https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting-1

function Base.BroadcastStyle(::Type{<:NamedDimsArray})
    return Broadcast.ArrayStyle{NamedDimsArray}()
end

function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{NamedDimsArray}},
    ::Type{ElType}
) where ElType

    L = broadcasted_names(bc)
    data = similar(Array{ElType}, axes(bc))
    return NamedDimsArray{L}(data)
end


broadcasted_names(bc::Base.Broadcast.Broadcasted) = broadcasted_names(bc.args...)
function broadcasted_names(a, bs...)
    a_name = broadcasted_names(a)
    b_name = broadcasted_names(bs...)
    combine_names_longest(a_name, b_name)
end
broadcasted_names(a::AbstractArray) = names(a)
broadcasted_names(a) = tuple()
