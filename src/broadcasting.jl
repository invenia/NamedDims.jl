# See: https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting-1

Base.BroadcastStyle(::Type{<:NamedDimsArray}) = Broadcast.ArrayStyle{NamedDimsArray}()

function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{NamedDimsArray}},
    ::Type{T}
) where T

    L = broadcasted_names(bc)
    data = similar(Array{T}, axes(bc))
    return NamedDimsArray{L}(data)
end


broadcasted_names(bc::Base.Broadcast.Broadcasted) = broadcasted_names(bc.args...)
function broadcasted_names(a, bs...)
    a_name = broadcasted_names(a)
    b_name = broadcasted_names(bs...)
    unify_names_longest(a_name, b_name)
end
broadcasted_names(a::AbstractArray) = names(a)
broadcasted_names(a) = tuple()
