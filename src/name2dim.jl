
function NamedNotFoundException(name, cands)
    return ArgumentError(
        "No dimensioned called $name exists. Dimension names: $(join(cands, ", "))"
    )
end

###################################

# We change the name into a `Val` because that will trigger constant propergation
# and allow this to resolve at compile time when the `name` is a constant
name2dim(namemap, name) = _name2dim(namemap, Val(name))

# `_name2dim` returns 0 for names not found, because throwing an error would make it inpure
# And then it couldn't run at compile time.

# Generic case
#TODO: for speed this could be made into a generated function, so it happens at compile time
function _name2dim(::Type{T}, ::Val{name}) where {T<:Tuple, name}
    return something(findfirst(isequal(name), T.parameters), 0)
end

## Hand roll out special cases to help this optimize to run at compile time
_name2dim(::Type{Tuple{A}}, ::Val{A}) where A = 1
_name2dim(::Type{Tuple{A}}, ::Val{name}) where {A,name} = 0

_name2dim(::Type{Tuple{A,B}}, ::Val{A}) where {A,B} = 1
_name2dim(::Type{Tuple{A,B}}, ::Val{B}) where {A,B} = 2
_name2dim(::Type{Tuple{A,B}}, ::Val{name}) where {A,B,name} = 0

_name2dim(::Type{Tuple{A,B,C}}, ::Val{A}) where {A,B,C} = 1
_name2dim(::Type{Tuple{A,B,C}}, ::Val{B}) where {A,B,C} = 2
_name2dim(::Type{Tuple{A,B,C}}, ::Val{C}) where {A,B,C} = 3
_name2dim(::Type{Tuple{A,B,C}}, ::Val{name}) where {A,B,C,name} = 0

_name2dim(::Type{Tuple{A,B,C,D}}, ::Val{A}) where {A,B,C,D} = 1
_name2dim(::Type{Tuple{A,B,C,D}}, ::Val{B}) where {A,B,C,D} = 2
_name2dim(::Type{Tuple{A,B,C,D}}, ::Val{C}) where {A,B,C,D} = 3
_name2dim(::Type{Tuple{A,B,C,D}}, ::Val{D}) where {A,B,C,D} = 4
_name2dim(::Type{Tuple{A,B,C,D}}, ::Val{name}) where {A,B,C,D, name} = 0

_name2dim(::Type{Tuple{A,B,C,D,E}}, ::Val{A}) where {A,B,C,D,E} = 1
_name2dim(::Type{Tuple{A,B,C,D,E}}, ::Val{B}) where {A,B,C,D,E} = 2
_name2dim(::Type{Tuple{A,B,C,D,E}}, ::Val{C}) where {A,B,C,D,E} = 3
_name2dim(::Type{Tuple{A,B,C,D,E}}, ::Val{D}) where {A,B,C,D,E} = 4
_name2dim(::Type{Tuple{A,B,C,D,E}}, ::Val{E}) where {A,B,C,D,E} = 5
_name2dim(::Type{Tuple{A,B,C,D,E}}, ::Val{name}) where {A,B,C,D,E, name} = 0
