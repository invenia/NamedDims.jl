"""
    dim(dimnames, [name])

For `dimnames` being a tuple of names (symbols) for the dimensions.
If called with just the tuple,
returns a named tuple, with each name mapped to a dimension.
e.g `dim((:a, :b)) == (a=1, b=2)`.

If the second `name` argument is given, them the dimension corresponding to that `name`,
is returned.
e.g. `dim((:a, :b), :b) == 2`
If that `name` is not found then `0` is returned.
"""
function dim(dimnames::Tuple)
    # Note: This code is runnable at compile time if input is a constant
    # If modified, make sure to recheck that it still can run at compile time
    # e.g. via `@code_llvm (()->dim((:a, :b)))()` which should be very short
    ndims = length(dimnames)
    return NamedTuple{dimnames, NTuple{ndims, Int}}(1:ndims)
end

function dim(dimnames::Tuple, name::Symbol)
    # Note: This code is runnable at compile time if inputs are constants
    # If modified, make sure to recheck that it still can run at compile time
    # e.g. via `@code_llvm (()->dim((:a, :b), :a))()` which should just say `return 1`
    this_namemap = NamedTuple{(name,), Tuple{Int}}((0,))  # 0 is default we will overwrite
    full_namemap = dim(dimnames)
    return first(merge(this_namemap, full_namemap))
end

function dim(dimnames::Tuple, names)
    # This handles things like `(:x, :y)` or `[:x, :y]`
    # or via the fallbacks `(1,2)`, or `1:5`
    return map(name->dim(dimnames, name), names)
end

function dim(dimnames::Tuple, d::Union{Integer, Colon})
    # This is the fallback that allows `NamedDimsArray`'s to be have dimensions
    # referred to by number. This is required to allow functions on `AbstractArray`s
    # and that use function like `sum(xs; dims=2)` to continue to work without changes
    # `:` is the default for most methods that take `dims`
    return d
end


"""
    default_inds(dimnames::Tuple)
This is the default value for all indexing expressions using the given dimnames.
Which is to say: take a full slice on everything
"""
function default_inds(dimnames::NTuple{N}) where N
    # Note: This code is runnable at compile time if input is a constant
    # If modified, make sure to recheck that it still can run at compile time
    values = ntuple(_->Colon(), N)
    return NamedTuple{dimnames, NTuple{N, Colon}}(values)
end


"""
    order_named_inds(dimnames::Tuple; named_inds...)

Returns the values of the `named_inds`, sorted as per the order they appear in `dimnames`,
with any missing dimnames, having there value set to `:`.
An error is thrown if any dimnames are given in `named_inds` that do not occur in `dimnames`.
"""
function order_named_inds(dimnames::Tuple; named_inds...)
    # Note: This code is runnable at compile time if input is a constant
    # If modified, make sure to recheck that it still can run at compile time

    slice_everything = default_inds(dimnames)
    full_named_inds = merge(slice_everything, named_inds)
    if length(full_named_inds) != length(dimnames)
        throw(DimensionMismatch("Expected $(dimnames), got $(keys(named_inds))"))
    end
    inds = Tuple(full_named_inds)
    return inds
end

"""
    determine_remaining_dim(dimnames::Tuple, inds...)
Given a tuple of dimension names, e.g.
and a set of index expressesion e.g `1, :, 1:3, [true, false]`,
determine which are not dropped.
Dimensions indexed with scalars are dropped
"""
@generated function determine_remaining_dim(dimnames::Tuple, inds)
    # TODO: This still allocates once, and it shouldn't have to
    # See: #@btime (()->determine_remaining_dim((:a, :b, :c), (:,390,:)))()
    ind_types = inds.parameters
    kept_dims = findall(keep_dim_ind_type, ind_types)
    keep_names = [:(getfield(dimnames, $ii)) for ii in kept_dims]
    return Expr(:tuple, keep_names...)
end
keep_dim_ind_type(::Type{<:Integer}) = false
keep_dim_ind_type(::Any) = true
