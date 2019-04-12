
"""
    name2dim(names, [name])

For `names` being a tuple of names (symbols) for dimenensions.
If called with just the tuple,
returns a named tuple, with each name maps to a dimension.
e.g `name2dim((:a, :b)) == (a=1, b=2)`.

If the second `name` argument is given, them the dimension corresponding to that `name`,
is returned.
e.g. `name2dim((:a, :b), :b) == 2`
If that `name` is not found then `0` is returned.
"""
function name2dim(names::Tuple)
    # Note: This code is runnable at compile time if input is a constant
    # If modified, make sure to recheck that it still can run at compile time
    # e.g. via `@code_llvm (()->name2dim((:a, :b)))()` which should be very short
    ndims = length(names)
    return NamedTuple{names, NTuple{ndims, Int}}(1:ndims)
end

function name2dim(names::Tuple, name)
    # Note: This code is runnable at compile time if inputs are constants
    # If modified, make sure to recheck that it still can run at compile time
    # e.g. via `@code_llvm (()->name2dim((:a, :b), :a))()` which should just say `return 1`
    this_namemap = NamedTuple{(name,), Tuple{Int}}((0,))  # 0 is default we will overwrite
    full_namemap = name2dim(names)
    dim = first(merge(this_namemap, full_namemap))
    return dim
end


"""
    default_inds(names::Tuple)
This is the defult value for all indexing expressions using the given names.
Which is to say: take a full slice on everything
"""
function default_inds(names::Tuple)
    # Note: This code is runnable at compile time if input is a constant
    # If modified, make sure to recheck that it still can run at compile time
    ndims = length(names)
    values = ntuple(_->Colon(), ndims)
    return NamedTuple{names, NTuple{ndims, Colon}}(values)
end


"""
    order_named_inds(names::Tuple; named_inds...)

Returns the values of the `named_inds`, sorted as per the order they appear in `names`,
with any missing names, having there value set to `:`.
An error is thrown if any names are given in `named_inds` that do not occur in `names`.
"""
function order_named_inds(names::Tuple; named_inds...)
    # Note: This code is runnable at compile time if input is a constant
    # If modified, make sure to recheck that it still can run at compile time
    keys(named_inds) ⊆ names || throw(
        DimensionMismatch("Expected $(names), got $(keys(named_inds))")
    )

    slice_everything = default_inds(names)
    full_named_inds = merge(slice_everything, named_inds)
    inds = Tuple(full_named_inds)
end
