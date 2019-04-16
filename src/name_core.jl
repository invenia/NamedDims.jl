# This is the core functionality for manipulating dimension names as express as a tuple
# It is central to the performance of this named dimensions as a zero cost abstraction
# For constant input all functions should either be nonallocting,
# or single allocation, if the return a tuple of symbols.
# Each function in this file is annotated with a comment and a benchmark as to which it is.
# See https://discourse.julialang.org/t/zero-allocation-tuple-subsetting/23122/8
# By ensuring this, we ensure that constant propagation should remove
# the computation entirely from most uses.
# Which lets us write faily naive code elsewhere in the package
# knowing the creation and destruction of NamedDimsArrays will be optimised away.


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
    # 0-Allocations see: `@btime (()->dim((:a, :b)))()`
    ndims = length(dimnames)
    return NamedTuple{dimnames, NTuple{ndims, Int}}(1:ndims)
end

function dim(dimnames::Tuple, name::Symbol)
    # 0-Allocations see: `@btime  (()->dim((:a, :b), :a))()`
    this_namemap = NamedTuple{(name,), Tuple{Int}}((0,))  # 0 is default we will overwrite
    full_namemap = dim(dimnames)
    return first(merge(this_namemap, full_namemap))
end

function dim(dimnames::Tuple, names)
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
    # 0-Allocations
    values = ntuple(_->Colon(), N)
    return NamedTuple{dimnames, NTuple{N, Colon}}(values)
end


function identity_namedtuple(tup::NTuple{N, Symbol}) where N
    # 0-Allocations
    return NamedTuple{tup, typeof(tup)}(tup)
end



"""
    order_named_inds(dimnames::Tuple; named_inds...)

Returns the values of the `named_inds`, sorted as per the order they appear in `dimnames`,
with any missing dimnames, having there value set to `:`.
An error is thrown if any dimnames are given in `named_inds` that do not occur in `dimnames`.
"""
function order_named_inds(dimnames::Tuple; named_inds...)
    # 0-Allocations

    slice_everything = default_inds(dimnames)
    full_named_inds = merge(slice_everything, named_inds)
    if length(full_named_inds) != length(dimnames)
        throw(DimensionMismatch("Expected $(dimnames), got $(keys(named_inds))"))
    end
    inds = Tuple(full_named_inds)
    return inds
end

function combine_names(names_a, names_b)
    # 0-Allocations if inputs are the same
    # 1-Allocation, if has a `:)_` see  `@btime (()->combine_names((:a, :b), (:a, :_)))()``

    names_a == names_b && return names_a

    # Error message should not include names until it is thrown, as othrwise
    # the interpolation allocates and slows everything down a lot.
    err_msg = "Attempted to combine arrays with incompatible dimension names."
    length(names_a) != length(names_b) && throw(DimensionMismatch(err_msg * "$names_a ≠ $names_b."))

    return ntuple(length(names_a)) do ii  # remove :_ wildcards
        a = getfield(names_a, ii)
        b = names_b[ii]
        a === :_ && return b
        b === :_ && return a
        a === b && return a
        throw(DimensionMismatch(err_msg * "$names_a ≠ $names_b."))
    end
end

"""
    remaining_dimnames_from_indexing(dimnames::Tuple, inds...)
Given a tuple of dimension names
and a set of index expressesion e.g `1, :, 1:3, [true, false]`,
determine which are not dropped.
Dimensions indexed with scalars are dropped
"""
@generated function remaining_dimnames_from_indexing(dimnames::Tuple, inds)
    # 1-Allocation see: @btime (()->determine_remaining_dim((:a, :b, :c), (:,390,:)))()
    ind_types = inds.parameters
    kept_dims = findall(keep_dim_ind_type, ind_types)
    keep_names = [:(getfield(dimnames, $ii)) for ii in kept_dims]
    return Expr(:tuple, keep_names...)
end
keep_dim_ind_type(::Type{<:Integer}) = false
keep_dim_ind_type(::Any) = true


"""
    remaining_dimnames_after_dropping(dimnames::Tuple, dropped_dims)
Given a tuple of dimension names, and either a collection of dimensions,
or a single dimension, expressed as a number,
Returns the dimension names with those dimensions dropped.
"""
function remaining_dimnames_after_dropping(dimnames::Tuple, dropped_dim::Integer)
    return remaining_dimnames_after_dropping(dimnames, (dropped_dim,))
end

function remaining_dimnames_after_dropping(dimnames::Tuple, dropped_dims)
    # 1-Allcoation see: `@btime remaining_dimnames_after_dropping((:a,:b,:c,:d,:e), (1,2,))


    anti_names = identity_namedtuple(map(x->dimnames[x], dropped_dims))
    full_names = identity_namedtuple(dimnames)

    # Now we construct a new named tuple, with all the names we want to remove at the start
    combine_names = merge(anti_names, full_names)
    n_skip = length(anti_names)
    ntuple(length(full_names) - n_skip) do ii
        combine_names[ii + n_skip]  # Skip over the ones we left as the start
    end
end
