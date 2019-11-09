# This is the core functionality for manipulating dimension names as express as a tuple
# It is central to the performance of this named dimensions as a zero cost abstraction
# For constant input all functions should constant propagate to constant output.
# Each function in this file is annotated with a comment and a benchmark
# running that benchmark should return 0 allocations,
# and you can swap its `@btime` for `@code_typed` to check if it is constant propergating
# This constant propagate lets us write faily naive code elsewhere in the package
# knowing the creation and destruction of NamedDimsArrays will be optimised away.

"""
    compile_time_return_hack(x)

This is a cunning hack to strongly encourage the compiler to use constant propagation,
when returning a `Tuple` of `Symbols`.
Under normal circumstances, constant propergation will not work to fully compute
returned tuples of non-bits types.
But because `Symbol`s are a allowed in type-parameters (unlike other nonbits types),
we can put them into a type-parameter, which triggers (??different??) constant propergation,
(for ??Reasons??)
and once something is a type-parameter, then extracting it and returning it is non-allocating.

See https://discourse.julialang.org/t/zero-allocation-tuple-subsetting/23122/8

The short version of this is:
if a function returns a tuple of symbols and is allocating when it looks like it shouldn't
then try wrapping its returned value in `compile_time_return_hack`.
and then look at the `@code_lowered` again.
"""
compile_time_return_hack(x::Tuple{Vararg{Symbol}}) = _compile_time_return_hack(Val{x}())
_compile_time_return_hack(::Val{X}) where {X} = X

"""
    dim(dimnames, name)

For `dimnames` being a tuple of names (symbols) for the dimensions.
and `name` being a one of those names
This returns the dimension corresponding to that `name`,
e.g. `dim((:a, :b), :b) == 2`
If that `name` is not one of the given `dimnames` then an error is thrown.
"""
function dim(dimnames::Tuple, name::Symbol)::Int
    # 0-Allocations see: `@btime  (()->dim((:a, :b), :b))()`
    dimnum = dim_noerror(dimnames, name)
    if dimnum === 0
        throw(ArgumentError("Specified name ($(repr(name))) does not match any dimension name ($dimnames)"))
    end
    return dimnum
end

function dim(dimnames::Tuple, names)
    # 0-Allocations see: `@btime (()->dim((:a,:b), (:a,:b)))()`
    return map(name -> dim(dimnames, name), names)
end

function dim(dimnames::Tuple, d::Union{Integer,Colon})
    # This is the fallback that allows `NamedDimsArray`'s to be have dimensions
    # referred to by number. This is required to allow functions on `AbstractArray`s
    # and that use function like `sum(xs; dims=2)` to continue to work without changes
    # `:` is the default for most methods that take `dims`
    return d
end

dim(dimnames::Tuple, ::Val{d}) where {d} = dim(dimnames, d)

Base.@pure function dim_noerror(dimnames::Tuple{Vararg{Symbol,N}}, name::Symbol) where {N}
    # 0-Allocations see: @btime  (()->dim_noerror((:a, :b, :c), :c))()
    for ii in 1:N
        getfield(dimnames, ii) === name && return ii
    end
    return 0
end

"""
    expand_dimnames(dimnames, name)

For `dimnames` being a tuple of names (symbols) for the dimensions.
and `name` being a name.
This expands the `dimnames` if `name` is not in `dimnames`.
e.g. `expand_dimnames((:a, :b), :c) == (:a, :b, :c)`
If `name` is already in `dimnames` then `dimnames` is returned.
"""
function expand_dimnames(dimnames::Tuple, name::Symbol)
    if dim_noerror(dimnames, name) > 0  # name in dimnames, but optimised
        return dimnames
    else
        return compile_time_return_hack((dimnames..., name))
    end
end

function expand_dimnames(dimnames::Tuple, name::Union{Colon,Tuple{}})
    return dimnames
end

function expand_dimnames(dimnames::Tuple, name::Integer)
    if name <= length(dimnames)
        return dimnames
    else
        extra_length = name - length(dimnames)
        new_dimnames = ntuple(i -> :_, extra_length)
        return compile_time_return_hack((dimnames..., new_dimnames...))
    end
end

function expand_dimnames(dimnames::Tuple, names)
    return expand_dimnames(expand_dimnames(dimnames, first(names)), Base.tail(names))
end

expand_dimnames(dimnames::Tuple, ::Val{d}) where {d} = expand_dimnames(dimnames, d)

"""
    permute_dimnames(dimnames, perm)

Reorder `dimnames` according `perm`.
`perm` should be ordered set of numerical indexs for the new position of the name.
Note: this does not throw errors if you give it a permutation that skips some positions
and duplicates others.
"""
function permute_dimnames(dimnames::NTuple{N,Symbol}, perm) where {N}
    # 0-Allocations, but does not seem to fully calculate at compile time
    # even with the `compile_time_return_hack`, though that is still required to
    # prevent allocations. `@code_typed permute_dimnames((:a,:b,:c), (1,3,2))`

    new_dimnames = ntuple(length(perm)) do ii
        ind = perm[ii]
        dimnames[ind]
    end
    return compile_time_return_hack(new_dimnames)
end

"""
    _rename(dimnames::Tuple, old_new::Vararg{Pair})
    _rename(dimname::Symbol, old_new::Vararg{Pair})

For each pair `old=>new`, replace all occurences of `old` in tuple `namea` with `new`.
If `dimname` is a Symbol, replace it with `new` of the pair with matching `old`, or
return `dimname` if no pairs match.
"""
function _rename(dimnames::Tuple, old_new::Vararg{Pair})
    return ntuple(i -> _rename(dimnames[i], old_new...), length(dimnames))
end

function _rename(dimname::Symbol, old_new::Vararg{Pair})
    # Avoid looping over pairs explicitly because that allocates.
    nt = ntuple(i -> dimname === first(old_new[i]) ? i : 0, length(old_new))
    which = sum(nt)
    which > length(old_new) &&
        throw(ArgumentError("Duplicate old names not permitted in `rename`"))
    return which == 0 ? dimname : last(old_new[which])
end

"""
    tuple_issubset
A version of `is_subset` sepecifically for `Tuple`s of `Symbol`s, that is `@pure`.
This helps it get optimised out of existance. It is less of an abuse of `@pure` than
most of the stuff for making `NamedTuples` work.
"""
Base.@pure function tuple_issubset(
    lhs::Tuple{Vararg{Symbol,N}}, rhs::Tuple{Vararg{Symbol,M}},
) where {N,M}
    N <= M || return false
    for a in lhs
        found = false
        for b in rhs
            found |= a === b
        end
        found || return false
    end
    return true
end

"""
    order_named_inds(Val(names); kw...)
    order_named_inds(Val(names), namedtuple)

Returns the tuple of index values for an array with `names`, when indexed by keywords.
Any dimensions not fixed are given as `:`, to make a slice.
An error is thrown if any keywords are used which do not occur in `nda`'s names.
"""
order_named_inds(val::Val{L}; kw...) where {L} = order_named_inds(val, kw.data)

@generated function order_named_inds(val::Val{L}, ni::NamedTuple{K}) where {L,K}
    tuple_issubset(K, L) || throw(DimensionMismatch("Expected subset of $L, got $K"))
    exs = map(L) do n
        if Base.sym_in(n, K)
            qn = QuoteNode(n)
            :(getfield(ni, $qn))
        else
            :(Colon())
        end
    end
    return Expr(:tuple, exs...)
end

"""
    incompatible_dimension_error(names_a, names_b)
Throws a `DimensionMismatch` explaining that these dimension names are not compatible.
"""
function incompatible_dimension_error(names_a, names_b)
    return throw(DimensionMismatch("Incompatible dimension names: $names_a ≠ $names_b"))
end

"""
    unify_names(a, b)
    unify_names(a, b, cs...)

Produces the merged set of names for tuples of names `a` and `b`,
or an error if it is not possibly to unify them.
Then continues with further names `cs`, if any.

Two tuples of names can be unified they are the same length
and if for each position the names are either the same, or one is a wildcard (`:_`).
When combining wildcard with non-wildcard the resulting name is the non-wildcard.
(This is somewhat like the very simplest case of unification in e.g prolog).

For example:
 - `(:a, :b)` and `(:a, :b)` can be unified to give `(:a, :b)`
 - similarly: `(:a, :_)` and `(:a, :b)` can also be unified to give `(:a, :b)`
 - `(:a, :b)` and `(:b, :c)` cannot be unified as the names at each position to not match.
 - `(:a, :b)` and `(:a, :b, :c)` cannot be unified as they have different lengths

This is the type of name combination used for binary array operations.
Where the dimensions of both arrays must be the same.

See also `names_are_unifiable(a, b)`, which returns `true` instead of the merged names,
and `false` instead of an error.
"""
function unify_names(names_a, names_b)
    # @btime (()->unify_names((:a, :b), (:a, :_)))()
    ret = try_unify_names(names_a, names_b)
    if ret === nothing
        incompatible_dimension_error(names_a, names_b)
    else
        return compile_time_return_hack(ret)
    end
end
unify_names(a) = a
unify_names(a, b, cs...) = unify_names(unify_names(a, b), cs...)
# @btime (()->unify_names((:a, :b), (:a, :_), (:_, :b)))()

names_are_unifiable(names_a, names_b) = try_unify_names(names_a, names_b) !== nothing

function try_unify_names(names_a, names_b)
    if names_a === names_b
        return names_a
    elseif length(names_a) !== length(names_b)
        return nothing
    end

    ret = ntuple(length(names_a)) do ii  # remove :_ wildcards
        a = getfield(names_a, ii)
        b = names_b[ii]
        a === :_ && return b
        b === :_ && return a
        a === b && return a
        return false  # mismatch occured, we mark this with a non-Symbol result
    end

    if ret isa Tuple{Vararg{Symbol}}
        return compile_time_return_hack(ret)
    else
        return nothing
    end
end

"""
    unify_names_longest(a, b)

This is the same as [`unify_names`](@ref), but with the equal length requirement removed.
It unifies the names up to the length of the shortest, and takes the named from the longest
for the remainder.
It can also be considered as padding the shorter of the two given tuples of names
with trailing wildcards (`:_`).
This is the type of name combination used for broadcating array operations.
Where the smaller (dimensionally) array is broadcast against the longer, repeating it for
all entries in the missing trailing dimensions.
"""
unify_names_longest(names, ::Tuple{}) = names
unify_names_longest(::Tuple{}, names) = names
unify_names_longest(::Tuple{}, ::Tuple{}) = tuple()
function unify_names_longest(names_a, names_b)
    # 0 Allocations: @btime (()-> unify_names_longest((:a,:b), (:a,)))()

    length(names_a) === length(names_b) && return unify_names(names_a, names_b)
    long, short =
        length(names_a) > length(names_b) ? (names_a, names_b) : (names_b, names_a)
    ret = ntuple(length(long)) do ii
        a = getfield(long, ii)
        b = ii <= length(short) ? short[ii] : :_
        a === :_ && return b
        b === :_ && return a
        a === b && return a
        return false  # mismatch occured, we mark this with a nonSymbol result
    end
    ret isa Tuple{Vararg{Symbol}} || incompatible_dimension_error(names_a, names_b)
    return compile_time_return_hack(ret)
end

function unify_names_longest(
    names_a, names_b, names_c::Vararg{<:NTuple{Nd,Symbol} where {Nd},N},
) where {N}
    return unify_names_longest(names_a, unify_names_longest(names_b, names_c...))
end

unify_names_shortest(names, ::Tuple{}) = ()
unify_names_shortest(::Tuple{}, names) = ()
unify_names_shortest(::Tuple{}, ::Tuple{}) = ()
function unify_names_shortest(names_a, names_b)
    # 0 Allocations: @btime (()-> unify_names_shortest((:a,:b), (:a,)))()

    length(names_a) === length(names_b) && return unify_names(names_a, names_b)
    long, short = if length(names_a) > length(names_b)
        (names_a, names_b)
    else
        (names_b, names_a)
    end
    ret = ntuple(length(short)) do ii
        a = getfield(long, ii)
        b = getfield(short, ii)
        a === :_ && return b
        b === :_ && return a
        a === b && return a
        return false  # mismatch occured, we mark this with a nonSymbol result
    end
    ret isa Tuple{Vararg{Symbol}} || incompatible_dimension_error(names_a, names_b)
    return compile_time_return_hack(ret)
end

function unify_names_shortest(
    names_a, names_b, names_c::Vararg{<:NTuple{Nd,Symbol} where {Nd},N},
) where {N}
    return unify_names_shortest(names_a, unify_names_shortest(names_b, names_c...))
end

# The following are helpers for remaining_dimnames_from_indexing
# as a generated function it can get unhappy if asked to use anon functions
# and it can only call function declared before it. So we declare them explictly here.
is_noninteger_type(::Type{<:Integer}) = false
is_noninteger_type(::Any) = true

"""
    remaining_dimnames_from_indexing(dimnames::Tuple, inds)

Given a tuple of dimension names,
and a tuple of indices e.g `(1, :, 1:3, [true, false])`,
this drops those indexed with scalars or `CartesianIndex`,
inserts `:_` for `newaxis = [CartesianIndex{0}()]`,
and returns another tuple of names.

It will return an empty tuple to indicate that all names should be dropped.
This happend for scalar indexing by integers, or one `CartesianIndex`.
It also happens e.g. when indexing a matrix by a `BitArray{2}` such as `mat[mat .> 0.5]`:
this returns a vector, the same as vec(mat)[vec(mat .> 0.5)], whose dimension isn't any
of the original dimensions, hence has no name.
"""
@generated function remaining_dimnames_from_indexing(dimnames::Tuple, inds::Tuple)
    # 0-Allocation see:
    # `@btime (()->remaining_dimnames_from_indexing((:a, :b, :c), (:,390,:)))()``
    keep_names = []
    dim_num = 0
    for type in inds.parameters
        if type <: Integer
            dim_num += 1
        elseif type <: CartesianIndex
            dim_num += type.parameters[1]
        elseif type <: AbstractVector{CartesianIndex{0}}
            push!(keep_names, QuoteNode(:_))
        elseif type <: AbstractArray{<:Integer} && ndims(type) > 1
            dim_num += 1
            for _ in 1:ndims(type)
                push!(keep_names, QuoteNode(:_))
            end
        else
            dim_num += 1
            push!(keep_names, :(getfield(dimnames, $dim_num)))
        end
    end
    return Expr(:call, :compile_time_return_hack, Expr(:tuple, keep_names...))
end

remaining_dimnames_from_indexing(dn::Tuple, inds::Tuple{Vararg{<:Integer}}) = ()
remaining_dimnames_from_indexing(dn::Tuple, ci::Tuple{CartesianIndex}) = ()

function remaining_dimnames_from_indexing(
    dimnames::Tuple{<:Any,<:Any,Vararg}, inds::Tuple{T},
) where {T<:Union{Base.LogicalIndex,AbstractVector{<:CartesianIndex}}}
    return ()
end

"""
    remaining_dimnames_after_dropping(dimnames::Tuple, dropped_dims)
Given a tuple of dimension names, and either a collection of dimensions,
or a single dimension, expressed as a number,
Returns the dimension names with those dimensions dropped.
"""
function remaining_dimnames_after_dropping(dimnames::Tuple, dropped_dim::Int)
    # 0 allocations. See `@btime remaining_dimnames_after_dropping((:a,:b,:c,:d,:e), 4)`
    return _remaining_dimnames_after_dropping(dimnames, Tuple{dropped_dim})
end

function remaining_dimnames_after_dropping(
    dimnames::NTuple{N,Symbol}, dropped_dims::Tuple{Vararg{Int}},
) where {N}
    # 0-Allocations see:
    # `@code_typed (()->remaining_dimnames_after_dropping((:a,:b,:c,:d,:e), (1,3)))()`
    return _remaining_dimnames_after_dropping(dimnames, Tuple{dropped_dims...})
end

# dropped_dims must be a Tuple type where the values are Int literal for the dimensions being dropped
@generated function _remaining_dimnames_after_dropping(
    dimnames::NTuple{N,Symbol}, dropped_dims::Type,
) where {N}
    dropped_dims_vals = dropped_dims.parameters[1].parameters
    keep_names = [:(getfield(dimnames, $ii)) for ii in 1:N if ii ∉ dropped_dims_vals]
    return Expr(:call, :compile_time_return_hack, Expr(:tuple, keep_names...))
end

"""
    tuple_cat(x, y, zs...)

This is like `vcat` for tuples, it splats everything into one long tuple.
"""
tuple_cat(x::Tuple, ys::Tuple...) = (x..., tuple_cat(ys...)...)
tuple_cat() = ()
# @btime tuple_cat((1, 2), (3, 4, 5), (6,)) # 0 allocations

"""
    replace_names(names, :a => :b)
    replace_names(names, :a => :b, :x => :y)

Replaces every `:a` in `names` with `:b`.
If given several rules, these are applied in sequence, left to right.
"""
function replace_names(dimnames::Tuple, pair::Pair)
    out = map(dimnames) do s
        s === first(pair) && return last(pair)
        s
    end
    return compile_time_return_hack(out)
end
# @btime NamedDims.replace_names((:a, :b), :b => :c) # 1.420 ns (0 allocations: 0 bytes)

function replace_names(dimnames::Tuple, pair::Pair, pairs::Pair...)
    step_one = replace_names(dimnames, pair)
    return replace_names(step_one, pairs...)
end
# @btime NamedDims.replace_names((:a, :b), :b => :c, :a => :z) # 1.420 ns (0 allocations: 0 bytes)
