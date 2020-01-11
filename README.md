# NamedDims

<!--
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/NamedDims.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://invenia.github.io/NamedDims.jl/dev)
-->
[![Build Status](https://travis-ci.com/invenia/NamedDims.jl.svg?branch=master)](https://travis-ci.com/invenia/NamedDims.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/invenia/NamedDims.jl?svg=true)](https://ci.appveyor.com/project/invenia/NamedDims-jl)
[![Codecov](https://codecov.io/gh/invenia/NamedDims.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/invenia/NamedDims.jl)
[![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/N/NamedDims.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html)
[![code style blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

`NamedDimsArray` is a zero-cost abstraction to add names to the dimensions of an array.

## Core functionality:

For `nda = NamedDimsArray{(:x, :y, :z)}(rand(10, 20, 30))`.

 - Indexing: `nda[y=2]` is the same as `nda[x=:, y=2, z=:]` which is the same as `nda[:, 2, :]`.
 - Functions taking a `dims` keyword: `sum(nda; dims=:y)` is the same as `sum(nda; dims=2)`.
 - Renaming: `rename(nda, new_names)` returns a new `NamedDimsArray` with the `new_names` but still wrapping the same data.
 - Unwrapping: `parent(nda)` returns the underlying `AbstractArray` that is wrapped by the `NamedDimsArray`.
 - Unnaming: `unname(a)` ensures an `AbstractArray` is _not_ a `NamedDimsArray`;
    if passed a `NamedDimsArray` it unwraps it, otherwise just returns the given `AbstractArray`.

### Dimensionally Safe Operations

Any operation of multiple `NamedDimArray`s must have compatible dimension names.
For example trying `NamedDimsArray{(:time,)}(ones(5)) + NamedDimsArray{(:place,)}(ones(5))`
will throw an error.
If you perform an operation between another `AbstractArray` and a `NamedDimsArray`, then
the result will take its names from the `NamedDimsArray`.
You can use this to bypass the protection,
 e.g. `NamedDimsArray{(:time,)}(ones(5)) + parent(NamedDimsArray{(:place,)}(ones(5)))`
 is allowed.

### Partially Named Dimensions (`:_`)

To allow for arrays where only some dimensions have names,
the name `:_` is treated as a wildcard.
Dimensions named with `:_` will not be protected against operating between dimensions of different names; in these cases the result will take the name from the non-wildcard name, if any of the operands had such a concrete name.
For example:
`NamedDimsArray{(:time,:_)}(ones(5,2)) + NamedDimsArray{(:_, :place,)}(ones(5,2))`
is allowed. and would have a result of:
`NamedDimsArray{(:time,:place)}(2*ones(5,2))`
As such, unless you want this wildcard behaviour, you should *not* use `:_` as a dimension name.
(Also that is a terrible dimension name, and goes against the whole point of this package.)

When you perform matrix multiplication between a `AbstractArray` and a `NamedDimsArray`
then the new dimensions name is given as the wildcard `:_`.
Similarly, when you take the transpose of a `AbstractVector`, the new first dimension
is named `:_`.

## Usage
### Writing functions that accept `NamedDimsArray`s or `AbstractArray`s

It is a common desire to be able to write code that anyone can call,
whether they are using `NamedDimsArray`s or not.
While also being able to use `NamedDimsArray`s internally in its definition;
and also getting the assertion when a `NamedDimsArray` _is_  passed in, that it has the
expected dimensions.
The way to do this is to call the `NamedDimsArray` constructor, with the expected names
within the function.
As in the following example:

```
function total_variance(data::AbstractMatrix)
    n_data = NamedDimsArray(data, (:times, :locations))
    location_variance = var(n_data; dims=:times)  # calculate variance at each location
    return sum(location_variance; dims=:locations)  # total them
end
```

If this function is given (say) a `Matrix`, then it will apply the names to it in `n_data`.
Thus the function will just work on unnamed types.
If `data` is a `NamedDimsArray`, with incompatible names an error will be thrown.
For example if it `data` was mistakenly transposed and so had the dimension names:
`(:locations, :times)` instead of `(:times, :locations)`.
If `data` was partially named, e.g. `(:_, :locations)`, then that name would be allowed to be
combined with the named from the constructor; yielding `n_data` with the expected names:
`(:times, :locations)`.
This pattern allows both assertions of correctness (for named inputs),
and convenience and compatibility (for unnamed input).
And since `NamedDimsArray` is a zero-cost abstraction, this will basically compile out of existence,
most of the time.

## Extending support for more functions
There are two common things to do to make a function support `NamedDimsArray`s.
These are:
 - Adding support for referring to a dimension by name to an existing function
 - Make the operation return a `NamedDimsArray` rather than a `Array`. (Many operations fallback to dropping the names)
Often they are done together.

They are illustrated by the following example:
```
function foo(nda::NamedDimsArray, args...; dims=:)
    numerical_dims = dim(nda, dims)  # convert any form of dims into numerical dims
    raw_result = foo(parent(nda), args...; dims=numerical_dims)  # call it on the backed data
    new_names = determine_foo_names(nda, args...)  # workout what the new names will be
    return NamedDimsArray{new_names)(raw_result)  # wrap the result up
end
```

You can do this to your own functions in your own packages, to add `NamedDimsArray` support.
If you implement it for any functions in a standard library, a PR would be very appreciated.

### Caveats

If multiple dimensions have the same names, indexing by name is considered undefined behaviour and should not be relied upon.
