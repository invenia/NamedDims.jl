# NamedDims

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/NamedDims.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://invenia.github.io/NamedDims.jl/dev)
[![Build Status](https://travis-ci.com/invenia/NamedDims.jl.svg?branch=master)](https://travis-ci.com/invenia/NamedDims.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/invenia/NamedDims.jl?svg=true)](https://ci.appveyor.com/project/invenia/NamedDims-jl)
[![Codecov](https://codecov.io/gh/invenia/NamedDims.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/invenia/NamedDims.jl)

`NamedDimsArray` is a zero-cost abstraction to add names to the dimensions of an array.

## Core functionality:
For `nda = NamedDimsArrays{(:x,:y,:z)}(rand(10,20,30))`.

 - Unwrapping: `parent(nda)`: returns the underlying `AbstractArray` that is wrapped by the `NamedDimsArray`
 - Indexing: `nda[y=2]`: the same as `nda[x=:, y=2, z=:]` which is the same as `nda[:,2,:]`
 - Functions taking a dims arg: `sum(nda; dims=:y)` is the same as `sum(nda; dims=2)`

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

Currently, if you have more than one wildcard dimension name,
functionality for referring to dimensions by name will not work.
See [issue #8](https://github.com/invenia/NamedDims.jl/issues/8).
