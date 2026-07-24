module AdaptExt

using Adapt
using NamedDims: NamedDimsArray, dimnames

Adapt.adapt_structure(to, x::NamedDimsArray) =
    NamedDimsArray(Adapt.adapt(to, parent(x)), dimnames(x))

end
