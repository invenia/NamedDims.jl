using Base: @deprecate

@deprecate names dimnames false
@deprecate (NamedDimsArray{L}(orig::NamedDimsArray) where L)  refine_names(orig, L) false
