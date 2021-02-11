"""
    @test_modern

Use this to mark tests that should pass on modern (>=v1.1) versions of julia,
but should be marked as `@test_broken` on Julia 1.0.
For more complicated cases, or where it shouldn't be defined at all on julia versions,
you can use `if VERSION` directly.
"""
macro test_modern(expr...)
    if VERSION >= v"1.4"
        return :(@test $(expr...))
    else
        return nothing
    end
end
