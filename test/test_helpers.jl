"""
    @test_modern

Alias for `@test`, kept for historical reasons (it used to mark tests that were
`@test_broken` on Julia 1.0).
"""
macro test_modern(expr...)
    return :(@test $(expr...)) |> esc
end
