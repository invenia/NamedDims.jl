@testset "chainrules.jl" begin
    test_rrule(NamedDimsArray, rand(2, 3), (:a, :b))
end
