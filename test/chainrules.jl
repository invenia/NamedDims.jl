@testset "chainrules.jl" begin
    test_rrule(NamedDimsArray, rand(2, 3), (:a, :b))
    test_rrule(NamedDimsArray{(:a, :b)}, (rand(2, 3)))
end
