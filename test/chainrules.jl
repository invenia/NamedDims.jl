@testset "chainrules.jl" begin
    test_rrule(NamedDimsArray, rand(2, 3), (:a, :b); check_inferred=VERSION >= v"1.6")
    test_rrule(NamedDimsArray{(:a, :b)}, (rand(2, 3)); check_inferred=VERSION >= v"1.6")
end
