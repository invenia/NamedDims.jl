@testset "FFT" begin
    ndv = NamedDimsArray(zeros(ComplexF64, 16), :μ)
    ndv[μ=2] = sqrt(2)

    @testset "vector" begin
        @test dimnames(fft(ndv, :μ => :κ)) == (:κ,)
        @test dimnames(ifft(ndv, :μ => :κ)) == (:κ,)
        @test dimnames(bfft(ndv, :μ => :κ)) == (:κ,)
        @test_throws ArgumentError fft(ndv, :nope => :κ)
        @test_throws ArgumentError ifft(ndv, :μ => :κ, :μ => :again)

        @test dimnames(fft(ndv)) == (:μ∿,)
        @test dimnames(ifft(ndv)) == (:μ∿,)
        @test dimnames(bfft(ndv)) == (:μ∿,)
        @test dimnames(ifft(fft(ndv))) == (:μ,)

        @test ifft(fft(ndv, :μ => :κ), :κ => :μ) ≈ ndv
        @test ifft(fft(ndv)) ≈ ndv
        @test bfft(fft(ndv)) ≈ 16 * ndv
    end

    @testset "vector plan" begin
        pv1 = plan_fft(ndv, :μ)
        pv2 = plan_fft(randn(ComplexF64, 16))
        @test pv2 isa FFTW.FFTWPlan

        @test dimnames(pv1 * ndv) == (:μ∿,)
        @test dimnames(pv2 * ndv) == (:μ∿,)
        @test dimnames(pv1 * randn(16)) == (:_,)

        pv3 = plan_ifft(pv2 * ndv)
        @test dimnames(pv3 * (pv2 * ndv)) == (:μ,)
        @test dimnames(inv(pv3) * ndv) == (:μ∿,)
        @test dimnames(inv(pv1) * (pv1 * ndv)) == (:μ,)

        @test dimnames(pv3 \ ndv) == (:μ∿,)
    end

    nda = NamedDimsArray(zeros(Float32, 4,4,4), (:a, :b′, :c))
    nda[1,2,3] = nda[2,3,4] = 1

    @testset "three dims" begin
        @test dimnames(fft(nda, :a => :k)) == (:k, :b′, :c)
        @test dimnames(ifft(nda, :a => :k, :c => :k′)) == (:k, :b′, :k′)

        @test dimnames(fft(nda)) == (:a∿, :b′∿, :c∿)
        @test dimnames(ifft(nda, 1)) == (:a∿, :b′, :c)
        @test dimnames(bfft(nda, :b′)) == (:a, :b′∿, :c)
        @test dimnames(fft(nda, 1:2)) == (:a∿, :b′∿, :c)
        @test dimnames(fft(nda, (:a, :c))) == (:a∿, :b′, :c∿)
    end

    @testset "three plan" begin
        p1 = plan_fft(nda, :a)
        p2 = plan_fft(zeros(Float32, 4,4,4), 2:2)
        @test_throws ArgumentError plan_fft(nda, :c => :C)
        p3 = plan_fft(nda, :c)

        @test dimnames(p1 * nda) == (:a∿, :b′, :c)
        @test dimnames(p2 * nda) == (:a, :b′∿, :c)
        @test dimnames(p3 * nda) == (:a, :b′, :c∿)
        @test dimnames(p2 * randn(4,4,4)) == (:_, :_, :_)

        @test dimnames(p3 * (p1 * nda)) == (:a∿, :b′, :c∿)

        @test_throws ArgumentError plan_fft(nda, :z => :Z)
    end

    @testset "wave_name" begin
        @test wave_name(:k) == :k∿
        @test wave_name((:k1, :k2∿)) == (:k1∿, :k2)
        @test wave_name((:k1, :k2, :k3), 2) == (:k1, :k2∿, :k3)
        @test wave_name((:k1, :k2, :k3), 1:2) == (:k1∿, :k2∿, :k3)
        @test wave_name((:k1, :k2, :k3), (1,3)) == (:k1∿, :k2, :k3∿)
    end

    @testset "other functions" begin
        nda = NamedDimsArray(rand(16, 16), (:x, :y))
        ndv = NamedDimsArray(randn(32), :z)

        for f in [fft, ifft, bfft, rfft]
            @test dimnames(f(nda)) == (:x∿, :y∿)
            @test dimnames(f(nda, :x)) == (:x∿, :y)
            @test dimnames(f(nda, :x => :k)) == (:k, :y)

            @test dimnames(f(ndv, 1)) == (:z∿,)
        end

        ndc = rfft(nda, 1) # size 9x16
        @test dimnames(irfft(ndc, 16, 1)) == (:x, :y)
        @test dimnames(brfft(ndc, 16, 1)) == (:x, :y)

        for f in [plan_fft, plan_ifft, plan_bfft]
            plan = f(nda, :x)
            @test plan isa AbstractFFTs.Plan
            @test dimnames(plan * nda) == (:x∿, :y)
        end

        @test dimnames(fftshift(nda)) == (:x, :y)
        @test fftshift(nda)[1,1] == nda[9,9]
        @test fftshift(nda, :x)[1,1] == nda[9,1]
    end
end

@testset "allocations: FFT" begin
    @test 0 == @ballocated wave_name(:k)
    @test 0 == @ballocated wave_name((:k1, :k2∿))
    if VERSION >= v"1.1"
        @test 0 == @ballocated wave_name((:k1, :k2, :k3), 2)
        @test 0 == @ballocated wave_name((:k1, :k2, :k3), (1,3))
    end
end
