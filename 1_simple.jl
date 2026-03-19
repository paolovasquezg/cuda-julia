using CUDA
using Test
using BenchmarkTools

N = 2^20
x = fill(1.0f0, N)
y = fill(2.0f0, N)
x_d = CUDA.fill(1.0f0, N)
y_d = CUDA.fill(2.0f0, N)

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end


function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

fill!(y_d, 2)

@cuda gpu_add1!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

@benchmark bench_gpu1!(y_d, x_d)