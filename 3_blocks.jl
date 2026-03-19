using CUDA
using Test

# Setup
N = 2^20
x_d = CUDA.fill(1.0f0, N)
y_d = CUDA.fill(2.0f0, N)

function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

kernel = @cuda launch=false gpu_add3!(y_d, x_d)
config = launch_configuration(kernel.fun)
threads = min(N, config.threads)
blocks = cld(N, threads)

println("Optimal threads: $threads")
println("Optimal blocks: $blocks")

fill!(y_d, 2)
@cuda threads=threads blocks=blocks gpu_add3!(y_d, x_d)
synchronize()
@test all(Array(y_d) .== 3.0f0)
println("Test passed!")