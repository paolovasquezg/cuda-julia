using CUDA
using Test

N = 2^20
x_d = CUDA.fill(1.0f0, N)
y_d = CUDA.fill(2.0f0, N)

# GPU kernel with threads
function gpu_add2!(y, x)
    index = threadIdx().x 
    stride = blockDim().x 
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

# Test
fill!(y_d, 2)
@cuda threads=256 gpu_add2!(y_d, x_d)
synchronize()
@test all(Array(y_d) .== 3.0f0)
println("Test passed!")