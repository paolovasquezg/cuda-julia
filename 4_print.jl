using CUDA
using Test

N = 2^20
x_d = CUDA.fill(1.0f0, N)
y_d = CUDA.fill(2.0f0, N)

function gpu_add2_print!(y, x)
    index = threadIdx().x
    stride = blockDim().x
    @cuprintln("thread: $index, block size: $stride")
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda threads=16 gpu_add2_print!(y_d, x_d)
synchronize()