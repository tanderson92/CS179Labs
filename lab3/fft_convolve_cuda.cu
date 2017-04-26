/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"

using std::cout;
using std::endl;


/*
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source:
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v,
    cufftComplex *out_data,
    int padded_length) {


    /* Implement point-wise multiplication and scaling for the FFT'd input
     * and impulse response.
     */

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Since the input data are const use local copies for manipulation
    cufftComplex c1, c2, res;
    while (tid < padded_length) {
        c1 = raw_data[tid];
        c2 = impulse_v[tid];
        // Complex multiplication; scale by padded length because cuFFT won't
        res.x = (c1.x * c2.x - c1.y * c2.y) / padded_length;
        res.y = (c1.x * c2.y + c1.y * c2.x) / padded_length;
        out_data[tid] = res;
        tid += blockDim.x * gridDim.x;
    }

}

/* The templating is a key insight from Mark Harris' approach. The blockSize is only
 * known at runtime, so we generate templated cudaMaximumKernels which handle all
 * power-of-2 block sizes less than or equal to 512 threads. Then the optimizing
 * compiler will discard all code not relevant to the templated parameter and call
 * the proper kernel at run-time. This is a significant optimization.
 */
template <int blockSize>
__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* NOTE: I rely heavily on Mark Harris' slideshow, linked during
     * lecture, and available at:
     *  http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/
     *      projects/reduction/doc/reduction.pdf
     *
     * My algorithm is essentially identical to the final reduction example,
     * but I correct what is a clear error in the loading of global memory
     * into shared memory. One must test i + blockSize against padded_length
     * since one is loading two at a time, the second being at position
     * blockSize + i.
     */


    /* Use shared memory buffer to allow an individual block to load
     * global memory and reduce to find the maximum within shared memory
     * for faster accesses. We also use sequential addressing (more below).
     *
     * sdata uses the externally-allocated shared memory allocation
     * because its size varies as the number of threadsPerBlock
     */
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    // These counters are multiplied by 2 because we are loading two elements
    // at a time below.
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = (blockSize * 2) * gridDim.x;
    // 0-initialize shmem.
    sdata[tid] = 0;

    /* We load two values from global memory; here we ensure we remain in-bounds.
     * This is a key optimization. Rather than leave half our threads idle after
     * the first reduction we load two at a time. We actually repeat this step
     * if there are fewer blocks available than needed. This is the purpose of the
     * loop. */
    while (blockSize + i < padded_length) {
        /* fmaxf and fabsf are CUDA functions. Select maximum value of out_data.
         * Note that out_data is real following a Forward/Inverse pair, so that
         * out_data[*] = out_data[*].x and we can simply do real float comparisons
         * for the logic here */
        sdata[tid] = fmaxf(fmaxf(fabsf(out_data[i].x), fabsf(out_data[blockSize + i].x)),
                         sdata[tid]);
        // Coalesced global memory accesses
        i += gridSize;
    }
    // Need a barrier here to ensure all global memory data has loaded
    // (see shmem accesses below)
    __syncthreads();

    /* Note how the shmem is loaded and read throughout the kernel so that a warp
     * reads a sequential portion of shared memory. We avoid bank conflicts by
     * doing this, since all threads access addresses in *different* banks. */

    // Notice that by using the templated parameter blockSize we can unroll what would
    // otherwise be a for loop up to blockSize. Exploit the fact that on current H/W,
    // we are limited to 512 threads, and are in powers of 2.
    // These accesses are not a single warp, so need to sync threads after every
    // step in the unroll.
    if (blockSize >= 512) {
        if (tid < 256)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + 256]);
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + 128]);
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + 64]);
        __syncthreads();
    }

    /* Notice that once we are to 64 values or fewer we are within a single
     * warp, since the reduction will compare 32 (one per thread) values against
     * the next 32 values. Since we're in the same warp now, no thread syncing
     * needed. */
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
        if (blockSize >= 32) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 16]);
        if (blockSize >= 16) sdata[tid] = fmaxf(sdata[tid], sdata[tid +  8]);
        if (blockSize >=  8) sdata[tid] = fmaxf(sdata[tid], sdata[tid +  4]);
        if (blockSize >=  4) sdata[tid] = fmaxf(sdata[tid], sdata[tid +  2]);
        if (blockSize >=  2) sdata[tid] = fmaxf(sdata[tid], sdata[tid +  1]);
    }

    // All reductions have been completed, use the provided atomicMax across
    // blocks; further optimization not possible (?). Notice that the maximum
    // from each block was reduced into the first element of shmem, and then
    // this is compared globally across blocks, an atomic operation.
    if (tid == 0)
        atomicMax(max_abs_val, sdata[0]);
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* Divide all data by the value pointed to by max_abs_val.

    This kernel is quite short.
    */

    // Use tid as consistent with cudaProdScaleKernel
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    while (tid < padded_length) {
        out_data[tid].x /= *max_abs_val;
        tid += blockDim.x * gridDim.x;
    }

}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {

    /* Calls the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {

    // A variable amount of shared memory is needed, one float per thread at
    // the first reduction stage. This number is used in the <<<.>>> below
    // as is the CUDA standard for dynamically-allocated shmem.
    int smemSize = threadsPerBlock * sizeof(float);

    /* Calls the max-finding kernel. */
    switch (threadsPerBlock) {
        case 512:
            cudaMaximumKernel<512><<<blocks, threadsPerBlock, smemSize>>>
                (out_data, max_abs_val, padded_length); break;
        case 256:
            cudaMaximumKernel<256><<<blocks, threadsPerBlock, smemSize>>>
                (out_data, max_abs_val, padded_length); break;
        case 128:
            cudaMaximumKernel<128><<<blocks, threadsPerBlock, smemSize>>>
                (out_data, max_abs_val, padded_length); break;
        case  64:
            cudaMaximumKernel< 64><<<blocks, threadsPerBlock, smemSize>>>
                (out_data, max_abs_val, padded_length); break;
        case  32:
            cudaMaximumKernel< 32><<<blocks, threadsPerBlock, smemSize>>>
                (out_data, max_abs_val, padded_length); break;
        case  16:
            cudaMaximumKernel< 16><<<blocks, threadsPerBlock, smemSize>>>
                (out_data, max_abs_val, padded_length); break;
        case   8:
            cudaMaximumKernel<  8><<<blocks, threadsPerBlock, smemSize>>>
                (out_data, max_abs_val, padded_length); break;
        case   4:
            cudaMaximumKernel<  4><<<blocks, threadsPerBlock, smemSize>>>
                (out_data, max_abs_val, padded_length); break;
        case   2:
            cudaMaximumKernel<  2><<<blocks, threadsPerBlock, smemSize>>>
                (out_data, max_abs_val, padded_length); break;
        case   1:
            cudaMaximumKernel<  1><<<blocks, threadsPerBlock, smemSize>>>
                (out_data, max_abs_val, padded_length); break;
        default:
            std::cout << "Invalid threadsPerBlock specified!" << std::endl;
    }

}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {

    /* Calls the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
}
