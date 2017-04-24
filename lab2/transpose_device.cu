#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // As described above, a single warp (each of 32) handles a 32x4
    // submatrix, with the individual threads each managing a 1x4
    // column, indexed by its threadIdx.y; note that since each thread
    // transposes a column, in column-major (C) format the read from
    // memory pointed to by `input' is coalesced. However, observe the
    // writes to memory pointed to by `output'; this cannot be a coalesced
    // write since each thread writes to consecutive  rows, meaning each
    // thread in the warp accesses its own cache line.
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    /* Use shared memory for the warp's accesses to global memory, ensuring
       entire cache lines are written into fast-access memory. We pad the rows
       with an extra column (65 columns per row) which ensures 0 bank conflicts. */
    __shared__ float data[65*64];

    // Indices to the input matrix
    const int i = threadIdx.x + 64 * blockIdx.x;
    // We removed the dependency from the naiveTransposeKernel version on j
    // here (and below) which seems to increase performance about ~.02ms.
    const int end_j = 4 * (threadIdx.y + 1) + 64 * blockIdx.y;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;

    // Indices to the output (transposed) matrix
    const int i_o = threadIdx.x + 64 * blockIdx.y;
    const int end_j_o = 4 * (threadIdx.y + 1) + 64 * blockIdx.x;
    int j_o = 4 * threadIdx.y + 64 * blockIdx.x;

    // Indices (local) to the shared memory array
    const int i_shared = threadIdx.x;
    int j_shared = 4 * threadIdx.y;

    // Use a while loop rather than a for loop to increment j at same time
    // as reading it in the body. 
    while (j < end_j)
        data[(j_shared++) + 65*i_shared] = input[i + n * (j++)];

    /* Ensure that all threads have filled the shared memory buffer before
       performing the actual transpose operation into global memory; necessary
       for correctness. */
    __syncthreads();
    // reset the shared data index for read out, since it was incremented above.
    j_shared = 4 * threadIdx.y;

    // Use a while loop rather than a for loop to increment j at same time
    // as reading it in the body. 
    while (j_o < end_j_o)
    /* Note that in contrast to the writes in naiveTransposeKernel, these
       writes to global memory are coalesced since they write consistently with
       column-major format. */
        output[i_o + n * (j_o++)] = data[i_shared + 65 * (j_shared++)];
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // This is based off of my shmemTransposeKernel. See the comments there
    // for the same memory access patterns.

    __shared__ float data[65*64];

    /* Declare all indices as const; since we unrolled the loops over the 1x4
       section, these are unchanged in the kernel (using GPU const memory
       yields no observed improvement. */

    // Indices to the input matrix
    const int i = threadIdx.x + 64 * blockIdx.x;
    const int j = 4 * threadIdx.y + 64 * blockIdx.y;

    // Indices to the output (transposed) matrix
    const int i_o = threadIdx.x + 64 * blockIdx.y;
    const int j_o = 4 * threadIdx.y + 64 * blockIdx.x;

    // Indices (local) to the shared memory array
    const int i_shared = threadIdx.x;
    const int j_shared = 4 * threadIdx.y;

    // This block yields no improvement despite the attempt to exploit ILP
    // This must be because the L1 cache is the same hardware as shmem

    //const float r1 = input[i + n * j];
    //const float r2 = input[i + n * (j + 1)];
    //const float r3 = input[i + n * (j + 2)];
    //const float r4 = input[i + n * (j + 3)];
    //data[ j_shared      + 65*i_shared] = r1;
    //data[(j_shared + 1) + 65*i_shared] = r2;
    //data[(j_shared + 2) + 65*i_shared] = r3;
    //data[(j_shared + 3) + 65*i_shared] = r4;

    // Unroll the first loop, writing into the shmem buffer
    data[ j_shared      + 65*i_shared] = input[i + n *  j     ];
    data[(j_shared + 1) + 65*i_shared] = input[i + n * (j + 1)];
    data[(j_shared + 2) + 65*i_shared] = input[i + n * (j + 2)];
    data[(j_shared + 3) + 65*i_shared] = input[i + n * (j + 3)];

    __syncthreads();

    // This block yields no improvement despite the attempt to exploit ILP
    // This must be because the L1 cache is the same hardware as shmem

    //const float r5 = data[i_shared + 65 * j_shared];
    //const float r6 = data[i_shared + 65 * (j_shared + 1)];
    //const float r7 = data[i_shared + 65 * (j_shared + 2)];
    //const float r8 = data[i_shared + 65 * (j_shared + 3)];
    //output[i_o + n *  j_o     ] = r5;
    //output[i_o + n * (j_o + 1)] = r6;
    //output[i_o + n * (j_o + 2)] = r7;
    //output[i_o + n * (j_o + 3)] = r8;

    // Unroll the second loop, writing into global memory.
    output[i_o + n *  j_o     ] = data[i_shared + 65 *  j_shared     ];
    output[i_o + n * (j_o + 1)] = data[i_shared + 65 * (j_shared + 1)];
    output[i_o + n * (j_o + 2)] = data[i_shared + 65 * (j_shared + 2)];
    output[i_o + n * (j_o + 3)] = data[i_shared + 65 * (j_shared + 3)];
}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}
