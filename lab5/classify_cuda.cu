#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "classify_cuda.cuh"

/*
 * Arguments:
 * data: Memory that contains both the review LSA coefficients and the labels.
 *       Format decided by implementation of classify.
 * batch_size: Size of mini-batch, how many elements to process at once
 * step_size: Step size for gradient descent. Tune this as needed. 1.0 is sane
 *            default.
 * weights: Pointer to weights vector of length REVIEW_DIM.
 * errors: Pointer to a single float used to describe the error for the batch.
 *         An output variable for the kernel. The kernel can either write the
 *         value of loss function over the batch or the misclassification rate
 *         in the batch to errors.
 */
__global__
void trainLogRegKernel(
    float *data,
    int batch_size,
    int step_size,
    float *weights,
    float *errors)
{
    /* As suggested in the assignment, have each thread classify a single point
     * and compute that point's contribution to the gradient. Use atomic adds
     * to accumulate across blocks (TODO: is there a way to avoid this??).
     */
    extern __shared__ float shmem[];
    float *weights_shmem = shmem;
    // pointer for gradient updates, offset by number of weights (above)
    float *grad_shmem = shmem + REVIEW_DIM;
    __shared__ float misclassifiedCount[1];

    // Fill shared memory, sync threads to ensure completion before processing
    {
        int tid = threadIdx.x;
        if (tid == 0)
            misclassifiedCount[0] = 0;

        while (tid < REVIEW_DIM) {
            weights_shmem[tid] = weights[tid]; // copy
            grad_shmem[tid] = 0; // initialize
            tid += blockDim.x;
        }
        __syncthreads();
    }

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // grad = (+1 / N) * sum_{n=1}^N (y_n * x_n) / (1 + exp(y_n w^T x_n)
    while (tid < batch_size) {
        // offset of the data point, REVIEW_DIM + 1 per line
        int offs = tid * (REVIEW_DIM + 1);
        // w: weights_shmem, x_n: data, y_n: yn
        float yn = data[offs + REVIEW_DIM];

        // Compute inner product; useful to have shmem here
        float inner_prod = 0; // w^T * x_n
        for (int i = 0; i < REVIEW_DIM; i++)
            inner_prod += weights_shmem[i] * data[offs + i];

        // Find misclassifications
        bool inner_prod_sgn;
        bool yn_sgn;
        if (inner_prod > 0) { inner_prod_sgn = true; } else { inner_prod_sgn = false; }
        if (yn > 0) { yn_sgn = true; } else { yn_sgn = false; }
        if (yn_sgn != inner_prod_sgn)
            atomicAdd(misclassifiedCount, 1.0);

        // Compute gradient correction, and (safely) update the gradient
        // Note we are missing a negative sign but we use atomicAdd rather
        // than atomicSub below in the weight update so these cancel in effect.
        for (int i = 0; i < REVIEW_DIM; i++) {
            float grad_correct = ( yn * data[offs + i] )
                / ( 1.0 + exp(yn * inner_prod) );
            atomicAdd(&grad_shmem[i], grad_correct);
        }
        tid += gridDim.x * blockDim.x;
    }

    // We have (atomically) updated the gradient per-thread, so ensure this has
    // completed
    __syncthreads();

    // Now accumulate at block level
    // w := w + step_size * grad
    if (threadIdx.x == 0) {
        // update the (device) weights
        for (int i = 0; i < REVIEW_DIM; i++)
            atomicAdd(&weights[i], step_size * (1.0 / batch_size) *
                        grad_shmem[i]);

        atomicAdd(errors, (misclassifiedCount[0]) / ( (float) batch_size ) );
    }
}

/*
 * All parameters have the same meaning as in docstring for trainLogRegKernel.
 * Notably, cudaClassify returns a float that quantifies the error in the
 * minibatch. This error should go down as more training occurs.
 */
float cudaClassify(
    float *data,
    int batch_size,
    float step_size,
    float *weights,
    cudaStream_t stream)
{
    int block_size = (batch_size < 1024) ? batch_size : 1024;

    // grid_size = CEIL(batch_size / block_size)
    int grid_size = (batch_size + block_size - 1) / block_size;
    int shmem_bytes = (REVIEW_DIM * 2 + 1) * sizeof(float);

    float *d_errors;
    cudaMalloc(&d_errors, sizeof(float));
    cudaMemset(d_errors, 0, sizeof(float));

    trainLogRegKernel<<<grid_size, block_size, shmem_bytes, stream>>>(
        data,
        batch_size,
        step_size,
        weights,
        d_errors);

    float h_errors = -1.0;
    cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDefault);
    cudaFree(d_errors);
    return h_errors;
}
