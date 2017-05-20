/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Cuda1DFDWave_cuda.cuh"

__global__
void
cuda1DWaveEqnKernel(const unsigned int numberOfNodes, const float cfl2,
    const float *oldDisplacements, const float *curDisplacements,
    float *newDisplacements, const float left_boundary_value) {

    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Warp divergence needed here: apply boundary condition at x_0^{n+1}
    if (tid == 0) {
        newDisplacements[tid] = left_boundary_value;
        tid += gridDim.x * blockDim.x;
    }

    // Apply the centered-time, centered-space finite difference method
    while (tid <= (numberOfNodes - 1) - 1) {
        newDisplacements[tid] = 2 * curDisplacements[tid] -
            oldDisplacements[tid]
            + cfl2 * (curDisplacements[tid+1] - 2*curDisplacements[tid] + curDisplacements[tid-1]);
        tid += gridDim.x * blockDim.x;
    }

    // Apply fixed x_{n} boundary condition
    if (tid == (numberOfNodes - 1)) {
        newDisplacements[tid] = 0;
    }
}


void cudaCall1DWaveEqnKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock, const unsigned int numberOfNodes,
    const float cfl, const float *oldDisplacements,
    const float *curDisplacements, float *newDisplacements,
    const float left_boundary_value) {

    const float cfl2 = cfl * cfl;
    cuda1DWaveEqnKernel<<<blocks, threadsPerBlock>>>(numberOfNodes, cfl2,
        oldDisplacements, curDisplacements, newDisplacements,
        left_boundary_value);
}
