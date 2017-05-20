#include <cuda_runtime.h>
#include <cstdio>
#include "gillespie_cuda.cuh"


/*
Atomic-min function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source:
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


// TODO: 2.1     Gillespie timestep implementation (25 pts)
__global__
void gillespieTimestepKernel(const float *rndReactions, const float *rndTimes,
    float *simTimes, float *simConcentrations, SysStat *simStates,
    const unsigned int nSims) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Iterate through MC simulations */
    while (idx < nSims) {
        float lambda;

        float rndReaction = rndReactions[idx];
        float Concentration = simConcentrations[idx];
        SysStat State = simStates[idx];

        if (State == Off) {
            // precompute these for ILP
            lambda = KOFF + Concentration * G;
            float crit = KON / lambda;

            if (rndReaction < crit) {
                simStates[idx] = On;
            } else { // decay prop. to concentration
                simConcentrations[idx]--;
            }
        } else { // State == On
            // precompute these for ILP
            lambda = (KOFF + B) + Concentration * G;
            float crit = KOFF / lambda;
            float crit1 = (KOFF + B) / lambda;

            if (rndReaction < crit) {
                simStates[idx] = Off;
            } else if (rndReaction < crit1) {
                simConcentrations[idx]++;
            } else {
                simConcentrations[idx]--;
            }
        }

        //simTimes[idx] += -log(rndTimes[idx]) / lambda;
        simTimes[idx] += log(rndTimes[idx]) / lambda;
        idx += gridDim.x * blockDim.x;
    }
}

// TODO: 2.2     Data resampling and stopping condition (25 pts)
__global__
void gillespieResampleKernel(int *prevSampleInd, float *simSamples,
    const float *simTimes, float *simMinTimes, const float *simConcentrations,
    const unsigned nSims)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < nSims) {
        // get the offsets into this simulation
        float *Sample = simSamples + idx * SAMPLE_SIZE;
        float Time = simTimes[idx];

        int prevSampleIndx = prevSampleInd[idx];
        int casted_curSampleIndx = Time / ( (float) SAMPLE_T / SAMPLE_SIZE );

        if ( (casted_curSampleIndx > prevSampleIndx)
            && (prevSampleIndx < SAMPLE_SIZE) ) {
            float Concentration = simConcentrations[idx];

            while ( (prevSampleIndx <= casted_curSampleIndx)
                && (prevSampleIndx < SAMPLE_SIZE) )

                Sample[prevSampleIndx++] = Concentration;
        }

        prevSampleInd[idx] = prevSampleIndx;
        idx += gridDim.x * blockDim.x;
    }

    // Determine stopping condition
    idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shmem[];
    float min_across_threads = INT_MAX;
    while (idx < nSims) {
        min_across_threads = min(min_across_threads, simTimes[idx]);
        idx += gridDim.x * blockDim.x;
    }
    shmem[threadIdx.x] = min_across_threads;

    // The threads should be synchronized across blocks before the reduction
    __syncthreads();

    if (threadIdx.x == 0) {
        float min_across_block = INT_MAX;

        for (unsigned i = 0; i < blockDim.x; i++) {
            min_across_block = min(min_across_block, shmem[i]);
        }

        atomicMin(simMinTimes, min_across_block);
    }
}

// TODO: 2.3a    Calculation of system mean (10 pts)
__global__
void gillespieMeanKernel(const unsigned i, float *simSamples,
    float *sample_mean_dev, const unsigned nSims)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float shmem[];
    shmem[idx] = 0;

    /* Iterate through simulations. */
    while (idx < nSims) {
        shmem[threadIdx.x] += simSamples[idx * SAMPLE_SIZE + i];

        idx += gridDim.x * blockDim.x;
    }

    __syncthreads();

    /* Use the first thread / warp to compute blocksum. */
    if (threadIdx.x == 0) {
        float blocksum = 0;

        for (unsigned j = 0; j < blockDim.x; j++) {
            blocksum += shmem[j];
        }

        // compute the mean
        blocksum /= ( (float) nSims );
        atomicAdd(sample_mean_dev + i, blocksum);
    }
}

void GillespieCallMeanKernel(const unsigned nBlocks,
    const unsigned nThreads, const unsigned i, float *simSamples, float *sample_mean_dev,
    const unsigned nSims) {
    gillespieMeanKernel<<<nBlocks, nThreads, nThreads * sizeof(float)>>>(
        i, simSamples, sample_mean_dev, nSims);
}

// TODO: 2.3b    Calculation of system varience (10 pts)
__global__
void gillespieVarianceKernel(const unsigned i, float *simSamples, float *sample_var_dev,
    float *sample_mean_dev, const unsigned nSims)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float shmem[];
    shmem[idx] = 0;

    float mean = sample_mean_dev[i];

    /* Iterate through simulations. */
    while (idx < nSims) {
        // variance is sum (X - mu)^2
        shmem[threadIdx.x] += powf(simSamples[idx * SAMPLE_SIZE + i] - mean, 2);

        idx += gridDim.x * blockDim.x;
    }

    __syncthreads();

    /* Use the first thread / warp to compute blocksum. */
    if (threadIdx.x == 0) {
        float blocksum = 0;

        for (unsigned j = 0; j < blockDim.x; j++) {
            blocksum += shmem[j];
        }

        blocksum /= ( (float) nSims );
        atomicAdd(sample_mean_dev + i, blocksum);
    }
}

void GillespieCallVarianceKernel(const unsigned nBlocks,
    const unsigned nThreads, const unsigned i, float *simSamples, float *sample_var_dev,
    float *sample_mean_dev, const unsigned nSims) {
    gillespieVarianceKernel<<<nBlocks, nThreads, nThreads * sizeof(float)>>>(
        i, simSamples, sample_var_dev, sample_mean_dev, nSims);
}

void GillespieCallTimestepKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock, const float *rndReactions,
    const float *rndTimes, float *simTimes, float *simConcentrations,
    SysStat *simStates, const unsigned int nSims) {

    gillespieTimestepKernel<<<blocks, threadsPerBlock>>>(rndReactions, rndTimes,
        simTimes, simConcentrations, simStates, nSims);
}

void GillespieCallResampleKernel( const unsigned nBlocks,
    const unsigned nThreads, int *prevSampleInd, float *simSamples,
    const float *simTimes, float *simMinTimes, float *min_time, const float *simConcentrations,
    int * completed, const unsigned nSims) {

    gillespieResampleKernel<<<nBlocks, nThreads, nThreads * sizeof(float)>>>
        (prevSampleInd, simSamples, simTimes, simMinTimes, simConcentrations,
            nSims);

    cudaMemcpy(min_time, simTimes, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", *min_time);
    if (*min_time >= SAMPLE_T)
        *completed = 1;

}
