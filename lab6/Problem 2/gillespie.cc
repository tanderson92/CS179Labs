#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include "ta_utilities.hpp"
#include "gillespie_cuda.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("usage: %s <threads per block> <number of blocks> <number of sims>\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    const unsigned nThreads = atoi(argv[1]);
    const unsigned nBlocks  = atoi(argv[2]);
    const unsigned nSims    = atoi(argv[3]);

    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 1000;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    /* State variables for the Monte Carlo simulations. */
    int simComplete;
    int *d_simComplete;

    /* Allocate memory on the CPU. */
    float min_time = 0.0;
    float max_float_val = INT_MAX;
    float *sample_mean = (float *) malloc(SAMPLE_SIZE * sizeof(float));
    float *sample_var  = (float *) malloc(SAMPLE_SIZE * sizeof(float));

    size_t simSize = nSims * sizeof(float);
    /* random inputs for computing transition, times, respectively */
    float *rndReactions, *rndTimes;
    /* Simulation vals */;
    float *simTimes, *simConcentrations, *simMinTimes;
    SysStat *simStates;

    /* Sampled vals */
    int *prevSampleInd;
    float *simSamples;
    /* Device versions of sample_mean, sample_var. */
    float *sample_mean_dev, *sample_var_dev;

    /* Allocate memory on the GPU. */

    gpuErrchk(cudaMalloc((void **) &rndReactions,      simSize));
    gpuErrchk(cudaMalloc((void **) &rndTimes,          simSize));

    gpuErrchk(cudaMalloc((void **) &prevSampleInd,     nSims * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &simTimes,          simSize));
    gpuErrchk(cudaMalloc((void **) &simConcentrations, simSize));
    gpuErrchk(cudaMalloc((void **) &simStates,         simSize));

    gpuErrchk(cudaMalloc((void **) &simMinTimes,   sizeof(float)));
    gpuErrchk(cudaMalloc((void **) &d_simComplete, sizeof(int)));

    gpuErrchk(cudaMalloc((void **) &sample_mean_dev, SAMPLE_SIZE * sizeof(float)));
    gpuErrchk(cudaMalloc((void **) &sample_var_dev,  SAMPLE_SIZE * sizeof(float)));

    gpuErrchk(cudaMalloc((void **) &simSamples,      SAMPLE_SIZE * simSize));

    /* Initialize memory on the GPU. */

    gpuErrchk(cudaMemset(prevSampleInd,     0, nSims * sizeof(int)));
    gpuErrchk(cudaMemset(simTimes,          0, simSize));
    gpuErrchk(cudaMemset(simConcentrations, 0, simSize));
    gpuErrchk(cudaMemset(simStates,         0, simSize));

    gpuErrchk(cudaMemset(rndReactions,      0, simSize));
    gpuErrchk(cudaMemset(rndTimes,          0, simSize));

    gpuErrchk(cudaMemset(sample_mean_dev, 0, SAMPLE_SIZE * sizeof(float)));
    gpuErrchk(cudaMemset(sample_var_dev,  0, SAMPLE_SIZE * sizeof(float)));

    gpuErrchk(cudaMemset(simSamples, 0, SAMPLE_SIZE * simSize));

    /* Perform curand initialization */
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345ull);

    /* Loop over each timestep in the simulation. */
    do {
        /* Generate random numbers for the simulation. */
        curandGenerateUniform(gen, rndReactions, nSims);
        curandGenerateUniform(gen, rndTimes, nSims);

        /* Execute a single timestep in the Gillespie simulation. */
        GillespieCallTimestepKernel(nBlocks, nThreads, rndReactions, rndTimes,
            simTimes, simConcentrations, simStates, nSims);

        /* Haven't completed yet. */
        gpuErrchk( cudaMemset(d_simComplete, 0, sizeof(int)) );

        /* Accumulate the results of the timestep. */
        //gillespieCallResampleKernel(nBlocks, nThreads, float *d_simComplete);
        gpuErrchk(cudaMemcpy(simMinTimes, &max_float_val, sizeof(float), cudaMemcpyHostToDevice));
        GillespieCallResampleKernel(nBlocks, nThreads, prevSampleInd,
            simSamples, simTimes, simMinTimes, &min_time, simConcentrations, &simComplete,
            nSims);

        /* Check if stopping condition has been reached. */
        //cudaMemcpy(&simComplete, d_simComplete, sizeof(int), cudaMemcpyDeviceToHost);

    } while (simComplete == false);

    /* Gather the results, per-sim. */
    for (unsigned i = 0; i < SAMPLE_SIZE; i++) {
        // Calculate mean of simulation_samples.
        GillespieCallMeanKernel(nBlocks, nThreads, i, simSamples,
                sample_mean_dev, nSims);

        // Calculate variance of simulation_samples.
        GillespieCallVarianceKernel(nBlocks, nThreads, i, simSamples,
                sample_var_dev, sample_mean_dev, nSims);
    }

    gpuErrchk(cudaMemcpy(sample_mean, sample_mean_dev,
        SAMPLE_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(sample_var, sample_var_dev,
        SAMPLE_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    printf("%%%% Computed Mean(s) %%%%\n");
    for (int i = 0; i < SAMPLE_SIZE; i++)
        printf("i: %d, %f\n", i, sample_mean[i]);
    printf("%%%% Computed Variance(s) %%%%\n");
    for (int i = 0; i < SAMPLE_SIZE; i++)
        printf("i: %d, %f\n", i, sample_var[i]);


    /* Free GPU memory */
    cudaFree(sample_mean_dev);
    cudaFree(sample_var_dev);
    cudaFree(prevSampleInd);
    cudaFree(simSamples);
    cudaFree(simTimes);
    cudaFree(simMinTimes);
    cudaFree(simConcentrations);
    cudaFree(simStates);

    /* Free CPU memory */
    free(sample_mean);
    free(sample_var);

}
