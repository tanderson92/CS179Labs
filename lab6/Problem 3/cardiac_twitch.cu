#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "curand.h"
#include "math.h"
#include <curand_kernel.h>
#include <time.h>
#include <fstream>
#include "transition_matrix.hpp"
#include "cardiac_twitch.hpp"
#include "ta_utilities.hpp"

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Write array of floats out to text file for plotting
void write_out(float * data, unsigned int num_elements)
{
    ofstream outfile;
    outfile.open("./results.csv");
    // for each elem
    for (unsigned int i = 0; i < num_elements; ++i)
    {
      outfile << data[i] << "\n";
    }
}

// Divides floats in an array by "scalar"
__global__ void
cuda_div(float *in, float scalar, int n)
{

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    while(globalIdx < n)
    {
        float temp = in[globalIdx] / scalar;
        in[globalIdx] = temp;
        globalIdx += blockDim.x * gridDim.x;
    }

}

// Kernel to perform cardiac tissue MCMC
__global__ void mcmc(const float * transMatrix, float *masterForces, unsigned int iterations, unsigned int reps)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    while (globalIdx < reps) 
    {
        unsigned int base_index = 0;
        float r = 0.0;
        float sum_contracted = 0;
        // All RUs set to initial state of 0
        unsigned int RU[NUM_RUS] = {0};
        // Variable to remember original RU state of current RU (see j loop) before it was updated
        unsigned int original_current_state	 = 0; 
        // Variable to remember original RU state of the left neighbor (see j loop) before it was updated
        unsigned int original_left_state	 = 0;
        // Initialize random number generator, seeding with globalIdx
        curandState s;
        curand_init (globalIdx , 0, 0, &s);

        for (unsigned int i = 0; i < iterations; i++)
        {
        	original_left_state	 = 0;
            // Only update non-edge RUs {1,25}
        	for(int j = 1; j < NUM_RUS - 1; j++)
        	{
                // Generate a single random number
        		r = curand_uniform(&s);
                original_current_state	 = RU[j];

                // linearization formula for a 5D matrix.
        		////index =  ((((leftNeighbor * dimen2 + rightNeighbor) * dimen3  + currentState) * dimen4 + MutantBinary) * dimen5)
        		base_index =  ((((original_left_state * 6 + (RU[j+1])) * 6  + (RU[j]) ) * 2 + 0) * 6);
                // Offset current RU state by using the cooresponding value of M. At most, 1 of these M's will be nonzero
                unsigned int M1 = (r < transMatrix[base_index]) * (transMatrix[base_index + 1]);
                unsigned int M2 = (! ( r < transMatrix[base_index])) * ( r < transMatrix[base_index + 2]) * (transMatrix[base_index + 3]);
                unsigned int M3 = (! ( r < transMatrix[base_index + 2])) * (r < transMatrix[base_index + 4]) * (transMatrix[base_index + 5]);
                RU[j] += M1 + M2 + M3;
                // Get ready for next j iteration
                original_left_state	 = original_current_state;
        	}
         
            // Count how many of the RU states, excluding edege RUs, are in the contractile state (5)
        	for(int z = 1; z < NUM_RUS - 1; z++)
        	{
        		sum_contracted += (RU[z] == 5);
        	}
            atomicAdd(masterForces + i, sum_contracted);
        	sum_contracted = 0.0;
        }
        globalIdx += blockDim.x * gridDim.x;
    }

}

int main() 
{
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 30;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    clock_t time1 = clock();
    unsigned int blockSize = 512;
    unsigned int num_blocks = 40;
    unsigned int iterations = 100000;
    unsigned int reps = 4096;

	// Host input vectors (transition matrix and force vector)
	float * h_TM;	
    float *h_F;

	// Device input vectors (transition matrix and force vector)
    float *d_F;
    float *d_TM;

	// Sizes of vectors. For any element at index i in the force vector, that element represents
    // the force fo the cardiac tissue at time i * dt where dt is defined in cardiac_twitch.hpp.
    size_t sizeF = iterations * sizeof(float);
    size_t sizeTM = transMatrixSize * sizeof(float);

    // Init host vectors
    h_F = (float*) calloc(iterations, sizeof(float));
    h_TM = gen_transition_matrix();

    // Allocate memory for each vector on GPU
    gpuErrchk( cudaMalloc(&d_F, sizeF) );
	gpuErrchk( cudaMalloc(&d_TM, sizeTM) );

    // Copy host vectors to device
    gpuErrchk( cudaMemcpy( d_TM, h_TM, sizeTM, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy( d_F, h_F, sizeF, cudaMemcpyHostToDevice) );

    // Execute the simulation
    mcmc<<<num_blocks, blockSize>>>(d_TM, d_F, iterations, reps);

    // Average % activation across all repitions
    float normalization_constant = reps * (NUM_RUS - 2);
    cuda_div<<<num_blocks, blockSize>>>(d_F, normalization_constant, iterations);
	 
    // Copy array back to host
    gpuErrchk( cudaMemcpy(h_F, d_F, sizeF, cudaMemcpyDeviceToHost) );

	// Release device memory
    gpuErrchk( cudaFree(d_F) );
    gpuErrchk( cudaFree(d_TM) );
 
    // Write results out to file for viewing
    write_out(h_F, iterations);

    // Release host memory
    free(h_F);
    free(h_TM);

    // Print time
    printf("Total time elapsed: %ld miliseconds\n", (clock() - time1) / (CLOCKS_PER_SEC / 1000));
};

