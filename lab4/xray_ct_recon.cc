
/*
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)

Modified by Jordan Bonilla and Matthew Cedeno (2016)
*/


#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>
#include <time.h>
#include "ta_utilities.hpp"
#include "xray_ct_cuda.cuh"

/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}



/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}


/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}

int main(int argc, char** argv){
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 60;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    // Begin timer and check for the correct number of inputs
    time_t start = clock();
    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Input sinogram text file's name > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output text file's name >\n");
        exit(EXIT_FAILURE);
    }


    /********** Parameters **********/


    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );

    int nAngles = atoi(argv[3]);


    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);


    // These are casts to int but truncating a zero frac. part
    int midpt_width = floor(width / 2.0);
    int midpt_height = floor(height / 2.0);
    // In sinogram coordinates, mid
    int midpt_width_sino = floor( (sinogram_width - 1.0) / 2.0 );


    /********** Data storage *********/


    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    float *dev_sinogram_float;
    float* output_dev;  // Image storage

    cufftReal *sinogram_host;

    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);

    /*********** Set up IO, Read in data ************/

    sinogram_host = (cufftReal *)malloc(  sinogram_width*nAngles*sizeof(cufftReal) );

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile,"%f",&sinogram_host[i]);
    }

    fclose(dataFile);


    /* Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */
    gpuErrchk(cudaMalloc( (void **) &dev_sinogram_cmplx, sinogram_width*nAngles * sizeof(cufftComplex)  ));

    gpuErrchk(cudaMemcpy( dev_sinogram_cmplx, sinogram_host,
        sinogram_width*nAngles * sizeof(cufftReal), cudaMemcpyHostToDevice));

    /* We use batched, in-place R2C / C2R transforms. Note this combination is
     * impossible with cufftPlan1D.
     * This is quite tricky to get right but the prescription below is correct
     * * The ranks and strides are 1 because we are performing a data-continuous
     *   1D transform
     * * The idist is sinogram_width because this is the length of a single batch
     *   element.
     * * The odist is (idist / 2) + 1 because a R2C transform of size N produces
     *   output of length N/2 + 1.
     * * inembed and onembed are ignored
     */
    cufftHandle plan_r2c, plan_c2r;
    int batch = nAngles;
    int rank = 1;
    int n[] = { sinogram_width };
    int istride = 1, ostride = 1;
    int idist = sinogram_width, odist = (sinogram_width / 2) + 1;
    int inembed[] = { 0 };
    int onembed[] = { 0 };
    if (cufftPlanMany(&plan_r2c, rank, n,
            inembed, istride, idist,
            onembed, ostride, odist,
            CUFFT_R2C, batch) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT R2C Plan Creation failed\n");
        return EXIT_FAILURE;
    }
    // Note that odist and idist are transposed from above, because
    // this is the inverse transform so the "output" distance from above
    // is the "input" distance here and vice versa.
    if (cufftPlanMany(&plan_c2r, rank, n,
            inembed, istride, odist,
            onembed, ostride, idist,
            CUFFT_C2R, batch) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT C2R Plan Creation failed\n");
        return EXIT_FAILURE;
    }

    // Forward transform in-place, treating dev_sinogram_cmplx as floats in input
    cufftExecR2C(plan_r2c, (cufftReal *) dev_sinogram_cmplx, dev_sinogram_cmplx);

    // Apply the spectral filter
    cudaCallHighPassFilter(nBlocks, threadsPerBlock,
        dev_sinogram_cmplx, sinogram_width, nAngles);

    // Back transform, producing output in cufftReal format
    cufftExecC2R(plan_c2r, dev_sinogram_cmplx, (cufftReal *) dev_sinogram_cmplx);

    // The C2R transform yields float data starting at dev_sinogram_cmplx;
    // use a cast to treat it as such
    dev_sinogram_float = (cufftReal *) dev_sinogram_cmplx;
    // Clean up
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);


    // Allocate and initialize memory for the output.
    gpuErrchk(cudaMalloc( (void **) &output_dev, size_result  ));
    gpuErrchk(cudaMemset(output_dev, 0, size_result ));

    // Perform the backprojection.
    cudaCallBackProjection(
        output_dev, dev_sinogram_float,
        sinogram_width, nAngles,
        width, height, midpt_width, midpt_height,
        midpt_width_sino
    );
    // Cleanup sinogram memory
    cudaFree(dev_sinogram_cmplx);

    gpuErrchk(cudaMemcpy( output_host, output_dev, size_result,
        cudaMemcpyDeviceToHost));
    cudaFree(output_dev);

    /* Export image data. */

    for (int j = 0; j < width; j++) {
        for(int i = 0; i < height; i++) {
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */
    free(sinogram_host);
    free(output_host);

    fclose(outputFile);
    printf("CT reconstruction complete. Total run time: %f seconds\n", (float) (clock() - start) / 1000.0 / 1000.0 );
    return 0;
}

