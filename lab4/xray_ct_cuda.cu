#include <cuda_runtime.h>
#include "xray_ct_cuda.cuh"

__global__
void
cudaHighPassFilter(cufftComplex *dev_sinogram_cmplx,
    const unsigned int sinogram_width, const unsigned int nAngles) {

    unsigned int len = sinogram_width*nAngles;
    float sinogram_spec_center = (sinogram_width - 1) / 2.0;
    unsigned int sinogram_spec_d_center;
    float filter;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < len) {
        /* Note that we use CUFFT's R2C transform with a batchsize of nAngles and
         * FFT length of N = sinogram_width, Now, R2C takes an N-vector of cufftReal
         * and yields an (N/2 + 1)-vector of cufftComplex. As a result all modes are
         * mod (N/2+1) with respect to the spectrum center.
         *
         * The center-to-freq distance for this mode, cast result to float before assignment
         */
        sinogram_spec_d_center =
            abs( (float) (tid % (sinogram_width / 2 + 1) - sinogram_spec_center) );
        // Compute filter factor for this mode
        filter = 1.0 - (float) sinogram_spec_d_center / sinogram_spec_center;
        // Apply high-pass spectral filter to the mode
        dev_sinogram_cmplx[tid].x *= filter;
        dev_sinogram_cmplx[tid].y *= filter;
        tid += blockDim.x * gridDim.x;
    }

}

void cudaCallHighPassFilter(unsigned int nBlocks, unsigned int threadsPerBlock,
    cufftComplex* dev_sinogram_cmplx, int sinogram_width, int nAngles) {

    cudaHighPassFilter<<<nBlocks, threadsPerBlock>>>(dev_sinogram_cmplx,
        sinogram_width, nAngles);
}

__global__
void
cudaBackProjection(float *output_dev, float *dev_sinogram_float, int sinogram_width,
    int nAngles, int width, int height, int midpt_width, int midpt_height,
    int midpt_width_sino) {

    // Coordinates for pixel for this thread
    int x_p;
    int y_p = blockIdx.y * blockDim.y + threadIdx.y;

    // Geometric (float) coordinates for pixel for this thread
    float x_g, y_g;
    // Calculated intersection point (x_i, y_i) with sinogram center
    float x_i, y_i;
    // distance of observation pt. from centerline d^2 = x_i^2 + y_i^2
    float d;
    // slope of centerline
    float m;
    // slope of perp. to centerline
    float q;
    // doh...
    float theta;


    for (x_p = blockIdx.x * blockDim.x + threadIdx.x;
         x_p < width; x_p += blockDim.x * gridDim.x) {
        for (y_p = blockIdx.y * blockDim.y + threadIdx.y;
             y_p < height; y_p += blockDim.y * gridDim.y) {
            for (int thetaIt = 0; thetaIt < nAngles; thetaIt++) {
                // Determine geometric coordinate for this pixel
                // since (0,0) in pixel coords is the upper left
                // and (0,0) in geo coords is the center
                x_g = x_p - midpt_width;
                y_g = midpt_height - y_p;

                theta = thetaIt * (PI / nAngles);
                if (theta == 0) { d = x_g; }
                else if (theta == PI/2) { d = y_g; }
                else {
                    // Slopes
                    m = -1.0 / tanf(theta);
                    q = -1.0 / m;

                    // Calculate intersection pt with sinogram center
                    x_i = (y_g - m * x_g) / (q - m);
                    y_i = q * x_i;

                    d = sqrtf( x_i * x_i + y_i * y_i );
                    // Implement |dt| feature
                    if ( (x_i < 0 && q > 0) || (x_i > 0 && q < 0) ) {
                        d = -d;
                    }
                }


                output_dev[x_p + y_p * width] +=
                    dev_sinogram_float[ (int) midpt_width_sino + (int) d +
                                        thetaIt * sinogram_width ];
            }
        }
    }
}

void cudaCallBackProjection(float *output, float *dev_sinogram_float,
    int sinogram_width, int nAngles, int width, int height,
    int midpt_width, int midpt_height, int midpt_width_sino) {

    dim3 blockNumThreads;
    dim3 blockSize;

    blockNumThreads.x = ceil( width / 32.0 );
    blockNumThreads.y = ceil( height / 32.0 );
    blockSize.x = 32;
    blockSize.y = 32;

    cudaBackProjection<<<blockNumThreads, blockSize>>>(output,
        dev_sinogram_float, sinogram_width, nAngles,
        width, height, midpt_width, midpt_height, midpt_width_sino);
}
