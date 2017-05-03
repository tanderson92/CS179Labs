#ifndef XRAY_CT_CUDA
#define XRAY_CT_CUDA

#include <cufft.h>
#define PI 3.14159265358979


void cudaCallHighPassFilter(unsigned int nBlocks, unsigned int threadsPerBlock,
    cufftComplex* dev_sinogram_cmplx, int sinogram_width, int nAngles);

void cudaCallBackProjection(float *output, float *dev_sinogram_float,
    int sinogram_width, int nAngles, int width, int height,
    int midpt_width, int midpt_height, int midpt_width_sino);

#endif
