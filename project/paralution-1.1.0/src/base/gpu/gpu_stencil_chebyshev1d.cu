// **************************************************************************
//
//    PARALUTION   www.paralution.com
//
//    Copyright (C) 2015  PARALUTION Labs UG (haftungsbeschränkt) & Co. KG
//                        Am Hasensprung 6, 76571 Gaggenau
//                        Handelsregister: Amtsgericht Mannheim, HRA 706051
//                        Vertreten durch:
//                        PARALUTION Labs Verwaltungs UG (haftungsbeschränkt)
//                        Am Hasensprung 6, 76571 Gaggenau
//                        Handelsregister: Amtsgericht Mannheim, HRB 721277
//                        Geschäftsführer: Dimitar Lukarski, Nico Trost
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
// **************************************************************************



// PARALUTION version 1.1.0


#include "../../utils/def.hpp"
#include "gpu_vector.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "../stencil_types.hpp"
#include "gpu_stencil_chebyshev1d.hpp"

#include "cufft.h"
#include "cuda_runtime.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <complex>

namespace paralution {

template <typename ValueType>
GPUAcceleratorStencilChebyshev1D<ValueType>::GPUAcceleratorStencilChebyshev1D() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
GPUAcceleratorStencilChebyshev1D<ValueType>::GPUAcceleratorStencilChebyshev1D(
                                    const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "GPUAcceleratorStencilChebyshev1D::GPUAcceleratorStencilChebyshev1D()",
            "constructor with local_backend");

  this->set_backend(local_backend);

  this->ndim_ = 1;
}

template <typename ValueType>
void GPUAcceleratorStencilChebyshev1D<ValueType>::CopyFromHost(const HostStencil<ValueType> &src) {
  LOG_INFO("no host<->accel copying");
  FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void GPUAcceleratorStencilChebyshev1D<ValueType>::CopyToHost(HostStencil<ValueType> *dst) const {
  LOG_INFO("no host<->accel copying");
  FATAL_ERROR(__FILE__, __LINE__);
}

__global__
void Plan_compute_freq_vecs_kern(int N, int *k) {

  int tid = blockIdx.x * gridDim.x + threadIdx.x;

  while (tid <= N) {
    if (tid != N) {
      k[tid] = tid;
    } else {
      k[tid] = 0;
    }
    tid += blockDim.x * gridDim.x;
  }

  tid = blockIdx.x * gridDim.x + threadIdx.x;
  while (tid < N - 1) {
    k[N + 1 + tid] = 1 - N + tid;
    tid += blockDim.x * gridDim.x;
  }
}

__global__
void Plan_compute_chebgrid_kern(int N, float *x) {
  int tid = blockIdx.x * gridDim.x + threadIdx.x;
  while (tid < N) {
    x[tid] = cos(tid * M_PI / ( (float) N ));
    tid += blockDim.x * gridDim.x;
  }
}

template <>
void GPUAcceleratorStencilChebyshev1D<float>::Plan(float *p_,
                                    float *q_, float *x_a_, float *x_b_) {

  int n = this->get_nrow();
  int N = n - 1;
  dim3 BlockSize(this->local_backend_.GPU_block_size);
  dim3 GridSize(n / this->local_backend_.GPU_block_size + 1);
  this->p = p_;
  this->q = q_;
  this->x_a = x_a_;
  this->x_b = x_b_;

  cudaMalloc(&arr_, (2*n - 2) * sizeof(cufftComplex) );
  cufftPlan1d(&plan, 2*n - 2, CUFFT_C2C, 1);

  cudaMalloc(&k, (2*n - 2) * sizeof(int));
  Plan_compute_freq_vecs_kern<<< GridSize, BlockSize >>>(N, k);

  cudaMalloc(&uder, n * sizeof(float) );
  cudaMalloc(&u2der, n * sizeof(float) );
  cudaMemset(uder, 0, n * sizeof(float) );
  cudaMemset(u2der, 0, n * sizeof(float) );

  cudaMalloc(&x, n * sizeof(float));
  Plan_compute_chebgrid_kern<<< GridSize, BlockSize >>>(N, x);
}

template <>
void GPUAcceleratorStencilChebyshev1D<double>::Plan(double *p_,
                                    double *q_, double *x_a_, double *x_b_) {


  LOG_INFO("no double type supported yet");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
GPUAcceleratorStencilChebyshev1D<ValueType>::~GPUAcceleratorStencilChebyshev1D() {

  LOG_DEBUG(this, "GPUAcceleratorStencilChebyshev1D::~GPUAcceleratorStencilChebyshev1D()",
            "destructor");

  cufftDestroy( plan );
  cudaFree(uder);
  cudaFree(u2der);
  cudaFree(k);
  cudaFree(arr_);
  cudaFree(x);

}

template <typename ValueType>
void GPUAcceleratorStencilChebyshev1D<ValueType>::info(void) const {

  LOG_INFO("Chebyshev 1D (GPUAccelerator) size=" << this->size_ << " dim=" << this->get_ndim());

}

template <typename ValueType>
int GPUAcceleratorStencilChebyshev1D<ValueType>::get_nnz(void) const {

  return pow(this->get_nrow(), 2);
}

__global__
void ChebFFT_init_arr_kernel(int n, const float *u, cufftComplex *arr) {
  int N = n - 1;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < n) {
    arr[tid].x = u[tid];
    arr[tid].y = 0.0;
    tid += blockDim.x * gridDim.x;
  }

  tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < N) {
    arr[n + tid].x = u[(n - 1 - 1) - tid];
    arr[n + tid].y = 0.0;
    tid += blockDim.x * gridDim.x;
  }
}

__global__
void ChebFFT_sum_bdry_pts_kernel(int n, float *uder, cufftComplex *arr, int *k) {
  int N = n - 1;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < N) {
    if (tid == N - 1) {
      atomicAdd(&uder[0], 0.5 * N * arr[tid].x);
    } else {
      atomicAdd(&uder[0], powf(k[tid], 2) * arr[tid + 1].x / ( (float) N ));
    }
    tid += blockDim.x * gridDim.x;
  }

  tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < N) {
    if (tid == N - 1) {
      atomicAdd(&uder[n - 1], 0.5 * powf(-1.0, N + 1) * N * arr[N].x);
    } else {
      atomicAdd(&uder[n - 1], powf(-1.0, k[tid] + 1) * powf(k[tid], 2) * arr[tid + 1].x
                                / ( (float) N ));
    }
    tid += blockDim.x * gridDim.x;
  }
}

__global__
void ChebFFT_apply_deriv_kernel(int N, int *k, cufftComplex *arr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < 2*N) {
    arr[tid].y = ( (float) k[tid] ) * arr[tid].x;
    arr[tid].x = 0.0;

    tid += blockDim.x * gridDim.x;
  }
}

__global__
void ChebFFT_compute_deriv_kernel(int n, float *uder, cufftComplex *arr, float *x,
                                  float *x_a, float *x_b) {
  int N = n - 1;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < N) {
    if (tid != 0) {
      uder[tid] = -1.0 * arr[tid].x / sqrt(1.0 - x[tid]*x[tid]) / ( (float) (2*N));
    }
    tid += blockDim.x * gridDim.x;
  }

  tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < n) {
    uder[tid] *= -2.0/(*x_b - *x_a);
    tid += blockDim.x * gridDim.x;
  }
}

// This algorithm computes the discrete cosine transform of the input signal, computes
// the derivative by multiplying by `ik' where k is the frequency vector. Then a 
// new transform is applied to bring the differentiated signal to physical space.
template <>
void GPUAcceleratorStencilChebyshev1D<float>::ChebFFT(const float* u, float *uder) const {

    int n = this->get_nrow();
    int N = n - 1;
    dim3 BlockSize(this->local_backend_.GPU_block_size);
    dim3 GridSize(n / this->local_backend_.GPU_block_size + 1);

    // arr_ = [ u; reverse(u(2:N)) ]
    ChebFFT_init_arr_kernel<<< GridSize, BlockSize >>>(n, u, arr_);

    cufftExecC2C(plan, arr_, arr_, CUFFT_FORWARD);
    ChebFFT_sum_bdry_pts_kernel<<< GridSize, BlockSize >>>(n, uder, arr_, k);
    ChebFFT_apply_deriv_kernel<<< GridSize, BlockSize >>>(N, k, arr_);
    cufftExecC2C(plan, arr_, arr_, CUFFT_INVERSE);
    ChebFFT_compute_deriv_kernel<<< GridSize, BlockSize >>>(n, uder, arr_, x,
                                    x_a, x_b);
}

template <>
void GPUAcceleratorStencilChebyshev1D<double>::ChebFFT(const double* u, double *uder) const {

  LOG_INFO("GPUAcceleratorStencilChebyshev1D::ChebFFT not implemented");
  FATAL_ERROR(__FILE__, __LINE__);

}

__global__
void ApplyChebyshev1D_kernel(int n, float *out, const float *u, float *uder,
                              float *u2der, float *p, float *q) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < n) {
    if (tid == 0) {
      out[tid] = u[tid];
    } else if (tid == n - 1) {
      out[tid] = u[tid];
    } else {
      out[tid] = u[tid] - p[tid] * uder[tid] - q[tid]*u2der[tid];
    }
    tid += blockDim.x * gridDim.x;
  }
}

// Computes the action of the operator B(u) = u - pu' - qu''
template <>
void GPUAcceleratorStencilChebyshev1D<float>::Apply(const BaseVector<float> &in, BaseVector<float> *out) const {

  if ((this->ndim_ > 0) && (this->size_ > 0)) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    int nrow = this->get_nrow();
    assert(in.  get_size() == nrow);
    assert(out->get_size() == nrow);
    assert(out->get_size() == in.  get_size());

    // Remove constness because we need to use fftw / cuFFT to access pointer arrays directly
    const GPUAcceleratorVector<float> *cast_in = dynamic_cast<const GPUAcceleratorVector<float>*> (&in);
    GPUAcceleratorVector<float> *cast_out      = dynamic_cast<      GPUAcceleratorVector<float>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    int n = this->get_nrow();
    const float* u = cast_in->GetDataPtr();
    this->ChebFFT(u, uder);
    this->ChebFFT(uder, u2der);

    dim3 BlockSize(this->local_backend_.GPU_block_size);
    dim3 GridSize(n / this->local_backend_.GPU_block_size + 1);
    ApplyChebyshev1D_kernel <<< GridSize, BlockSize >>>(n, cast_out->vec_,
                                                        u, uder, u2der, p, q);
  }

}

template <>
void GPUAcceleratorStencilChebyshev1D<double>::Apply(const BaseVector<double> &in, BaseVector<double> *out) const {

  LOG_INFO("GPUAcceleratorStencilChebyshev1D::Apply not implemented");
  FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void GPUAcceleratorStencilChebyshev1D<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                         BaseVector<ValueType> *out) const {

  LOG_INFO("GPUAcceleratorStencilChebyshev1D::ApplyAdd not implemented");
  FATAL_ERROR(__FILE__, __LINE__);
}


template class GPUAcceleratorStencilChebyshev1D<double>;
template class GPUAcceleratorStencilChebyshev1D<float>;
#ifdef SUPPORT_COMPLEX
//template class GPUAcceleratorStencilChebyshev1D<std::complex<double> >;
//template class GPUAcceleratorStencilChebyshev1D<std::complex<float> >;
#endif

}
