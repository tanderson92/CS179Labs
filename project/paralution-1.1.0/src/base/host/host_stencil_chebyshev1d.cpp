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
#include "host_vector.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "../stencil_types.hpp"
#include "host_stencil_chebyshev1d.hpp"

#include "fftw3.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <complex>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num);
#endif

namespace paralution {

template <typename ValueType>
HostStencilChebyshev1D<ValueType>::HostStencilChebyshev1D() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HostStencilChebyshev1D<ValueType>::HostStencilChebyshev1D(
                                    const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HostStencilChebyshev1D::HostStencilChebyshev1D()",
            "constructor with local_backend");

  this->set_backend(local_backend);

  this->ndim_ = 1;
}

template <>
void HostStencilChebyshev1D<float>::Plan(float *p_,
                                    float *q_, float *x_a_, float *x_b_) {

  int n = this->get_nrow();
  this->p = p_;
  this->q = q_;
  this->x_a = x_a_;
  this->x_b = x_b_;

  this->arr_ =  (fftwf_complex *) fftwf_malloc( (2*n - 2) * sizeof(fftwf_complex));
  this->plan_fwd_ = fftwf_plan_dft_1d(2*n - 2, arr_, arr_, FFTW_FORWARD, FFTW_MEASURE);
  this->plan_bak_ = fftwf_plan_dft_1d(2*n - 2, arr_, arr_, FFTW_BACKWARD, FFTW_MEASURE);
}

template <>
void HostStencilChebyshev1D<double>::Plan(double *p_,
                                    double *q_, double *x_a_, double *x_b_) {


  LOG_INFO("no double type supported yet");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HostStencilChebyshev1D<ValueType>::~HostStencilChebyshev1D() {

  LOG_DEBUG(this, "HostStencilChebyshev1D::~HostStencilChebyshev1D()",
            "destructor");

  fftwf_destroy_plan ( plan_fwd_ );
  fftwf_destroy_plan ( plan_bak_ );

}

template <typename ValueType>
void HostStencilChebyshev1D<ValueType>::info(void) const {

  LOG_INFO("Chebyshev 1D (Host) size=" << this->size_ << " dim=" << this->get_ndim());

}

template <typename ValueType>
int HostStencilChebyshev1D<ValueType>::get_nnz(void) const {

  return pow(this->get_nrow(), 2);
}

template <typename ValueType>
void HostStencilChebyshev1D<ValueType>::ChebFFT(const ValueType* u, ValueType *uder) const {

    int n = this->get_nrow();
    int N = n - 1;

    // arr_ = [ u; reverse(u(2:N)) ]
    for (int i = 0; i < n; i++) {
      arr_[i][0] = (ValueType) u[i];
      arr_[i][1] = (float) 0.0;
    }
    for (int i = 0; i < N; i++) {
      arr_[n + i][0] = (ValueType) u[(n - 1 - 1) - i];
      arr_[n + i][0] = (float) 0.0;
    }

    fftwf_execute(plan_fwd_);

    int *k = (int *) malloc( (2*n - 2) * sizeof(ValueType));
    {
      for (int i = 0; i < N; i++)
        k[i] = i;
      k[N] = 0;
      for (int i = 0; i < N - 1; i++)
        k[N+1 + i] = 1 - N + i;
    }

    uder[0] = 0.5 * N * arr_[N][0];
    for (int i = 0; i < N - 1; i++) {
      uder[0] += pow(k[i], 2) * arr_[i + 1][0] / ( (float) N );
    }
    uder[0] /= (float) (2*n - 2);
    uder[n - 1] = 0.5 * pow(-1.0, N + 1) * N * arr_[N][0];
    for (int i = 0; i < N - 1; i++) {
      uder[n - 1] += pow(-1.0, k[i] + 1) * pow(k[i], 2) * arr_[i + 1][0]
                  / ( (float) N );
    }
    uder[n - 1] /= (float) (2*n - 2);

    for (int i = 0; i < 2*N; i++) {
      arr_[i][1] = ( (float) k[i]) * arr_[i][0];
      arr_[i][0] = (float) 0.0;
    }

    fftwf_execute(plan_bak_);

    ValueType *x = (ValueType *) malloc(n * sizeof(ValueType));
    for (int i = 0; i < N; i++) {
      x[i] = cos(i * M_PI / ( (float) N));
    }
    for (int i = 1; i < N; i++) {
      uder[i] = -1 * arr_[i][0] / sqrt(1 - x[i]*x[i]) / ( (float) (2*n - 2) );
    }

    for (int i = 0; i < n; i++) {
      uder[i] *= -2.0/(*x_b - *x_a);
    }
    free(x);
    free(k);
}

template <typename ValueType>
void HostStencilChebyshev1D<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if ((this->ndim_ > 0) && (this->size_ > 0)) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    int nrow = this->get_nrow();
    assert(in.  get_size() == nrow);
    assert(out->get_size() == nrow);
    assert(out->get_size() == in.  get_size());

    // Remove constness because we need to use fftw / cuFFT to access pointer arrays directly
    const HostVector<ValueType> *cast_in = dynamic_cast<const HostVector<ValueType>*> (&in);
    HostVector<ValueType> *cast_out      = dynamic_cast<      HostVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    int n = this->get_nrow();
    const ValueType* u = cast_in->GetDataPtr();
    /*for (int i = 0; i < n - 1; i++) {
        printf("i: %d, u[i]: %f\n", i, u[i]);
    }*/
    ValueType* uder  = (ValueType *) malloc( n * sizeof(ValueType) );
    ValueType* u2der = (ValueType *) malloc( n * sizeof(ValueType) );
    this->ChebFFT(u, uder);
    this->ChebFFT(uder, u2der);
    _set_omp_backend_threads(this->local_backend_, nrow);

    cast_out->vec_[0] = u[0];
    for (int i = 1; i < n - 1; i++) {
      cast_out->vec_[i] = u[i] - this->p[i] * uder[i] - this->q[i] * u2der[i];
    }
    cast_out->vec_[n - 1] = u[n - 1];
  }

}

template <typename ValueType>
void HostStencilChebyshev1D<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                         BaseVector<ValueType> *out) const {

  LOG_INFO("HostStencilChebyshev1D::ApplyAdd not implemented");
  FATAL_ERROR(__FILE__, __LINE__);
}


template class HostStencilChebyshev1D<double>;
template class HostStencilChebyshev1D<float>;
#ifdef SUPPORT_COMPLEX
//template class HostStencilChebyshev1D<std::complex<double> >;
//template class HostStencilChebyshev1D<std::complex<float> >;
#endif

}
