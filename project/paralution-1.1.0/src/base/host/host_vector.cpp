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
#include "../base_vector.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/math_functions.hpp"

#include <typeinfo>
#include <fstream>
#include <math.h>
#include <limits>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_set_nested(num)  ;
#endif

#ifdef SUPPORT_MKL
#include <mkl.h>
#include <mkl_spblas.h>
#endif

namespace paralution {

template <typename ValueType>
HostVector<ValueType>::HostVector() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HostVector<ValueType>::HostVector(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HostVector::HostVector()",
            "constructor with local_backend");

  this->vec_ = NULL;
  this->set_backend(local_backend);

}

template <typename ValueType>
HostVector<ValueType>::~HostVector() {

  LOG_DEBUG(this, "HostVector::~HostVector()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void HostVector<ValueType>::info(void) const {

  LOG_INFO("HostVector<ValueType>, OpenMP threads: " << this->local_backend_.OpenMP_threads);

}

template <typename ValueType>
bool HostVector<ValueType>::Check(void) const {

  bool check = true;

  if (this->size_ > 0) {

    for (int i=0; i<this->size_; ++i)
      if ((paralution_abs(this->vec_[i]) ==
           std::numeric_limits<ValueType>::infinity()) || // inf
          (this->vec_[i] != this->vec_[i])) { // NaN
        LOG_VERBOSE_INFO(2,"*** error: Vector:Check - problems with vector data");
        return false;
      }

    if ((paralution_abs(this->size_) ==
         std::numeric_limits<int>::infinity()) || // inf
        ( this->size_ != this->size_)) { // NaN
      LOG_VERBOSE_INFO(2,"*** error: Vector:Check - problems with vector size");
      return false;
    }

  } else {

    assert(this->size_ == 0);
    assert(this->vec_ == NULL);

  }
  return check;

}

template <typename ValueType>
void HostVector<ValueType>::Allocate(const int n) {

  assert(n >= 0);

  if (this->size_ >0)
    this->Clear();

  if (n > 0) {

    allocate_host(n, &this->vec_);

    set_to_zero_host(n, this->vec_);

    this->size_ = n;

  }

}

template <typename ValueType>
void HostVector<ValueType>::SetDataPtr(ValueType **ptr, const int size) {

  assert(*ptr != NULL);
  assert(size > 0);

  this->Clear();

  this->vec_ = *ptr;
  this->size_ = size;

}

template <typename ValueType>
void HostVector<ValueType>::LeaveDataPtr(ValueType **ptr) {

  assert(this->size_ > 0);

  // see free_host function for details
  *ptr = this->vec_;
  this->vec_ = NULL;

  this->size_ = 0 ;

}

template <typename ValueType>
const ValueType* HostVector<ValueType>::GetDataPtr() const {

  assert(this->size_ > 0);

  return this->vec_;

}

template <typename ValueType>
void HostVector<ValueType>::CopyFromData(const ValueType *data) {

  if (this->size_ > 0) {

    _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
    for (int i=0; i<this->size_; ++i)
      this->vec_[i] = data[i];

  }

}

template <typename ValueType>
void HostVector<ValueType>::CopyToData(ValueType *data) const {

  if (this->size_ > 0) {

    _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
    for (int i=0; i<this->size_; ++i)
      data[i] = this->vec_[i];

  }

}

template <typename ValueType>
void HostVector<ValueType>::Clear(void) {

  if (this->size_ >0) {

    free_host(&this->vec_);

    this->size_ = 0 ;

  }

}

template <typename ValueType>
void HostVector<ValueType>::CopyFrom(const BaseVector<ValueType> &vec) {

  if (this != &vec)  {

    if (const HostVector<ValueType> *cast_vec = dynamic_cast<const HostVector<ValueType>*> (&vec)) {

      if (this->size_ == 0)
        this->Allocate(cast_vec->size_);

      assert(cast_vec->size_ == this->size_);

      _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
      for (int i=0; i<this->size_; ++i)
        this->vec_[i] = cast_vec->vec_[i];

    } else {

      // non-host type

    vec.CopyTo(this);

    }

  }

}

template <typename ValueType>
void HostVector<ValueType>::CopyTo(BaseVector<ValueType> *vec) const {

  vec->CopyFrom(*this);

}

template <typename ValueType>
void HostVector<ValueType>::CopyFromFloat(const BaseVector<float> &vec) {

  LOG_INFO("Mixed precision for non-complex to complex casting is not allowed");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <>
void HostVector<double>::CopyFromFloat(const BaseVector<float> &vec) {

  if (const HostVector<float> *cast_vec = dynamic_cast<const HostVector<float>*> (&vec)) {

    if (this->size_ == 0)
      this->Allocate(cast_vec->size_);

    assert(cast_vec->size_ == this->size_);

    _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
    for (int i=0; i<this->size_; ++i)
      this->vec_[i] = double(cast_vec->vec_[i]);

  } else {

    LOG_INFO("No cross backend casting");
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HostVector<ValueType>::CopyFromDouble(const BaseVector<double> &vec) {

  LOG_INFO("Mixed precision for non-complex to complex casting is not allowed");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <>
void HostVector<float>::CopyFromDouble(const BaseVector<double> &vec) {

  if (const HostVector<double> *cast_vec = dynamic_cast<const HostVector<double>*> (&vec)) {

    if (this->size_ == 0)
      this->Allocate(cast_vec->size_);

    assert(cast_vec->size_ == this->size_);

    _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
    for (int i=0; i<this->size_; ++i)
      this->vec_[i] = float(cast_vec->vec_[i]);

  } else {

    LOG_INFO("No cross backend casting");
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HostVector<ValueType>::Zeros(void) {

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = ValueType(0.0);

}

template <typename ValueType>
void HostVector<ValueType>::Ones(void) {

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = ValueType(1.0);

}

template <typename ValueType>
void HostVector<ValueType>::SetValues(const ValueType val) {

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = val;

}

template <typename ValueType>
void HostVector<ValueType>::SetRandom(const ValueType a, const ValueType b, const int seed) {

  _set_omp_backend_threads(this->local_backend_, this->size_);

  // Fill this with random data from interval [a,b]
  srand(seed);
#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = a + (ValueType)rand() / (ValueType)RAND_MAX * (b - a);

}

template <>
void HostVector<std::complex<float> >::SetRandom(const std::complex<float> a, const std::complex<float> b,
                                                 const int seed) {

  _set_omp_backend_threads(this->local_backend_, this->size_);

  // Fill this with random data from interval [a,b]
  srand(seed);
#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = a + (float)rand() / (float)RAND_MAX * (b - a);

}

template <typename ValueType>
void HostVector<ValueType>::ReadFileASCII(const std::string filename) {

  std::ifstream file;
  std::string line;
  int n = 0;

  LOG_INFO("ReadFileASCII: filename="<< filename << "; reading...");

  file.open((char*)filename.c_str(), std::ifstream::in);

  if (!file.is_open()) {
    LOG_INFO("Can not open vector file [read]:" << filename);
    FATAL_ERROR(__FILE__, __LINE__);
  }

  this->Clear();

  // get the size of the vector
  while (std::getline(file, line))
    ++n;

  this->Allocate(n);

  file.clear();
  file.seekg(0, std::ios_base::beg);

  for (int i=0; i<n; ++i)
    file >> this->vec_[i];

  file.close();

  LOG_INFO("ReadFileASCII: filename="<< filename << "; done");

}

template <typename ValueType>
void HostVector<ValueType>::WriteFileASCII(const std::string filename) const {

  std::ofstream file;
  std::string line;

  LOG_INFO("WriteFileASCII: filename="<< filename << "; writing...");

  file.open((char*)filename.c_str(), std::ifstream::out);

  if (!file.is_open()) {
    LOG_INFO("Can not open vector file [write]:" << filename);
    FATAL_ERROR(__FILE__, __LINE__);
  }

  file.setf(std::ios::scientific);

  for (int n=0; n<this->size_; n++)
    file << this->vec_[n] << std::endl;

  file.close();

  LOG_INFO("WriteFileASCII: filename="<< filename << "; done");

}

template <typename ValueType>
void HostVector<ValueType>::ReadFileBinary(const std::string filename) {

  LOG_INFO("ReadFileBinary: filename="<< filename << "; reading...");

  std::ifstream out(filename.c_str(), std::ios::in | std::ios::binary);

  this->Clear();

  int n;
  out.read((char*)&n, sizeof(int));

  this->Allocate(n);

  out.read((char*)this->vec_, n*sizeof(ValueType));

  LOG_INFO("ReadFileBinary: filename="<< filename << "; done");

  out.close();

}

template <typename ValueType>
void HostVector<ValueType>::WriteFileBinary(const std::string filename) const {

  LOG_INFO("WriteFileBinary: filename="<< filename << "; writing...");

  std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);

  out.write((char*)&this->size_, sizeof(int));
  out.write((char*)this->vec_, this->size_*sizeof(ValueType));

  out.close();

  LOG_INFO("WriteFileBinary: filename="<< filename << "; done");

}

#ifdef SUPPORT_MKL

template <>
void HostVector<double>::AddScale(const BaseVector<double> &x, const double alpha) {

  assert(&x != NULL);

  const HostVector<double> *cast_x = dynamic_cast<const HostVector<double>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  cblas_daxpy(this->size_, alpha, cast_x->vec_, 1, this->vec_, 1);

}

template <>
void HostVector<float>::AddScale(const BaseVector<float> &x, const float alpha) {

  assert(&x != NULL);

  const HostVector<float> *cast_x = dynamic_cast<const HostVector<float>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  cblas_saxpy(this->size_, alpha, cast_x->vec_, 1, this->vec_, 1);

}

template <>
void HostVector<std::complex<double> >::AddScale(const BaseVector<std::complex<double> > &x, const std::complex<double> alpha) {

  assert(&x != NULL);

  const HostVector<std::complex<double> > *cast_x = dynamic_cast<const HostVector<std::complex<double> >*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  cblas_zaxpy(this->size_, &alpha, cast_x->vec_, 1, this->vec_, 1);

}

template <>
void HostVector<std::complex<float> >::AddScale(const BaseVector<std::complex<float> > &x, const std::complex<float> alpha) {

  assert(&x != NULL);

  const HostVector<std::complex<float> > *cast_x = dynamic_cast<const HostVector<std::complex<float> >*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  cblas_caxpy(this->size_, &alpha, cast_x->vec_, 1, this->vec_, 1);

}

template <>
void HostVector<int>::AddScale(const BaseVector<int> &x, const int alpha) {

  assert(&x != NULL);

  const HostVector<int> *cast_x = dynamic_cast<const HostVector<int>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  LOG_INFO("No int MKL axpy function");
  FATAL_ERROR(__FILE__, __LINE__);

}

#else

template <typename ValueType>
void HostVector<ValueType>::AddScale(const BaseVector<ValueType> &x, const ValueType alpha) {

  assert(&x != NULL);

  const HostVector<ValueType> *cast_x = dynamic_cast<const HostVector<ValueType>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = this->vec_[i] + alpha*cast_x->vec_[i];

}

#endif

template <typename ValueType>
void HostVector<ValueType>::ScaleAdd(const ValueType alpha, const BaseVector<ValueType> &x) {

  assert(&x != NULL);

  const HostVector<ValueType> *cast_x = dynamic_cast<const HostVector<ValueType>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = alpha*this->vec_[i] + cast_x->vec_[i];

}

template <typename ValueType>
void HostVector<ValueType>::ScaleAddScale(const ValueType alpha, const BaseVector<ValueType> &x, const ValueType beta) {

  assert(&x != NULL);

  const HostVector<ValueType> *cast_x = dynamic_cast<const HostVector<ValueType>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = alpha*this->vec_[i] + beta*cast_x->vec_[i];

}

template <typename ValueType>
void HostVector<ValueType>::ScaleAddScale(const ValueType alpha, const BaseVector<ValueType> &x, const ValueType beta,
                                          const int src_offset, const int dst_offset,const int size) {

  assert(&x != NULL);

  const HostVector<ValueType> *cast_x = dynamic_cast<const HostVector<ValueType>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_   > 0);
  assert(cast_x->size_ > 0);
  assert(size          > 0);
  assert(src_offset + size <= cast_x->size_);
  assert(dst_offset + size <= this->size_);

  _set_omp_backend_threads(this->local_backend_, size);

#pragma omp parallel for
  for (int i=0; i<size; ++i)
    this->vec_[i+dst_offset] = alpha*this->vec_[i+dst_offset] + beta*cast_x->vec_[i+src_offset];

}


template <typename ValueType>
void HostVector<ValueType>::ScaleAdd2(const ValueType alpha, const BaseVector<ValueType> &x, const ValueType beta, const BaseVector<ValueType> &y, const ValueType gamma) {

  assert(&x != NULL);
  assert(&y != NULL);

  const HostVector<ValueType> *cast_x = dynamic_cast<const HostVector<ValueType>*> (&x);
  const HostVector<ValueType> *cast_y = dynamic_cast<const HostVector<ValueType>*> (&y);

  assert(cast_x != NULL);
  assert(cast_y != NULL);
  assert(this->size_ == cast_x->size_);
  assert(this->size_ == cast_y->size_);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = alpha*this->vec_[i] + beta*cast_x->vec_[i] + gamma*cast_y->vec_[i];

}

#ifdef SUPPORT_MKL

template <>
void HostVector<double>::Scale(const double alpha) {

  cblas_dscal(this->size_, alpha, this->vec_, 1);

}

template <>
void HostVector<float>::Scale(const float alpha) {

  cblas_sscal(this->size_, alpha, this->vec_, 1);

}

template <>
void HostVector<std::complex<double> >::Scale(const std::complex<double> alpha) {

  cblas_zscal(this->size_, &alpha, this->vec_, 1);

}

template <>
void HostVector<std::complex<float> >::Scale(const std::complex<float> alpha) {

  cblas_cscal(this->size_, &alpha, this->vec_, 1);

}

template <>
void HostVector<int>::Scale(const int alpha) {

  LOG_INFO("No int MKL scale function");
  FATAL_ERROR(__FILE__, __LINE__);  

}

#else

template <typename ValueType>
void HostVector<ValueType>::Scale(const ValueType alpha) {

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] *= alpha;

}

#endif

template <typename ValueType>
void HostVector<ValueType>::ExclusiveScan(const BaseVector<ValueType> &x) {

  assert(&x != NULL);

  const HostVector<ValueType> *cast_x = dynamic_cast<const HostVector<ValueType>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  this->vec_[0] = cast_x->vec_[0];

#pragma omp parallel for
  for (int i=1; i<this->size_; ++i)
    this->vec_[i] = cast_x->vec_[i] + this->vec_[i-1];

}

#ifdef SUPPORT_MKL

template <>
double HostVector<double>::Dot(const BaseVector<double> &x) const {

  assert(&x != NULL);

  const HostVector<double> *cast_x = dynamic_cast<const HostVector<double>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  return cblas_ddot(this->size_, this->vec_, 1, cast_x->vec_, 1);

}

template <>
float HostVector<float>::Dot(const BaseVector<float> &x) const {

  assert(&x != NULL);

  const HostVector<float> *cast_x = dynamic_cast<const HostVector<float>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  return cblas_sdot(this->size_, this->vec_, 1, cast_x->vec_, 1);

}

template <>
std::complex<double> HostVector<std::complex<double> >::Dot(const BaseVector<std::complex<double> > &x) const {

  assert(&x != NULL);

  const HostVector<std::complex<double> > *cast_x = dynamic_cast<const HostVector<std::complex<double> >*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  std::complex<double> res;

  cblas_zdotc_sub(this->size_, this->vec_, 1, cast_x->vec_, 1, &res);

  return res;

}

template <>
std::complex<float> HostVector<std::complex<float> >::Dot(const BaseVector<std::complex<float> > &x) const {

  assert(&x != NULL);

  const HostVector<std::complex<float> > *cast_x = dynamic_cast<const HostVector<std::complex<float> >*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  std::complex<float> res;

  cblas_cdotc_sub(this->size_, this->vec_, 1, cast_x->vec_, 1, &res);

  return res;

}

template <>
int HostVector<int>::Dot(const BaseVector<int> &x) const {

  assert(&x != NULL);

  const HostVector<int> *cast_x = dynamic_cast<const HostVector<int>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  LOG_INFO("No int MKL dot function");
  FATAL_ERROR(__FILE__, __LINE__);

}

#else

template <typename ValueType>
ValueType HostVector<ValueType>::Dot(const BaseVector<ValueType> &x) const {

  assert(&x != NULL);

  const HostVector<ValueType> *cast_x = dynamic_cast<const HostVector<ValueType>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  ValueType dot = ValueType(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:dot)
  for (int i=0; i<this->size_; ++i)
    dot += this->vec_[i]*cast_x->vec_[i];

  return dot;

}

template <>
std::complex<float> HostVector<std::complex<float> >::Dot(const BaseVector<std::complex<float> > &x) const {

  assert(&x != NULL);

  const HostVector<std::complex<float> > *cast_x = dynamic_cast<const HostVector<std::complex<float> >*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  float dot_real = float(0.0);
  float dot_imag = float(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:dot_real, dot_imag)
  for (int i=0; i<this->size_; ++i) {

    dot_real += this->vec_[i].real() * cast_x->vec_[i].real() + this->vec_[i].imag() * cast_x->vec_[i].imag();
    dot_imag += this->vec_[i].real() * cast_x->vec_[i].imag() - this->vec_[i].imag() * cast_x->vec_[i].real();

  }

  return std::complex<float>(dot_real, dot_imag);

}

template <>
std::complex<double> HostVector<std::complex<double> >::Dot(const BaseVector<std::complex<double> > &x) const {

  assert(&x != NULL);

  const HostVector<std::complex<double> > *cast_x = dynamic_cast<const HostVector<std::complex<double> >*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  double dot_real = double(0.0);
  double dot_imag = double(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:dot_real, dot_imag)
  for (int i=0; i<this->size_; ++i) {

    dot_real += this->vec_[i].real() * cast_x->vec_[i].real() + this->vec_[i].imag() * cast_x->vec_[i].imag();
    dot_imag += this->vec_[i].real() * cast_x->vec_[i].imag() - this->vec_[i].imag() * cast_x->vec_[i].real();

  }

  return std::complex<double>(dot_real, dot_imag);

}

#endif

template <typename ValueType>
ValueType HostVector<ValueType>::DotNonConj(const BaseVector<ValueType> &x) const {

  return this->Dot(x);

}

#ifdef SUPPORT_MKL

template <>
std::complex<double> HostVector<std::complex<double> >::DotNonConj(const BaseVector<std::complex<double> > &x) const {

  assert(&x != NULL);

  const HostVector<std::complex<double> > *cast_x = dynamic_cast<const HostVector<std::complex<double> >*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  std::complex<double> res;

  cblas_zdotu_sub(this->size_, this->vec_, 1, cast_x->vec_, 1, &res);

  return res;

}

template <>
std::complex<float> HostVector<std::complex<float> >::DotNonConj(const BaseVector<std::complex<float> > &x) const {

  assert(&x != NULL);

  const HostVector<std::complex<float> > *cast_x = dynamic_cast<const HostVector<std::complex<float> >*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  std::complex<float> res;

  cblas_cdotu_sub(this->size_, this->vec_, 1, cast_x->vec_, 1, &res);

  return res;

}

#else

template <>
std::complex<float> HostVector<std::complex<float> >::DotNonConj(const BaseVector<std::complex<float> > &x) const {

  assert(&x != NULL);

  const HostVector<std::complex<float> > *cast_x = dynamic_cast<const HostVector<std::complex<float> >*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  float dot_real = float(0.0);
  float dot_imag = float(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:dot_real, dot_imag)
  for (int i=0; i<this->size_; ++i) {

    dot_real += this->vec_[i].real() * cast_x->vec_[i].real() - this->vec_[i].imag() * cast_x->vec_[i].imag();
    dot_imag += this->vec_[i].real() * cast_x->vec_[i].imag() + this->vec_[i].imag() * cast_x->vec_[i].real();

  }

  return std::complex<float>(dot_real, dot_imag);

}

template <>
std::complex<double> HostVector<std::complex<double> >::DotNonConj(const BaseVector<std::complex<double> > &x) const {

  assert(&x != NULL);

  const HostVector<std::complex<double> > *cast_x = dynamic_cast<const HostVector<std::complex<double> >*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  double dot_real = double(0.0);
  double dot_imag = double(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:dot_real, dot_imag)
  for (int i=0; i<this->size_; ++i) {

    dot_real += this->vec_[i].real() * cast_x->vec_[i].real() - this->vec_[i].imag() * cast_x->vec_[i].imag();
    dot_imag += this->vec_[i].real() * cast_x->vec_[i].imag() + this->vec_[i].imag() * cast_x->vec_[i].real();

  }

  return std::complex<double>(dot_real, dot_imag);

}

#endif

template <typename ValueType>
ValueType HostVector<ValueType>::Asum(void) const {

  ValueType asum = ValueType(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:asum)
  for (int i=0; i<this->size_; ++i)
    asum += paralution_abs(this->vec_[i]);

  return asum;

}

template <>
std::complex<float> HostVector<std::complex<float> >::Asum(void) const {

  float asum_real = float(0.0);
  float asum_imag = float(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:asum_real, asum_imag)
  for (int i=0; i<this->size_; ++i) {

    asum_real += paralution_abs(this->vec_[i].real());
    asum_imag += paralution_abs(this->vec_[i].imag());

  }

  return std::complex<float>(asum_real, asum_imag);

}

template <>
std::complex<double> HostVector<std::complex<double> >::Asum(void) const {

  double asum_real = double(0.0);
  double asum_imag = double(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:asum_real, asum_imag)
  for (int i=0; i<this->size_; ++i) {

    asum_real += paralution_abs(this->vec_[i].real());
    asum_imag += paralution_abs(this->vec_[i].imag());

  }

  return std::complex<double>(asum_real, asum_imag);

}

template <typename ValueType>
int HostVector<ValueType>::Amax(ValueType &value) const {

  int index = 0;
  value = ValueType(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i) {
    ValueType val = paralution_abs(this->vec_[i]);
    if (val > value)
#pragma omp critical
{
      if (val > value) {
        value = val;
        index = i;
      }
}
  }

  return index;

}

#ifdef SUPPORT_MKL

template <>
double HostVector<double>::Norm(void) const {

  return cblas_dnrm2(this->size_, this->vec_, 1);

}

template <>
float HostVector<float>::Norm(void) const {

  return cblas_snrm2(this->size_, this->vec_, 1);

}

template <>
std::complex<double> HostVector<std::complex<double> >::Norm(void) const {

  return cblas_dznrm2(this->size_, this->vec_, 1);

}

template <>
std::complex<float> HostVector<std::complex<float> >::Norm(void) const {

  return cblas_scnrm2(this->size_, this->vec_, 1);

}

#else 

template <typename ValueType>
ValueType HostVector<ValueType>::Norm(void) const {

  ValueType norm2 = ValueType(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:norm2)
  for (int i=0; i<this->size_; ++i)
    norm2 += this->vec_[i] * this->vec_[i];

  return sqrt(norm2);

}

template <>
std::complex<float> HostVector<std::complex<float> >::Norm(void) const {

  float norm2 = float(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:norm2)
  for (int i=0; i<this->size_; ++i)
    norm2 += this->vec_[i].real() * this->vec_[i].real() + this->vec_[i].imag() * this->vec_[i].imag();

  std::complex<float> res(sqrt(norm2), float(0.0));

  return res;

}

template <>
std::complex<double> HostVector<std::complex<double> >::Norm(void) const {

  double norm2 = double(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:norm2)
  for (int i=0; i<this->size_; ++i)
    norm2 += this->vec_[i].real() * this->vec_[i].real() + this->vec_[i].imag() * this->vec_[i].imag();

  std::complex<double> res(sqrt(norm2), double(0.0));

  return res;

}

#endif

template <>
int HostVector<int>::Norm(void) const {

  LOG_INFO("What is int HostVector<ValueType>::Norm(void) const?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
ValueType HostVector<ValueType>::Reduce(void) const {

  ValueType reduce = ValueType(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:reduce)
  for (int i=0; i<this->size_; ++i)
    reduce += this->vec_[i];

  return reduce;

}

template <>
std::complex<float> HostVector<std::complex<float> >::Reduce(void) const {

  float reduce_real = float(0.0);
  float reduce_imag = float(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:reduce_real, reduce_imag)
  for (int i=0; i<this->size_; ++i) {

    reduce_real += this->vec_[i].real();
    reduce_imag += this->vec_[i].imag();

  }

  return  std::complex<float>(reduce_real, reduce_imag);

}

template <>
std::complex<double> HostVector<std::complex<double> >::Reduce(void) const {

  double reduce_real = double(0.0);
  double reduce_imag = double(0.0);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for reduction(+:reduce_real, reduce_imag)
  for (int i=0; i<this->size_; ++i) {

    reduce_real += this->vec_[i].real();
    reduce_imag += this->vec_[i].imag();

  }

  return  std::complex<double>(reduce_real, reduce_imag);

}

template <typename ValueType>
void HostVector<ValueType>::PointWiseMult(const BaseVector<ValueType> &x) {

  assert(&x != NULL);

  const HostVector<ValueType> *cast_x = dynamic_cast<const HostVector<ValueType>*> (&x);

  assert(cast_x != NULL);
  assert(this->size_ == cast_x->size_);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = this->vec_[i]*cast_x->vec_[i];

}

template <typename ValueType>
void HostVector<ValueType>::PointWiseMult(const BaseVector<ValueType> &x, const BaseVector<ValueType> &y) {

  assert(&x != NULL);
  assert(&y != NULL);

  const HostVector<ValueType> *cast_x = dynamic_cast<const HostVector<ValueType>*> (&x);
  const HostVector<ValueType> *cast_y = dynamic_cast<const HostVector<ValueType>*> (&y);

  assert(cast_x != NULL);
  assert(cast_y != NULL);
  assert(this->size_ == cast_x->size_);
  assert(this->size_ == cast_y->size_);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = cast_y->vec_[i]*cast_x->vec_[i];

}

template <typename ValueType>
void HostVector<ValueType>::CopyFrom(const BaseVector<ValueType> &src,
                                     const int src_offset,
                                     const int dst_offset,
                                     const int size) {

  assert(&src != NULL);

  const HostVector<ValueType> *cast_src = dynamic_cast<const HostVector<ValueType>*> (&src);

  assert(cast_src != NULL);
  //TOOD check always for == this?
  assert(&src != this);
  assert(this->size_     > 0);
  assert(cast_src->size_ > 0);
  assert(size            > 0);
  assert(src_offset + size <= cast_src->size_);
  assert(dst_offset + size <= this->size_);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<size; ++i)
    this->vec_[i+dst_offset] = cast_src->vec_[i+src_offset];

}

template <typename ValueType>
void HostVector<ValueType>::Permute(const BaseVector<int> &permutation) {
  
  assert(&permutation != NULL);

  const HostVector<int> *cast_perm = dynamic_cast<const HostVector<int>*> (&permutation);

  assert(cast_perm != NULL);
  assert(this->size_ == cast_perm->size_);

  HostVector<ValueType> vec_tmp(this->local_backend_);
  vec_tmp.Allocate(this->size_);
  vec_tmp.CopyFrom(*this);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i) {
    assert_dbg(cast_perm->vec_[i] >= 0);
    assert_dbg(cast_perm->vec_[i] < this->size_);
    this->vec_[ cast_perm->vec_[i] ] = vec_tmp.vec_[i];
  }

}

template <typename ValueType>
void HostVector<ValueType>::PermuteBackward(const BaseVector<int> &permutation) {

  assert(&permutation != NULL);

  const HostVector<int> *cast_perm = dynamic_cast<const HostVector<int>*> (&permutation);

  assert(cast_perm != NULL);
  assert(this->size_ == cast_perm->size_);

  HostVector<ValueType> vec_tmp(this->local_backend_);
  vec_tmp.Allocate(this->size_);
  vec_tmp.CopyFrom(*this);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i) {
    assert_dbg(cast_perm->vec_[i] >= 0);
    assert_dbg(cast_perm->vec_[i] < this->size_);
    this->vec_[i] = vec_tmp.vec_[ cast_perm->vec_[i] ];
  }
}

template <typename ValueType>
void HostVector<ValueType>::CopyFromPermute(const BaseVector<ValueType> &src,
                                            const BaseVector<int> &permutation) {

  assert(this != &src);

  const HostVector<ValueType> *cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src);
  const HostVector<int> *cast_perm      = dynamic_cast<const HostVector<int>*> (&permutation);
  assert(cast_perm != NULL);
  assert(cast_vec  != NULL);

  assert(cast_vec ->size_ == this->size_);
  assert(cast_perm->size_ == this->size_);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[ cast_perm->vec_[i] ] = cast_vec->vec_[i];

}

template <typename ValueType>
void HostVector<ValueType>::CopyFromPermuteBackward(const BaseVector<ValueType> &src,
                                                    const BaseVector<int> &permutation) {

  assert(this != &src);

  const HostVector<ValueType> *cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src);
  const HostVector<int> *cast_perm      = dynamic_cast<const HostVector<int>*> (&permutation);
  assert(cast_perm != NULL);
  assert(cast_vec  != NULL);

  assert(cast_vec ->size_ == this->size_);
  assert(cast_perm->size_ == this->size_);

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = cast_vec->vec_[ cast_perm->vec_[i] ];

}

template <typename ValueType>
bool HostVector<ValueType>::Restriction(const BaseVector<ValueType> &vec_fine, const BaseVector<int> &map) {

  assert(this != &vec_fine);
  
  const HostVector<ValueType> *cast_vec = dynamic_cast<const HostVector<ValueType>*> (&vec_fine);
  const HostVector<int> *cast_map       = dynamic_cast<const HostVector<int>*> (&map);
  assert(cast_map != NULL);
  assert(cast_vec != NULL);
  assert(cast_map->size_ == cast_vec->size_);

  this->Zeros();

  for (int i=0; i<cast_vec->size_; ++i)
    if (cast_map->vec_[i] != -1)
      this->vec_[cast_map->vec_[i]] += cast_vec->vec_[i];

  return true;

}

template <typename ValueType>
bool HostVector<ValueType>::Prolongation(const BaseVector<ValueType> &vec_coarse, const BaseVector<int> &map) {

  assert(this != &vec_coarse);
  
  const HostVector<ValueType> *cast_vec = dynamic_cast<const HostVector<ValueType>*> (&vec_coarse);
  const HostVector<int> *cast_map       = dynamic_cast<const HostVector<int>*> (&map);
  assert(cast_map != NULL);
  assert(cast_vec != NULL);
  assert(cast_map->size_ == this->size_);
  
  for (int i=0; i<this->size_; ++i)
    if (cast_map->vec_[i] != -1)
      this->vec_[i] += cast_vec->vec_[cast_map->vec_[i]];

  return true;

}


template <typename ValueType>
void HostVector<ValueType>::Assemble(const int *ii, const ValueType *v,
                                     int size, const int n) {

  assert(ii != NULL);
  assert(v != NULL);
  assert(size > 0);
  assert(n >= 0);

  _set_omp_backend_threads(this->local_backend_, this->size_);
  const int nThreads = omp_get_max_threads();

  int N = n;

  if (N == 0) {

#pragma omp parallel for
    for (int i=0; i<size; ++i) {

      assert(ii[i] >= 0);

      int val = ii[i]+1;

      if (val > N) {
#pragma omp critical
{
        N = val;
}
      }

    }

    this->Clear();
    this->Allocate(N);

  }

  if (nThreads <= 2) {

    // serial
    for (int i=0; i<size; ++i)
      this->vec_[ ii[i] ] += v[i];

  } else {

    // parallel
    ValueType **v_red;

    v_red = (ValueType **) malloc(nThreads*sizeof(ValueType*));

    for (int k=0; k<nThreads; ++k) {

      v_red[k] = NULL;
      allocate_host(N, &v_red[k]);
      set_to_zero_host(N, v_red[k]);

    }

#pragma omp parallel
{
    const int me = omp_get_thread_num();
    const int istart = size*me/nThreads;
    const int iend = size*(me+1)/nThreads;

    for (int i = istart; i < iend; i++)
      v_red[me][ii[i]] += v[i];
}

#pragma omp parallel
{
    const int me = omp_get_thread_num();
    const int istart = N*me/nThreads;
    const int iend = N*(me+1)/nThreads;

    for (int i = istart; i < iend; ++i)
      for (int k=0; k<nThreads; ++k)
        this->vec_[i] += v_red[k][i];

}

    for (int k=0; k<nThreads; ++k)
      free_host(&v_red[k]);

    free(v_red);

  }

}

template <typename ValueType>
void HostVector<ValueType>::Power(const double power) {

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = pow(this->vec_[i], ValueType(power));

}

template <>
void HostVector<std::complex<float> >::Power(const double power) {

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i)
    this->vec_[i] = pow(this->vec_[i], std::complex<float>(float(power)));

}

template <>
void HostVector<int>::Power(const double power) {

  _set_omp_backend_threads(this->local_backend_, this->size_);

#pragma omp parallel for
  for (int i=0; i<this->size_; ++i) {

    int value = 1;
    for (int j=0; j<power; ++j)
      value *= this->vec_[i];
    
    this->vec_[i] = value;
  }

}


template class HostVector<double>;
template class HostVector<float>;
#ifdef SUPPORT_COMPLEX
template class HostVector<std::complex<double> >;
template class HostVector<std::complex<float> >;
#endif

template class HostVector<int>;

}
