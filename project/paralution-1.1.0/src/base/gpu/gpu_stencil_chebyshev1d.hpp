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


#ifndef PARALUTION_GPU_STENCIL_CHEBYSHEV1D_HPP_
#define PARALUTION_GPU_STENCIL_CHEBYSHEV1D_HPP_

#include "../base_vector.hpp"
#include "../base_stencil.hpp"
#include "../stencil_types.hpp"

#include "cufft.h"

namespace paralution {

template <typename ValueType>
class GPUAcceleratorStencilChebyshev1D : public GPUAcceleratorStencil<ValueType> {

public:

  GPUAcceleratorStencilChebyshev1D();
  GPUAcceleratorStencilChebyshev1D(const Paralution_Backend_Descriptor local_backend);
  virtual ~GPUAcceleratorStencilChebyshev1D();

  virtual int get_nnz(void) const;
  virtual void info(void) const;
  virtual unsigned int get_stencil_id(void) const { return Chebyshev1D; }


  virtual void Plan(ValueType *p, ValueType *q, ValueType *x_a, ValueType *x_b);
  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const;

  virtual void CopyFromHost(const HostStencil<ValueType> &src);
  virtual void CopyToHost(HostStencil<ValueType> *dst) const;

private:

  void ChebFFT(const ValueType *u, ValueType *uder) const;

  friend class BaseVector<ValueType>;
  friend class GPUAcceleratorVector<ValueType>;

  mutable cufftHandle plan;
  ValueType *p, *q;
  ValueType *x_a, *x_b;
  ValueType *uder, *u2der;
  ValueType *x;
  int *k;

  mutable cufftComplex* arr_;
  //  friend class GPUAcceleratorStencilChebyshev1D<ValueType>;
  //  friend class OCLAcceleratorStencilChebyshev1D<ValueType>;
  //  friend class MICAcceleratorStencilChebyshev1D<ValueType>;

};


}

#endif // PARALUTION_GPU_STENCIL_CHEBYSHEV1D_HPP_
