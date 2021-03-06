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


#ifndef PARALUTION_OCL_ALLOCATE_FREE_HPP_
#define PARALUTION_OCL_ALLOCATE_FREE_HPP_

#if defined(__APPLE__) && defined(__MACH__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace paralution {

/// Allocate device memory
template <typename DataType>
void allocate_ocl(const int size, cl_context ocl_context, DataType **ptr);

/// Free device memory
template <typename DataType>
void free_ocl(DataType **ptr);

/// Set device object to specific values
template <typename DataType>
void ocl_set_to(const int size, const DataType val, DataType *ptr, cl_command_queue cmdQueue);

/// Copy object from host to device memory
template <typename DataType>
void ocl_host2dev(const int size, const DataType *src, DataType *dst, cl_command_queue ocl_cmdQueue);

/// Copy object from device to host memory
template <typename DataType>
void ocl_dev2host(const int size, const DataType *src, DataType *dst, cl_command_queue ocl_cmdQueue);

/// Copy object from device to device (intra) memory
template <typename DataType>
void ocl_dev2dev(const int size, const DataType *src, DataType *dst, cl_command_queue ocl_cmdQueue);


}

#endif // PARALUTION_OCL_ALLOCATE_FREE_HPP_
