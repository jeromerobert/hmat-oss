/*
  HMat-OSS (HMatrix library, open source software)

  Copyright (C) 2014-2015 Airbus Group SAS

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

  http://github.com/jeromerobert/hmat-oss
*/

#include "config.h"
#include "disable_threading.hpp"

#ifdef HAVE_MKL_H
#include <mkl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef OPENBLAS_DISABLE_THREADS
#include <cblas.h>
extern "C" {
// This function is private in openblas
int  goto_get_num_procs(void);
void openblas_set_num_threads(int num_threads);
}
#endif

namespace hmat {

DisableThreadingInBlock::DisableThreadingInBlock()
  : mklNumThreads(1)
  , ompNumThreads(1)
  , openblasNumThreads(1)
{
#if defined(HAVE_MKL_H)
    mklNumThreads = mkl_get_max_threads();
    mkl_set_num_threads(1);
#endif
#ifdef _OPENMP
    ompNumThreads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
#ifdef OPENBLAS_DISABLE_THREADS
    openblasNumThreads = goto_get_num_procs();
    openblas_set_num_threads(1);
#endif
    // Silence compiler warnings about unused private members
    (void) mklNumThreads;
    (void) ompNumThreads;
    (void) openblasNumThreads;
}

DisableThreadingInBlock::~DisableThreadingInBlock() {
#if defined(HAVE_MKL_H)
    mkl_set_num_threads(mklNumThreads);
#endif
#ifdef _OPENMP
    omp_set_num_threads(ompNumThreads);
#endif
#ifdef OPENBLAS_DISABLE_THREADS
    openblas_set_num_threads(openblasNumThreads);
#endif
}

}  // end namespace hmat
