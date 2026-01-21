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

#include <algorithm>
#include <cstring> // memset
#include <complex>

#include "lapack_operations.hpp"
#include "blas_overloads.hpp"
#include "full_matrix.hpp"
#include "common/context.hpp"
#include "common/my_assert.h"
#include "lapack_overloads.hpp"
#include "blas_overloads.hpp"
#include "lapack_exception.hpp"

using namespace std;

namespace hmat {
template <typename T>
int svdCall(char jobu, char jobv, int m, int n, T *a, int lda,
            typename hmat::Types<T>::real *sigma, T *u, int ldu, T *vt,
            int ldvt) {
  auto superb = new typename hmat::Types<T>::real[min(m, n)];
  int result = proxy_lapack::gesvd(jobu, jobv, m, n, a, lda, sigma, u, ldu, vt,
                                   ldvt, superb);
  if (result != 0)
    throw hmat::LapackException("gesvd", result);
  delete[] superb;
  return result;
}

template <typename T>
int sddCall(char jobz, int m, int n, T *a, int lda,
            typename hmat::Types<T>::real *sigma, T *u, int ldu, T *vt,
            int ldvt) {
  int result = proxy_lapack::gesdd(jobz, m, n, a, lda, sigma, u, ldu, vt, ldvt);
  if (result != 0)
    throw hmat::LapackException("gesdd", result);
  return result;
}

template int svdCall(char jobu, char jobv, int m, int n, S_t *a, int lda,
                     S_t *sigma, S_t *u, int ldu, S_t *vt, int ldvt);
template int svdCall(char jobu, char jobv, int m, int n, D_t *a, int lda,
                     D_t *sigma, D_t *u, int ldu, D_t *vt, int ldvt);
template int svdCall(char jobu, char jobv, int m, int n, C_t *a, int lda,
                     S_t *sigma, C_t *u, int ldu, C_t *vt, int ldvt);
template int svdCall(char jobu, char jobv, int m, int n, Z_t *a, int lda,
                     D_t *sigma, Z_t *u, int ldu, Z_t *vt, int ldvt);

template int sddCall(char jobz, int m, int n, S_t* a, int lda, S_t* sigma, S_t* u, int ldu, S_t* vt, int ldvt);
template int sddCall(char jobz, int m, int n, D_t* a, int lda, D_t* sigma, D_t* u, int ldu, D_t* vt, int ldvt);
template int sddCall(char jobz, int m, int n, C_t* a, int lda, S_t* sigma, C_t* u, int ldu, C_t* vt, int ldvt);
template int sddCall(char jobz, int m, int n, Z_t* a, int lda, D_t* sigma, Z_t* u, int ldu, Z_t* vt, int ldvt);
}  // end namespace hmat
