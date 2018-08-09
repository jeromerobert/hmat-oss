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

// Implementation

template<> int svdCall<hmat::S_t>(char jobu, char jobv, int m, int n, hmat::S_t* a,
                            int lda, double* sigma, hmat::S_t* u, int ldu, hmat::S_t* vt,
                            int ldvt) {
  int result;
  int p = min(m, n);
  float* sigmaFloat = new float[p];
  int workSize;
  hmat::S_t workSize_S;

  result = proxy_lapack::gesvd(jobu, jobv, m, n, a, lda, sigmaFloat, u, ldu, vt, ldvt, &workSize_S, -1);
  if(result != 0)
      throw hmat::LapackException("gesvd", result);
  workSize = (int) workSize_S + 1;
  hmat::S_t* work = new hmat::S_t[workSize];
  HMAT_ASSERT(work) ;
  result = proxy_lapack::gesvd(jobu, jobv, m, n, a, lda, sigmaFloat, u, ldu, vt, ldvt, work, workSize);
  if(result != 0)
      throw hmat::LapackException("gesvd", result);
  delete[] work;

  for (int i = 0; i < p; i++) {
    sigma[i] = sigmaFloat[i];
  }
  delete[] sigmaFloat;
  return result;
}
template<> int svdCall<hmat::D_t>(char jobu, char jobv, int m, int n, hmat::D_t* a,
                            int lda, double* sigma, hmat::D_t* u, int ldu, hmat::D_t* vt,
                            int ldvt) {
  int workSize;
  hmat::D_t workSize_D;
  int result;

  // We request the right size for WORK
  result = proxy_lapack::gesvd(jobu, jobv, m, n, a, lda, sigma, u, ldu, vt, ldvt, &workSize_D, -1);
  if(result != 0)
      throw hmat::LapackException("gesvd", result);
  workSize = (int) workSize_D + 1;
  hmat::D_t* work = new hmat::D_t[workSize];
  HMAT_ASSERT(work) ;
  result = proxy_lapack::gesvd(jobu, jobv, m, n, a, lda, sigma, u, ldu, vt, ldvt, work, workSize);
  if(result != 0)
      throw hmat::LapackException("gesvd", result);
  delete[] work;
  return result;
}
template<> int svdCall<hmat::C_t>(char jobu, char jobv, int m, int n, hmat::C_t* a,
                            int lda, double* sigma, hmat::C_t* u, int ldu, hmat::C_t* vt,
                            int ldvt) {
  int result;
  int workSize;
  hmat::C_t workSize_C;
  int p = min(m, n);
  float* sigmaFloat = new float[p];

  // We request the right size for WORK
  result = proxy_lapack::gesvd(jobu, jobv, m, n, a, lda, sigmaFloat, u, ldu, vt, ldvt, &workSize_C, -1);
  if(result != 0)
      throw hmat::LapackException("gesvd", result);
  workSize = (int) workSize_C.real() + 1;
  hmat::C_t* work = new hmat::C_t[workSize];
  HMAT_ASSERT(work) ;
  result = proxy_lapack::gesvd(jobu, jobv, m, n, a, lda, sigmaFloat, u, ldu, vt, ldvt, work, workSize);
  if(result != 0)
      throw hmat::LapackException("gesvd", result);
  delete[] work;

  for (int i = 0; i < p; i++) {
    sigma[i] = sigmaFloat[i];
  }
  delete[] sigmaFloat;
  return result;
}
template<> int svdCall<hmat::Z_t>(char jobu, char jobv, int m, int n, hmat::Z_t* a,
                            int lda, double* sigma, hmat::Z_t* u, int ldu, hmat::Z_t* vt,
                            int ldvt) {
  int result;
  int workSize;
  hmat::Z_t workSize_Z;

  // We request the right size for WORK
  result = proxy_lapack::gesvd(jobu, jobv, m, n, a, lda, sigma, u, ldu, vt, ldvt, &workSize_Z, -1);
  if(result != 0)
      throw hmat::LapackException("gesvd", result);
  workSize = (int) workSize_Z.real() + 1;
  hmat::Z_t* work = new hmat::Z_t[workSize];
  HMAT_ASSERT(work) ;
  result = proxy_lapack::gesvd(jobu, jobv, m, n, a, lda, sigma, u, ldu, vt, ldvt, work, workSize);
  if(result != 0)
      throw hmat::LapackException("gesvd", result);
  delete[] work;
  return result;
}

// Implementation
template<> int sddCall<hmat::S_t>(char jobz, int m, int n, hmat::S_t* a, int lda,
                            double* sigma, hmat::S_t* u, int ldu, hmat::S_t* vt, int ldvt) {
  int result;
  int p = min(m, n);
  float* sigmaFloat = new float[p];
  int workSize;
  hmat::S_t workSize_S;
  int* iwork = new int[8*p];

  result = proxy_lapack::gesdd(jobz, m, n, a, lda, sigmaFloat, u, ldu, vt, ldvt, &workSize_S, -1, iwork);
  HMAT_ASSERT(!result);
  workSize = (int) workSize_S + 1;
  hmat::S_t* work = new hmat::S_t[workSize];
  HMAT_ASSERT(work) ;
  result = proxy_lapack::gesdd(jobz, m, n, a, lda, sigmaFloat, u, ldu, vt, ldvt, work, workSize, iwork);
  HMAT_ASSERT(!result);
  delete[] work;
  delete[] iwork;

  for (int i = 0; i < p; i++) {
    sigma[i] = sigmaFloat[i];
  }
  delete[] sigmaFloat;
  return result;
}

template<> int sddCall<hmat::D_t>(char jobz, int m, int n, hmat::D_t* a, int lda,
                            double* sigma, hmat::D_t* u, int ldu, hmat::D_t* vt, int ldvt) {
  int workSize;
  hmat::D_t workSize_D;
  int result;
  int* iwork = new int[8*min(m, n)];

  // We request the right size for WORK
  result = proxy_lapack::gesdd(jobz, m, n, a, lda, sigma, u, ldu, vt, ldvt, &workSize_D, -1, iwork);
  HMAT_ASSERT(!result);
  workSize = (int) workSize_D + 1;
  hmat::D_t* work = new hmat::D_t[workSize];
  HMAT_ASSERT(work) ;
  result = proxy_lapack::gesdd(jobz, m, n, a, lda, sigma, u, ldu, vt, ldvt, work, workSize, iwork);
  HMAT_ASSERT(!result);
  delete[] work;
  delete[] iwork;
  return result;
}

template<> int sddCall<hmat::C_t>(char jobz, int m, int n, hmat::C_t* a, int lda,
                            double* sigma, hmat::C_t* u, int ldu, hmat::C_t* vt, int ldvt) {
  int result;
  int workSize;
  hmat::C_t workSize_C;
  int p = min(m, n);
  float* sigmaFloat = new float[p];
  int* iwork = new int[8*p];

  // We request the right size for WORK
  result = proxy_lapack::gesdd(jobz, m, n, a, lda, sigmaFloat, u, ldu, vt, ldvt, &workSize_C, -1, iwork);
  HMAT_ASSERT(!result);
  workSize = (int) workSize_C.real() + 1;
  hmat::C_t* work = new hmat::C_t[workSize];
  HMAT_ASSERT(work) ;
  result = proxy_lapack::gesdd(jobz, m, n, a, lda, sigmaFloat, u, ldu, vt, ldvt, work, workSize, iwork);
  HMAT_ASSERT(!result);
  delete[] work;
  delete[] iwork;

  for (int i = 0; i < p; i++) {
    sigma[i] = sigmaFloat[i];
  }
  delete[] sigmaFloat;
  return result;
}

template<> int sddCall<hmat::Z_t>(char jobz, int m, int n, hmat::Z_t* a, int lda,
                            double* sigma, hmat::Z_t* u, int ldu, hmat::Z_t* vt, int ldvt) {
  int result;
  int workSize;
  hmat::Z_t workSize_Z;
  int* iwork = new int[8*min(m,n)];

  // We request the right size for WORK
  result = proxy_lapack::gesdd(jobz, m, n, a, lda, sigma, u, ldu, vt, ldvt, &workSize_Z, -1, iwork);
  HMAT_ASSERT(!result);
  workSize = (int) workSize_Z.real() + 1;
  hmat::Z_t* work = new hmat::Z_t[workSize];
  HMAT_ASSERT(work) ;
  result = proxy_lapack::gesdd(jobz, m, n, a, lda, sigma, u, ldu, vt, ldvt, work, workSize, iwork);
  HMAT_ASSERT(!result);
  delete[] work;
  delete[] iwork;
  return result;
}

}  // end namespace hmat
