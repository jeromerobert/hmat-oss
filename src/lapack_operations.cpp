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

template<typename T> int modifiedGramSchmidt( ScalarArray<T> *a, ScalarArray<T> *result, double prec ) {
  DECLARE_CONTEXT;
  {
    size_t mm = a->rows;
    size_t n = a->cols;
    size_t multiplications = 2*mm*n*n;
    size_t additions = 2*mm*n*n;
    increment_flops(Multipliers<T>::mul * multiplications + Multipliers<T>::add * additions);
  }
  int rank;
  double relative_epsilon;
  static const double LOWEST_EPSILON = 1.0e-6;

  const int original_rank(result->lda);
  assert(original_rank == result->rows);
  assert(original_rank == result->cols);
  assert(original_rank >= a->cols);
  int* perm = new int[original_rank];
  for(int k = 0; k < original_rank; ++k) {
    perm[k] = k;
  }
  // Temporary arrays
  ScalarArray<T> r(original_rank, original_rank);
  Vector<T> buffer(std::max(original_rank, a->rows));

  // Lower threshold for relative precision
  if(prec < LOWEST_EPSILON) {
    prec = LOWEST_EPSILON;
  }

  // Init.
  Vector<double> norm2(a->cols);
  rank = 0;
  relative_epsilon = 0.0;
  for(int j=0; j < a->cols; ++j) {
    const Vector<T> aj(a, j);
    norm2[j] = aj.normSqr();
    relative_epsilon = std::max(relative_epsilon, norm2[j]);
  }
  relative_epsilon *= prec * prec;

  // Modified Gram-Schmidt process with column pivoting
  for(int j = 0; j < a->cols; ++j) {
    // Find the largest pivot
    const int pivot = norm2.absoluteMaxIndex(j);
    const double pivmax = norm2[pivot];

    // Stopping criterion
    if (pivmax > relative_epsilon) {
      ++rank;

      // Pivoting
      if (j != pivot) {
        std::swap(perm[j], perm[pivot]);
        std::swap(norm2[j], norm2[pivot]);

        // Exchange the column 'j' and 'pivot' in a[] using buffer as temp space
        memcpy(buffer.m,              a->m + j * a->lda,     a->rows*sizeof(T));
        memcpy(a->m + j * a->lda,     a->m + pivot * a->lda, a->rows*sizeof(T));
        memcpy(a->m + pivot * a->lda, buffer.m,              a->rows*sizeof(T));

        // Idem for r[]
        memcpy(buffer.m,            r.m + j * r.lda,     a->cols*sizeof(T));
        memcpy(r.m +  j * r.lda,    r.m + pivot * r.lda, a->cols*sizeof(T));
        memcpy(r.m + pivot * r.lda, buffer.m,            a->cols*sizeof(T));
      }

      // Normalisation of qj
      r.get(j,j) = sqrt(norm2[j]);
      Vector<T> aj(a, j);
      T coef = Constants<T>::pone / r.get(j,j);
      aj.scale(coef);

      // Remove the qj-component from vectors bk (k=j+1,...,n-1)
      for(int k = j + 1; k < a->cols; ++k) {
        // Scalar product of qj and bk
        Vector<T> ak(a, k);
        r.get(j,k) = Vector<T>::dot(&aj, &ak);
        coef = - r.get(j,k);
        ak.axpy(coef, &aj);
        norm2[k] -= std::abs(r.get(j,k)) * std::abs(r.get(j,k));
      }
    }
  }

  // Apply perm to result
  for(int j = 0; j < result->cols; ++j) {
    memcpy(result->m + perm[j] * result->lda, r.m + j * result->lda, result->lda*sizeof(T));
  }
  // Update matrix dimensions
  a->cols = rank;
  result->rows = rank;
  // Clean up
  delete[] perm;
  /* end of modified Gram-Schmidt */
  return rank;
}
// Explicit instantiations
template int modifiedGramSchmidt( ScalarArray<S_t> *a, ScalarArray<S_t> *r, double prec );
template int modifiedGramSchmidt( ScalarArray<D_t> *a, ScalarArray<D_t> *r, double prec );
template int modifiedGramSchmidt( ScalarArray<C_t> *a, ScalarArray<C_t> *r, double prec );
template int modifiedGramSchmidt( ScalarArray<Z_t> *a, ScalarArray<Z_t> *r, double prec );

}  // end namespace hmat
