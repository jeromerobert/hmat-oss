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

namespace {

// Implementation
template<typename T> int svdCall(char jobu, char jobv, int m, int n, T* a,
                                 int lda, double* sigma, T* u, int ldu, T* vt,
                                 int ldvt);

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
template<typename T> int sddCall(char jobz, int m, int n, T* a, int lda,
                                 double* sigma, T* u, int ldu, T* vt, int ldvt);

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

}  // end anonymous namespace

namespace hmat {

template<typename T> int truncatedSdd(ScalarArray<T>* m, ScalarArray<T>** u, Vector<double>** sigma, ScalarArray<T>** vt) {
  DECLARE_CONTEXT;


  // Allocate free space for U, S, V
  int rows = m->rows;
  int cols = m->cols;
  int p = min(rows, cols);

  *u = new ScalarArray<T>(rows, p);
  *sigma = new Vector<double>(p);
  *vt = new ScalarArray<T>(p, cols);

  assert(m->lda >= m->rows);

  char jobz = 'S';
  int mm = rows;
  int n = cols;
  T* a = m->m;
  int lda = rows;
  int info;

  {
    const size_t _m = mm, _n = n;
    // Warning: These quantities are a rough approximation.
    // What's wrong with these estimates:
    //  - Golub only gives 14 * M*N*N + 8 N*N*N
    //  - This is for real numbers
    //  - We assume the same number of * and +
    size_t adds = 7 * _m * _n * _n + 4 * _n * _n * _n;
    size_t muls = 7 * _m * _n * _n + 4 * _n * _n * _n;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  info = sddCall<T>(jobz, mm, n, a, lda, (*sigma)->m, (*u)->m,
                    (*u)->lda, (*vt)->m, (*vt)->lda);
  HMAT_ASSERT_MSG(!info, "Error in ?gesdd, info=%d", info);
  return info;
}

template<typename T> int truncatedSvd(ScalarArray<T>* m, ScalarArray<T>** u, Vector<double>** sigma, ScalarArray<T>** vt) {
  DECLARE_CONTEXT;
  static char * useGESSD = getenv("HMAT_GESSD");
  if(useGESSD)
      return truncatedSdd(m, u, sigma, vt);

  // Allocate free space for U, S, V
  int rows = m->rows;
  int cols = m->cols;
  int p = min(rows, cols);

  *u = new ScalarArray<T>(rows, p);
  *sigma = new Vector<double>(p);
  *vt = new ScalarArray<T>(p, cols);

  assert(m->lda >= m->rows);

  char jobz = 'S';
  int mm = rows;
  int n = cols;
  T* a = m->m;
  int lda = rows;
  int info;

  {
    const size_t _m = mm, _n = n;
    // Warning: These quantities are a rough approximation.
    // What's wrong with these estimates:
    //  - Golub only gives 14 * M*N*N + 8 N*N*N
    //  - This is for real numbers
    //  - We assume the same number of * and +
    size_t adds = 7 * _m * _n * _n + 4 * _n * _n * _n;
    size_t muls = 7 * _m * _n * _n + 4 * _n * _n * _n;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  info = svdCall<T>(jobz, jobz, mm, n, a, lda, (*sigma)->m, (*u)->m,
                    (*u)->lda, (*vt)->m, (*vt)->lda);
  if (info) {
    cerr << "Erreur dans xGESVD: " << info << endl;
  }
  HMAT_ASSERT(!info);
  return info;
}

// Explicit instantiations
template int truncatedSvd(ScalarArray<S_t>* m, ScalarArray<S_t>** u, Vector<double>** sigma, ScalarArray<S_t>** vt);
template int truncatedSvd(ScalarArray<D_t>* m, ScalarArray<D_t>** u, Vector<double>** sigma, ScalarArray<D_t>** vt);
template int truncatedSvd(ScalarArray<C_t>* m, ScalarArray<C_t>** u, Vector<double>** sigma, ScalarArray<C_t>** vt);
template int truncatedSvd(ScalarArray<Z_t>* m, ScalarArray<Z_t>** u, Vector<double>** sigma, ScalarArray<Z_t>** vt);

template<typename T> T* qrDecomposition(ScalarArray<T>* m) {
  DECLARE_CONTEXT;
  //  SUBROUTINE DGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
  int rows = m->rows;
  int cols = m->cols;
  T* tau = (T*) calloc(min(rows, cols), sizeof(T));
  {
    size_t mm = max(rows, cols);
    size_t n = min(rows, cols);
    size_t multiplications = mm * n * n - (n * n * n) / 3 + mm * n + (n * n) / 2 + (29 * n) / 6;
    size_t additions = mm * n * n + (n * n * n) / 3 + 2 * mm * n - (n * n) / 2 + (5 * n) / 6;
    increment_flops(Multipliers<T>::mul * multiplications + Multipliers<T>::add * additions);
  }
  int info;
  int workSize;
  T workSize_S;
  // int info = LAPACKE_sgeqrf(LAPACK_COL_MAJOR, rows, cols, m->m, rows, *tau);
  info = proxy_lapack::geqrf(rows, cols, m->m, rows, tau, &workSize_S, -1);
  HMAT_ASSERT(!info);
  workSize = (int) hmat::real(workSize_S) + 1;
  T* work = new T[workSize];// TODO Mettre dans la pile ??
  HMAT_ASSERT(work) ;
  info = proxy_lapack::geqrf(rows, cols, m->m, rows, tau, work, workSize);
  delete[] work;

  HMAT_ASSERT(!info);
  return tau;
}

// templates declaration
template S_t* qrDecomposition<S_t>(ScalarArray<S_t>* m);
template D_t* qrDecomposition<D_t>(ScalarArray<D_t>* m);
template C_t* qrDecomposition<C_t>(ScalarArray<C_t>* m);
template Z_t* qrDecomposition<Z_t>(ScalarArray<Z_t>* m);



template<typename T>
void myTrmm(ScalarArray<T>* aFull, ScalarArray<T>* bTri) {
  DECLARE_CONTEXT;
  int mm = aFull->rows;
  int n = aFull->rows;
  T alpha = Constants<T>::pone;
  T *aData = bTri->m;
  int lda = bTri->rows;
  T *bData = aFull->m;
  int ldb = aFull->rows;
  {
    size_t m_ = mm;
    size_t nn = n;
    size_t multiplications = m_ * nn  * (nn + 1) / 2;
    size_t additions = m_ * nn  * (nn - 1) / 2;
    increment_flops(Multipliers<T>::mul * multiplications + Multipliers<T>::add * additions);
  }
  proxy_cblas::trmm('R', 'U', 'T', 'N', mm, n, alpha, aData, lda, bData, ldb);
}
// Explicit instantiations
template void myTrmm(ScalarArray<S_t>* aFull, ScalarArray<S_t>* bTri);
template void myTrmm(ScalarArray<D_t>* aFull, ScalarArray<D_t>* bTri);
template void myTrmm(ScalarArray<C_t>* aFull, ScalarArray<C_t>* bTri);
template void myTrmm(ScalarArray<Z_t>* aFull, ScalarArray<Z_t>* bTri);

template<typename T>
int productQ(char side, char trans, ScalarArray<T>* qr, T* tau, ScalarArray<T>* c) {
  DECLARE_CONTEXT;
  int m = c->rows;
  int n = c->cols;
  int k = qr->cols;
  T* a = qr->m;
  assert((side == 'L') ? qr->rows == m : qr->rows == n);
  int ldq = qr->lda;
  int ldc = c->lda;
  int info;
  int workSize;
  T workSize_req;
  {
    size_t _m = m, _n = n, _k = k;
    size_t muls = 2 * _m * _n * _k - _n * _k * _k + 2 * _n * _k;
    size_t adds = 2 * _m * _n * _k - _n * _k * _k + _n * _k;
    increment_flops(Multipliers<T>::mul * muls + Multipliers<T>::add * adds);
  }
  info = proxy_lapack_convenience::or_un_mqr(side, trans, m, n, k, a, ldq, tau, c->m, ldc, &workSize_req, -1);
  HMAT_ASSERT(!info);
  workSize = (int) hmat::real(workSize_req) + 1;
  T* work = new T[workSize];
  HMAT_ASSERT(work);
  info = proxy_lapack_convenience::or_un_mqr(side, trans, m, n, k, a, ldq, tau, c->m, ldc, work, workSize);
  HMAT_ASSERT(!info);
  delete[] work;
  return 0;
}
// Explicit instantiations
template int productQ(char side, char trans, ScalarArray<S_t>* qr, S_t* tau, ScalarArray<S_t>* c);
template int productQ(char side, char trans, ScalarArray<D_t>* qr, D_t* tau, ScalarArray<D_t>* c);
template int productQ(char side, char trans, ScalarArray<C_t>* qr, C_t* tau, ScalarArray<C_t>* c);
template int productQ(char side, char trans, ScalarArray<Z_t>* qr, Z_t* tau, ScalarArray<Z_t>* c);

template<typename T> int modifiedGramSchmidt( ScalarArray<T> *a, ScalarArray<T> *result, double prec, double maxNorm ) {
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
    const Vector<T> aj(&a->get(0, j), a->rows);
    norm2.m[j] = aj.normSqr();
    relative_epsilon = std::max(relative_epsilon, norm2.m[j]);
  }
  if(maxNorm>0){relative_epsilon=maxNorm;}
  relative_epsilon *= prec * prec;

  // Modified Gram-Schmidt process with column pivoting
  for(int j = 0; j < a->cols; ++j) {
    // Find the largest pivot
    const int pivot = norm2.absoluteMaxIndex(j);
    const double pivmax = norm2.m[pivot];

    // Stopping criterion
    if (pivmax <= relative_epsilon)
      break;

    ++rank;

    // Pivoting
    if (j != pivot) {
      std::swap(perm[j], perm[pivot]);
      std::swap(norm2.m[j], norm2.m[pivot]);

      memcpy(&buffer.get(0, 0), &a->get(0, j), a->rows*sizeof(T));
      memcpy(&a->get(0, j), &a->get(0, pivot), a->rows*sizeof(T));
      memcpy(&a->get(0, pivot), &buffer.get(0, 0), a->rows*sizeof(T));

      memcpy(&buffer.get(0, 0), &r.get(0, j), a->cols*sizeof(T));
      memcpy(&r.get(0, j), &r.get(0, pivot), a->cols*sizeof(T));
      memcpy(&r.get(0, pivot), &buffer.get(0, 0), a->cols*sizeof(T));
    }

    // Normalisation of qj
    r.get(j, j) = sqrt(norm2.get(j, 0));
    Vector<T> aj(&a->get(0, j), a->rows);
    T coef = Constants<T>::pone / r.get(j, j);
    aj.scale(coef);

    // Remove the qj-component from vectors bk (k=j+1,...,n-1)
    for(int k = j + 1; k < a->cols; ++k) {
      // Scalar product of qj and bk
      Vector<T> ak(&a->get(0, k), a->rows);
      T dot_jk = Vector<T>::dot(&aj, &ak);
      r.get(j, k) = dot_jk;
      ak.axpy(- dot_jk, &aj);
      norm2.m[k] -= std::abs(dot_jk) * std::abs(dot_jk);
    }
  }

  // Apply perm to result
  for(int j = 0; j < result->cols; ++j) {
    memcpy(&result->get(0, perm[j]), &r.get(0, j), result->lda*sizeof(T));
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
template int modifiedGramSchmidt( ScalarArray<S_t> *a, ScalarArray<S_t> *r, double prec, double maxNorm );
template int modifiedGramSchmidt( ScalarArray<D_t> *a, ScalarArray<D_t> *r, double prec, double maxNorm );
template int modifiedGramSchmidt( ScalarArray<C_t> *a, ScalarArray<C_t> *r, double prec, double maxNorm );
template int modifiedGramSchmidt( ScalarArray<Z_t> *a, ScalarArray<Z_t> *r, double prec, double maxNorm );

template<typename T> int blockedMGS( ScalarArray<T> *a, ScalarArray<T> *result, double prec, const int nb ) {
  DECLARE_CONTEXT;

  const int mm = a->rows;
  const int nn = a->cols;

  ScalarArray<T> rtmp(nn, nn);

  int rank=0;
  int nBlocks = a->cols / nb;
  const int remain = a->cols % nb;
  if(remain > 0) {
    ++nBlocks;
  }

  int *permBlock = new int[nBlocks];
  int *blockRanks = new int[nBlocks];
  int *blockSizes = new int[nBlocks];
  for(int p = 0; p < nBlocks; ++p) {
    permBlock[p] = p;
    blockSizes[p] = nb;
  }
  if(remain > 0) {
    blockSizes[nBlocks-1] = remain;
  }

  double maxNorm = -1.0;

  Vector<double> blockNorms2(nBlocks);
  // Initialization: computing column norms and panel norms
  double maxCol = 0.;
  double maxBlockNorm = 0.;
  for(int k = 0; k < nBlocks; ++k) {
    blockNorms2.m[k] = 0;
    for(int p = 0; p < blockSizes[k]; ++p) {
      ScalarArray<T> ak(&a->get(0, k*nb+p), a->rows, 1);
      const double norm2_ak = ak.normSqr();
      blockNorms2.m[k] += norm2_ak;
      if(norm2_ak > maxCol) {
        maxCol = norm2_ak;
      }
    }
    if(blockNorms2.m[k] > maxBlockNorm) {
      maxBlockNorm = blockNorms2.m[k];
    }
  }
  maxNorm = maxCol;

  const double relative_epsilon = maxBlockNorm * prec * prec;
  ScalarArray<T> buffer(a->rows,nb);

  rank = 0;
  for(int k = 0; k < nBlocks; ++k) {
    // Find the largest pivot
    const int pivot = blockNorms2.absoluteMaxIndex(k);
    const double pivmax = blockNorms2.m[pivot];

    if(pivmax < relative_epsilon)
      break;

    if(pivot != k) {
      std::swap(blockNorms2.m[k], blockNorms2.m[pivot]);
      std::swap(blockSizes[k], blockSizes[pivot]);
      std::swap(permBlock[k], permBlock[pivot]);
    }

    const int kk = permBlock[k];
    ScalarArray<T> ak(&a->get(0, kk*nb), a->rows, blockSizes[k]);
    ScalarArray<T> r(nb, nb);

    const int rkk = modifiedGramSchmidt( &ak, &r, prec, maxNorm );
    if(rkk == 0) {
      blockRanks[k] = rkk;
      continue;
    }

    for(int q = 0; q < blockSizes[k]; ++q) {
      for(int p = 0; p < rkk; ++p) {
        rtmp.get(kk*nb+p, kk*nb+q) = r.get(p, q);
      }
    }
    blockRanks[k] = rkk;
    ak.cols = rkk;

    for(int j = k+1; j < nBlocks; ++j) {
      const int jj = permBlock[j];
      ScalarArray<T> Rkj(rkk, blockSizes[j]);
      ScalarArray<T> aj(&a->get(0, jj*nb), a->rows, blockSizes[j]);

      // Rkj = ak^* . aj
      Rkj.gemm('C','N',Constants<T>::pone, &ak, &aj, Constants<T>::pone);

      // Write Rkj in buffer rtmp
      for(int q = 0; q < blockSizes[j]; ++q) {
        for(int p = 0; p < rkk; ++p) {
          rtmp.get(kk*nb+p, jj*nb+q) = Rkj.get(p, q);
        }
      }
      const double norm_Rkj2 = Rkj.normSqr();

      // Update column aj
      aj.gemm('N','N',Constants<T>::mone, &ak, &Rkj , Constants<T>::pone);

      // Update Frobenius norm
      blockNorms2.m[j] -= norm_Rkj2;
    }
    rank += rkk;
  }

  /* No 'physical' copies during the orthonornamisation so we do it now.
     The matrix result is overwritten with the adequate blocks of rtmp according
     to the permutation obtained.
     Matrix a is overwritten with the orthonormal matrix composed of the panels computed.
     Panels have been orthonormalised in place so we do not copy the unused columns.
  */
  for(int p = 0; p < nBlocks; ++p) {
    blockSizes[p]=nb;
  }
  if(remain > 0) {
    blockSizes[nBlocks-1] = remain;
  }

  int jbStart = 0;
  for(int jb = 0; jb < nBlocks; ++jb) {
    int ibStart = 0;
    for(int ib = 0; ib < nBlocks; ++ib) {
      for(int p = 0; p < blockSizes[jb]; ++p) {
         memcpy(&result->get(ibStart, jbStart+p), &rtmp.get(permBlock[ib]*nb,jb*nb+p), blockRanks[ib]*sizeof(T));
      }
      ibStart += blockRanks[ib];
    }
    jbStart += blockSizes[jb];
  }
  result->rows = rank;

  T *newMat = new T[mm*rank];
  int toCol_start = 0;
  for(int p = 0; p < nBlocks; ++p) {
    int fromCol_start = permBlock[p]*nb;
    memcpy(&newMat[toCol_start*mm], &a->get(0, fromCol_start), a->lda*blockRanks[p]*sizeof(T));
    toCol_start += blockRanks[p];
  }

  // Free memory
  delete[] permBlock;
  delete[] blockRanks;
  delete[] blockSizes;

  // a is overwritten by qa
  delete[] a->m;
  a->m = newMat;
  a->lda = mm;
  a->rows = mm;
  a->cols = rank;

  /* end of blocked Modified Gram-Schmidt */
  return rank;
}
// Explicit instantiations
template int blockedMGS( ScalarArray<S_t> *a, ScalarArray<S_t> *r, double prec, const int nb );
template int blockedMGS( ScalarArray<D_t> *a, ScalarArray<D_t> *r, double prec, const int nb );
template int blockedMGS( ScalarArray<C_t> *a, ScalarArray<C_t> *r, double prec, const int nb );
template int blockedMGS( ScalarArray<Z_t> *a, ScalarArray<Z_t> *r, double prec, const int nb );

}  // end namespace hmat
