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

/* Declarations of BLAS and MKL functions used by hmat */
#ifndef _BLAS_OVERLOADS_HPP
#define _BLAS_OVERLOADS_HPP

#include "config.h"

#include "data_types.hpp"

#ifdef HAVE_MKL_H
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#ifdef HAVE_MKL_IMATCOPY
#include "mkl_trans.h"
#endif

namespace proxy_cblas {

#if defined(OPENBLAS_COMPLEX_STRUCT) || defined(OPENBLAS_COMPLEX_C99)
 #define _C(x) (_C_T::value_type*)x
 #define _CC(T, x) (T*)x
#else
 #define _C(x) x
 #define _CC(T, x) x
#endif
  // Level 1
template<typename T>
void axpy(const int n, const T& alpha, const T* x, const int incx, T* y, const int incy);

inline
void axpy(const int n, const S_t& alpha, const S_t* x, const int incx, S_t* y, const int incy) {
  cblas_saxpy(n, alpha, x, incx, y, incy);
}
inline
void axpy(const int n, const D_t& alpha, const D_t* x, const int incx, D_t* y, const int incy) {
  cblas_daxpy(n, alpha, x, incx, y, incy);
}
inline
void axpy(const int n, const C_t& alpha, const C_t* x, const int incx, C_t* y, const int incy) {
  // WARNING: &alpha instead of alpha for complex values
  #define _C_T C_t
  cblas_caxpy(n, _C(&alpha), _C(x), incx, _C(y), incy);
  #undef _C_T
}
inline
void axpy(const int n, const Z_t& alpha, const Z_t* x, const int incx, Z_t* y, const int incy) {
  #define _C_T Z_t
  // WARNING: &alpha instead of alpha for complex values
  cblas_zaxpy(n, _C(&alpha), _C(x), incx, _C(y), incy);
  #undef _C_T
}

template<typename T>
void copy(const int n, const T* x, const int incx, T* y, const int incy);

inline
void copy(const int n, const S_t* x, int incx, S_t* y, const int incy) {
  cblas_scopy(n, x, incx, y, incy);
}
inline
void copy(const int n, const D_t* x, int incx, D_t* y, const int incy) {
  cblas_dcopy(n, x, incx, y, incy);
}
inline
void copy(const int n, const C_t* x, int incx, C_t* y, const int incy) {
  #define _C_T C_t
  cblas_ccopy(n, _C(x), incx, _C(y), incy);
  #undef _C_T
}
inline
void copy(const int n, const Z_t* x, int incx, Z_t* y, const int incy) {
  #define _C_T Z_t
  cblas_zcopy(n, _C(x), incx, _C(y), incy);
  #undef _C_T
}

template<typename T>
T dot(const int n, const T* x, const int incx, const T* y, const int incy);

inline
S_t dot(const int n, const S_t* x, const int incx, const S_t* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}
inline
D_t dot(const int n, const D_t* x, const int incx, const D_t* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template<typename T>
T dotc(const int n, const T* x, const int incx, const T* y, const int incy);

inline
C_t dotc(const int n, const C_t* x, const int incx, const C_t* y, const int incy) {
#ifdef HAVE_CBLAS_CDOTC
  return cblas_cdotc(n, x, incx, y, incy);
#else
  C_t result = Constants<C_t>::zero;
  #define _C_T C_t
  cblas_cdotc_sub(n, _C(x), incx, _C(y), incy, _CC(openblas_complex_float, &result));
  #undef _C_T
  return result;
#endif
}
inline
Z_t dotc(const int n, const Z_t* x, const int incx, const Z_t* y, const int incy) {
#ifdef HAVE_CBLAS_CDOTC
  return cblas_zdotc(n, x, incx, y, incy);
#else
  Z_t result = Constants<Z_t>::zero;
  #define _C_T Z_t
  cblas_zdotc_sub(n, _C(x), incx, _C(y), incy, _CC(openblas_complex_double, &result));
  #undef _C_T
  return result;
#endif
}

template<typename T>
int i_amax(const int n, const T* x, const int incx);

inline
int i_amax(const int n, const S_t* x, const int incx) {
  return cblas_isamax(n, x, incx);
}
inline
int i_amax(const int n, const D_t* x, const int incx) {
  return cblas_idamax(n, x, incx);
}
inline
int i_amax(const int n, const C_t* x, const int incx) {
  #define _C_T C_t
  return cblas_icamax(n, _C(x), incx);
  #undef _C_T
}
inline
int i_amax(const int n, const Z_t* x, const int incx) {
  #define _C_T Z_t
  return cblas_izamax(n, _C(x), incx);
  #undef _C_T
}

template<typename T>
void scal(const int n, const T& alpha, T* x, const int incx);

inline
void scal(const int n, const S_t& alpha, S_t* x, const int incx) {
  cblas_sscal(n, alpha, x, incx);
}
inline
void scal(const int n, const D_t& alpha, D_t* x, const int incx) {
  cblas_dscal(n, alpha, x, incx);
}
inline
void scal(const int n, const C_t& alpha, C_t* x, const int incx) {
  // WARNING: &alpha instead of alpha for complex values
  #define _C_T C_t
  cblas_cscal(n, _C(&alpha), _C(x), incx);
  #undef _C_T
}
inline
void scal(const int n, const Z_t& alpha, Z_t* x, const int incx) {
  // WARNING: &alpha instead of alpha for complex values
  #define _C_T Z_t
  cblas_zscal(n, _C(&alpha), _C(x), incx);
  #undef _C_T
}

  // Level 2
template<typename T>
void gemv(const char trans, const int m, const int n, const T& alpha, const T* a, const int lda,
          const T* x, const int incx, const T& beta, T* y, const int incy);

inline
void gemv(const char trans, const int m, const int n, const S_t& alpha, const S_t* a, const int lda,
          const S_t* x, const int incx, const S_t& beta, S_t* y, const int incy) {
  const CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  cblas_sgemv(CblasColMajor, t, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline
void gemv(const char trans, const int m, const int n, const D_t& alpha, const D_t* a, const int lda,
          const D_t* x, const int incx, const D_t& beta, D_t* y, const int incy) {
  const CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  cblas_dgemv(CblasColMajor, t, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline
void gemv(const char trans, const int m, const int n, const C_t& alpha, const C_t* a, const int lda,
          const C_t* x, const int incx, const C_t& beta, C_t* y, const int incy) {
  const CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  // WARNING: &alpha/&beta instead of alpha/beta for complex values
  #define _C_T C_t
  cblas_cgemv(CblasColMajor, t, m, n, _C(&alpha), _C(a), lda, _C(x), incx, _C(&beta), _C(y), incy);
  #undef _C_T
}
inline
void gemv(const char trans, const int m, const int n, const Z_t& alpha, const Z_t* a, const int lda,
          const Z_t* x, const int incx, const Z_t& beta, Z_t* y, const int incy) {
  const CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  // WARNING: &alpha/&beta instead of alpha/beta for complex values
  #define _C_T Z_t
  cblas_zgemv(CblasColMajor, t, m, n, _C(&alpha), _C(a), lda, _C(x), incx, _C(&beta), _C(y), incy);
  #undef _C_T
}

  // Level 3
template<typename T>
void gemm(const char transA, const char transB, const int m, const int n, const int k,
          const T& alpha, const T* a, const int lda, const T* b, const int ldb,
          const T& beta, T* c, const int ldc);

inline
void gemm(const char transA, const char transB, const int m, const int n, const int k,
          const S_t& alpha, const S_t* a, const int lda, const S_t* b, const int ldb,
          const S_t& beta, S_t* c, const int ldc) {
  const CBLAS_TRANSPOSE tA = (transA == 'N' ? CblasNoTrans : CblasTrans);
  const CBLAS_TRANSPOSE tB = (transB == 'N' ? CblasNoTrans : CblasTrans);
  cblas_sgemm(CblasColMajor, tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline
void gemm(const char transA, const char transB, const int m, const int n, const int k,
          const D_t& alpha, const D_t* a, const int lda, const D_t* b, const int ldb,
          const D_t& beta, D_t* c, const int ldc) {
  const CBLAS_TRANSPOSE tA = (transA == 'N' ? CblasNoTrans : CblasTrans);
  const CBLAS_TRANSPOSE tB = (transB == 'N' ? CblasNoTrans : CblasTrans);
  cblas_dgemm(CblasColMajor, tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline
void gemm(const char transA, const char transB, const int m, const int n, const int k,
          const C_t& alpha, const C_t* a, const int lda, const C_t* b, const int ldb,
          const C_t& beta, C_t* c, const int ldc) {
  const CBLAS_TRANSPOSE tA = (transA == 'N' ? CblasNoTrans : CblasTrans);
  const CBLAS_TRANSPOSE tB = (transB == 'N' ? CblasNoTrans : CblasTrans);
  // WARNING: &alpha/&beta instead of alpha/beta for complex values
#define _C_T C_t
#ifdef HAVE_ZGEMM3M
  cblas_cgemm3m(CblasColMajor, tA, tB, m, n, k, _C(&alpha), _C(a), lda, _C(b), ldb, _C(&beta), _C(c), ldc);
#else
  cblas_cgemm(CblasColMajor, tA, tB, m, n, k, _C(&alpha), _C(a), lda, _C(b), ldb, _C(&beta), _C(c), ldc);
#endif
#undef _C_T
}
inline
void gemm(const char transA, const char transB, const int m, const int n, int k,
          const Z_t& alpha, const Z_t* a, const int lda, const Z_t* b, const int ldb,
          const Z_t& beta, Z_t* c, const int ldc) {
  const CBLAS_TRANSPOSE tA = (transA == 'N' ? CblasNoTrans : CblasTrans);
  const CBLAS_TRANSPOSE tB = (transB == 'N' ? CblasNoTrans : CblasTrans);
  // WARNING: &alpha/&beta instead of alpha/beta for complex values
#define _C_T Z_t
#ifdef HAVE_ZGEMM3M
  cblas_zgemm3m(CblasColMajor, tA, tB, m, n, k, _C(&alpha), _C(a), lda, _C(b), ldb, _C(&beta), _C(c), ldc);
#else
  cblas_zgemm(CblasColMajor, tA, tB, m, n, k, _C(&alpha), _C(a), lda, _C(b), ldb, _C(&beta), _C(c), ldc);
#endif
#undef _C_T
}

template<typename T>
void trmm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const T& alpha, const T* a, const int lda,
          T* b, const int ldb);

inline
void trmm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const S_t& alpha, const S_t* a, const int lda,
          S_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  cblas_strmm(CblasColMajor, s, u, t, d, m, n, alpha, a, lda, b, ldb);
}
inline
void trmm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const D_t& alpha, const D_t* a, const int lda,
          D_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  cblas_dtrmm(CblasColMajor, s, u, t, d, m, n, alpha, a, lda, b, ldb);
}
inline
void trmm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const C_t& alpha, const C_t* a, const int lda,
          C_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  // WARNING: &alpha instead of alpha for complex values
#define _C_T C_t
  cblas_ctrmm(CblasColMajor, s, u, t, d, m, n, _C(&alpha), _C(a), lda, _C(b), ldb);
#undef _C_T
}
inline
void trmm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const Z_t& alpha, const Z_t* a, const int lda,
          Z_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  // WARNING: &alpha instead of alpha for complex values
#define _C_T Z_t
  cblas_ztrmm(CblasColMajor, s, u, t, d, m, n, _C(&alpha), _C(a), lda, _C(b), ldb);
#undef _C_T
}

template<typename T>
void trsm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const T& alpha, const T* a, const int lda,
          T* b, const int ldb);

inline
void trsm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const S_t& alpha, const S_t* a, const int lda,
          S_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  cblas_strsm(CblasColMajor, s, u, t, d, m, n, alpha, a, lda, b, ldb);
}
inline
void trsm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const D_t& alpha, const D_t* a, const int lda,
          D_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  cblas_dtrsm(CblasColMajor, s, u, t, d, m, n, alpha, a, lda, b, ldb);
}
inline
void trsm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const C_t& alpha, const C_t* a, const int lda,
          C_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  // WARNING: &alpha instead of alpha for complex values
#define _C_T C_t
  cblas_ctrsm(CblasColMajor, s, u, t, d, m, n, _C(&alpha), _C(a), lda, _C(b), ldb);
#undef _C_T
}
inline
void trsm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const Z_t& alpha, const Z_t* a, const int lda,
          Z_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  // WARNING: &alpha instead of alpha for complex values
#define _C_T Z_t
  cblas_ztrsm(CblasColMajor, s, u, t, d, m, n, _C(&alpha), _C(a), lda, _C(b), ldb);
#undef _C_T
}

} // End namespace proxy_cblas

// Some BLAS routines only exist for real or complex matrices; for instance,
// when computing norms, one want to call dot on real vectors and dotc on
// complex vectors.

namespace proxy_cblas_convenience {

template<typename T>
T dot_c(const int n, const T* x, const int incx, const T* y, const int incy);

inline
S_t dot_c(const int n, const S_t* x, const int incx, const S_t* y, const int incy) {
  return proxy_cblas::dot(n, x, incx, y, incy);
}
inline
D_t dot_c(const int n, const D_t* x, const int incx, const D_t* y, const int incy) {
  return proxy_cblas::dot(n, x, incx, y, incy);
}
inline
C_t dot_c(const int n, const C_t* x, const int incx, const C_t* y, const int incy) {
  return proxy_cblas::dotc(n, x, incx, y, incy);
}
inline
Z_t dot_c(const int n, const Z_t* x, const int incx, const Z_t* y, const int incy) {
  return proxy_cblas::dotc(n, x, incx, y, incy);
}

} // end namespace proxy_cblas_convenience

#ifdef HAVE_MKL_IMATCOPY
namespace proxy_mkl {

template<typename T>
void imatcopy(const size_t rows, const size_t cols, T* m);

inline
void imatcopy(const size_t rows, const size_t cols, S_t* m) {
  mkl_simatcopy('C', 'T', rows, cols, Constants<S_t>::pone, m, rows, cols);
}
inline
void imatcopy(const size_t rows, const size_t cols, D_t* m) {
  mkl_dimatcopy('C', 'T', rows, cols, Constants<D_t>::pone, m, rows, cols);
}
inline
void imatcopy(const size_t rows, const size_t cols, C_t* m) {
  const MKL_Complex8 pone = {1., 0.};
  mkl_cimatcopy('C', 'T', rows, cols, pone, (MKL_Complex8*) m, rows, cols);
}
inline
void imatcopy(const size_t rows, const size_t cols, Z_t* m) {
  const MKL_Complex16 pone = {1., 0.};
  mkl_zimatcopy('C', 'T', rows, cols, pone, (MKL_Complex16*) m, rows, cols);
}

template<typename T>
void omatcopy(size_t rows, size_t cols, const T* m, T* copy);

inline
void omatcopy(size_t rows, size_t cols, const S_t* m, S_t* copy) {
  mkl_somatcopy('C', 'T', rows, cols, Constants<S_t>::pone, m, rows, copy, cols);
}
inline
void omatcopy(size_t rows, size_t cols, const D_t* m, D_t* copy) {
  mkl_domatcopy('C', 'T', rows, cols, Constants<D_t>::pone, m, rows, copy, cols);
}
inline
void omatcopy(size_t rows, size_t cols, const C_t* m, C_t* copy) {
  const MKL_Complex8 pone = {1., 0.};
  mkl_comatcopy('C', 'T', rows, cols, pone, (const MKL_Complex8*) m, rows, (MKL_Complex8*) copy, cols);
}
inline
void omatcopy(size_t rows, size_t cols, const Z_t* m, Z_t* copy) {
  const MKL_Complex16 pone = {1., 0.};
  mkl_zomatcopy('C', 'T', rows, cols, pone, (const MKL_Complex16*) m, rows, (MKL_Complex16*) copy, cols);
}

} // End namespace proxy_mkl
#endif

#endif // _BLAS_OVERLOADS_HPP
