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

#include <assert.h>
#include "data_types.hpp"

#ifdef HAVE_MKL_CBLAS_H
  #define MKL_Complex8 hmat::C_t
  #define MKL_Complex16 hmat::Z_t
  #include <mkl_lapacke.h>
#else
  #define armpl_singlecomplex_t hmat::C_t
  #define armpl_doublecomplex_t hmat::Z_t
  #define lapack_complex_float hmat::C_t
  #define lapack_complex_double hmat::Z_t
  // OpenBLAS have a dirty lapack_make_complex_float in a public header
  #ifdef __clang__
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
  #endif
  #include <lapacke.h>
  #ifdef __clang__
    #pragma clang diagnostic pop
  #endif
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
void axpy(const int n, const hmat::S_t& alpha, const hmat::S_t* x, const int incx, hmat::S_t* y, const int incy) {
  cblas_saxpy(n, alpha, x, incx, y, incy);
}
inline
void axpy(const int n, const hmat::D_t& alpha, const hmat::D_t* x, const int incx, hmat::D_t* y, const int incy) {
  cblas_daxpy(n, alpha, x, incx, y, incy);
}
inline
void axpy(const int n, const hmat::C_t& alpha, const hmat::C_t* x, const int incx, hmat::C_t* y, const int incy) {
  // WARNING: &alpha instead of alpha for complex values
  #define _C_T hmat::C_t
  cblas_caxpy(n, _C(&alpha), _C(x), incx, _C(y), incy);
  #undef _C_T
}
inline
void axpy(const int n, const hmat::Z_t& alpha, const hmat::Z_t* x, const int incx, hmat::Z_t* y, const int incy) {
  #define _C_T hmat::Z_t
  // WARNING: &alpha instead of alpha for complex values
  cblas_zaxpy(n, _C(&alpha), _C(x), incx, _C(y), incy);
  #undef _C_T
}

template<typename T>
T dot(const int n, const T* x, const int incx, const T* y, const int incy);

inline
hmat::S_t dot(const int n, const hmat::S_t* x, const int incx, const hmat::S_t* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}
inline
hmat::D_t dot(const int n, const hmat::D_t* x, const int incx, const hmat::D_t* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

inline hmat::C_t dot(const int n, const hmat::C_t *x, const int incx,
                     const hmat::C_t *y, const int incy) {
    hmat::C_t result = 0;
#define _C_T hmat::C_t
    cblas_cdotu_sub(n, _C(x), incx, _C(y), incy,
                    _CC(openblas_complex_float, &result));
#undef _C_T
    return result;
}

inline hmat::Z_t dot(const int n, const hmat::Z_t *x, const int incx,
                     const hmat::Z_t *y, const int incy) {
    hmat::Z_t result = 0;
#define _C_T hmat::Z_t
    cblas_zdotu_sub(n, _C(x), incx, _C(y), incy,
                    _CC(openblas_complex_double, &result));
#undef _C_T
    return result;
}

template<typename T>
T dotc(const int n, const T* x, const int incx, const T* y, const int incy);

inline
hmat::C_t dotc(const int n, const hmat::C_t* x, const int incx, const hmat::C_t* y, const int incy) {
#ifdef HAVE_CBLAS_CDOTC
  return cblas_cdotc(n, x, incx, y, incy);
#else
  hmat::C_t result = 0;
  #define _C_T hmat::C_t
  cblas_cdotc_sub(n, _C(x), incx, _C(y), incy, _CC(openblas_complex_float, &result));
  #undef _C_T
  return result;
#endif
}
inline
hmat::Z_t dotc(const int n, const hmat::Z_t* x, const int incx, const hmat::Z_t* y, const int incy) {
#ifdef HAVE_CBLAS_CDOTC
  return cblas_zdotc(n, x, incx, y, incy);
#else
  hmat::Z_t result = 0;
  #define _C_T hmat::Z_t
  cblas_zdotc_sub(n, _C(x), incx, _C(y), incy, _CC(openblas_complex_double, &result));
  #undef _C_T
  return result;
#endif
}

template<typename T>
int i_amax(const int n, const T* x, const int incx);

inline
int i_amax(const int n, const hmat::S_t* x, const int incx) {
  return cblas_isamax(n, x, incx);
}
inline
int i_amax(const int n, const hmat::D_t* x, const int incx) {
  return cblas_idamax(n, x, incx);
}
inline
int i_amax(const int n, const hmat::C_t* x, const int incx) {
  #define _C_T hmat::C_t
  return cblas_icamax(n, _C(x), incx);
  #undef _C_T
}
inline
int i_amax(const int n, const hmat::Z_t* x, const int incx) {
  #define _C_T hmat::Z_t
  return cblas_izamax(n, _C(x), incx);
  #undef _C_T
}

template<typename T>
void scal(const int n, const T& alpha, T* x, const int incx);

inline
void scal(const int n, const hmat::S_t& alpha, hmat::S_t* x, const int incx) {
  cblas_sscal(n, alpha, x, incx);
}
inline
void scal(const int n, const hmat::D_t& alpha, hmat::D_t* x, const int incx) {
  cblas_dscal(n, alpha, x, incx);
}
inline
void scal(const int n, const hmat::C_t& alpha, hmat::C_t* x, const int incx) {
  // WARNING: &alpha instead of alpha for complex values
  #define _C_T hmat::C_t
  cblas_cscal(n, _C(&alpha), _C(x), incx);
  #undef _C_T
}
inline
void scal(const int n, const hmat::Z_t& alpha, hmat::Z_t* x, const int incx) {
  // WARNING: &alpha instead of alpha for complex values
  #define _C_T hmat::Z_t
  cblas_zscal(n, _C(&alpha), _C(x), incx);
  #undef _C_T
}

  // Level 2
template<typename T>
void gemv(const char trans, const int m, const int n, const T& alpha, const T* a, const int lda,
          const T* x, const int incx, const T& beta, T* y, const int incy);

inline
void gemv(const char trans, const int m, const int n, const hmat::S_t& alpha, const hmat::S_t* a, const int lda,
          const hmat::S_t* x, const int incx, const hmat::S_t& beta, hmat::S_t* y, const int incy) {
  const CBLAS_TRANSPOSE t = (trans == 'C' ? CblasConjTrans : (trans == 'T' ? CblasTrans : CblasNoTrans));
  cblas_sgemv(CblasColMajor, t, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline
void gemv(const char trans, const int m, const int n, const hmat::D_t& alpha, const hmat::D_t* a, const int lda,
          const hmat::D_t* x, const int incx, const hmat::D_t& beta, hmat::D_t* y, const int incy) {
  const CBLAS_TRANSPOSE t = (trans == 'C' ? CblasConjTrans : (trans == 'T' ? CblasTrans : CblasNoTrans));
  cblas_dgemv(CblasColMajor, t, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline
void gemv(const char trans, const int m, const int n, const hmat::C_t& alpha, const hmat::C_t* a, const int lda,
          const hmat::C_t* x, const int incx, const hmat::C_t& beta, hmat::C_t* y, const int incy) {
  const CBLAS_TRANSPOSE t = (trans == 'C' ? CblasConjTrans : (trans == 'T' ? CblasTrans : CblasNoTrans));
  // WARNING: &alpha/&beta instead of alpha/beta for complex values
  #define _C_T hmat::C_t
  cblas_cgemv(CblasColMajor, t, m, n, _C(&alpha), _C(a), lda, _C(x), incx, _C(&beta), _C(y), incy);
  #undef _C_T
}
inline
void gemv(const char trans, const int m, const int n, const hmat::Z_t& alpha, const hmat::Z_t* a, const int lda,
          const hmat::Z_t* x, const int incx, const hmat::Z_t& beta, hmat::Z_t* y, const int incy) {
  const CBLAS_TRANSPOSE t = (trans == 'C' ? CblasConjTrans : (trans == 'T' ? CblasTrans : CblasNoTrans));
  // WARNING: &alpha/&beta instead of alpha/beta for complex values
  #define _C_T hmat::Z_t
  cblas_zgemv(CblasColMajor, t, m, n, _C(&alpha), _C(a), lda, _C(x), incx, _C(&beta), _C(y), incy);
  #undef _C_T
}

template<typename T>
void ger(const int m, const int n, const T& alpha, const T* x, const int incx,
         const T* y, const int incy, T* a, const int lda);

inline
void ger(const int m, const int n, const hmat::S_t& alpha, const hmat::S_t* x, const int incx,
         const hmat::S_t* y, const int incy, hmat::S_t* a, const int lda) {
  cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
}
inline
void ger(const int m, const int n, const hmat::D_t& alpha, const hmat::D_t* x, const int incx,
         const hmat::D_t* y, const int incy, hmat::D_t* a, const int lda) {
  cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
}
inline
void ger(const int m, const int n, const hmat::C_t& alpha, const hmat::C_t* x, const int incx,
         const hmat::C_t* y, const int incy, hmat::C_t* a, const int lda) {
  // WARNING: &alpha instead of alpha for complex values
  #define _C_T hmat::C_t
  cblas_cgeru(CblasColMajor, m, n, _C(&alpha), _C(x), incx, _C(y), incy, _C(a), lda);
  #undef _C_T
}
inline
void ger(const int m, const int n, const hmat::Z_t& alpha, const hmat::Z_t* x, const int incx,
         const hmat::Z_t* y, const int incy, hmat::Z_t* a, const int lda) {
  // WARNING: &alpha instead of alpha for complex values
  #define _C_T hmat::Z_t
  cblas_zgeru(CblasColMajor, m, n, _C(&alpha), _C(x), incx, _C(y), incy, _C(a), lda);
  #undef _C_T
}

  // Level 3
template<typename T>
void gemm(const char transA, const char transB, const int m, const int n, const int k,
          const T& alpha, const T* a, const int lda, const T* b, const int ldb,
          const T& beta, T* c, const int ldc);

inline
void gemm(const char transA, const char transB, const int m, const int n, const int k,
          const hmat::S_t& alpha, const hmat::S_t* a, const int lda, const hmat::S_t* b, const int ldb,
          const hmat::S_t& beta, hmat::S_t* c, const int ldc) {
  const CBLAS_TRANSPOSE tA = (transA == 'C' ? CblasConjTrans : (transA == 'T' ? CblasTrans : CblasNoTrans));
  const CBLAS_TRANSPOSE tB = (transB == 'C' ? CblasConjTrans : (transB == 'T' ? CblasTrans : CblasNoTrans));
  cblas_sgemm(CblasColMajor, tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline
void gemm(const char transA, const char transB, const int m, const int n, const int k,
          const hmat::D_t& alpha, const hmat::D_t* a, const int lda, const hmat::D_t* b, const int ldb,
          const hmat::D_t& beta, hmat::D_t* c, const int ldc) {
  const CBLAS_TRANSPOSE tA = (transA == 'C' ? CblasConjTrans : (transA == 'T' ? CblasTrans : CblasNoTrans));
  const CBLAS_TRANSPOSE tB = (transB == 'C' ? CblasConjTrans : (transB == 'T' ? CblasTrans : CblasNoTrans));
  cblas_dgemm(CblasColMajor, tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline
void gemm(const char transA, const char transB, const int m, const int n, const int k,
          const hmat::C_t& alpha, const hmat::C_t* a, const int lda, const hmat::C_t* b, const int ldb,
          const hmat::C_t& beta, hmat::C_t* c, const int ldc) {
  const CBLAS_TRANSPOSE tA = (transA == 'C' ? CblasConjTrans : (transA == 'T' ? CblasTrans : CblasNoTrans));
  const CBLAS_TRANSPOSE tB = (transB == 'C' ? CblasConjTrans : (transB == 'T' ? CblasTrans : CblasNoTrans));
  // WARNING: &alpha/&beta instead of alpha/beta for complex values
#define _C_T hmat::C_t
#ifdef HAVE_ZGEMM3M
  static char* noGemm3m=getenv("HMAT_NO_GEMM3M"); // forbids inlining :-(
  if (!noGemm3m)
    cblas_cgemm3m(CblasColMajor, tA, tB, m, n, k, _C(&alpha), _C(a), lda, _C(b), ldb, _C(&beta), _C(c), ldc);
  else
#endif
    cblas_cgemm(CblasColMajor, tA, tB, m, n, k, _C(&alpha), _C(a), lda, _C(b), ldb, _C(&beta), _C(c), ldc);
#undef _C_T
}
inline
void gemm(const char transA, const char transB, const int m, const int n, int k,
          const hmat::Z_t& alpha, const hmat::Z_t* a, const int lda, const hmat::Z_t* b, const int ldb,
          const hmat::Z_t& beta, hmat::Z_t* c, const int ldc) {
  const CBLAS_TRANSPOSE tA = (transA == 'C' ? CblasConjTrans : (transA == 'T' ? CblasTrans : CblasNoTrans));
  const CBLAS_TRANSPOSE tB = (transB == 'C' ? CblasConjTrans : (transB == 'T' ? CblasTrans : CblasNoTrans));
  // WARNING: &alpha/&beta instead of alpha/beta for complex values
#define _C_T hmat::Z_t
#ifdef HAVE_ZGEMM3M
  static char* noGemm3m=getenv("HMAT_NO_GEMM3M"); // forbids inlining :-(
  if (!noGemm3m)
    cblas_zgemm3m(CblasColMajor, tA, tB, m, n, k, _C(&alpha), _C(a), lda, _C(b), ldb, _C(&beta), _C(c), ldc);
  else
#endif
    cblas_zgemm(CblasColMajor, tA, tB, m, n, k, _C(&alpha), _C(a), lda, _C(b), ldb, _C(&beta), _C(c), ldc);
#undef _C_T
}

template<typename T>
void trmm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const T& alpha, const T* a, const int lda,
          T* b, const int ldb);

inline
void trmm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const hmat::S_t& alpha, const hmat::S_t* a, const int lda,
          hmat::S_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'C' ? CblasConjTrans : (trans == 'T' ? CblasTrans : CblasNoTrans));
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  cblas_strmm(CblasColMajor, s, u, t, d, m, n, alpha, a, lda, b, ldb);
}
inline
void trmm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const hmat::D_t& alpha, const hmat::D_t* a, const int lda,
          hmat::D_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'C' ? CblasConjTrans : (trans == 'T' ? CblasTrans : CblasNoTrans));
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  cblas_dtrmm(CblasColMajor, s, u, t, d, m, n, alpha, a, lda, b, ldb);
}
inline
void trmm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const hmat::C_t& alpha, const hmat::C_t* a, const int lda,
          hmat::C_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'C' ? CblasConjTrans : (trans == 'T' ? CblasTrans : CblasNoTrans));
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  // WARNING: &alpha instead of alpha for complex values
#define _C_T hmat::C_t
  cblas_ctrmm(CblasColMajor, s, u, t, d, m, n, _C(&alpha), _C(a), lda, _C(b), ldb);
#undef _C_T
}
inline
void trmm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const hmat::Z_t& alpha, const hmat::Z_t* a, const int lda,
          hmat::Z_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'C' ? CblasConjTrans : (trans == 'T' ? CblasTrans : CblasNoTrans));
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  // WARNING: &alpha instead of alpha for complex values
#define _C_T hmat::Z_t
  cblas_ztrmm(CblasColMajor, s, u, t, d, m, n, _C(&alpha), _C(a), lda, _C(b), ldb);
#undef _C_T
}

template<typename T>
void trsm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const T& alpha, const T* a, const int lda,
          T* b, const int ldb);

inline
void trsm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const hmat::S_t& alpha, const hmat::S_t* a, const int lda,
          hmat::S_t* b, const int ldb) {
  assert(lda >= m || side != 'L');
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'C' ? CblasConjTrans : (trans == 'T' ? CblasTrans : CblasNoTrans));
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  cblas_strsm(CblasColMajor, s, u, t, d, m, n, alpha, a, lda, b, ldb);
}
inline
void trsm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const hmat::D_t& alpha, const hmat::D_t* a, const int lda,
          hmat::D_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'C' ? CblasConjTrans : (trans == 'T' ? CblasTrans : CblasNoTrans));
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  cblas_dtrsm(CblasColMajor, s, u, t, d, m, n, alpha, a, lda, b, ldb);
}
inline
void trsm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const hmat::C_t& alpha, const hmat::C_t* a, const int lda,
          hmat::C_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'C' ? CblasConjTrans : (trans == 'T' ? CblasTrans : CblasNoTrans));
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  // WARNING: &alpha instead of alpha for complex values
#define _C_T hmat::C_t
  cblas_ctrsm(CblasColMajor, s, u, t, d, m, n, _C(&alpha), _C(a), lda, _C(b), ldb);
#undef _C_T
}
inline
void trsm(const char side, const char uplo, const char trans, const char diag,
          const int m, const int n, const hmat::Z_t& alpha, const hmat::Z_t* a, const int lda,
          hmat::Z_t* b, const int ldb) {
  const CBLAS_SIDE s = (side == 'L' ? CblasLeft : CblasRight);
  const CBLAS_UPLO u = (uplo == 'U' ? CblasUpper : CblasLower);
  const CBLAS_TRANSPOSE t = (trans == 'C' ? CblasConjTrans : (trans == 'T' ? CblasTrans : CblasNoTrans));
  const CBLAS_DIAG d = (diag == 'N' ?  CblasNonUnit : CblasUnit );
  // WARNING: &alpha instead of alpha for complex values
#define _C_T hmat::Z_t
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
hmat::S_t dot_c(const int n, const hmat::S_t* x, const int incx, const hmat::S_t* y, const int incy) {
  return proxy_cblas::dot(n, x, incx, y, incy);
}
inline
hmat::D_t dot_c(const int n, const hmat::D_t* x, const int incx, const hmat::D_t* y, const int incy) {
  return proxy_cblas::dot(n, x, incx, y, incy);
}
inline
hmat::C_t dot_c(const int n, const hmat::C_t* x, const int incx, const hmat::C_t* y, const int incy) {
  return proxy_cblas::dotc(n, x, incx, y, incy);
}
inline
hmat::Z_t dot_c(const int n, const hmat::Z_t* x, const int incx, const hmat::Z_t* y, const int incy) {
  return proxy_cblas::dotc(n, x, incx, y, incy);
}

} // end namespace proxy_cblas_convenience

#ifdef HAVE_MKL_IMATCOPY
namespace proxy_mkl {

template<typename T>
void imatcopy(const size_t rows, const size_t cols, T* m);

inline
void imatcopy(const size_t rows, const size_t cols, hmat::S_t* m) {
  mkl_simatcopy('C', 'T', rows, cols, 1, m, rows, cols);
}
inline
void imatcopy(const size_t rows, const size_t cols, hmat::D_t* m) {
  mkl_dimatcopy('C', 'T', rows, cols, 1, m, rows, cols);
}
inline
void imatcopy(const size_t rows, const size_t cols, hmat::C_t* m) {
  const MKL_Complex8 pone = {1., 0.};
  mkl_cimatcopy('C', 'T', rows, cols, pone, (MKL_Complex8*) m, rows, cols);
}
inline
void imatcopy(const size_t rows, const size_t cols, hmat::Z_t* m) {
  const MKL_Complex16 pone = {1., 0.};
  mkl_zimatcopy('C', 'T', rows, cols, pone, (MKL_Complex16*) m, rows, cols);
}

template<typename T>
void omatcopy(size_t rows, size_t cols, const T* m, T* copy);

inline
void omatcopy(size_t rows, size_t cols, const hmat::S_t* m, hmat::S_t* copy) {
  mkl_somatcopy('C', 'T', rows, cols, 1, m, rows, copy, cols);
}
inline
void omatcopy(size_t rows, size_t cols, const hmat::D_t* m, hmat::D_t* copy) {
  mkl_domatcopy('C', 'T', rows, cols, 1, m, rows, copy, cols);
}
inline
void omatcopy(size_t rows, size_t cols, const hmat::C_t* m, hmat::C_t* copy) {
  const MKL_Complex8 pone = {1., 0.};
  mkl_comatcopy('C', 'T', rows, cols, pone, (const MKL_Complex8*) m, rows, (MKL_Complex8*) copy, cols);
}
inline
void omatcopy(size_t rows, size_t cols, const hmat::Z_t* m, hmat::Z_t* copy) {
  const MKL_Complex16 pone = {1., 0.};
  mkl_zomatcopy('C', 'T', rows, cols, pone, (const MKL_Complex16*) m, rows, (MKL_Complex16*) copy, cols);
}

} // End namespace proxy_mkl
#endif

#endif // _BLAS_OVERLOADS_HPP
