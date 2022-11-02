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

/* Declarations of LAPACK functions used by hmat*/
#ifndef _LAPACK_OVERLOADS_HPP
#define _LAPACK_OVERLOADS_HPP
#include "config.h"

#include "data_types.hpp"
#include <algorithm>

#ifdef HAVE_MKL_H
  #define MKL_Complex8 hmat::C_t
  #define MKL_Complex16 hmat::Z_t
  #include <mkl_lapacke.h>
#else
  #define armpl_singlecomplex_t hmat::C_t
  #define armpl_doublecomplex_t hmat::Z_t
  #define LAPACK_COMPLEX_CUSTOM
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

namespace {
inline int gesddRworkSize(char jobz, int m, int n) {
    if (jobz == 'N')
        return 7 * std::min(m, n);
    else
        return std::min(m, n) * std::max(5 * std::min(m, n) +
            7, 2 * std::max(m, n) + 2 * std::min(m, n) + 1);
}
}

namespace proxy_lapack {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/*      SUBROUTINE SGETRF( M, N, A, LDA, IPIV, INFO )*/
/*      SGETRF COMPUTES AN LU FACTORIZATION OF A GENERAL M-BY-N MATRIX A USING PARTIAL PIVOTING WITH ROW INTERCHANGES.*/

template <typename T> int getrf(int m, int n, T *a, int lda, int *ipiv);

template <>
inline int getrf<hmat::S_t>(int m, int n, hmat::S_t *a, int lda, int *ipiv) {
  return LAPACKE_sgetrf(LAPACK_COL_MAJOR, m, n, a, lda, ipiv);
}

template <>
inline int getrf<hmat::D_t>(int m, int n, hmat::D_t *a, int lda, int *ipiv) {
  return LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, a, lda, ipiv);
}
template <>
inline int getrf<hmat::C_t>(int m, int n, hmat::C_t *a, int lda, int *ipiv) {
  return LAPACKE_cgetrf(LAPACK_COL_MAJOR, m, n, a, lda, ipiv);
}
template <>
inline int getrf<hmat::Z_t>(int m, int n, hmat::Z_t *a, int lda, int *ipiv) {
  return LAPACKE_zgetrf(LAPACK_COL_MAJOR, m, n, a, lda, ipiv);
}

/*      SUBROUTINE SGETRI( N, A, LDA, IPIV, WORK, LWORK, INFO )*/
/*      SGETRI COMPUTES THE INVERSE OF A MATRIX USING THE LU FACTORIZATION COMPUTED BY SGETRF.*/

template <typename T> int getri(int n, T *a, int lda, const int *ipiv);

template <>
inline int getri<hmat::S_t>(int n, hmat::S_t *a, int lda, const int *ipiv) {
  return LAPACKE_sgetri(LAPACK_COL_MAJOR, n, a, lda, ipiv);
}
template <>
inline int getri<hmat::D_t>(int n, hmat::D_t *a, int lda, const int *ipiv) {
  return LAPACKE_dgetri(LAPACK_COL_MAJOR, n, a, lda, ipiv);
}
template <>
inline int getri<hmat::C_t>(int n, hmat::C_t *a, int lda, const int *ipiv) {
  return LAPACKE_cgetri(LAPACK_COL_MAJOR, n, a, lda, ipiv);
}
template <>
inline int getri<hmat::Z_t>(int n, hmat::Z_t *a, int lda, const int *ipiv) {
  return LAPACKE_zgetri(LAPACK_COL_MAJOR, n, a, lda, ipiv);
}

/*      SUBROUTINE SGETRS( TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO)*/
/* Solves a system of linear equations with an LU-factored square matrix, with
   multiple right-hand sides */

template <typename T>
int getrs(char trans, int n, int nrhs, const T *a, int lda, const int *ipiv,
          T *b, int ldb);

template <>
inline int getrs<hmat::S_t>(char trans, int n, int nrhs, const hmat::S_t *a,
                            int lda, const int *ipiv, hmat::S_t *b, int ldb) {
  return LAPACKE_sgetrs(LAPACK_COL_MAJOR, trans, n, nrhs, a, lda, ipiv, b, ldb);
}
template <>
inline int getrs<hmat::D_t>(char trans, int n, int nrhs, const hmat::D_t *a,
                            int lda, const int *ipiv, hmat::D_t *b, int ldb) {
  return LAPACKE_dgetrs(LAPACK_COL_MAJOR, trans, n, nrhs, a, lda, ipiv, b, ldb);
}
template <>
inline int getrs<hmat::C_t>(char trans, int n, int nrhs, const hmat::C_t *a,
                            int lda, const int *ipiv, hmat::C_t *b, int ldb) {
  return LAPACKE_cgetrs(LAPACK_COL_MAJOR, trans, n, nrhs, a, lda, ipiv, b, ldb);
}
template <>
inline int getrs<hmat::Z_t>(char trans, int n, int nrhs, const hmat::Z_t *a,
                            int lda, const int *ipiv, hmat::Z_t *b, int ldb) {
  return LAPACKE_zgetrs(LAPACK_COL_MAJOR, trans, n, nrhs, a, lda, ipiv, b, ldb);
}

//       SUBROUTINE ZGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
//  ZGEQRF computes a QR factorization of a real M-by-N matrix A:
//  A = Q * R.
template <typename T> int geqrf(int m, int n, T *a, int lda, T *tau);

template <>
inline int geqrf<hmat::S_t>(int m, int n, hmat::S_t *a, int lda,
                            hmat::S_t *tau) {
  return LAPACKE_sgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
}
template <>
inline int geqrf<hmat::D_t>(int m, int n, hmat::D_t *a, int lda,
                            hmat::D_t *tau) {
  return LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
}
template <>
inline int geqrf<hmat::C_t>(int m, int n, hmat::C_t *a, int lda,
                            hmat::C_t *tau) {
  return LAPACKE_cgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
}
template <>
inline int geqrf<hmat::Z_t>(int m, int n, hmat::Z_t *a, int lda,
                            hmat::Z_t *tau) {
  return LAPACKE_zgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
}

/*      SUBROUTINE SGESDD( JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT,*/
/*                         WORK, LWORK, IWORK, INFO )                     */
/*      ZGESDD computes the singular value decomposition (SVD) of a complex */
/*      M-by-N matrix A, optionally computing the left and/or right singular */
/*      vectors   */
template <typename T, typename Treal>
int gesdd(char jobz, int m, int n, T *a, int lda, Treal *s, T *u, int ldu,
          T *vt, int ldvt);

template <>
inline int gesdd<hmat::S_t, hmat::S_t>(char jobz, int m, int n, hmat::S_t *a,
                                       int lda, hmat::S_t *s, hmat::S_t *u,
                                       int ldu, hmat::S_t *vt, int ldvt) {
  return LAPACKE_sgesdd(LAPACK_COL_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt,
                        ldvt);
}
template <>
inline int gesdd<hmat::D_t, hmat::D_t>(char jobz, int m, int n, hmat::D_t *a,
                                       int lda, hmat::D_t *s, hmat::D_t *u,
                                       int ldu, hmat::D_t *vt, int ldvt) {
  return LAPACKE_dgesdd(LAPACK_COL_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt,
                        ldvt);
}
template <>
inline int gesdd<hmat::C_t, hmat::S_t>(char jobz, int m, int n, hmat::C_t *a,
                                       int lda, hmat::S_t *s, hmat::C_t *u,
                                       int ldu, hmat::C_t *vt, int ldvt) {
  return LAPACKE_cgesdd(LAPACK_COL_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt,
                        ldvt);
}
template <>
inline int gesdd<hmat::Z_t, hmat::D_t>(char jobz, int m, int n, hmat::Z_t *a,
                                       int lda, hmat::D_t *s, hmat::Z_t *u,
                                       int ldu, hmat::Z_t *vt, int ldvt) {
  return LAPACKE_zgesdd(LAPACK_COL_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt,
                        ldvt);
}

/*      SUBROUTINE SGESVD( JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT,*/
/*                         WORK, LWORK, INFO )                     */
/*      ZGESVD computes the singular value decomposition (SVD) of a complex */
/*      M-by-N matrix A, optionally computing the left and/or right singular */
/*      vectors   */
template <typename T>
int gesvd(char jobu, char jobvt, int m, int n, T *a, int lda,
          typename hmat::Types<T>::real *s, T *u, int ldu, T *vt, int ldvt,
          typename hmat::Types<T>::real *superb);

template <>
inline int gesvd<hmat::S_t>(char jobu, char jobvt, int m, int n, hmat::S_t *a,
                            int lda, hmat::S_t *s, hmat::S_t *u, int ldu,
                            hmat::S_t *vt, int ldvt, hmat::S_t *superb) {
  return LAPACKE_sgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, a, lda, s, u, ldu,
                        vt, ldvt, superb);
}
template <>
inline int gesvd<hmat::D_t>(char jobu, char jobvt, int m, int n, hmat::D_t *a,
                            int lda, hmat::D_t *s, hmat::D_t *u, int ldu,
                            hmat::D_t *vt, int ldvt, hmat::D_t *superb) {
  return LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, a, lda, s, u, ldu,
                        vt, ldvt, superb);
}
template <>
inline int gesvd<hmat::C_t>(char jobu, char jobvt, int m, int n, hmat::C_t *a,
                            int lda, hmat::S_t *s, hmat::C_t *u, int ldu,
                            hmat::C_t *vt, int ldvt, hmat::S_t *superb) {
  return LAPACKE_cgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, a, lda, s, u, ldu,
                        vt, ldvt, superb);
}
template <>
inline int gesvd<hmat::Z_t>(char jobu, char jobvt, int m, int n, hmat::Z_t *a,
                            int lda, hmat::D_t *s, hmat::Z_t *u, int ldu,
                            hmat::Z_t *vt, int ldvt, hmat::D_t *superb) {
  return LAPACKE_zgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, a, lda, s, u, ldu,
                        vt, ldvt, superb);
}

//       SUBROUTINE DORMQR( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
//      $                   WORK, LWORK, INFO )
//  DORMQR overwrites the general real M-by-N matrix C with
//
//                  SIDE = 'L'     SIDE = 'R'
//  TRANS = 'N':      Q * C          C * Q
//  TRANS = 'T':      Q**T * C       C * Q**T
//
//  where Q is a real orthogonal matrix defined as the product of k
//  elementary reflectors
//
//        Q = H(1) H(2) . . . H(k)
//
//  as returned by DGEQRF. Q is of order M if SIDE = 'L' and of order N
//  if SIDE = 'R'.
template<typename T>
int ormqr(char side, char trans, int m, int n, int k, const T* a, int lda, const T* tau, T* c, int ldc);

template<>
inline int ormqr<hmat::S_t>(char side, char trans, int m, int n, int k, const hmat::S_t* a, int lda, const hmat::S_t* tau, hmat::S_t* c, int ldc) {
  return LAPACKE_sormqr(LAPACK_COL_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc);
}
template<>
inline int
ormqr<hmat::D_t>(char side, char trans, int m, int n, int k, const hmat::D_t* a, int lda, const hmat::D_t* tau, hmat::D_t* c, int ldc) {
  return LAPACKE_dormqr(LAPACK_COL_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc);
}

//       SUBROUTINE CUNMQR( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
//      $                   WORK, LWORK, INFO )
//  CUNMQR overwrites the general complex M-by-N matrix C with
//
//                  SIDE = 'L'     SIDE = 'R'
//  TRANS = 'N':      Q * C          C * Q
//  TRANS = 'C':      Q**H * C       C * Q**H
//
//  where Q is a complex unitary matrix defined as the product of k
//  elementary reflectors
//
//        Q = H(1) H(2) . . . H(k)
//
//  as returned by CGEQRF. Q is of order M if SIDE = 'L' and of order N
//  if SIDE = 'R'.

template <typename T>
int unmqr(char side, char trans, int m, int n, int k, const T *a, int lda,
          const T *tau, T *c, int ldc);

template <>
inline int unmqr<hmat::C_t>(char side, char trans, int m, int n, int k,
                            const hmat::C_t *a, int lda, const hmat::C_t *tau,
                            hmat::C_t *c, int ldc) {
  return LAPACKE_cunmqr(LAPACK_COL_MAJOR, side, trans, m, n, k, a, lda, tau, c,
                        ldc);
}
template <>
inline int unmqr<hmat::Z_t>(char side, char trans, int m, int n, int k,
                            const hmat::Z_t *a, int lda, const hmat::Z_t *tau,
                            hmat::Z_t *c, int ldc) {
  return LAPACKE_zunmqr(LAPACK_COL_MAJOR, side, trans, m, n, k, a, lda, tau, c,
                        ldc);
}

template <typename T>
void laswp(int n, T *a, int lda, int k1, int k2, const int *ipiv, int incx);

template <>
inline void laswp<hmat::S_t>(int n, hmat::S_t *a, int lda, int k1, int k2,
                             const int *ipiv, int incx) {
  LAPACKE_slaswp(LAPACK_COL_MAJOR, n, a, lda, k1, k2, ipiv, incx);
}
template <>
inline void laswp<hmat::D_t>(int n, hmat::D_t *a, int lda, int k1, int k2,
                             const int *ipiv, int incx) {
  LAPACKE_dlaswp(LAPACK_COL_MAJOR, n, a, lda, k1, k2, ipiv, incx);
}
template <>
inline void laswp<hmat::C_t>(int n, hmat::C_t *a, int lda, int k1, int k2,
                             const int *ipiv, int incx) {
  LAPACKE_claswp(LAPACK_COL_MAJOR, n, a, lda, k1, k2, ipiv, incx);
}
template <>
inline void laswp<hmat::Z_t>(int n, hmat::Z_t *a, int lda, int k1, int k2,
                             const int *ipiv, int incx) {
  LAPACKE_zlaswp(LAPACK_COL_MAJOR, n, a, lda, k1, k2, ipiv, incx);
}

template <typename T> int potrf(char uplo, int n, T *a, int lda);

template <>
inline int potrf<hmat::S_t>(char uplo, int n, hmat::S_t *a, int lda) {
  return LAPACKE_spotrf(LAPACK_COL_MAJOR, uplo, n, a, lda);
}
template <>
inline int potrf<hmat::D_t>(char uplo, int n, hmat::D_t *a, int lda) {
  return LAPACKE_dpotrf(LAPACK_COL_MAJOR, uplo, n, a, lda);
}
template <>
inline int potrf<hmat::C_t>(char uplo, int n, hmat::C_t *a, int lda) {
  return LAPACKE_cpotrf(LAPACK_COL_MAJOR, uplo, n, a, lda);
}
template <>
inline int potrf<hmat::Z_t>(char uplo, int n, hmat::Z_t *a, int lda) {
  return LAPACKE_zpotrf(LAPACK_COL_MAJOR, uplo, n, a, lda);
}

template<typename T> void lacgv(int n, T* a, int incx);
template<> inline void lacgv<hmat::S_t>(int, hmat::S_t*, int) {}
template<> inline void lacgv<hmat::D_t>(int, hmat::D_t*, int) {}
template<> inline void lacgv<hmat::C_t>(int n, hmat::C_t* a, int incx) { LAPACKE_clacgv(n, a, incx); }
template<> inline void lacgv<hmat::Z_t>(int n, hmat::Z_t* a, int incx) { LAPACKE_zlacgv(n, a, incx); }

}  // end namespace proxy_lapack

// Some LAPACK routines only exist for real or complex matrices; for instance,
// ORMQR is for real orthogonal matrices, and UNMQR for unitary complex matrices.
namespace proxy_lapack_convenience {

// WARNING: trans equals N or T for real matrices, but N or C for complex matrices.
//          or_un_mqr accepts N/T and replace T by C for complex matrices.
template<typename T>
int or_un_mqr(char side, char trans, int m, int n, int k, const T* a, int lda, const T* tau, T* c, int ldc);

template<>
inline int
or_un_mqr<hmat::S_t>(char side, char trans, int m, int n, int k, const hmat::S_t* a, int lda, const hmat::S_t* tau, hmat::S_t* c, int ldc) {
  return proxy_lapack::ormqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}
template<>
inline int
or_un_mqr<hmat::D_t>(char side, char trans, int m, int n, int k, const hmat::D_t* a, int lda, const hmat::D_t* tau, hmat::D_t* c, int ldc) {
  return proxy_lapack::ormqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}
template<>
inline int
or_un_mqr<hmat::C_t>(char side, char trans, int m, int n, int k, const hmat::C_t* a, int lda, const hmat::C_t* tau, hmat::C_t* c, int ldc) {
  const char t = (trans == 'N' ? 'N' : 'C');
  return proxy_lapack::unmqr(side, t, m, n, k, a, lda, tau, c, ldc);
}
template<>
inline int
or_un_mqr<hmat::Z_t>(char side, char trans, int m, int n, int k, const hmat::Z_t* a, int lda, const hmat::Z_t* tau, hmat::Z_t* c, int ldc) {
  const char t = (trans == 'N' ? 'N' : 'C');
  return proxy_lapack::unmqr(side, t, m, n, k, a, lda, tau, c, ldc);
}

}  // end namespace proxy_lapack_convenience

namespace hmat {

template<typename T> double real(const T x);
template<> inline double real(const hmat::C_t x) {
  return x.real();
}
template<> inline double real(const hmat::Z_t x) {
  return x.real();
}
template<typename T> inline double real(const T x) {
  return x;
}
template<typename T> double imag(const T x);
template<> inline double imag(const hmat::C_t x) {
  return x.imag();
}
template<> inline double imag(const hmat::Z_t x) {
  return x.imag();
}
template<typename T> inline double imag(const T x) {
  return 0;
}

}  // end namespace hmat

#endif // DOXYGEN_SHOULD_SKIP_THIS
#endif // _LAPACK_OVERLOADS_HPP
