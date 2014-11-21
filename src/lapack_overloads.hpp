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

#include "data_types.hpp"

#define F77_FUNC(a, b) a ##_

namespace proxy_lapack {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/*      SUBROUTINE SGETRF( M, N, A, LDA, IPIV, INFO )*/
/*      SGETRF COMPUTES AN LU FACTORIZATION OF A GENERAL M-BY-N MATRIX A USING PARTIAL PIVOTING WITH ROW INTERCHANGES.*/
#define _SGETRF_ F77_FUNC(sgetrf,SGETRF)
#define _DGETRF_ F77_FUNC(dgetrf,DGETRF)
#define _CGETRF_ F77_FUNC(cgetrf,CGETRF)
#define _ZGETRF_ F77_FUNC(zgetrf,ZGETRF)
extern "C" {
void _SGETRF_(int*, int*, S_t*, int*, int*, int*);
void _DGETRF_(int*, int*, D_t*, int*, int*, int*);
void _CGETRF_(int*, int*, C_t*, int*, int*, int*);
void _ZGETRF_(int*, int*, Z_t*, int*, int*, int*);
}

template <typename T>
int getrf(int m, int n, T* a, int lda, int* ipiv);

template <>
inline int
getrf<S_t>(int m, int n, S_t* a, int lda, int* ipiv) {
  int info = 0;
  _SGETRF_(&m, &n, a, &lda, ipiv, &info);
  return info;
}
template <>
inline int
getrf<D_t>(int m, int n, D_t* a, int lda, int* ipiv) {
  int info = 0;
  _DGETRF_(&m, &n, a, &lda, ipiv, &info);
  return info;
}
template <>
inline int
getrf<C_t>(int m, int n, C_t* a, int lda, int* ipiv) {
  int info = 0;
  _CGETRF_(&m, &n, a, &lda, ipiv, &info);
  return info;
}
template <>
inline int
getrf<Z_t>(int m, int n, Z_t* a, int lda, int* ipiv) {
  int info = 0;
  _ZGETRF_(&m, &n, a, &lda, ipiv, &info);
  return info;
}
#undef _SGETRF_
#undef _DGETRF_
#undef _CGETRF_
#undef _ZGETRF_

/*      SUBROUTINE SGETRI( N, A, LDA, IPIV, WORK, LWORK, INFO )*/
/*      SGETRI COMPUTES THE INVERSE OF A MATRIX USING THE LU FACTORIZATION COMPUTED BY SGETRF.*/
#define _SGETRI_ F77_FUNC(sgetri,SGETRI)
#define _DGETRI_ F77_FUNC(dgetri,DGETRI)
#define _CGETRI_ F77_FUNC(cgetri,CGETRI)
#define _ZGETRI_ F77_FUNC(zgetri,ZGETRI)
extern "C" void _SGETRI_(int*, S_t*, int*, const int*, S_t*, int*, int*);
extern "C" void _DGETRI_(int*, D_t*, int*, const int*, D_t*, int*, int*);
extern "C" void _CGETRI_(int*, C_t*, int*, const int*, C_t*, int*, int*);
extern "C" void _ZGETRI_(int*, Z_t*, int*, const int*, Z_t*, int*, int*);

template<typename T>
int getri(int n, T* a, int lda, const int* ipiv, T* work, int lwork);

template<>
inline int
getri<S_t>(int n, S_t* a, int lda, const int* ipiv, S_t* work, int lwork) {
  int info = 0;
  _SGETRI_(&n, a, &lda, ipiv, work, &lwork, &info);
  return info;
}
template<>
inline int
getri<D_t>(int n, D_t* a, int lda, const int* ipiv, D_t* work, int lwork) {
  int info = 0;
  _DGETRI_(&n, a, &lda, ipiv, work, &lwork, &info);
  return info;
}
template<>
inline int
getri<C_t>(int n, C_t* a, int lda, const int* ipiv, C_t* work, int lwork) {
  int info = 0;
  _CGETRI_(&n, a, &lda, ipiv, work, &lwork, &info);
  return info;
}
template<>
inline int
getri<Z_t>(int n, Z_t* a, int lda, const int* ipiv, Z_t* work, int lwork) {
  int info = 0;
  _ZGETRI_(&n, a, &lda, ipiv, work, &lwork, &info);
  return info;
}
#undef _SGETRI_
#undef _DGETRI_
#undef _CGETRI_
#undef _ZGETRI_

/*      SUBROUTINE SGETRS( TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO)*/
/* Solves a system of linear equations with an LU-factored square matrix, with
   multiple right-hand sides */
#define _SGETRS_ F77_FUNC(sgetrs,SGETRS)
#define _DGETRS_ F77_FUNC(dgetrs,DGETRS)
#define _CGETRS_ F77_FUNC(cgetrs,CGETRS)
#define _ZGETRS_ F77_FUNC(zgetrs,ZGETRS)
extern "C" void _SGETRS_(char*, int*, int*, const S_t*, int*, const int*, S_t*, int*, int*);
extern "C" void _DGETRS_(char*, int*, int*, const D_t*, int*, const int*, D_t*, int*, int*);
extern "C" void _CGETRS_(char*, int*, int*, const C_t*, int*, const int*, C_t*, int*, int*);
extern "C" void _ZGETRS_(char*, int*, int*, const Z_t*, int*, const int*, Z_t*, int*, int*);

template<typename T>
int getrs(char trans, int n, int nrhs, const T* a, int lda, const int* ipiv, T* b, int ldb);

template<>
inline int
getrs<S_t>(char trans, int n, int nrhs, const S_t* a, int lda, const int* ipiv, S_t* b, int ldb) {
  int info = 0;
  _SGETRS_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}
template<>
inline int
getrs<D_t>(char trans, int n, int nrhs, const D_t* a, int lda, const int* ipiv, D_t* b, int ldb) {
  int info = 0;
  _DGETRS_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}
template<>
inline int
getrs<C_t>(char trans, int n, int nrhs, const C_t* a, int lda, const int* ipiv, C_t* b, int ldb) {
  int info = 0;
  _CGETRS_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}
template<>
inline int
getrs<Z_t>(char trans, int n, int nrhs, const Z_t* a, int lda, const int* ipiv, Z_t* b, int ldb) {
  int info = 0;
  _ZGETRS_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}
#undef _SGETRS_
#undef _DGETRS_
#undef _CGETRS_
#undef _ZGETRS_

//       SUBROUTINE ZGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
//  ZGEQRF computes a QR factorization of a real M-by-N matrix A:
//  A = Q * R.
#define _SGEQRF_ F77_FUNC(sgeqrf,SGEQRF)
#define _DGEQRF_ F77_FUNC(dgeqrf,DGEQRF)
#define _CGEQRF_ F77_FUNC(cgeqrf,CGEQRF)
#define _ZGEQRF_ F77_FUNC(zgeqrf,ZGEQRF)
extern "C" void _SGEQRF_(int*, int*, S_t*, int*, S_t*, S_t*, int*, int*);
extern "C" void _DGEQRF_(int*, int*, D_t*, int*, D_t*, D_t*, int*, int*);
extern "C" void _CGEQRF_(int*, int*, C_t*, int*, C_t*, C_t*, int*, int*);
extern "C" void _ZGEQRF_(int*, int*, Z_t*, int*, Z_t*, Z_t*, int*, int*);

template<typename T>
int geqrf(int m, int n, T* a, int lda, T* tau, T* work, int lwork);

template<>
inline int
geqrf<S_t>(int m, int n, S_t* a, int lda, S_t* tau, S_t* work, int lwork) {
  int info = 0;
  _SGEQRF_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}
template<>
inline int
geqrf<D_t>(int m, int n, D_t* a, int lda, D_t* tau, D_t* work, int lwork) {
  int info = 0;
  _DGEQRF_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}
template<>
inline int
geqrf<C_t>(int m, int n, C_t* a, int lda, C_t* tau, C_t* work, int lwork) {
  int info = 0;
  _CGEQRF_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}
template<>
inline int
geqrf<Z_t>(int m, int n, Z_t* a, int lda, Z_t* tau, Z_t* work, int lwork) {
  int info = 0;
  _ZGEQRF_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}
#undef _SGEQRF_
#undef _DGEQRF_
#undef _CGEQRF_
#undef _ZGEQRF_

/*      SUBROUTINE SGESVD( JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT,*/
/*                         WORK, LWORK, INFO )                     */
/*      ZGESVD computes the singular value decomposition (SVD) of a complex */
/*      M-by-N matrix A, optionally computing the left and/or right singular */
/*      vectors   */
#define _SGESVD_ F77_FUNC(sgesvd,SGESVD)
#define _DGESVD_ F77_FUNC(dgesvd,DGESVD)
#define _CGESVD_ F77_FUNC(cgesvd,CGESVD)
#define _ZGESVD_ F77_FUNC(zgesvd,ZGESVD)
extern "C" void _SGESVD_(char*, char*, int*, int*, S_t*, int*,  float*, S_t*, int*, S_t*, int*, S_t*, int*, int*);
extern "C" void _DGESVD_(char*, char*, int*, int*, D_t*, int*, double*, D_t*, int*, D_t*, int*, D_t*, int*, int*);
extern "C" void _CGESVD_(char*, char*, int*, int*, C_t*, int*,  float*, C_t*, int*, C_t*, int*, C_t*, int*,  float*, int*);
extern "C" void _ZGESVD_(char*, char*, int*, int*, Z_t*, int*, double*, Z_t*, int*, Z_t*, int*, Z_t*, int*, double*, int*);

template<typename T, typename Treal>
int gesvd(char jobu, char jobvt, int m, int n, T* a, int lda,  Treal* s, T* u, int ldu, T* vt, int ldvt, T* work, int lwork);

template<>
inline int
gesvd<S_t, S_t>(char jobu, char jobvt, int m, int n, S_t* a, int lda,  S_t* s, S_t* u, int ldu, S_t* vt, int ldvt, S_t* work, int lwork) {
  int info = 0;
  _SGESVD_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
  return info;
}
template<>
inline int
gesvd<D_t, D_t>(char jobu, char jobvt, int m, int n, D_t* a, int lda,  D_t* s, D_t* u, int ldu, D_t* vt, int ldvt, D_t* work, int lwork) {
  int info = 0;
  _DGESVD_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
  return info;
}
template<>
inline int
gesvd<C_t, S_t>(char jobu, char jobvt, int m, int n, C_t* a, int lda,  S_t* s, C_t* u, int ldu, C_t* vt, int ldvt, C_t* work, int lwork) {
  int info = 0;
  S_t* rwork = (lwork == -1 ? NULL : new S_t[5 * std::min(m, n)]);
  _CGESVD_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info);
  if (rwork) delete [] rwork;
  return info;
}
template<>
inline int
gesvd<Z_t, D_t>(char jobu, char jobvt, int m, int n, Z_t* a, int lda,  D_t* s, Z_t* u, int ldu, Z_t* vt, int ldvt, Z_t* work, int lwork) {
  int info = 0;
  D_t* rwork = (lwork == -1 ? NULL : new D_t[5 * std::min(m, n)]);
  _ZGESVD_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info);
  if (rwork) delete [] rwork;
  return info;
}
#undef _SGESVD_
#undef _DGESVD_
#undef _CGESVD_
#undef _ZGESVD_

//      SUBROUTINE SORGQR( M, N, K, A, LDA, TAU, WORK, LWORK, INFO )
//   SORGQR generates an M-by-N real matrix Q with orthonormal columns,
//   which is defined as the first N columns of a product of K elementary
//   reflectors of order M
//
//         Q  =  H(1) H(2) . . . H(k)
//
//   as returned by SGEQRF.
#define _SORGQR_ F77_FUNC(sorgqr,SORGQR)
#define _DORGQR_ F77_FUNC(dorgqr,DORGQR)
extern "C" void _SORGQR_(int*, int*, int*, S_t*, int*, const S_t*, S_t*, int*, int*);
extern "C" void _DORGQR_(int*, int*, int*, D_t*, int*, const D_t*, D_t*, int*, int*);

template<typename T>
int orgqr(int m, int n, int k, T* a, int lda, const T* tau, T* work, int lwork);

template<>
inline int
orgqr<S_t>(int m, int n, int k, S_t* a, int lda, const S_t* tau, S_t* work, int lwork) {
  int info = 0;
  _SORGQR_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}
template<>
inline int
orgqr<D_t>(int m, int n, int k, D_t* a, int lda, const D_t* tau, D_t* work, int lwork) {
  int info = 0;
  _DORGQR_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}
#undef _SORGQR_
#undef _DORGQR_

//       SUBROUTINE CUNGQR( M, N, K, A, LDA, TAU, WORK, LWORK, INFO )
//  CUNGQR generates an M-by-N complex matrix Q with orthonormal columns,
//  which is defined as the first N columns of a product of K elementary
//  reflectors of order M
//
//        Q  =  H(1) H(2) . . . H(k)
//
//  as returned by CGEQRF.
#define _CUNGQR_ F77_FUNC(cungqr,CUNGQR)
#define _ZUNGQR_ F77_FUNC(zungqr,ZUNGQR)
extern "C" void _CUNGQR_(int*, int*, int*, C_t*, int*, const C_t*, C_t*, int*, int*);
extern "C" void _ZUNGQR_(int*, int*, int*, Z_t*, int*, const Z_t*, Z_t*, int*, int*);

template<typename T>
int ungqr(int m, int n, int k, T* a, int lda, const T* tau, T* work, int lwork);

template<>
inline int
ungqr<C_t>(int m, int n, int k, C_t* a, int lda, const C_t* tau, C_t* work, int lwork) {
  int info = 0;
  _CUNGQR_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}
template<>
inline int
ungqr<Z_t>(int m, int n, int k, Z_t* a, int lda, const Z_t* tau, Z_t* work, int lwork) {
  int info = 0;
  _ZUNGQR_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}
#undef _CUNGQR_
#undef _ZUNGQR_

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
#define _SORMQR_ F77_FUNC(sormqr,SORMQR)
#define _DORMQR_ F77_FUNC(dormqr,DORMQR)
extern "C" void _SORMQR_(char*, char*, int*, int*, int*, const S_t*, int*, const S_t*, S_t*, int*, S_t*, int*, int*);
extern "C" void _DORMQR_(char*, char*, int*, int*, int*, const D_t*, int*, const D_t*, D_t*, int*, D_t*, int*, int*);

template<typename T>
int ormqr(char side, char trans, int m, int n, int k, const T* a, int lda, const T* tau, T* c, int ldc, T* work, int lwork);

template<>
inline int
ormqr<S_t>(char side, char trans, int m, int n, int k, const S_t* a, int lda, const S_t* tau, S_t* c, int ldc, S_t* work, int lwork) {
  int info = 0;
  _SORMQR_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}
template<>
inline int
ormqr<D_t>(char side, char trans, int m, int n, int k, const D_t* a, int lda, const D_t* tau, D_t* c, int ldc, D_t* work, int lwork) {
  int info = 0;
  _DORMQR_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}
#undef _SORMQR_
#undef _DORMQR_

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
#define _CUNMQR_ F77_FUNC(cunmqr,CUNMQR)
#define _ZUNMQR_ F77_FUNC(zunmqr,ZUNMQR)
extern "C" void _CUNMQR_(char*, char*, int*, int*, int*, const C_t*, int*, const C_t*, C_t*, int*, C_t*, int*, int*);
extern "C" void _ZUNMQR_(char*, char*, int*, int*, int*, const Z_t*, int*, const Z_t*, Z_t*, int*, Z_t*, int*, int*);

template<typename T>
int unmqr(char side, char trans, int m, int n, int k, const T* a, int lda, const T* tau, T* c, int ldc, T* work, int lwork);

template<>
inline int
unmqr<C_t>(char side, char trans, int m, int n, int k, const C_t* a, int lda, const C_t* tau, C_t* c, int ldc, C_t* work, int lwork) {
  int info = 0;
  _CUNMQR_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}
template<>
inline int
unmqr<Z_t>(char side, char trans, int m, int n, int k, const Z_t* a, int lda, const Z_t* tau, Z_t* c, int ldc, Z_t* work, int lwork) {
  int info = 0;
  _ZUNMQR_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}
#undef _CUNMQR_
#undef _ZUNMQR_

#define _SLASWP_ F77_FUNC(slaswp,SLASWP)
#define _DLASWP_ F77_FUNC(dlaswp,DLASWP)
#define _CLASWP_ F77_FUNC(claswp,CLASWP)
#define _ZLASWP_ F77_FUNC(zlaswp,ZLASWP)
extern "C" void _SLASWP_(int*, S_t*, int*, int*, int*, const int*, int*);
extern "C" void _DLASWP_(int*, D_t*, int*, int*, int*, const int*, int*);
extern "C" void _CLASWP_(int*, C_t*, int*, int*, int*, const int*, int*);
extern "C" void _ZLASWP_(int*, Z_t*, int*, int*, int*, const int*, int*);

template<typename T>
void laswp(int n, T* a, int lda, int k1, int k2, const int* ipiv, int incx);

template<>
inline void
laswp<S_t>(int n, S_t* a, int lda, int k1, int k2, const int* ipiv, int incx) {
  _SLASWP_(&n, a, &lda, &k1, &k2, ipiv, &incx);
}
template<>
inline void
laswp<D_t>(int n, D_t* a, int lda, int k1, int k2, const int* ipiv, int incx) {
  _DLASWP_(&n, a, &lda, &k1, &k2, ipiv, &incx);
}
template<>
inline void
laswp<C_t>(int n, C_t* a, int lda, int k1, int k2, const int* ipiv, int incx) {
  _CLASWP_(&n, a, &lda, &k1, &k2, ipiv, &incx);
}
template<>
inline void
laswp<Z_t>(int n, Z_t* a, int lda, int k1, int k2, const int* ipiv, int incx) {
  _ZLASWP_(&n, a, &lda, &k1, &k2, ipiv, &incx);
}
#undef _SLASWP_
#undef _DLASWP_
#undef _CLASWP_
#undef _ZLASWP_

#define _SPOTRF_ F77_FUNC(spotrf,SPOTRF)
#define _DPOTRF_ F77_FUNC(dpotrf,DPOTRF)
#define _CPOTRF_ F77_FUNC(cpotrf,CPOTRF)
#define _ZPOTRF_ F77_FUNC(zpotrf,ZPOTRF)
extern "C" {
void _SPOTRF_(char * uplo, int* n, S_t* a, int* lda, int* info);
void _DPOTRF_(char * uplo, int* n, D_t* a, int* lda, int* info);
void _CPOTRF_(char * uplo, int* n, C_t* a, int* lda, int* info);
void _ZPOTRF_(char * uplo, int* n, Z_t* a, int* lda, int* info);
}

template <typename T> int potrf(char uplo, int n, T* a, int lda);

template <> inline int potrf<S_t>(char uplo, int n, S_t* a, int lda) {
  int info = 0;
  _SPOTRF_(&uplo, &n, a, &lda, &info);
  return info;
}
template <> inline int potrf<D_t>(char uplo, int n, D_t* a, int lda) {
  int info = 0;
  _DPOTRF_(&uplo, &n, a, &lda, &info);
  return info;
}
template <> inline int potrf<C_t>(char uplo, int n, C_t* a, int lda) {
  int info = 0;
  _CPOTRF_(&uplo, &n, a, &lda, &info);
  return info;
}
template <> inline int potrf<Z_t>(char uplo, int n, Z_t* a, int lda) {
  int info = 0;
  _ZPOTRF_(&uplo, &n, a, &lda, &info);
  return info;
}
#undef _SPOTRF_
#undef _DPOTRF_
#undef _CPOTRF_
#undef _ZPOTRF_


}  // end namespace proxy_lapack

// Some LAPACK routines only exist for real or complex matrices; for instance,
// ORMQR is for real orthogonal matrices, and UNMQR for unitary complex matrices.
namespace proxy_lapack_convenience {

// WARNING: trans equals N or T for real matrices, but N or C for complex matrices.
//          or_un_mqr accepts N/T and replace T by C for complex matrices.
template<typename T>
int or_un_mqr(char side, char trans, int m, int n, int k, const T* a, int lda, const T* tau, T* c, int ldc, T* work, int lwork);

template<>
inline int
or_un_mqr<S_t>(char side, char trans, int m, int n, int k, const S_t* a, int lda, const S_t* tau, S_t* c, int ldc, S_t* work, int lwork) {
  return proxy_lapack::ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
}
template<>
inline int
or_un_mqr<D_t>(char side, char trans, int m, int n, int k, const D_t* a, int lda, const D_t* tau, D_t* c, int ldc, D_t* work, int lwork) {
  return proxy_lapack::ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
}
template<>
inline int
or_un_mqr<C_t>(char side, char trans, int m, int n, int k, const C_t* a, int lda, const C_t* tau, C_t* c, int ldc, C_t* work, int lwork) {
  const char t = (trans == 'N' ? 'N' : 'C');
  return proxy_lapack::unmqr(side, t, m, n, k, a, lda, tau, c, ldc, work, lwork);
}
template<>
inline int
or_un_mqr<Z_t>(char side, char trans, int m, int n, int k, const Z_t* a, int lda, const Z_t* tau, Z_t* c, int ldc, Z_t* work, int lwork) {
  const char t = (trans == 'N' ? 'N' : 'C');
  return proxy_lapack::unmqr(side, t, m, n, k, a, lda, tau, c, ldc, work, lwork);
}

}  // end namespace proxy_lapack_convenience

namespace hmat {

template<typename T> double real(const T x);
template<> inline double real(const C_t x) {
  return x.real();
}
template<> inline double real(const Z_t x) {
  return x.real();
}
template<typename T> inline double real(const T x) {
  return x;
}

}  // end namespace hmat

#endif // DOXYGEN_SHOULD_SKIP_THIS
#endif // _LAPACK_OVERLOADS_HPP
