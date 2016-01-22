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
#include <algorithm>

#define F77_FUNC(a, b) a ##_

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
#define _SGETRF_ F77_FUNC(sgetrf,SGETRF)
#define _DGETRF_ F77_FUNC(dgetrf,DGETRF)
#define _CGETRF_ F77_FUNC(cgetrf,CGETRF)
#define _ZGETRF_ F77_FUNC(zgetrf,ZGETRF)
extern "C" {
void _SGETRF_(int*, int*, hmat::S_t*, int*, int*, int*);
void _DGETRF_(int*, int*, hmat::D_t*, int*, int*, int*);
void _CGETRF_(int*, int*, hmat::C_t*, int*, int*, int*);
void _ZGETRF_(int*, int*, hmat::Z_t*, int*, int*, int*);
}

template <typename T>
int getrf(int m, int n, T* a, int lda, int* ipiv);

template <>
inline int
getrf<hmat::S_t>(int m, int n, hmat::S_t* a, int lda, int* ipiv) {
  int info = 0;
  _SGETRF_(&m, &n, a, &lda, ipiv, &info);
  return info;
}
template <>
inline int
getrf<hmat::D_t>(int m, int n, hmat::D_t* a, int lda, int* ipiv) {
  int info = 0;
  _DGETRF_(&m, &n, a, &lda, ipiv, &info);
  return info;
}
template <>
inline int
getrf<hmat::C_t>(int m, int n, hmat::C_t* a, int lda, int* ipiv) {
  int info = 0;
  _CGETRF_(&m, &n, a, &lda, ipiv, &info);
  return info;
}
template <>
inline int
getrf<hmat::Z_t>(int m, int n, hmat::Z_t* a, int lda, int* ipiv) {
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
extern "C" void _SGETRI_(int*, hmat::S_t*, int*, const int*, hmat::S_t*, int*, int*);
extern "C" void _DGETRI_(int*, hmat::D_t*, int*, const int*, hmat::D_t*, int*, int*);
extern "C" void _CGETRI_(int*, hmat::C_t*, int*, const int*, hmat::C_t*, int*, int*);
extern "C" void _ZGETRI_(int*, hmat::Z_t*, int*, const int*, hmat::Z_t*, int*, int*);

template<typename T>
int getri(int n, T* a, int lda, const int* ipiv, T* work, int lwork);

template<>
inline int
getri<hmat::S_t>(int n, hmat::S_t* a, int lda, const int* ipiv, hmat::S_t* work, int lwork) {
  int info = 0;
  _SGETRI_(&n, a, &lda, ipiv, work, &lwork, &info);
  return info;
}
template<>
inline int
getri<hmat::D_t>(int n, hmat::D_t* a, int lda, const int* ipiv, hmat::D_t* work, int lwork) {
  int info = 0;
  _DGETRI_(&n, a, &lda, ipiv, work, &lwork, &info);
  return info;
}
template<>
inline int
getri<hmat::C_t>(int n, hmat::C_t* a, int lda, const int* ipiv, hmat::C_t* work, int lwork) {
  int info = 0;
  _CGETRI_(&n, a, &lda, ipiv, work, &lwork, &info);
  return info;
}
template<>
inline int
getri<hmat::Z_t>(int n, hmat::Z_t* a, int lda, const int* ipiv, hmat::Z_t* work, int lwork) {
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
extern "C" void _SGETRS_(char*, int*, int*, const hmat::S_t*, int*, const int*, hmat::S_t*, int*, int*);
extern "C" void _DGETRS_(char*, int*, int*, const hmat::D_t*, int*, const int*, hmat::D_t*, int*, int*);
extern "C" void _CGETRS_(char*, int*, int*, const hmat::C_t*, int*, const int*, hmat::C_t*, int*, int*);
extern "C" void _ZGETRS_(char*, int*, int*, const hmat::Z_t*, int*, const int*, hmat::Z_t*, int*, int*);

template<typename T>
int getrs(char trans, int n, int nrhs, const T* a, int lda, const int* ipiv, T* b, int ldb);

template<>
inline int
getrs<hmat::S_t>(char trans, int n, int nrhs, const hmat::S_t* a, int lda, const int* ipiv, hmat::S_t* b, int ldb) {
  int info = 0;
  _SGETRS_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}
template<>
inline int
getrs<hmat::D_t>(char trans, int n, int nrhs, const hmat::D_t* a, int lda, const int* ipiv, hmat::D_t* b, int ldb) {
  int info = 0;
  _DGETRS_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}
template<>
inline int
getrs<hmat::C_t>(char trans, int n, int nrhs, const hmat::C_t* a, int lda, const int* ipiv, hmat::C_t* b, int ldb) {
  int info = 0;
  _CGETRS_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}
template<>
inline int
getrs<hmat::Z_t>(char trans, int n, int nrhs, const hmat::Z_t* a, int lda, const int* ipiv, hmat::Z_t* b, int ldb) {
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
extern "C" void _SGEQRF_(int*, int*, hmat::S_t*, int*, hmat::S_t*, hmat::S_t*, int*, int*);
extern "C" void _DGEQRF_(int*, int*, hmat::D_t*, int*, hmat::D_t*, hmat::D_t*, int*, int*);
extern "C" void _CGEQRF_(int*, int*, hmat::C_t*, int*, hmat::C_t*, hmat::C_t*, int*, int*);
extern "C" void _ZGEQRF_(int*, int*, hmat::Z_t*, int*, hmat::Z_t*, hmat::Z_t*, int*, int*);

template<typename T>
int geqrf(int m, int n, T* a, int lda, T* tau, T* work, int lwork);

template<>
inline int
geqrf<hmat::S_t>(int m, int n, hmat::S_t* a, int lda, hmat::S_t* tau, hmat::S_t* work, int lwork) {
  int info = 0;
  _SGEQRF_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}
template<>
inline int
geqrf<hmat::D_t>(int m, int n, hmat::D_t* a, int lda, hmat::D_t* tau, hmat::D_t* work, int lwork) {
  int info = 0;
  _DGEQRF_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}
template<>
inline int
geqrf<hmat::C_t>(int m, int n, hmat::C_t* a, int lda, hmat::C_t* tau, hmat::C_t* work, int lwork) {
  int info = 0;
  _CGEQRF_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}
template<>
inline int
geqrf<hmat::Z_t>(int m, int n, hmat::Z_t* a, int lda, hmat::Z_t* tau, hmat::Z_t* work, int lwork) {
  int info = 0;
  _ZGEQRF_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}
#undef _SGEQRF_
#undef _DGEQRF_
#undef _CGEQRF_
#undef _ZGEQRF_

/*      SUBROUTINE SGESDD( JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT,*/
/*                         WORK, LWORK, IWORK, INFO )                     */
/*      ZGESDD computes the singular value decomposition (SVD) of a complex */
/*      M-by-N matrix A, optionally computing the left and/or right singular */
/*      vectors   */
#define _SGESDD_ F77_FUNC(sgesdd,SGESDD)
#define _DGESDD_ F77_FUNC(dgesdd,DGESDD)
#define _CGESDD_ F77_FUNC(cgesdd,CGESDD)
#define _ZGESDD_ F77_FUNC(zgesdd,ZGESDD)
extern "C" void _SGESDD_(char*, int*, int*, hmat::S_t*, int*,  float*, hmat::S_t*, int*, hmat::S_t*, int*, hmat::S_t*, int*, int*, int*);
extern "C" void _DGESDD_(char*, int*, int*, hmat::D_t*, int*, double*, hmat::D_t*, int*, hmat::D_t*, int*, hmat::D_t*, int*, int*, int*);
extern "C" void _CGESDD_(char*, int*, int*, hmat::C_t*, int*,  float*, hmat::C_t*, int*, hmat::C_t*, int*, hmat::C_t*, int*, float*, int*, int*);
extern "C" void _ZGESDD_(char*, int*, int*, hmat::Z_t*, int*, double*, hmat::Z_t*, int*, hmat::Z_t*, int*, hmat::Z_t*, int*, double*, int*, int*);

template<typename T, typename Treal>
int gesdd(char jobz, int m, int n, T* a, int lda,  Treal* s, T* u, int ldu, T* vt, int ldvt, T* work, int lwork, int* iwork);

template<>
inline int
gesdd<hmat::S_t, hmat::S_t>(char jobz, int m, int n, hmat::S_t* a, int lda,  hmat::S_t* s, hmat::S_t* u, int ldu, hmat::S_t* vt, int ldvt, hmat::S_t* work, int lwork, int* iwork) {
  int info = 0;
  _SGESDD_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
  return info;
}
template<>
inline int
gesdd<hmat::D_t, hmat::D_t>(char jobz, int m, int n, hmat::D_t* a, int lda,  hmat::D_t* s, hmat::D_t* u, int ldu, hmat::D_t* vt, int ldvt, hmat::D_t* work, int lwork, int* iwork) {
  int info = 0;
  _DGESDD_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
  return info;
}
template<>
inline int
gesdd<hmat::C_t, hmat::S_t>(char jobz, int m, int n, hmat::C_t* a, int lda,  hmat::S_t* s, hmat::C_t* u, int ldu, hmat::C_t* vt, int ldvt, hmat::C_t* work, int lwork, int* iwork) {
  int info = 0;
  hmat::S_t* rwork = (lwork == -1 ? NULL : new hmat::S_t[gesddRworkSize(jobz, m, n)]);
  _CGESDD_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info);
  if (rwork) delete [] rwork;
  return info;
}
template<>
inline int
gesdd<hmat::Z_t, hmat::D_t>(char jobz, int m, int n, hmat::Z_t* a, int lda,  hmat::D_t* s, hmat::Z_t* u, int ldu, hmat::Z_t* vt, int ldvt, hmat::Z_t* work, int lwork, int* iwork) {
  int info = 0;
  hmat::D_t* rwork = (lwork == -1 ? NULL : new hmat::D_t[gesddRworkSize(jobz, m, n)]);
  _ZGESDD_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info);
  if (rwork) delete [] rwork;
  return info;
}
#undef _SGESDD_
#undef _DGESDD_
#undef _CGESDD_
#undef _ZGESDD_

/*      SUBROUTINE SGESVD( JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT,*/
/*                         WORK, LWORK, INFO )                     */
/*      ZGESVD computes the singular value decomposition (SVD) of a complex */
/*      M-by-N matrix A, optionally computing the left and/or right singular */
/*      vectors   */
#define _SGESVD_ F77_FUNC(sgesvd,SGESVD)
#define _DGESVD_ F77_FUNC(dgesvd,DGESVD)
#define _CGESVD_ F77_FUNC(cgesvd,CGESVD)
#define _ZGESVD_ F77_FUNC(zgesvd,ZGESVD)
extern "C" void _SGESVD_(char*, char*, int*, int*, hmat::S_t*, int*,  float*, hmat::S_t*, int*, hmat::S_t*, int*, hmat::S_t*, int*, int*);
extern "C" void _DGESVD_(char*, char*, int*, int*, hmat::D_t*, int*, double*, hmat::D_t*, int*, hmat::D_t*, int*, hmat::D_t*, int*, int*);
extern "C" void _CGESVD_(char*, char*, int*, int*, hmat::C_t*, int*,  float*, hmat::C_t*, int*, hmat::C_t*, int*, hmat::C_t*, int*,  float*, int*);
extern "C" void _ZGESVD_(char*, char*, int*, int*, hmat::Z_t*, int*, double*, hmat::Z_t*, int*, hmat::Z_t*, int*, hmat::Z_t*, int*, double*, int*);

template<typename T, typename Treal>
int gesvd(char jobu, char jobvt, int m, int n, T* a, int lda,  Treal* s, T* u, int ldu, T* vt, int ldvt, T* work, int lwork);

template<>
inline int
gesvd<hmat::S_t, hmat::S_t>(char jobu, char jobvt, int m, int n, hmat::S_t* a, int lda,  hmat::S_t* s, hmat::S_t* u, int ldu, hmat::S_t* vt, int ldvt, hmat::S_t* work, int lwork) {
  int info = 0;
  _SGESVD_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
  return info;
}
template<>
inline int
gesvd<hmat::D_t, hmat::D_t>(char jobu, char jobvt, int m, int n, hmat::D_t* a, int lda,  hmat::D_t* s, hmat::D_t* u, int ldu, hmat::D_t* vt, int ldvt, hmat::D_t* work, int lwork) {
  int info = 0;
  _DGESVD_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
  return info;
}
template<>
inline int
gesvd<hmat::C_t, hmat::S_t>(char jobu, char jobvt, int m, int n, hmat::C_t* a, int lda,  hmat::S_t* s, hmat::C_t* u, int ldu, hmat::C_t* vt, int ldvt, hmat::C_t* work, int lwork) {
  int info = 0;
  const int mn = (m < n ? m : n);
  hmat::S_t* rwork = (lwork == -1 ? NULL : new hmat::S_t[5 * mn]);
  _CGESVD_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info);
  if (rwork) delete [] rwork;
  return info;
}
template<>
inline int
gesvd<hmat::Z_t, hmat::D_t>(char jobu, char jobvt, int m, int n, hmat::Z_t* a, int lda,  hmat::D_t* s, hmat::Z_t* u, int ldu, hmat::Z_t* vt, int ldvt, hmat::Z_t* work, int lwork) {
  int info = 0;
  const int mn = (m < n ? m : n);
  hmat::D_t* rwork = (lwork == -1 ? NULL : new hmat::D_t[5 * mn]);
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
extern "C" void _SORGQR_(int*, int*, int*, hmat::S_t*, int*, const hmat::S_t*, hmat::S_t*, int*, int*);
extern "C" void _DORGQR_(int*, int*, int*, hmat::D_t*, int*, const hmat::D_t*, hmat::D_t*, int*, int*);

template<typename T>
int orgqr(int m, int n, int k, T* a, int lda, const T* tau, T* work, int lwork);

template<>
inline int
orgqr<hmat::S_t>(int m, int n, int k, hmat::S_t* a, int lda, const hmat::S_t* tau, hmat::S_t* work, int lwork) {
  int info = 0;
  _SORGQR_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}
template<>
inline int
orgqr<hmat::D_t>(int m, int n, int k, hmat::D_t* a, int lda, const hmat::D_t* tau, hmat::D_t* work, int lwork) {
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
extern "C" void _CUNGQR_(int*, int*, int*, hmat::C_t*, int*, const hmat::C_t*, hmat::C_t*, int*, int*);
extern "C" void _ZUNGQR_(int*, int*, int*, hmat::Z_t*, int*, const hmat::Z_t*, hmat::Z_t*, int*, int*);

template<typename T>
int ungqr(int m, int n, int k, T* a, int lda, const T* tau, T* work, int lwork);

template<>
inline int
ungqr<hmat::C_t>(int m, int n, int k, hmat::C_t* a, int lda, const hmat::C_t* tau, hmat::C_t* work, int lwork) {
  int info = 0;
  _CUNGQR_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}
template<>
inline int
ungqr<hmat::Z_t>(int m, int n, int k, hmat::Z_t* a, int lda, const hmat::Z_t* tau, hmat::Z_t* work, int lwork) {
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
extern "C" void _SORMQR_(char*, char*, int*, int*, int*, const hmat::S_t*, int*, const hmat::S_t*, hmat::S_t*, int*, hmat::S_t*, int*, int*);
extern "C" void _DORMQR_(char*, char*, int*, int*, int*, const hmat::D_t*, int*, const hmat::D_t*, hmat::D_t*, int*, hmat::D_t*, int*, int*);

template<typename T>
int ormqr(char side, char trans, int m, int n, int k, const T* a, int lda, const T* tau, T* c, int ldc, T* work, int lwork);

template<>
inline int
ormqr<hmat::S_t>(char side, char trans, int m, int n, int k, const hmat::S_t* a, int lda, const hmat::S_t* tau, hmat::S_t* c, int ldc, hmat::S_t* work, int lwork) {
  int info = 0;
  _SORMQR_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}
template<>
inline int
ormqr<hmat::D_t>(char side, char trans, int m, int n, int k, const hmat::D_t* a, int lda, const hmat::D_t* tau, hmat::D_t* c, int ldc, hmat::D_t* work, int lwork) {
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
extern "C" void _CUNMQR_(char*, char*, int*, int*, int*, const hmat::C_t*, int*, const hmat::C_t*, hmat::C_t*, int*, hmat::C_t*, int*, int*);
extern "C" void _ZUNMQR_(char*, char*, int*, int*, int*, const hmat::Z_t*, int*, const hmat::Z_t*, hmat::Z_t*, int*, hmat::Z_t*, int*, int*);

template<typename T>
int unmqr(char side, char trans, int m, int n, int k, const T* a, int lda, const T* tau, T* c, int ldc, T* work, int lwork);

template<>
inline int
unmqr<hmat::C_t>(char side, char trans, int m, int n, int k, const hmat::C_t* a, int lda, const hmat::C_t* tau, hmat::C_t* c, int ldc, hmat::C_t* work, int lwork) {
  int info = 0;
  _CUNMQR_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}
template<>
inline int
unmqr<hmat::Z_t>(char side, char trans, int m, int n, int k, const hmat::Z_t* a, int lda, const hmat::Z_t* tau, hmat::Z_t* c, int ldc, hmat::Z_t* work, int lwork) {
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
extern "C" void _SLASWP_(int*, hmat::S_t*, int*, int*, int*, const int*, int*);
extern "C" void _DLASWP_(int*, hmat::D_t*, int*, int*, int*, const int*, int*);
extern "C" void _CLASWP_(int*, hmat::C_t*, int*, int*, int*, const int*, int*);
extern "C" void _ZLASWP_(int*, hmat::Z_t*, int*, int*, int*, const int*, int*);

template<typename T>
void laswp(int n, T* a, int lda, int k1, int k2, const int* ipiv, int incx);

template<>
inline void
laswp<hmat::S_t>(int n, hmat::S_t* a, int lda, int k1, int k2, const int* ipiv, int incx) {
  _SLASWP_(&n, a, &lda, &k1, &k2, ipiv, &incx);
}
template<>
inline void
laswp<hmat::D_t>(int n, hmat::D_t* a, int lda, int k1, int k2, const int* ipiv, int incx) {
  _DLASWP_(&n, a, &lda, &k1, &k2, ipiv, &incx);
}
template<>
inline void
laswp<hmat::C_t>(int n, hmat::C_t* a, int lda, int k1, int k2, const int* ipiv, int incx) {
  _CLASWP_(&n, a, &lda, &k1, &k2, ipiv, &incx);
}
template<>
inline void
laswp<hmat::Z_t>(int n, hmat::Z_t* a, int lda, int k1, int k2, const int* ipiv, int incx) {
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
void _SPOTRF_(char * uplo, int* n, hmat::S_t* a, int* lda, int* info);
void _DPOTRF_(char * uplo, int* n, hmat::D_t* a, int* lda, int* info);
void _CPOTRF_(char * uplo, int* n, hmat::C_t* a, int* lda, int* info);
void _ZPOTRF_(char * uplo, int* n, hmat::Z_t* a, int* lda, int* info);
}

template <typename T> int potrf(char uplo, int n, T* a, int lda);

template <> inline int potrf<hmat::S_t>(char uplo, int n, hmat::S_t* a, int lda) {
  int info = 0;
  _SPOTRF_(&uplo, &n, a, &lda, &info);
  return info;
}
template <> inline int potrf<hmat::D_t>(char uplo, int n, hmat::D_t* a, int lda) {
  int info = 0;
  _DPOTRF_(&uplo, &n, a, &lda, &info);
  return info;
}
template <> inline int potrf<hmat::C_t>(char uplo, int n, hmat::C_t* a, int lda) {
  int info = 0;
  _CPOTRF_(&uplo, &n, a, &lda, &info);
  return info;
}
template <> inline int potrf<hmat::Z_t>(char uplo, int n, hmat::Z_t* a, int lda) {
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
or_un_mqr<hmat::S_t>(char side, char trans, int m, int n, int k, const hmat::S_t* a, int lda, const hmat::S_t* tau, hmat::S_t* c, int ldc, hmat::S_t* work, int lwork) {
  return proxy_lapack::ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
}
template<>
inline int
or_un_mqr<hmat::D_t>(char side, char trans, int m, int n, int k, const hmat::D_t* a, int lda, const hmat::D_t* tau, hmat::D_t* c, int ldc, hmat::D_t* work, int lwork) {
  return proxy_lapack::ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
}
template<>
inline int
or_un_mqr<hmat::C_t>(char side, char trans, int m, int n, int k, const hmat::C_t* a, int lda, const hmat::C_t* tau, hmat::C_t* c, int ldc, hmat::C_t* work, int lwork) {
  const char t = (trans == 'N' ? 'N' : 'C');
  return proxy_lapack::unmqr(side, t, m, n, k, a, lda, tau, c, ldc, work, lwork);
}
template<>
inline int
or_un_mqr<hmat::Z_t>(char side, char trans, int m, int n, int k, const hmat::Z_t* a, int lda, const hmat::Z_t* tau, hmat::Z_t* c, int ldc, hmat::Z_t* work, int lwork) {
  const char t = (trans == 'N' ? 'N' : 'C');
  return proxy_lapack::unmqr(side, t, m, n, k, a, lda, tau, c, ldc, work, lwork);
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

}  // end namespace hmat

#endif // DOXYGEN_SHOULD_SKIP_THIS
#endif // _LAPACK_OVERLOADS_HPP
