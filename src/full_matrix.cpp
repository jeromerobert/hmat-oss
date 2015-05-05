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

/*! \file
  \ingroup HMatrix
  \brief Dense Matrix implementation.
*/
#include "config.h"

#ifdef __INTEL_COMPILER
#include <mathimf.h>
#else
#include <cmath>
#endif

#include "full_matrix.hpp"

#include "data_types.hpp"
#include "lapack_overloads.hpp"
#include "blas_overloads.hpp"
#include "lapack_exception.hpp"
#include "common/memory_instrumentation.hpp"
#include "system_types.h"
#include "common/my_assert.h"
#include "common/context.hpp"

#include <cstring> // memset
#include <algorithm> // swap
#include <iostream>
#include <fstream>
#include <cmath>
#include <fcntl.h>

#ifndef _WIN32
#include <sys/mman.h> // mmap
#endif

#include <sys/stat.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include <stdlib.h>

#ifdef _MSC_VER
// Intel compiler defines isnan in global namespace
// MSVC defines _isnan
# ifndef __INTEL_COMPILER
#  define isnan _isnan
# endif
#elif defined(__MINGW32__) || defined(__APPLE__)
using std::isnan;
#endif

namespace hmat {

/** FullMatrix */
template<typename T>
FullMatrix<T>::FullMatrix(T* _m, int _rows, int _cols, int _lda)
  : ownsMemory(false), m(_m), rows(_rows), cols(_cols), lda(_lda), pivots(NULL),
    diagonal(NULL), isTriUpper(false), isTriLower(false) {
  if (lda == -1) {
    lda = rows;
  }
  myAssert(lda >= rows);
}

// #define POISON_ALLOCATION
#ifdef POISON_ALLOCATION
/*! \brief Fill an array with NaNs.

  The purpose of this function is to help spotting initialized memory
  earlier by making sure that any code using uninitialized memory
  encounters NaNs.
 */
template<typename T> void poisonArray(T* array, size_t n);

template<> static void poisonArray(S_t* array, size_t n) {
  const float nanFloat = nanf("");
  for (size_t i = 0; i < n; i++) {
    array[i] = nanFloat;
  }
}
template<> static void poisonArray(D_t* array, size_t n) {
  const double nanDouble = nan("");
  for (size_t i = 0; i < n; i++) {
    array[i] = nanDouble;
  }
}
template<> static void poisonArray(C_t* array, size_t n) {
  poisonArray<S_t>((S_t*) array, 2 * n);
}
template<> static void poisonArray(Z_t* array, size_t n) {
  poisonArray<D_t>((D_t*) array, 2 * n);
}
#endif

template<typename T>
FullMatrix<T>::FullMatrix(int _rows, int _cols)
  : ownsMemory(true), rows(_rows), cols(_cols), lda(_rows), pivots(NULL),
    diagonal(NULL), isTriUpper(false), isTriLower(false) {
  size_t size = ((size_t) rows) * cols * sizeof(T);
  m = (T*) calloc(size, 1);
  REGISTER_ALLOC(m, size);
  strongAssert(m);
#ifdef POISON_ALLOCATION
  // This memory is not initialized, fill it with NaNs to force a
  // crash when using it.
  poisonArray<T>(m, ((size_t) rows) * cols);
#endif
}

template<typename T>
FullMatrix<T>* FullMatrix<T>::Zero(int rows, int cols) {
  FullMatrix<T>* result = new FullMatrix<T>(rows, cols);
#ifdef POISON_ALLOCATION
  // The memory was poisoned in FullMatrix<T>::FullMatrix(), set it back to 0;
  result->clear();
#endif
  return result;
}

template<typename T> FullMatrix<T>::~FullMatrix() {
  if (ownsMemory) {
    size_t size = ((size_t) rows) * cols * sizeof(T);
    REGISTER_FREE(m, size);
    free(m);
    m = NULL;
  }
  if (pivots) {
    free(pivots);
  }
  if (diagonal) {
    delete diagonal;
  }
}

template<typename T> void FullMatrix<T>::clear() {
  myAssert(lda == rows);
  size_t size = ((size_t) rows) * cols * sizeof(T);
  memset(m, 0, size);
  if (diagonal) {
    memset(diagonal->v, 0, rows * sizeof(T));
  }
}

template<typename T> void FullMatrix<T>::scale(T alpha) {
  increment_flops(Multipliers<T>::mul * ((size_t) rows) * cols);
  if (lda == rows) {
    if (alpha == Constants<T>::zero) {
      this->clear();
    } else {
      // Warning: check for overflow
      size_t nm = ((size_t) rows) * cols;
      const size_t block_size_blas = 1 << 30;
      while (nm > block_size_blas) {
        proxy_cblas::scal(block_size_blas, alpha, m + nm - block_size_blas, 1);
        nm -= block_size_blas;
      }
      proxy_cblas::scal(nm, alpha, m, 1);
    }
  } else {
    T* x = m;
    if (alpha == Constants<T>::zero) {
      for (int col = 0; col < cols; col++) {
        memset(x, 0, sizeof(T) * rows);
        x += lda;
      }
    } else {
      for (int col = 0; col < cols; col++) {
        proxy_cblas::scal(rows, alpha, x, 1);
        x += lda;
      }
    }
  }
  if (diagonal) {
    proxy_cblas::scal(rows, alpha, diagonal->v, 1);
  }
}

template<typename T> void FullMatrix<T>::transpose() {
  myAssert(lda == rows);
  myAssert(m);
#ifdef HAVE_MKL_IMATCOPY
  proxy_mkl::imatcopy(rows, cols, m);
  std::swap(rows, cols);
  lda = rows;
#else
  if (rows == cols) {
    // "Fast" path
    for (int col = 0; col < cols; col++) {
      for (int row = 0; row < col; row++) {
        T tmp = get(row, col);
        get(row, col) = get(col, row);
        get(col, row) = tmp;
      }
    }
  } else {
    FullMatrix<T> tmp(rows, cols);
    tmp.copyMatrixAtOffset(this, 0, 0);
    std::swap(rows, cols);
    lda = rows;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        get(i, j) = tmp.get(j, i);
      }
    }
  }
#endif
  if (isTriUpper) {
    isTriUpper = false;
    isTriLower = true;
  } else if (isTriLower) {
    isTriLower = false;
    isTriUpper = true;
  }
}

template<typename T> FullMatrix<T>* FullMatrix<T>::copy() const {
  FullMatrix<T>* result = new FullMatrix<T>(rows, cols);

  if (lda == rows) {
    size_t size = ((size_t) rows) * cols * sizeof(T);
    memcpy(result->m, m, size);
  } else {
    for (int col = 0; col < cols; col++) {
      size_t resultOffset = ((size_t) result->rows) * col;
      size_t offset = ((size_t) lda) * col;
      memcpy(result->m + resultOffset, m + offset, rows * sizeof(T));
    }
  }
  return result;
}

template<typename T> FullMatrix<T>* FullMatrix<T>::copyAndTranspose() const {
  FullMatrix<T>* result = new FullMatrix<T>(cols, rows);
  result->clear();
#ifdef HAVE_MKL_IMATCOPY
  if (lda == rows) {
    proxy_mkl::omatcopy(rows, cols, m, result->m);
  } else {
#endif
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result->get(j, i) = get(i, j);
    }
  }
#ifdef HAVE_MKL_IMATCOPY
  }
#endif
  return result;
}


template<typename T>
void FullMatrix<T>::gemm(char transA, char transB, T alpha,
                         const FullMatrix<T>* a, const FullMatrix<T>* b,
                         T beta) {
  int m = (transA == 'N' ? a->rows : a->cols);
  int n = (transB == 'N' ? b->cols : b->rows);
  int k = (transA == 'N' ? a->cols : a->rows);
  myAssert(a->lda >= (transA == 'N' ? m : k));
  myAssert(b->lda >= (transB == 'N' ? k : n));
  myAssert(rows == m);
  myAssert(cols == n);
  {
    const size_t _m = m, _n = n, _k = k;
    const size_t adds = _m * _n * _k;
    const size_t muls = _m * _n * _k;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  proxy_cblas::gemm(transA, transB, m, n, k, alpha, a->m, a->lda, b->m, b->lda,
                    beta, this->m, this->lda);
}

template<typename T>
void FullMatrix<T>::multiplyWithDiagOrDiagInv(const Vector<T>* d, bool inverse, bool left) {
  myAssert(left || (this->cols == d->rows));
  myAssert(!left || (this->rows == d->rows));

  T* diag = d->v;
  {
    const size_t _rows = rows, _cols = cols;
    increment_flops(Multipliers<T>::mul * _rows * _cols);
  }
  if (left) {
    if (inverse) {
      // In this case, copying is a good idea since it avoids repeated
      // computations of 1 / diag[i].
      diag = (T*) malloc(d->rows * sizeof(T));
      strongAssert(diag);
      memcpy(diag, d->v, d->rows * sizeof(T));
      for (int i = 0; i < d->rows; i++) {
        diag[i] = Constants<T>::pone / diag[i];
      }
    }
    // TODO: Test with scale to see if it is better.
    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        m[i + j * ((size_t) lda)] *= diag[i];
      }
    }
    if (inverse) {
      free(diag);
    }
  } else {
    for (int j = 0; j < cols; j++) {
      proxy_cblas::scal(rows, diag[j], m + j * ((size_t) lda), 1);
    }
  }
}

template<typename T>
void FullMatrix<T>::ldltDecomposition() {
  int n = this->rows;
  diagonal = new Vector<T>(n);
  strongAssert(diagonal);
  myAssert(this->rows == this->cols); // We expect a square matrix

  // Standard LDLt factorization algorithm is:
  //  diag[j] = A(j,j) - sum_{k < j} L(j,k)^2 diag[k]
  //  L(i,j) = (A(i,j) - sum_{k < j} (L(i,k)L(j,k)diag[k])) / diag[j]
  // See for instance http://en.wikipedia.org/wiki/Cholesky_decomposition
  // An auxiliary array is introduced in order to perform less multiplications,
  // see  algorithm 1 in http://icl.cs.utk.edu/projectsfiles/plasma/pubs/ICL-UT-11-03.pdf
  T* v = new T[n];
  strongAssert(v);
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < j; i++)
      v[i] = get(j,i) * get(i,i);

    v[j] = get(j,j);
    for (int i = 0; i < j; i++)
      v[j] -= get(j,i) * v[i];

    get(j,j) = v[j];
    for (int i = 0; i < j; i++)
      for (int k = j+1; k < n; k++)
        get(k,j) -= get(k,i) * v[i];

    for (int k = j+1; k < n; k++) {
      get(k,j) /= v[j];
    }
  }

  for(int i = 0; i < n; i++) {
    diagonal->v[i] = get(i,i);
    get(i,i) = Constants<T>::pone;
    for (int j = i + 1; j < n; j++)
      get(i,j) = Constants<T>::zero;
  }

  isTriLower = true;
  myAssert(!isTriUpper);
  delete[] v;
}

template<typename T> void FullMatrix<T>::lltDecomposition() {
    // from http://www.netlib.org/lapack/lawnspdf/lawn41.pdf page 120
    const size_t n2 = ((size_t) rows) * rows;
    const size_t n3 = n2 * rows;
    const size_t muls = n3 / 6 + n2 / 2 + rows / 3;
    const size_t adds = n3 / 6 - rows / 6;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);

    int info = proxy_lapack::potrf('L', this->rows, this->m, this->lda);
    if(info != 0)
        // throw a pointer to be compliant with the Status class
        throw hmat::LapackException("potrf", info);
    isTriLower = true;
    for (int j = 0; j < this->cols; j++) {
        for(int i = 0; i < j; i++) {
            get(i,j) = Constants<T>::zero;
        }
    }
}

template<typename T>
void FullMatrix<T>::luDecomposition() {
  pivots = (int*) calloc(rows, sizeof(int));
  strongAssert(pivots);
  int info;
  {
    const size_t _m = rows, _n = cols;
    const size_t muls = _m * _n *_n / 2 - _n *_n*_n / 6 + _m * _n / 2 - _n*_n / 2 + 2 * _n / 3;
    const size_t adds = _m * _n *_n / 2 - _n *_n*_n / 6 + _m * _n / 2 + _n / 6;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  info = proxy_lapack::getrf(rows, cols, m, lda, pivots);
  strongAssert(!info);
}

// The following code is very close to that of ZGETRS in LAPACK.
// However, the resolution here is divided in the two parties.

// Warning! The matrix has been obtained with ZGETRF therefore it is
// permuted! We have the factorization A = P L U  with P the
// permutation matrix. So to solve L X = B, we must
// solve LX = (P ^ -1 B), which is done by ZLASWP with
// the permutation. we used it just like in ZGETRS.
template<typename T>
void FullMatrix<T>::solveLowerTriangularLeft(FullMatrix<T>* x, bool unitriangular) const {
  {
    const size_t _m = rows, _n = x->cols;
    const size_t adds = _n * _m * (_m - 1) / 2;
    const size_t muls = _n * _m * (_m + 1) / 2;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  if (pivots)
    proxy_lapack::laswp(x->cols, x->m, x->lda, 1, rows, pivots, 1);
  proxy_cblas::trsm('L', 'L', 'N', unitriangular ? 'U' : 'N', rows, x->cols, Constants<T>::pone, m, lda, x->m, x->lda);
}


// The resolution of the upper triangular system does not need to
//  change the order of columns.
//  The pivots are not necessary here, but this helps to check
//  the matrix was factorized before.

template<typename T>
void FullMatrix<T>::solveUpperTriangularRight(FullMatrix<T>* x, bool unitriangular, bool lowerStored) const {
  {
    const size_t _m = rows, _n = x->cols;
    const size_t adds = _n * _m * (_m - 1) / 2;
    const size_t muls = _n * _m * (_m + 1) / 2;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  proxy_cblas::trsm('R', lowerStored ? 'L' : 'U', lowerStored ? 'T' : 'N', unitriangular ? 'U' : 'N',
    x->rows, x->cols, Constants<T>::pone, m, lda, x->m, x->lda);
}

template<typename T>
void FullMatrix<T>::solveUpperTriangularLeft(FullMatrix<T>* x, bool unitriangular, bool lowerStored) const {
  {
    const size_t _m = rows, _n = x->cols;
    const size_t adds = _n * _m * (_n - 1) / 2;
    const size_t muls = _n * _m * (_n + 1) / 2;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  proxy_cblas::trsm('L', lowerStored ? 'L' : 'U', lowerStored ? 'T' : 'N', unitriangular ? 'U' : 'N',
    x->rows, x->cols, Constants<T>::pone, m, lda, x->m, x->lda);
}

template<typename T>
void FullMatrix<T>::solve(FullMatrix<T>* x) const {
  myAssert(pivots);
  int ierr = 0;
  {
    const size_t nrhs = x->cols;
    const size_t n = rows;
    const size_t adds = n * n * nrhs;
    const size_t muls = (n * n - n) * nrhs;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  ierr = proxy_lapack::getrs('N', rows, x->cols, m, lda, pivots, x->m, x->rows);
  strongAssert(!ierr);
}


template<typename T>
void FullMatrix<T>::inverse() {

  // The inversion is done in two steps with dgetrf for LU decomposition and
  // dgetri for inversion of triangular matrices

  myAssert(rows == cols);

  int *ipiv = new int[rows];
  int info;
  {
    size_t vn = cols, vm = cols;
    // getrf
    size_t additions = (vm*vn*vn)/2 - (vn*vn*vn)/6 - (vm*vn)/2 + vn/6;
    size_t multiplications = (vm*vn*vn)/2 - (vn*vn*vn)/6 + (vm*vn)/2
      - (vn*vn)/2 + 2*vn/3;
    increment_flops(Multipliers<T>::add * additions + Multipliers<T>::mul * multiplications);
    // getri
    additions = (2*vn*vn*vn)/3 - (3*vn*vn)/2 + (5*vn)/6;
    multiplications = (2*vn*vn*vn)/3 + (vn*vn)/2 + (5*vn)/6;
    increment_flops(Multipliers<T>::add * additions + Multipliers<T>::mul * multiplications);
  }
  info = proxy_lapack::getrf(rows, cols, m, lda, ipiv);
  strongAssert(!info);
  // We call it twice: the first time to know the optimal size of
  // temporary arrays, and the second time for real calculation.
  int workSize;
  T workSize_req;
  info = proxy_lapack::getri(rows, m, lda, ipiv, &workSize_req, -1);
  workSize = (int) hmat::real(workSize_req) + 1;
  T* work = new T[workSize];
  strongAssert(work);
  info = proxy_lapack::getri(rows, m, lda, ipiv, work, workSize);
  delete[] work;
  strongAssert(!info);
  delete[] ipiv;
}


template<typename T>
void FullMatrix<T>::copyMatrixAtOffset(const FullMatrix<T>* a,
                                       int rowOffset, int colOffset) {
  myAssert(rowOffset + a->rows <= rows);
  myAssert(colOffset + a->cols <= cols);


  // Use memcpy when copying the whole matrix. This avoids BLAS calls.
  if ((rowOffset == 0) && (colOffset == 0)
      && (a->rows == rows) && (a->cols == cols)
      && (a->lda == a->rows) && (lda == rows)) {
    size_t size = ((size_t) rows) * cols;
    memcpy(m, a->m, size * sizeof(T));
    return;
  }

  for (int col = 0; col < a->cols; col++) {
    proxy_cblas::copy(a->rows, a->m + col * a->lda, 1,
                m + rowOffset + ((colOffset + col) * lda), 1);
  }
}

template<typename T>
void FullMatrix<T>::copyMatrixAtOffset(const FullMatrix<T>* a,
                                       int rowOffset, int colOffset,
                                       int rowsToCopy, int colsToCopy) {
  myAssert(rowOffset + rowsToCopy <= rows);
  myAssert(colOffset + colsToCopy <= cols);
  for (int col = 0; col < colsToCopy; col++) {
    proxy_cblas::copy(rowsToCopy, a->m + col * a->lda, 1,
                (m + rowOffset + ((colOffset + col) * lda)), 1);
  }
}

template<typename T>
void FullMatrix<T>::axpy(T alpha, const FullMatrix<T>* a) {
  myAssert(rows == a->rows);
  myAssert(cols == a->cols);
  size_t size = ((size_t) rows) * cols;

  increment_flops(Multipliers<T>::add * size
		  + (alpha == Constants<T>::pone ? 0 : Multipliers<T>::mul * size));
  // Fast path
  if ((lda == rows) && (a->lda == a->rows) && (size < 1000000000)) {
    proxy_cblas::axpy(size, alpha, a->m, 1, m, 1);
    return;
  }

  for (int col = 0; col < cols; col++) {
    proxy_cblas::axpy(rows, alpha, a->m + ((size_t) col) * a->lda, 1, m + ((size_t) col) * lda, 1);
  }
}

template<typename T>
double FullMatrix<T>::norm() const {
  size_t size = ((size_t) rows) * cols;
  T result = Constants<T>::zero;

  // Fast path
  if ((size < 1000000000) && (lda == rows)) {
    result += proxy_cblas_convenience::dot_c(size, m, 1, m, 1);
    return sqrt(hmat::real(result));
  }
  for (int col = 0; col < cols; col++) {
    result += proxy_cblas_convenience::dot_c(rows, m + col * lda, 1, m + col * lda, 1);
  }
  return sqrt(hmat::real(result));
}


template<typename T> void FullMatrix<T>::toFile(const char *filename) const {
  int ierr;
  int fd;
  size_t size = ((size_t) rows) * cols * sizeof(T) + 5 * sizeof(int);

  strongAssert(lda == rows);

  fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
  strongAssert(fd != -1);
  ierr = lseek(fd, size - 1, SEEK_SET);
  strongAssert(ierr != -1);
  ierr = write(fd, "", 1);
  strongAssert(ierr == 1);
#ifndef _WIN32
  void* mmapedFile = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ierr = (mmapedFile == MAP_FAILED) ? 1 : 0;
  strongAssert(!ierr);
  int *asIntArray = (int*) mmapedFile;
  asIntArray[0] = Constants<T>::code;
  asIntArray[1] = rows;
  asIntArray[2] = cols;
  asIntArray[3] = sizeof(T);
  asIntArray[4] = 0;
  asIntArray += 5;
  T* mat = (T*) asIntArray;
  memcpy(mat, m, size - 5 * sizeof(int));
  close(fd);
  munmap(mmapedFile, size);
#else
#ifdef __GNUC__
#warning MMAP: TO BE REMOVED OR FIXED
#else
#pragma message("Warning: MMAP: TO BE REMOVED OR FIXED")
#endif
#endif

}

template<typename T> size_t FullMatrix<T>::memorySize() const {
   return ((size_t) rows) * cols * sizeof(T);
}

template<typename T> void checkNanReal(const FullMatrix<T>* m) {
  for (int col = 0; col < m->cols; col++) {
    for (int row = 0; row < m->rows; row++) {
      strongAssert(!isnan(m->get(row, col)));
    }
  }
}

template<typename T> void checkNanComplex(const FullMatrix<T>* m) {
  for (int col = 0; col < m->cols; col++) {
    for (int row = 0; row < m->rows; row++) {
      strongAssert(!isnan(m->get(row, col).real()));
      strongAssert(!isnan(m->get(row, col).imag()));
    }
  }
}

template<> void FullMatrix<S_t>::checkNan() const {
  checkNanReal(this);
}
template<> void FullMatrix<D_t>::checkNan() const {
  checkNanReal(this);
}
template<> void FullMatrix<C_t>::checkNan() const {
  checkNanComplex(this);
}
template<> void FullMatrix<Z_t>::checkNan() const {
  checkNanComplex(this);
}


// MmapedFullMatrix
template<typename T>
MmapedFullMatrix<T>::MmapedFullMatrix(int rows, int cols, const char* filename)
  : m(NULL, rows, cols), mmapedFile(NULL), fd(-1), size(0) {
#ifdef _WIN32
  strongAssert(false); // no mmap() on Windows
#else
  int ierr;

  size = ((size_t) rows) * cols * sizeof(T) + 5 * sizeof(int);
  fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
  strongAssert(fd != -1);
  ierr = lseek(fd, size - 1, SEEK_SET);
  strongAssert(ierr != -1);
  ierr = write(fd, "", 1);
  strongAssert(ierr == 1);
  mmapedFile = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ierr = (mmapedFile == MAP_FAILED) ? 1 : 0;
  strongAssert(!ierr);
  int *asIntArray = (int*) mmapedFile;
  asIntArray[0] = 0;
  asIntArray[1] = rows;
  asIntArray[2] = cols;
  asIntArray[3] = sizeof(T);
  asIntArray[4] = 0;
  asIntArray += 5;
  T* mat = (T*) asIntArray;
  m.m = mat;
#endif
}

template<typename T>
MmapedFullMatrix<T>::~MmapedFullMatrix() {
#ifndef _WIN32
  close(fd);
  munmap(mmapedFile, size);
#endif
}

template<typename T>
MmapedFullMatrix<T>* MmapedFullMatrix<T>::fromFile(const char* filename) {
  MmapedFullMatrix<T>* result = new MmapedFullMatrix();

#ifdef _WIN32
  strongAssert(false); // no mmap() on Windows
#else
  int ierr;
  result->fd = open(filename, O_RDONLY);
  strongAssert(result->fd != -1);
  struct stat fileStat;
  ierr = fstat(result->fd, &fileStat);
  strongAssert(!ierr);
  size_t fileSize = fileStat.st_size;

  result->mmapedFile = mmap(0, fileSize, PROT_READ, MAP_SHARED, result->fd, 0);
  ierr = (result->mmapedFile == MAP_FAILED) ? 1 : 0;
  strongAssert(!ierr);
  int* header = (int*) result->mmapedFile;
  // Check the consistency of the file
  strongAssert(header[0] == Constants<T>::code);
  strongAssert(header[3] == sizeof(T));
  strongAssert(header[1] * ((size_t) header[2]) * sizeof(T) + (5 * sizeof(int)) == fileSize);
  result->m.lda = result->m.rows = header[1];
  result->m.cols = header[2];
  result->m.m = (T*) (header + 5);
#endif
  return result;
}


/** Vector */
template<typename T> Vector<T>::Vector(T* _v, int _rows)
  : ownsMemory(false), v(_v), rows(_rows) {}

template<typename T> Vector<T>::Vector(int _rows)
  : ownsMemory(true), rows(_rows) {
  size_t size = rows * sizeof(T);
  v = (T*) calloc(size, 1);
  REGISTER_ALLOC(v, size);
  strongAssert(v);
}

template<typename T> Vector<T>::~Vector() {
  if (ownsMemory) {
    size_t size = rows * sizeof(T);
    free(v);
    REGISTER_FREE(v, size);
  }
  v = NULL;
}

template<typename T> Vector<T>* Vector<T>::Zero(int rows) {
  Vector<T> *result = new Vector<T>(rows);
  return result;
}

template<typename T>
void Vector<T>::gemv(char trans, T alpha,
                     const FullMatrix<T>* a,
                     const Vector* x, T beta)
{
  int matRows = a->rows;
  int matCols = a->cols;
  int lda = matRows;
  CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
  int64_t ops = (Multipliers<T>::add + Multipliers<T>::mul) * ((int64_t) matRows) * matCols;
  increment_flops(ops);

  if (trans == 'N') {
    myAssert(rows == a->rows);
    myAssert(x->rows == a->cols);
  } else {
    myAssert(rows == a->cols);
    myAssert(x->rows == a->rows);
  }
  proxy_cblas::gemv(t, matRows, matCols, alpha, a->m, lda, x->v, 1, beta, v, 1);
}

template<typename T>
void Vector<T>::axpy(T alpha, const Vector* x) {
  myAssert(rows == x->rows);
  proxy_cblas::axpy(rows, alpha, x->v, 1, this->v, 1);
}

template<typename T>
int Vector<T>::absoluteMaxIndex() const {
  return proxy_cblas::i_amax(rows, v, 1);
}

template<typename T>
T Vector<T>::dot(const Vector<T>* x, const Vector<T>* y) {
  myAssert(x->rows == y->rows);
  // TODO: Beware of large vectors (>2 billion elements) !
  return proxy_cblas_convenience::dot_c(x->rows, x->v, 1, y->v, 1);
}

template<typename T> void Vector<T>::addToMe(const Vector<T>* x) {
  myAssert(rows == x->rows);
  axpy(Constants<T>::pone, x);
}

template<typename T> void Vector<T>::subToMe(const Vector<T>* x) {
  myAssert(rows == x->rows);
  axpy(Constants<T>::mone, x);
}

template<typename T> double Vector<T>::norm() const {
  T result = dot(this, this);
  return sqrt(hmat::real(result));
}

template<typename T> void Vector<T>::clear() {
  memset(this->v, 0, sizeof(T) * this->rows);
}

template<typename T> void Vector<T>::scale(T alpha) {
  if (alpha == Constants<T>::zero) {
    memset(v, 0, sizeof(T) * rows);
  } else {
    proxy_cblas::scal(rows, alpha, v, 1);
  }
}

// the classes declaration
template class FullMatrix<S_t>;
template class FullMatrix<D_t>;
template class FullMatrix<C_t>;
template class FullMatrix<Z_t>;

template class MmapedFullMatrix<S_t>;
template class MmapedFullMatrix<D_t>;
template class MmapedFullMatrix<C_t>;
template class MmapedFullMatrix<Z_t>;

template class Vector<S_t>;
template class Vector<D_t>;
template class Vector<C_t>;
template class Vector<Z_t>;

}  // end namespace hmat

