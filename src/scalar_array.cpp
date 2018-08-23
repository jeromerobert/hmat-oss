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

#include "scalar_array.hpp"

#include "data_types.hpp"
#include "lapack_overloads.hpp"
#include "blas_overloads.hpp"
#include "lapack_exception.hpp"
#include "common/memory_instrumentation.hpp"
#include "system_types.h"
#include "common/my_assert.h"
#include "common/context.hpp"
#include "lapack_operations.hpp"

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

#ifdef HAVE_JEMALLOC
#define JEMALLOC_NO_DEMANGLE
#include <jemalloc/jemalloc.h>
#endif

#ifdef _MSC_VER
// Intel compiler defines isnan in global namespace
// MSVC defines _isnan
# ifndef __INTEL_COMPILER
#  define isnan _isnan
# endif
#elif __GLIBC__ == 2 && __GLIBC_MINOR__ < 23
// https://sourceware.org/bugzilla/show_bug.cgi?id=19439
#elif __cplusplus >= 201103L || !defined(__GLIBC__)
using std::isnan;
#endif

namespace hmat {

/** ScalarArray */
template<typename T>
ScalarArray<T>::ScalarArray(T* _m, int _rows, int _cols, int _lda)
  : ownsMemory(false), m(_m), rows(_rows), cols(_cols), lda(_lda) {
  if (lda == -1) {
    lda = rows;
  }
  assert(lda >= rows);
}

template<typename T>
ScalarArray<T>::ScalarArray(int _rows, int _cols)
  : ownsMemory(true), rows(_rows), cols(_cols), lda(_rows) {
  size_t size = ((size_t) rows) * cols * sizeof(T);
#ifdef HAVE_JEMALLOC
  m = (T*) je_calloc(size, 1);
#else
  m = (T*) calloc(size, 1);
#endif
  HMAT_ASSERT_MSG(m, "Trying to allocate %ldb of memory failed (rows=%d cols=%d sizeof(T)=%d)", size, rows, cols, sizeof(T));
  MemoryInstrumenter::instance().alloc(size, MemoryInstrumenter::FULL_MATRIX);
}

template<typename T> ScalarArray<T>::~ScalarArray() {
  if (ownsMemory) {
    size_t size = ((size_t) rows) * cols * sizeof(T);
    MemoryInstrumenter::instance().free(size, MemoryInstrumenter::FULL_MATRIX);
#ifdef HAVE_JEMALLOC
    je_free(m);
#else
    free(m);
#endif
    m = NULL;
  }
}

template<typename T> void ScalarArray<T>::clear() {
  assert(lda == rows);
  std::fill(m, m + ((size_t) rows) * cols, Constants<T>::zero);
}

template<typename T> size_t ScalarArray<T>::storedZeros() const {
  size_t result = 0;
  for (int col = 0; col < cols; col++) {
    for (int row = 0; row < rows; row++) {
      if (std::abs(get(row, col)) < 1e-16) {
        result++;
      }
    }
  }
  return result;
}

template<typename T> void ScalarArray<T>::scale(T alpha) {
  increment_flops(Multipliers<T>::mul * ((size_t) rows) * cols);
  if (lda == rows) {
    if (alpha == Constants<T>::zero) {
      this->clear();
    } else {
      // Warning: check for overflow
      size_t nm = ((size_t) rows) * cols;
      const size_t block_size_blas = 1 << 30;
      while (nm > block_size_blas) {
        proxy_cblas::scal(block_size_blas, alpha, ptr() + nm - block_size_blas, 1);
        nm -= block_size_blas;
      }
      proxy_cblas::scal(nm, alpha, ptr(), 1);
    }
  } else {
    T* x = ptr();
    if (alpha == Constants<T>::zero) {
      for (int col = 0; col < cols; col++) {
        std::fill(x, x + rows, Constants<T>::zero);
        x += lda;
      }
    } else {
      for (int col = 0; col < cols; col++) {
        proxy_cblas::scal(rows, alpha, x, 1);
        x += lda;
      }
    }
  }
}

template<typename T> void ScalarArray<T>::transpose() {
  assert(lda == rows);
#ifdef HAVE_MKL_IMATCOPY
  proxy_mkl::imatcopy(rows, cols, ptr());
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
    ScalarArray<T> *tmp=copy();
    std::swap(rows, cols);
    lda = rows;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        get(i, j) = tmp->get(j, i);
      }
    }
    delete(tmp);
  }
#endif
}

template<typename T> void ScalarArray<T>::conjugate() {
  if (lda == rows) {
    // Warning: check for overflow
    size_t nm = ((size_t) rows) * cols;
    const size_t block_size_blas = 1 << 30;
    while (nm > block_size_blas) {
      proxy_lapack::lacgv(block_size_blas, m + nm - block_size_blas, 1);
      nm -= block_size_blas;
    }
    proxy_lapack::lacgv(nm, m, 1);
  } else {
    T* x = m;
    for (int col = 0; col < cols; col++) {
      proxy_lapack::lacgv(rows, x, 1);
      x += lda;
    }
  }
}

template<typename T> ScalarArray<T>* ScalarArray<T>::copy(ScalarArray<T>* result) const {
  if(result == NULL)
    result = new ScalarArray<T>(rows, cols);

  if (lda == rows && result->lda == result->rows) {
    size_t size = ((size_t) rows) * cols * sizeof(T);
    memcpy(result->ptr(), const_ptr(), size);
  } else {
    for (int col = 0; col < cols; col++) {
      size_t resultOffset = ((size_t) result->lda) * col;
      size_t offset = ((size_t) lda) * col;
      memcpy(result->ptr() + resultOffset, const_ptr() + offset, rows * sizeof(T));
    }
  }

  return result;
}

template<typename T> ScalarArray<T>* ScalarArray<T>::copyAndTranspose(ScalarArray<T>* result) const {
  if(result == NULL)
    result = new ScalarArray<T>(cols, rows);
#ifdef HAVE_MKL_IMATCOPY
  if (lda == rows && result->lda == result->rows) {
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
ScalarArray<T> ScalarArray<T>::rowsSubset(const int rowsOffset, const int rowsSize) const { //TODO: remove it, we have a constructor to do this
  assert(rowsOffset + rowsSize <= rows);
  return ScalarArray<T>(*this, rowsOffset, rowsSize, 0, cols);
}

template<typename T>
void ScalarArray<T>::gemm(char transA, char transB, T alpha,
                         const ScalarArray<T>* a, const ScalarArray<T>* b,
                         T beta) {
  int aRows  = (transA == 'N' ? a->rows : a->cols);
  int n  = (transB == 'N' ? b->cols : b->rows);
  int k  = (transA == 'N' ? a->cols : a->rows);
  assert(a->lda >= (transA == 'N' ? aRows : k));
  assert(b->lda >= (transB == 'N' ? k : n));
  assert(rows == aRows);
  assert(cols == n);
  assert(k == (transB == 'N' ? b->rows : b->cols));
  {
    const size_t _m = aRows, _n = n, _k = k;
    const size_t adds = _m * _n * _k;
    const size_t muls = _m * _n * _k;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  proxy_cblas::gemm(transA, transB, aRows, n, k, alpha, a->const_ptr(), a->lda, b->const_ptr(), b->lda,
                    beta, this->ptr(), this->lda);
}

template<typename T>
void ScalarArray<T>::copyMatrixAtOffset(const ScalarArray<T>* a,
                                       int rowOffset, int colOffset) {
  assert(rowOffset + a->rows <= rows);
  assert(colOffset + a->cols <= cols);


  // Use memcpy when copying the whole matrix. This avoids BLAS calls.
  if ((rowOffset == 0) && (colOffset == 0)
      && (a->rows == rows) && (a->cols == cols)
      && (a->lda == a->rows) && (lda == rows)) {
    size_t size = ((size_t) rows) * cols;
    memcpy(ptr(), a->const_ptr(), size * sizeof(T));
    return;
  }

  for (int col = 0; col < a->cols; col++) {
    proxy_cblas::copy(a->rows, a->const_ptr() + col * a->lda, 1,
                ptr() + rowOffset + ((colOffset + col) * lda), 1);
  }
}

template<typename T>
void ScalarArray<T>::copyMatrixAtOffset(const ScalarArray<T>* a,
                                       int rowOffset, int colOffset,
                                       int rowsToCopy, int colsToCopy) { // NOT USED
  assert(rowOffset + rowsToCopy <= rows);
  assert(colOffset + colsToCopy <= cols);
  for (int col = 0; col < colsToCopy; col++) {
    proxy_cblas::copy(rowsToCopy, a->const_ptr() + col * a->lda, 1,
                (ptr() + rowOffset + ((colOffset + col) * lda)), 1);
  }
}

template<typename T> void ScalarArray<T>::addRand(double epsilon) {
  DECLARE_CONTEXT;
  if (lda == rows) {
    for (size_t i = 0; i < ((size_t) rows) * cols; ++i) {
      get(i) *= 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
    }
  } else {
    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < rows; ++row) {
        get(row, col) *= 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
      }
    }
  }
}

template<> void ScalarArray<C_t>::addRand(double epsilon) {
  DECLARE_CONTEXT;

  if (lda == rows) {
    for (size_t i = 0; i < ((size_t) rows) * cols; ++i) {
      float c1 = 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
      float c2 = 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
      get(i) *= C_t(c1, c2);
    }
  } else {
    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < rows; ++row) {
        float c1 = 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
        float c2 = 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
        get(row, col) *= C_t(c1, c2);
      }
    }
  }
}

template<> void ScalarArray<Z_t>::addRand(double epsilon) {
  DECLARE_CONTEXT;

  if (lda == rows) {
    for (size_t i = 0; i < ((size_t) rows) * cols; ++i) {
      double c1 = 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
      double c2 = 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
      get(i) *= Z_t(c1, c2);
    }
  } else {
    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < rows; ++row) {
        double c1 = 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
        double c2 = 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
        get(row, col) *= Z_t(c1, c2);
      }
    }
  }
}

template<typename T>
void ScalarArray<T>::axpy(T alpha, const ScalarArray<T>* a) {
  assert(rows == a->rows);
  assert(cols == a->cols);
  size_t size = ((size_t) rows) * cols;

  increment_flops(Multipliers<T>::add * size
		  + (alpha == Constants<T>::pone ? 0 : Multipliers<T>::mul * size));
  // Fast path
  if ((lda == rows) && (a->lda == a->rows) && (size < 1000000000)) {
    proxy_cblas::axpy(size, alpha, a->const_ptr(), 1, ptr(), 1);
    return;
  }

  for (int col = 0; col < cols; col++) {
    proxy_cblas::axpy(rows, alpha, a->const_ptr() + ((size_t) col) * a->lda, 1, ptr() + ((size_t) col) * lda, 1);
  }
}

template<typename T>
double ScalarArray<T>::normSqr() const {
  size_t size = ((size_t) rows) * cols;
  T result = Constants<T>::zero;

  // Fast path
  if ((size < 1000000000) && (lda == rows)) {
    result += proxy_cblas_convenience::dot_c(size, const_ptr(), 1, const_ptr(), 1);
    return real(result);
  }
  for (int col = 0; col < cols; col++) {
    result += proxy_cblas_convenience::dot_c(rows, const_ptr() + col * lda, 1, const_ptr() + col * lda, 1);
  }
  return real(result);
}

template<typename T> double ScalarArray<T>::norm() const {
  return sqrt(normSqr());
}

// Compute squared Frobenius norm of a.b^t (a=this)
template<typename T> double ScalarArray<T>::norm_abt_Sqr(const ScalarArray<T> &b) const {
  double result = 0;
  const int k = cols;
  for (int i = 1; i < k; ++i) {
    for (int j = 0; j < i; ++j) {
      result += real(proxy_cblas_convenience::dot_c(rows, const_ptr() + i*lda, 1, const_ptr() + j*lda, 1) *
                           proxy_cblas_convenience::dot_c(b.rows, b.const_ptr() + i*b.lda, 1, b.const_ptr() + j*b.lda, 1));
    }
  }
  result *= 2.0;
  for (int i = 0; i < k; ++i) {
    result += real(proxy_cblas_convenience::dot_c(rows, const_ptr() + i*lda, 1, const_ptr() + i*lda, 1) *
                         proxy_cblas_convenience::dot_c(b.rows, b.const_ptr() + i*b.lda, 1, b.const_ptr() + i*b.lda, 1));
  }
  return result;
}

// Compute dot product between this[i,*] and b[j,*]
template<typename T> T ScalarArray<T>::dot_aibj(int i, const ScalarArray<T> &b, int j) const {
  return proxy_cblas::dot(cols, &get(i,0), lda, &b.get(j,0), b.lda);
}

template<typename T> void ScalarArray<T>::fromFile(const char * filename) {
  FILE * f = fopen(filename, "rb");
  /* Read the header before data : [stype, rows, cols, sieof(T), 0] */
  int code;
  int r = fread(&code, sizeof(int), 1, f);
  HMAT_ASSERT(r == 1);
  HMAT_ASSERT(code == Constants<T>::code);
  r = fread(&rows, sizeof(int), 1, f);
  lda = rows;
  HMAT_ASSERT(r == 1);
  r = fread(&cols, sizeof(int), 1, f);
  HMAT_ASSERT(r == 1);
  r = fseek(f, 2 * sizeof(int), SEEK_CUR);
  HMAT_ASSERT(r == 0);
  if(m)
      free(m);
  size_t size = ((size_t) rows) * cols * sizeof(T);
  m = (T*) calloc(size, 1);
  r = fread(ptr(), size, 1, f);
  fclose(f);
  HMAT_ASSERT(r == 1);
}

template<typename T> void ScalarArray<T>::toFile(const char *filename) const {
  int ierr;
  int fd;
  size_t size = ((size_t) rows) * cols * sizeof(T) + 5 * sizeof(int);

  HMAT_ASSERT(lda == rows);

  fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
  HMAT_ASSERT(fd != -1);
  ierr = lseek(fd, size - 1, SEEK_SET);
  HMAT_ASSERT(ierr != -1);
  ierr = write(fd, "", 1);
  HMAT_ASSERT(ierr == 1);
#ifndef _WIN32
  void* mmapedFile = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ierr = (mmapedFile == MAP_FAILED) ? 1 : 0;
  HMAT_ASSERT(!ierr);
  /* Write the header before data : [stype, rows, cols, sieof(T), 0] */
  int *asIntArray = (int*) mmapedFile;
  asIntArray[0] = Constants<T>::code;
  asIntArray[1] = rows;
  asIntArray[2] = cols;
  asIntArray[3] = sizeof(T);
  asIntArray[4] = 0;
  asIntArray += 5;
  T* mat = (T*) asIntArray;
  memcpy(mat, const_ptr(), size - 5 * sizeof(int));
  close(fd);
  munmap(mmapedFile, size);
#else
  HMAT_ASSERT_MSG(false, "mmap not available on this platform");
#endif

}

template<typename T> size_t ScalarArray<T>::memorySize() const {
   return ((size_t) rows) * cols * sizeof(T);
}

template<typename T> void checkNanReal(const ScalarArray<T>* m) {
  for (int col = 0; col < m->cols; col++) {
    for (int row = 0; row < m->rows; row++) {
      HMAT_ASSERT(!isnan(m->get(row, col)));
    }
  }
}

template<typename T> void checkNanComplex(const ScalarArray<T>* m) {
  for (int col = 0; col < m->cols; col++) {
    for (int row = 0; row < m->rows; row++) {
      HMAT_ASSERT(!isnan(m->get(row, col).real()));
      HMAT_ASSERT(!isnan(m->get(row, col).imag()));
    }
  }
}

template<> void ScalarArray<S_t>::checkNan() const {
  checkNanReal(this);
}
template<> void ScalarArray<D_t>::checkNan() const {
  checkNanReal(this);
}
template<> void ScalarArray<C_t>::checkNan() const {
  checkNanComplex(this);
}
template<> void ScalarArray<Z_t>::checkNan() const {
  checkNanComplex(this);
}

template<typename T> bool ScalarArray<T>::isZero() const {
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
      if (get(i, j) != Constants<T>::zero)
        return false;
  return true;
}

// this := alpha*x*b^T + this
template<typename T> void ScalarArray<T>::rankOneUpdate(const T alpha, const ScalarArray<T> &x, const ScalarArray<T> &b){
  assert(x.rows==rows);
  assert(x.cols==1);
  assert(b.rows==cols);
  assert(b.cols==1);
  proxy_cblas::ger(rows, cols, alpha, x.const_ptr(), 1, b.const_ptr(), 1, ptr(), lda);
}

// this := alpha*x*b + this
template<typename T> void ScalarArray<T>::rankOneUpdateT(const T alpha, const ScalarArray<T> &x, const ScalarArray<T> &tb){
  assert(x.rows==rows);
  assert(x.cols==1);
  assert(tb.rows==1);
  assert(tb.cols==cols);
  proxy_cblas::ger(rows, cols, alpha, x.const_ptr(), 1, tb.const_ptr(), tb.lda, ptr(), lda);
}

template<typename T> void ScalarArray<T>::writeArray(hmat_iostream writeFunc, void * userData) const{
  assert(lda == rows);
  size_t s = (size_t)rows * cols;
  // We use m instead of const_ptr() because writeFunc() expects a void*, not a const void*
  writeFunc(m, sizeof(T) * s, userData);
}

template<typename T> void ScalarArray<T>::readArray(hmat_iostream readFunc, void * userData) {
  assert(lda == rows);
  size_t s = (size_t)rows * cols;
  readFunc(ptr(), sizeof(T) * s, userData);
}

template<typename T> void ScalarArray<T>::luDecomposition(int *pivots) {
  int info;
  {
    const size_t _m = rows, _n = cols;
    const size_t muls = _m * _n *_n / 2 - _n *_n*_n / 6 + _m * _n / 2 - _n*_n / 2 + 2 * _n / 3;
    const size_t adds = _m * _n *_n / 2 - _n *_n*_n / 6 + _m * _n / 2 + _n / 6;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  info = proxy_lapack::getrf(rows, cols, ptr(), lda, pivots);
  if (info)
    throw LapackException("getrf", info);
}

// The following code is very close to that of ZGETRS in LAPACK.
// However, the resolution here is divided in the two parties.

// Warning! The matrix has been obtained with ZGETRF therefore it is
// permuted! We have the factorization A = P L U  with P the
// permutation matrix. So to solve L X = B, we must
// solve LX = (P ^ -1 B), which is done by ZLASWP with
// the permutation. we used it just like in ZGETRS.
template<typename T>
void ScalarArray<T>::solveLowerTriangularLeft(ScalarArray<T>* x, int* pivots, bool unitriangular) const {
  {
    const size_t _m = rows, _n = x->cols;
    const size_t adds = _n * _m * (_m - 1) / 2;
    const size_t muls = _n * _m * (_m + 1) / 2;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  if (pivots)
    proxy_lapack::laswp(x->cols, x->ptr(), x->lda, 1, rows, pivots, 1);
  proxy_cblas::trsm('L', 'L', 'N', unitriangular ? 'U' : 'N', rows, x->cols, Constants<T>::pone, const_ptr(), lda, x->ptr(), x->lda);
}

// The resolution of the upper triangular system does not need to
//  change the order of columns.
//  The pivots are not necessary here, but this helps to check
//  the matrix was factorized before.

template<typename T>
void ScalarArray<T>::solveUpperTriangularRight(ScalarArray<T>* x, bool unitriangular, bool lowerStored) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;

  {
    const size_t _m = rows, _n = x->cols;
    const size_t adds = _n * _m * (_m - 1) / 2;
    const size_t muls = _n * _m * (_m + 1) / 2;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  proxy_cblas::trsm('R', lowerStored ? 'L' : 'U', lowerStored ? 'T' : 'N', unitriangular ? 'U' : 'N',
    x->rows, x->cols, Constants<T>::pone, const_ptr(), lda, x->ptr(), x->lda);
}

template<typename T>
void ScalarArray<T>::solveUpperTriangularLeft(ScalarArray<T>* x, bool unitriangular, bool lowerStored) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;

  {
    const size_t _m = rows, _n = x->cols;
    const size_t adds = _n * _m * (_n - 1) / 2;
    const size_t muls = _n * _m * (_n + 1) / 2;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  proxy_cblas::trsm('L', lowerStored ? 'L' : 'U', lowerStored ? 'T' : 'N', unitriangular ? 'U' : 'N',
    x->rows, x->cols, Constants<T>::pone, const_ptr(), lda, x->ptr(), x->lda);
}

template<typename T>
void ScalarArray<T>::solve(ScalarArray<T>* x, int *pivots) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;

  int ierr = 0;
  {
    const size_t nrhs = x->cols;
    const size_t n = rows;
    const size_t adds = n * n * nrhs;
    const size_t muls = (n * n - n) * nrhs;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  ierr = proxy_lapack::getrs('N', rows, x->cols, const_ptr(), lda, pivots, x->ptr(), x->rows);
  if (ierr)
    throw LapackException("getrs", ierr);
}


template<typename T>
void ScalarArray<T>::inverse() {

  // The inversion is done in two steps with dgetrf for LU decomposition and
  // dgetri for inversion of triangular matrices

  assert(rows == cols);

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
  info = proxy_lapack::getrf(rows, cols, ptr(), lda, ipiv);
  HMAT_ASSERT(!info);
  // We call it twice: the first time to know the optimal size of
  // temporary arrays, and the second time for real calculation.
  int workSize;
  T workSize_req;
  info = proxy_lapack::getri(rows, ptr(), lda, ipiv, &workSize_req, -1);
  workSize = (int) real(workSize_req) + 1;
  T* work = new T[workSize];
  HMAT_ASSERT(work);
  info = proxy_lapack::getri(rows, ptr(), lda, ipiv, work, workSize);
  delete[] work;
  if (info)
    throw LapackException("getri", info);
  delete[] ipiv;
}

template<typename T> int ScalarArray<T>::svdDecomposition(ScalarArray<T>** u, Vector<double>** sigma, ScalarArray<T>** v) const {
  DECLARE_CONTEXT;
  static char * useGESDD = getenv("HMAT_GESDD");

  // Allocate free space for U, S, V
  int p = std::min(rows, cols);

  *u = new ScalarArray<T>(rows, p);
  *sigma = new Vector<double>(p);
  *v = new ScalarArray<T>(p, cols); // We create v in transposed shape (as expected by lapack zgesvd)

  assert(lda >= rows);

  char jobz = 'S';
  int info;

  {
    const size_t _m = rows, _n = cols;
    // Warning: These quantities are a rough approximation.
    // What's wrong with these estimates:
    //  - Golub only gives 14 * M*N*N + 8 N*N*N
    //  - This is for real numbers
    //  - We assume the same number of * and +
    size_t adds = 7 * _m * _n * _n + 4 * _n * _n * _n;
    size_t muls = 7 * _m * _n * _n + 4 * _n * _n * _n;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  if(useGESDD)
    info = sddCall(jobz, rows, cols, ptr(), lda, (*sigma)->ptr(), (*u)->ptr(),
                      (*u)->lda, (*v)->ptr(), (*v)->lda);
  else
    info = svdCall(jobz, jobz, rows, cols, ptr(), lda, (*sigma)->ptr(), (*u)->ptr(),
                      (*u)->lda, (*v)->ptr(), (*v)->lda);

  (*v)->transpose();

  return info;
}

template<typename T> T* ScalarArray<T>::qrDecomposition() {
  DECLARE_CONTEXT;
  //  SUBROUTINE DGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
  T* tau = (T*) calloc(std::min(rows, cols), sizeof(T));
  {
    size_t mm = std::max(rows, cols);
    size_t n = std::min(rows, cols);
    size_t multiplications = mm * n * n - (n * n * n) / 3 + mm * n + (n * n) / 2 + (29 * n) / 6;
    size_t additions = mm * n * n + (n * n * n) / 3 + 2 * mm * n - (n * n) / 2 + (5 * n) / 6;
    increment_flops(Multipliers<T>::mul * multiplications + Multipliers<T>::add * additions);
  }
  int info;
  int workSize;
  T workSize_S;
  // int info = LAPACKE_sgeqrf(LAPACK_COL_MAJOR, rows, cols, m, rows, *tau);
  info = proxy_lapack::geqrf(rows, cols, ptr(), rows, tau, &workSize_S, -1);
  HMAT_ASSERT(!info);
  workSize = (int) hmat::real(workSize_S) + 1;
  T* work = new T[workSize];// TODO Mettre dans la pile ??
  HMAT_ASSERT(work) ;
  info = proxy_lapack::geqrf(rows, cols, ptr(), rows, tau, work, workSize);
  delete[] work;

  HMAT_ASSERT(!info);
  return tau;
}

// aFull <- aFull.bTri^t with aFull=this and bTri upper triangular matrix
template<typename T>
void ScalarArray<T>::myTrmm(const ScalarArray<T>* bTri) {
  DECLARE_CONTEXT;
  int mm = rows;
  int n = rows;
  T alpha = Constants<T>::pone;
  const T *aData = bTri->const_ptr();
  int lda = bTri->rows;
  int ldb = rows;
  {
    size_t m_ = mm;
    size_t nn = n;
    size_t multiplications = m_ * nn  * (nn + 1) / 2;
    size_t additions = m_ * nn  * (nn - 1) / 2;
    increment_flops(Multipliers<T>::mul * multiplications + Multipliers<T>::add * additions);
  }
  proxy_cblas::trmm('R', 'U', 'T', 'N', mm, n, alpha, aData, lda, ptr(), ldb);
}

template<typename T>
int ScalarArray<T>::productQ(char side, char trans, T* tau, ScalarArray<T>* c) const {
  DECLARE_CONTEXT;
  assert((side == 'L') ? rows == c->rows : rows == c->cols);
  int info;
  int workSize;
  T workSize_req;
  {
    size_t _m = c->rows, _n = c->cols, _k = cols;
    size_t muls = 2 * _m * _n * _k - _n * _k * _k + 2 * _n * _k;
    size_t adds = 2 * _m * _n * _k - _n * _k * _k + _n * _k;
    increment_flops(Multipliers<T>::mul * muls + Multipliers<T>::add * adds);
  }
  info = proxy_lapack_convenience::or_un_mqr(side, trans, c->rows, c->cols, cols, const_ptr(), lda, tau, c->m, c->lda, &workSize_req, -1);
  HMAT_ASSERT(!info);
  workSize = (int) hmat::real(workSize_req) + 1;
  T* work = new T[workSize];
  HMAT_ASSERT(work);
  info = proxy_lapack_convenience::or_un_mqr(side, trans, c->rows, c->cols, cols, const_ptr(), lda, tau, c->m, c->lda, work, workSize);
  HMAT_ASSERT(!info);
  delete[] work;
  return 0;
}

template<typename T>
void ScalarArray<T>::gemv(char trans, T alpha,
                     const ScalarArray<T>* a,
                     const ScalarArray<T>* x, T beta)
{
  assert(this->cols==1);
  assert(x->cols==1);
  int matRows = a->rows;
  int matCols = a->cols;
  int aLda = a->lda;
  int64_t ops = (Multipliers<T>::add + Multipliers<T>::mul) * ((int64_t) matRows) * matCols;
  increment_flops(ops);

  if (trans == 'N') {
    assert(this->rows == a->rows);
    assert(x->rows == a->cols);
  } else {
    assert(this->rows == a->cols);
    assert(x->rows == a->rows);
  }
  proxy_cblas::gemv(trans, matRows, matCols, alpha, a->const_ptr(), aLda, x->const_ptr(), 1, beta, this->ptr(), 1);
}

template<typename T> int ScalarArray<T>::modifiedGramSchmidt(ScalarArray<T> *result, double prec ) {
  DECLARE_CONTEXT;
  {
    size_t mm = rows;
    size_t n = cols;
    size_t multiplications = 2*mm*n*n;
    size_t additions = 2*mm*n*n;
    increment_flops(Multipliers<T>::mul * multiplications + Multipliers<T>::add * additions);
  }
  int rank;
  double relative_epsilon;
  static const double LOWEST_EPSILON = 1.0e-6;

  const int original_rank(result->rows);
  assert(original_rank == result->cols);
  assert(original_rank >= cols);
  int* perm = new int[original_rank];
  for(int k = 0; k < original_rank; ++k) {
    perm[k] = k;
  }
  // Temporary arrays
  ScalarArray<T> r(original_rank, original_rank);
  Vector<T> buffer(std::max(original_rank, rows));

  // Lower threshold for relative precision
  if(prec < LOWEST_EPSILON) {
    prec = LOWEST_EPSILON;
  }

  // Init.
  Vector<double> norm2(cols); // Norm^2 of the columns of 'this' during computation
  Vector<double> norm2_orig(cols); // Norm^2 of the original columns of 'this'
  Vector<double> norm2_update(cols); // Norm^2 of the columns of 'this' normalized (will be 1. at the beginning, then decrease)
  // The actual norm^2 of column j during computation (before it is finally normalized) will be norm2[j] = norm2_orig[j] * norm2_update[j]
  // The choice of the pivot will be based on norm2_update[] or norm2[]
  rank = 0;
  relative_epsilon = 0.0;
  for(int j=0; j < cols; ++j) {
    const Vector<T> aj(*this, j);
    norm2[j] = aj.normSqr();
    relative_epsilon = std::max(relative_epsilon, norm2[j]);
    norm2_orig[j] = norm2[j];
    norm2_update[j] = 1.0 ;
    if(norm2_orig[j]==0) { // Neutralize the null columns
      norm2_orig[j] = 1.;
      norm2_update[j] = 0.0 ;
    }
  }
  relative_epsilon *= prec * prec;

  // Modified Gram-Schmidt process with column pivoting
  for(int j = 0; j < cols; ++j) {
    // Find the largest pivot
    int pivot = norm2.absoluteMaxIndex(j);
    double pivmax = norm2[pivot];

    static char *newPivot = getenv("HMAT_MGS_ALTPIV");
    if (newPivot) {
      pivot = norm2_update.absoluteMaxIndex(j);
      pivmax = norm2_update[pivot];
      relative_epsilon = prec * prec;
    }

    // Stopping criterion
    if (pivmax > relative_epsilon) {
      ++rank;

      // Pivoting
      if (j != pivot) {
        std::swap(perm[j], perm[pivot]);
        std::swap(norm2[j], norm2[pivot]);
        std::swap(norm2_orig[j], norm2_orig[pivot]);
        std::swap(norm2_update[j], norm2_update[pivot]);

        // Exchange the column 'j' and 'pivot' in this[] using buffer as temp space
        memcpy(buffer.ptr(),  const_ptr(0, j),     rows*sizeof(T));
        memcpy(ptr(0, j),     const_ptr(0, pivot), rows*sizeof(T));
        memcpy(ptr(0, pivot), buffer.const_ptr(),  rows*sizeof(T));

        // Idem for r[]
        memcpy(buffer.ptr(),    r.const_ptr(0, j),     cols*sizeof(T));
        memcpy(r.ptr(0, j),     r.const_ptr(0, pivot), cols*sizeof(T));
        memcpy(r.ptr(0, pivot), buffer.const_ptr(),    cols*sizeof(T));
      }

      // Normalisation of qj
      r.get(j,j) = sqrt(norm2[j]);
      Vector<T> aj(*this, j);
      T coef = Constants<T>::pone / r.get(j,j);
      aj.scale(coef);

      // Remove the qj-component from vectors bk (k=j+1,...,n-1)
      if (j<cols-1) {
        ScalarArray<T> bK(*this, 0, rows, j+1, cols-j-1); // All the columns of 'this' after column 'j'
        ScalarArray<T> aj_bK(r, j, 1, j+1, cols-j-1); // In 'r': row 'j', all the columns after column 'j'
        // Compute in 1 operation all the scalar products between aj and a_j+1, ..., a_n
        aj_bK.gemm('C', 'N', Constants<T>::pone, &aj, &bK, Constants<T>::zero);
        // Update a_j+1, ..., a_n
        bK.rankOneUpdateT(Constants<T>::mone, aj, aj_bK);
        // Update the norms
        for(int k = j + 1; k < cols; ++k) {
          double rjk = std::abs(r.get(j,k));
          norm2[k] -= rjk * rjk;
          norm2_update[k] -= rjk * rjk / norm2_orig[k];
        }
      }
    } else
      break;
  }

  // Update matrix dimensions
  cols = rank;
  result->rows = rank;

  // Apply perm to result
  for(int j = 0; j < result->cols; ++j) {
    // Copy the column j of r into the column perm[j] of result (only 'rank' rows)
    memcpy(result->ptr(0, perm[j]), r.const_ptr(0, j), result->rows*sizeof(T));
  }
  // Clean up
  delete[] perm;
  /* end of modified Gram-Schmidt */
  return rank;
}

template<typename T>
void ScalarArray<T>::multiplyWithDiagOrDiagInv(const ScalarArray<T>* d, bool inverse, bool left) {
  assert(d);
  assert(left || (cols == d->rows));
  assert(!left || (rows == d->rows));
  assert(d->cols==1);

  {
    const size_t _rows = rows, _cols = cols;
    increment_flops(Multipliers<T>::mul * _rows * _cols);
  }
  if (left) { // line i is multiplied by d[i] or 1/d[i]
    // TODO: Test with scale to see if it is better.
    if (inverse) {
      ScalarArray<T> *d2 = new ScalarArray<T>(rows,1);
      for (int i = 0; i < rows; i++)
        d2->get(i) = Constants<T>::pone / d->get(i);
      d = d2;
    }
    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        get(i, j) *= d->get(i);
      }
    }
    if (inverse) delete(d);
  } else { // column j is multiplied by d[j] or 1/d[j]
    for (int j = 0; j < cols; j++) {
      T diag_val = inverse ? Constants<T>::pone / d->get(j,0) : d->get(j);
      proxy_cblas::scal(rows, diag_val, &get(0,j), 1);
    }
  }
}

template<typename T>
void ScalarArray<T>::multiplyWithDiag(const ScalarArray<double>* d) {
  assert(d);
  assert(cols <= d->rows); // d can be larger than needed
  assert(d->cols==1);

  {
    const size_t _rows = rows, _cols = cols;
    increment_flops(Multipliers<T>::mul * _rows * _cols);
  }
  for (int j = 0; j < cols; j++) {
    T diag_val = T(d->get(j));
    proxy_cblas::scal(rows, diag_val, m+j*lda, 1); // We don't use ptr() on purpose, because is_ortho is preserved here
  }
}

template<typename T>
T Vector<T>::dot(const Vector<T>* x, const Vector<T>* y) {
  assert(x->cols == 1);
  assert(y->cols == 1);
  assert(x->rows == y->rows);
  // TODO: Beware of large vectors (>2 billion elements) !
  return proxy_cblas_convenience::dot_c(x->rows, x->const_ptr(), 1, y->const_ptr(), 1);
}

template<typename T>
int Vector<T>::absoluteMaxIndex(int startIndex) const {
  assert(this->cols == 1);
  return startIndex + proxy_cblas::i_amax(this->rows - startIndex, this->const_ptr() + startIndex, 1);
}

// the classes declaration
template class ScalarArray<S_t>;
template class ScalarArray<D_t>;
template class ScalarArray<C_t>;
template class ScalarArray<Z_t>;

// the classes declaration
template class Vector<S_t>;
template class Vector<D_t>;
template class Vector<C_t>;
template class Vector<Z_t>;
}  // end namespace hmat
