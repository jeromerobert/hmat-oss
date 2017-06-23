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

template<typename T> size_t ScalarArray<T>::storedZeros() {
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
        proxy_cblas::scal(block_size_blas, alpha, m + nm - block_size_blas, 1);
        nm -= block_size_blas;
      }
      proxy_cblas::scal(nm, alpha, m, 1);
    }
  } else {
    T* x = m;
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
  assert(m);
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
    ScalarArray<T> tmp(rows, cols);
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
    memcpy(result->m, m, size);
  } else {
    for (int col = 0; col < cols; col++) {
      size_t resultOffset = ((size_t) result->lda) * col;
      size_t offset = ((size_t) lda) * col;
      memcpy(result->m + resultOffset, m + offset, rows * sizeof(T));
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
ScalarArray<T> ScalarArray<T>::rowsSubset(const int rowsOffset, const int rowsSize) const {
  assert(rowsOffset + rowsSize <= rows);
  return ScalarArray<T>(m + rowsOffset, rowsSize, cols, lda);
}

template<typename T>
void ScalarArray<T>::gemm(char transA, char transB, T alpha,
                         const ScalarArray<T>* a, const ScalarArray<T>* b,
                         T beta) {
DECLARE_CONTEXT;
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
  proxy_cblas::gemm(transA, transB, aRows, n, k, alpha, a->m, a->lda, b->m, b->lda,
                    beta, this->m, this->lda);
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
    memcpy(m, a->m, size * sizeof(T));
    return;
  }

  for (int col = 0; col < a->cols; col++) {
    proxy_cblas::copy(a->rows, a->m + col * a->lda, 1,
                m + rowOffset + ((colOffset + col) * lda), 1);
  }
}

template<typename T>
void ScalarArray<T>::copyMatrixAtOffset(const ScalarArray<T>* a,
                                       int rowOffset, int colOffset,
                                       int rowsToCopy, int colsToCopy) {
  assert(rowOffset + rowsToCopy <= rows);
  assert(colOffset + colsToCopy <= cols);
  for (int col = 0; col < colsToCopy; col++) {
    proxy_cblas::copy(rowsToCopy, a->m + col * a->lda, 1,
                (m + rowOffset + ((colOffset + col) * lda)), 1);
  }
}

template<typename T> void ScalarArray<T>::addRand(double epsilon) {
  DECLARE_CONTEXT;
  if (lda == rows) {
    for (size_t i = 0; i < ((size_t) rows) * cols; ++i) {
      m[i] *= 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
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
      m[i] *= C_t(c1, c2);
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
      m[i] *= Z_t(c1, c2);
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
    proxy_cblas::axpy(size, alpha, a->m, 1, m, 1);
    return;
  }

  for (int col = 0; col < cols; col++) {
    proxy_cblas::axpy(rows, alpha, a->m + ((size_t) col) * a->lda, 1, m + ((size_t) col) * lda, 1);
  }
}

template<typename T>
double ScalarArray<T>::normSqr() const {
  size_t size = ((size_t) rows) * cols;
  T result = Constants<T>::zero;

  // Fast path
  if ((size < 1000000000) && (lda == rows)) {
    result += proxy_cblas_convenience::dot_c(size, m, 1, m, 1);
    return hmat::real(result);
  }
  for (int col = 0; col < cols; col++) {
    result += proxy_cblas_convenience::dot_c(rows, m + col * lda, 1, m + col * lda, 1);
  }
  return hmat::real(result);
}

template<typename T> double ScalarArray<T>::norm() const {
  return sqrt(normSqr());
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
  r = fread(m, size, 1, f);
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
  memcpy(mat, m, size - 5 * sizeof(int));
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

template<typename T>
void Vector<T>::gemv(char trans, T alpha,
                     const ScalarArray<T>* a,
                     const Vector* x, T beta)
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
  proxy_cblas::gemv(trans, matRows, matCols, alpha, a->m, aLda, x->m, 1, beta, this->m, 1);
}

template<typename T>
void Vector<T>::addToMe(const Vector<T>* x) {
  ScalarArray<T>::axpy(Constants<T>::pone, x);
}

template<typename T>
void Vector<T>::subToMe(const Vector<T>* x) {
  ScalarArray<T>::axpy(Constants<T>::mone, x);
}

template<typename T>
T Vector<T>::dot(const Vector<T>* x, const Vector<T>* y) {
  assert(x->cols == 1);
  assert(y->cols == 1);
  assert(x->rows == y->rows);
  // TODO: Beware of large vectors (>2 billion elements) !
  return proxy_cblas_convenience::dot_c(x->rows, x->m, 1, y->m, 1);
}

template<typename T>
int Vector<T>::absoluteMaxIndex(int startIndex) const {
  assert(this->cols == 1);
  return startIndex + proxy_cblas::i_amax(this->rows - startIndex, this->m + startIndex, 1);
}

// MmapedScalarArray
template<typename T>
MmapedScalarArray<T>::MmapedScalarArray(int rows, int cols, const char* filename)
  : m(NULL, rows, cols), mmapedFile(NULL), fd(-1), size(0) {
#ifdef _WIN32
  HMAT_ASSERT(false); // no mmap() on Windows
#else
  int ierr;

  size = ((size_t) rows) * cols * sizeof(T) + 5 * sizeof(int);
  fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
  HMAT_ASSERT(fd != -1);
  ierr = lseek(fd, size - 1, SEEK_SET);
  HMAT_ASSERT(ierr != -1);
  ierr = write(fd, "", 1);
  HMAT_ASSERT(ierr == 1);
  mmapedFile = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ierr = (mmapedFile == MAP_FAILED) ? 1 : 0;
  HMAT_ASSERT(!ierr);
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
MmapedScalarArray<T>::~MmapedScalarArray() {
#ifndef _WIN32
  close(fd);
  munmap(mmapedFile, size);
#endif
}

template<typename T>
MmapedScalarArray<T>* MmapedScalarArray<T>::fromFile(const char* filename) {
  MmapedScalarArray<T>* result = new MmapedScalarArray();

#ifdef _WIN32
  HMAT_ASSERT(false); // no mmap() on Windows
#else
  int ierr;
  result->fd = open(filename, O_RDONLY);
  HMAT_ASSERT(result->fd != -1);
  struct stat fileStat;
  ierr = fstat(result->fd, &fileStat);
  HMAT_ASSERT(!ierr);
  size_t fileSize = fileStat.st_size;

  result->mmapedFile = mmap(0, fileSize, PROT_READ, MAP_SHARED, result->fd, 0);
  ierr = (result->mmapedFile == MAP_FAILED) ? 1 : 0;
  HMAT_ASSERT(!ierr);
  int* header = (int*) result->mmapedFile;
  // Check the consistency of the file
  HMAT_ASSERT(header[0] == Constants<T>::code);
  HMAT_ASSERT(header[3] == sizeof(T));
  HMAT_ASSERT(header[1] * ((size_t) header[2]) * sizeof(T) + (5 * sizeof(int)) == fileSize);
  result->m.lda = result->m.rows = header[1];
  result->m.cols = header[2];
  result->m.m = (T*) (header + 5);
#endif
  return result;
}

// the classes declaration
template class ScalarArray<S_t>;
template class ScalarArray<D_t>;
template class ScalarArray<C_t>;
template class ScalarArray<Z_t>;

template class MmapedScalarArray<S_t>;
template class MmapedScalarArray<D_t>;
template class MmapedScalarArray<C_t>;
template class MmapedScalarArray<Z_t>;

// the classes declaration
template class Vector<S_t>;
template class Vector<D_t>;
template class Vector<C_t>;
template class Vector<Z_t>;
}  // end namespace hmat
