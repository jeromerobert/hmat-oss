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
#include "scalar_array.hpp"

#include "data_types.hpp"
#include "lapack_overloads.hpp"
#include "blas_overloads.hpp"
#include "lapack_exception.hpp"
#include "common/memory_instrumentation.hpp"
#include "system_types.h"
#include "common/my_assert.h"
#include "common/context.hpp"
#include "common/timeline.hpp"
#include "lapack_operations.hpp"

#include <cstring> // memset
#include <algorithm> // swap
#include <iostream>
#include <fstream>
#include <cmath>
#include <fcntl.h>
#include <limits>

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

namespace {
struct EnvVarSA {
  bool sumCriterion;
  bool gessd;
  bool mgsBlas3;
  bool initPivot;
  bool mgsAltPivot;
  bool testOrtho;
  EnvVarSA() {
    sumCriterion = getenv("HMAT_SUM_CRITERION") != nullptr;
    gessd = getenv("HMAT_GESDD") != nullptr;
    mgsBlas3 = getenv("HMAT_MGS_BLAS3") != nullptr;
    initPivot = getenv("HMAT_TRUNC_INITPIV") != nullptr;
    mgsAltPivot = getenv("HMAT_MGS_ALTPIV") != nullptr;
    testOrtho = getenv("HMAT_TEST_ORTHO") != nullptr;
  }
};
const EnvVarSA envSA;

/*! \brief Returns the number of singular values to keep.

     The stop criterion is (assuming that the singular value
     are in descending order):
         sigma [k] / sigma[0] <epsilon
     except if env. var. HMAT_SUM_CRITERION is set, in which case the criterion is:
         sigma [k] / SUM (sigma) <epsilon

     \param sigma table of singular values at least maxK elements.
     \param epsilon tolerance.
     \return int the number of singular values to keep.
 */
template <typename T>
int findK(hmat::Vector<T> &sigma, double epsilon) {
  assert(epsilon >= 0.);
  double threshold_eigenvalue = 0.0;
  if (envSA.sumCriterion) {
    for (int i = 0; i < sigma.rows; i++) {
      threshold_eigenvalue += sigma[i];
    }
  } else {
    threshold_eigenvalue = sigma[0];
  }
  threshold_eigenvalue *= epsilon;
  int i = 0;
  for (i = 0; i < sigma.rows; i++) {
    if (sigma[i] <= threshold_eigenvalue){
      break;
    }
  }
  return i;
}

struct SigmaPrinter {
  bool enabled;
  SigmaPrinter() {
    enabled = getenv("HMAT_PRINT_SIGMA") != nullptr;
  }
  template <typename T> void print(const hmat::Vector<T> & sigma) const {
    if(!enabled)
      return;
    // Use a buffer and printf for better output in multi-thread
    std::stringstream buf;
    for(int i = 0; i < sigma.rows; i++) {
      buf << sigma[i] << " ";
    }
    printf("[SIGMA] %s\n", buf.str().c_str());
  }
};
static SigmaPrinter sigmaPrinter;
}

namespace hmat {

Factorization convert_int_to_factorization(int t) {
    switch (t) {
    case hmat_factorization_none:
        return Factorization::NONE;
    case hmat_factorization_lu:
        return Factorization::LU;
    case hmat_factorization_ldlt:
        return Factorization::LDLT;
    case hmat_factorization_llt:
        return Factorization::LLT;
    case hmat_factorization_hodlr:
        return Factorization::HODLR;
    case hmat_factorization_hodlrsym:
        return Factorization::HODLRSYM;
    default:
        HMAT_ASSERT(false);
    }
}

int convert_factorization_to_int(Factorization f) {
    switch (f) {
    case Factorization::NONE:
        return hmat_factorization_none;
    case Factorization::LU:
        return hmat_factorization_lu;
    case Factorization::LDLT:
        return hmat_factorization_ldlt;
    case Factorization::LLT:
        return hmat_factorization_llt;
    case Factorization::HODLR:
        return hmat_factorization_hodlr;
    default:
        HMAT_ASSERT(false);
    }
}

/** ScalarArray */
template<typename T>
ScalarArray<T>::ScalarArray(T* _m, int _rows, int _cols, int _lda)
  : ownsMemory(false), m(_m), rows(_rows), cols(_cols), lda(_lda) {
  if (lda == -1) {
    lda = rows;
  }
  ownsFlag = true ;
#ifdef HMAT_SCALAR_ARRAY_ORTHO
  is_ortho = (int*)calloc(1, sizeof(int));
#endif
  assert(lda >= rows);
}

template<typename T>
ScalarArray<T>::ScalarArray(int _rows, int _cols, bool initzero)
  : ownsMemory(true), ownsFlag(true), rows(_rows), cols(_cols), lda(_rows) {
  size_t size = sizeof(T) * rows * cols;
  if(size == 0) {
    m = nullptr;
    return;
  }
  void * p;
#ifdef HAVE_JEMALLOC
  p = initzero ? je_calloc(size, 1) : je_malloc(size);
#else
  p = initzero ? calloc(size, 1) : malloc(size);
#endif
  m = static_cast<T*>(p);
#ifdef HMAT_SCALAR_ARRAY_ORTHO
  is_ortho = (int*)calloc(1, sizeof(int));
  setOrtho(initzero ? 1 : 0); // buffer filled with 0 is orthogonal
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
#ifdef HMAT_SCALAR_ARRAY_ORTHO
  if (ownsFlag) {
    free(is_ortho);
    is_ortho = NULL;
  }
#endif
}

template<typename T> void ScalarArray<T>::resize(int col_num) {
  assert(ownsFlag);
  if(col_num > cols)
    setOrtho(0);
  int diffcol = col_num - cols;
  if(diffcol > 0)
    MemoryInstrumenter::instance().alloc(sizeof(T) * rows * diffcol,
                                         MemoryInstrumenter::FULL_MATRIX);
  else
    MemoryInstrumenter::instance().free(sizeof(T) * rows * -diffcol,
                                        MemoryInstrumenter::FULL_MATRIX);

  size_t new_size_bytes = sizeof(T) * rows * col_num;
#ifdef HAVE_JEMALLOC
  void * p = je_realloc(m, new_size_bytes);
#else
  void * p = realloc(m, new_size_bytes);
#endif

if(p == NULL && new_size_bytes > 0) {
  if(diffcol > 0)
    MemoryInstrumenter::instance().free(sizeof(T) * lda * diffcol,
                                          MemoryInstrumenter::FULL_MATRIX);


  throw std::bad_alloc();
}

  m = static_cast<T*>(p);
  cols = col_num;
}

template<typename T> void ScalarArray<T>::clear() {
  assert(lda == rows);
  std::fill(m, m + ((size_t) rows) * cols, 0);
  setOrtho(1); // we dont use ptr(): buffer filled with 0 is orthogonal
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
    if (alpha == T(0)) {
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
    if (alpha == T(0)) {
      for (int col = 0; col < cols; col++) {
        std::fill(x, x + rows, 0);
        x += lda;
      }
    } else {
      for (int col = 0; col < cols; col++) {
        proxy_cblas::scal(rows, alpha, x, 1);
        x += lda;
      }
    }
  }
  if (alpha == T(0)) setOrtho(1); // buffer filled with 0 is orthogonal
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
      // We don't use c->ptr() on purpose, because is_ortho is preserved here
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
    result = new ScalarArray<T>(rows, cols, false);

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
  result->setOrtho(getOrtho());
  return result;
}

template<typename T> ScalarArray<T>* ScalarArray<T>::copyAndTranspose(ScalarArray<T>* result) const {
  if(result == NULL)
    result = new ScalarArray<T>(cols, rows);
#ifdef HAVE_MKL_IMATCOPY
  if (lda == rows && result->lda == result->rows) {
    proxy_mkl::omatcopy(rows, cols, const_ptr(), result->ptr());
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

template <typename T>
ScalarArray<T> ScalarArray<T>::colsSubset(const int colOffset, const int colSize) const
{
  assert(colOffset + colSize <= cols);
  return ScalarArray<T>(*this, 0, rows, colOffset, colSize);
}

template<typename T>
void ScalarArray<T>::gemm(char transA, char transB, T alpha,
                         const ScalarArray<T>* a, const ScalarArray<T>* b,
                         T beta) {
  const int aRows  = (transA == 'N' ? a->rows : a->cols);
  const int n  = (transB == 'N' ? b->cols : b->rows);
  const int k  = (transA == 'N' ? a->cols : a->rows);
  const int tA = (transA == 'N' ? 0 : ( transA == 'T' ? 1 : 2 ));
  const int tB = (transB == 'N' ? 0 : ( transB == 'T' ? 1 : 2 ));
  Timeline::Task t(Timeline::BLASGEMM, &rows, &cols, &k, &tA, &tB);
  (void)t;
  assert(rows == aRows);
  assert(cols == n);
  assert(k == (transB == 'N' ? b->rows : b->cols));
  {
    const size_t _m = aRows, _n = n, _k = k;
    const size_t adds = _m * _n * _k;
    const size_t muls = _m * _n * _k;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  assert(a->lda >= a->rows);
  assert(b->lda >= b->rows);
  assert(a->lda > 0);
  assert(b->lda > 0);

  if (n > 1 || transB != 'N')
    proxy_cblas::gemm(transA, transB, aRows, n, k, alpha, a->const_ptr(), a->lda, b->const_ptr(), b->lda,
                      beta, this->ptr(), this->lda);
  else
    proxy_cblas::gemv(transA, a->rows, a->cols, alpha, a->const_ptr(), a->lda, b->const_ptr(), 1, beta, this->ptr(), 1);
}

template<typename T>
void ScalarArray<T>::copyMatrixAtOffset(const ScalarArray<T>* a,
                                       int rowOffset, int colOffset) {
  assert(rowOffset + a->rows <= rows);
  assert(colOffset + a->cols <= cols);

  if (rowOffset == 0 && a->rows == rows &&
      a->lda == a->rows && lda == rows) {
    memcpy(ptr() + colOffset * lda, a->const_ptr(), sizeof(T) * rows * a->cols);
    // If I copy the whole matrix, I copy this flag
    if(a->cols == cols)
      setOrtho(a->getOrtho());
  } else {
    for (int col = 0; col < a->cols; col++) {
      memcpy(ptr() + rowOffset + (colOffset + col) * lda,
             a->const_ptr() + col * a->lda,
             sizeof(T) * a->rows);
    }
  }
}

template<typename T, typename std::enable_if<hmat::Types<T>::IS_REAL::value, T*>::type = nullptr>
void addRandSFINAE(ScalarArray<T>& a, double epsilon) {
  if (a.lda == a.rows) {
    for (size_t i = 0; i < ((size_t) a.rows) * a.cols; ++i) {
      a.get(i) *= 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
    }
  } else {
    for (int col = 0; col < a.cols; ++col) {
      for (int row = 0; row < a.rows; ++row) {
        a.get(row, col) *= 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
      }
    }
  }
}

template<typename T, typename std::enable_if<!hmat::Types<T>::IS_REAL::value, T*>::type = nullptr>
void addRandSFINAE(ScalarArray<T>& a, double epsilon) {
  if (a.lda == a.rows) {
    for (size_t i = 0; i < ((size_t) a.rows) * a.cols; ++i) {
      typename T::value_type c1 = 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
      typename T::value_type c2 = 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
      a.get(i) *= T(c1, c2);
    }
  } else {
    for (int col = 0; col < a.cols; ++col) {
      for (int row = 0; row < a.rows; ++row) {
        typename T::value_type c1 = 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
        typename T::value_type c2 = 1.0 + epsilon*(1.0-2.0*rand()/(double)RAND_MAX);
        a.get(row, col) *= T(c1, c2);
      }
    }
  }
}

template<typename T>
void ScalarArray<T>::addRand(double epsilon) {
  DECLARE_CONTEXT;
  addRandSFINAE<T>(*this, epsilon);
}

template<typename T>
void ScalarArray<T>::axpy(T alpha, const ScalarArray<T>* a) {
  assert(rows == a->rows);
  assert(cols == a->cols);
  size_t size = ((size_t) rows) * cols;

  increment_flops(Multipliers<T>::add * size
		  + (alpha == T(1) ? 0 : Multipliers<T>::mul * size));
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
  T result = 0;

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
      result += real(proxy_cblas_convenience::dot_c(rows, const_ptr(0, i), 1, const_ptr(0, j), 1) *
                     proxy_cblas_convenience::dot_c(b.rows, b.const_ptr(0, i), 1, b.const_ptr(0, j), 1));
    }
  }
  result *= 2.0;
  for (int i = 0; i < k; ++i) {
    result += real(proxy_cblas_convenience::dot_c(rows, const_ptr(0, i), 1, const_ptr(0, i), 1) *
                   proxy_cblas_convenience::dot_c(b.rows, b.const_ptr(0, i), 1, b.const_ptr(0, i), 1));
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

template<typename T, typename std::enable_if<hmat::Types<T>::IS_REAL::value, T*>::type = nullptr>
void checkNanSFINAE(const ScalarArray<T>* m) {
  for (int col = 0; col < m->cols; col++) {
    for (int row = 0; row < m->rows; row++) {
      HMAT_ASSERT(swIsFinite(m->get(row, col)));
    }
  }
}

template<typename T, typename std::enable_if<!hmat::Types<T>::IS_REAL::value, T*>::type = nullptr>
void checkNanSFINAE(const ScalarArray<T>* m) {
  for (int col = 0; col < m->cols; col++) {
    for (int row = 0; row < m->rows; row++) {
      HMAT_ASSERT(swIsFinite(m->get(row, col).real()));
      HMAT_ASSERT(swIsFinite(m->get(row, col).imag()));
    }
  }
}

template<typename T>
void ScalarArray<T>::checkNan() const {
  checkNanSFINAE<T>(this);
}

template<typename T> bool ScalarArray<T>::isZero() const {
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
      if (get(i, j) != T(0))
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

template<typename T>
class InvalidDiagonalException: public LapackException {
    std::string invalidDiagonalMessage_;
public:
    InvalidDiagonalException(const T value, const int j, const char * where)
      : LapackException(where, -1)
    {
        std::stringstream sstm;
        sstm << "In " << where << ", diagonal index "<< j << " has an invalid value " << value;
        invalidDiagonalMessage_ = sstm.str();
    }

    virtual const char* what() const noexcept {
        return invalidDiagonalMessage_.c_str();
    }
};

template<typename T, typename std::enable_if<hmat::Types<T>::IS_REAL::value, T*>::type = nullptr>
void assertPositive(const T v, const int j, const char * const where) {
    if(!(v > 0))
      throw InvalidDiagonalException<S_t>(v, j, where);
}

template<typename T, typename std::enable_if<!hmat::Types<T>::IS_REAL::value, T*>::type = nullptr>
void assertPositive(const T v, const int j, const char * const where) {
    if(v == T(0))
      throw InvalidDiagonalException<T>(v, j, where);
}

template<typename T>
void ScalarArray<T>::ldltDecomposition(Vector<T>& diagonal) {
  assert(rows == cols); // We expect a square matrix
  //TODO : add flops counter

  // Standard LDLt factorization algorithm is:
  //  diag[j] = A(j,j) - sum_{k < j} L(j,k)^2 diag[k]
  //  L(i,j) = (A(i,j) - sum_{k < j} (L(i,k)L(j,k)diag[k])) / diag[j]
  // See for instance http://en.wikipedia.org/wiki/Cholesky_decomposition
  // An auxiliary array is introduced in order to perform less multiplications,
  // see  algorithm 1 in http://icl.cs.utk.edu/projectsfiles/plasma/pubs/ICL-UT-11-03.pdf
  int n = rows;
  T* v = new T[n];
  HMAT_ASSERT(v);
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < j; i++)
      v[i] = get(j,i) * get(i,i);

    v[j] = get(j,j);
    for (int i = 0; i < j; i++)
      // Do not use the -= operator because it's buggy in the intel compiler
      v[j] = v[j] - get(j,i) * v[i];

    get(j,j) = v[j];
    for (int i = 0; i < j; i++)
      for (int k = j+1; k < n; k++)
        get(k,j) -= get(k,i) * v[i];

    for (int k = j+1; k < n; k++) {
      if (v[j] == T(0))
        throw InvalidDiagonalException<T>(v[j], j, "ldltDecomposition");
      get(k,j) /= v[j];
    }
  }

  for(int i = 0; i < n; i++) {
    diagonal[i] = get(i,i);
    get(i,i) = 1;
    for (int j = i + 1; j < n; j++)
      get(i,j) = 0;
  }

  delete[] v;
}

template<typename T>
void ScalarArray<T>::lltDecomposition() {
  assert(rows == cols); // We expect a square matrix

  // Standard LLt factorization algorithm is:
  //  L(j,j) = sqrt( A(j,j) - sum_{k < j} L(j,k)^2)
  //  L(i,j) = ( A(i,j) - sum_{k < j} L(i,k)L(j,k) ) / L(j,j)

    // from http://www.netlib.org/lapack/lawnspdf/lawn41.pdf page 120
  int n = rows;
  const size_t n2 = (size_t) n*n;
  const size_t n3 = n2 * n;
  const size_t muls = n3 / 6 + n2 / 2 + n / 3;
  const size_t adds = n3 / 6 - n / 6;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  if(hmat::Types<T>::IS_REAL::value) {
    // For real matrices, we can use the lapack version.
    // (There is no L.Lt factorisation for complex matrices in lapack)
    int info = proxy_lapack::potrf('L', rows, m, lda);
    if(info != 0)
      assertPositive(T(-1), info, "potrf");
  } else {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < j; k++)
        get(j,j) -= get(j,k) * get(j,k);
      assertPositive(get(j, j), j, "lltDecomposition");

      get(j,j) = std::sqrt(get(j,j));

      for (int k = 0; k < j; k++)
        for (int i = j+1; i < n; i++)
          get(i,j) -= get(i,k) * get(j,k);

      for (int i = j+1; i < n; i++) {
        get(i,j) /= get(j,j);
      }
    }
  }

  for (int j = 0; j < n; j++) {
        for(int i = 0; i < j; i++) {
            get(i,j) = 0;
        }
    }
}

// Helper functions for solveTriangular
inline char to_blas(const Side side) {
    return side == Side::LEFT ? 'L' : 'R';
}

inline char to_blas(const Uplo uplo) {
    return uplo == Uplo::LOWER ? 'L' : 'U';
}

inline char to_blas(const Diag diag) {
    return diag == Diag::UNIT ? 'U' : 'N';
}


// The following code is very close to that of ZGETRS in LAPACK.
// However, the resolution here is divided in the two parties.

// Warning! The matrix has been obtained with ZGETRF therefore it is
// permuted! We have the factorization A = P L U  with P the
// permutation matrix. So to solve L X = B, we must
// solve LX = (P ^ -1 B), which is done by ZLASWP with
// the permutation. we used it just like in ZGETRS.
template<typename T>
void ScalarArray<T>::solveLowerTriangularLeft(ScalarArray<T>* x, const FactorizationData<T>& context,
  Diag diag, Uplo uplo) const {
  if (context.algo == Factorization::LU && uplo == Uplo::LOWER)
    proxy_lapack::laswp(x->cols, x->ptr(), x->lda, 1, rows, context.data.pivots, 1);
  x->trsm(Side::LEFT, uplo, uplo == Uplo::LOWER ? 'N' : 'T', diag, 1, this);
}

// The resolution of the upper triangular system does not need to
//  change the order of columns.
//  The pivots are not necessary here, but this helps to check
//  the matrix was factorized before.

template<typename T>
void ScalarArray<T>::solveUpperTriangularRight(ScalarArray<T>* x, const FactorizationData<T>&,
  Diag diag, Uplo uplo) const {
  x->trsm(Side::RIGHT, uplo, uplo == Uplo::LOWER ? 'T' : 'N', diag, 1, this);
}

template<typename T>
void ScalarArray<T>::solveUpperTriangularLeft(ScalarArray<T>* x, const FactorizationData<T>&,
  Diag diag, Uplo uplo) const {
  x->trsm(Side::LEFT, uplo, uplo == Uplo::LOWER ? 'T' : 'N', diag, 1, this);
}

template<typename T>
void ScalarArray<T>::solve(ScalarArray<T>* x, const FactorizationData<T>& context) const {
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
  HMAT_ASSERT(context.algo == Factorization::LU);
  ierr = proxy_lapack::getrs('N', rows, x->cols, const_ptr(), lda, context.data.pivots, x->ptr(), x->rows);
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
  info = proxy_lapack::getri(rows, ptr(), lda, ipiv);
  if (info)
    throw LapackException("getri", info);
  delete[] ipiv;
}

template<typename T> int ScalarArray<T>::truncatedSvdDecomposition(ScalarArray<T>** u, ScalarArray<T>** v, double epsilon, bool workAroundFailures, Vector<typename Types<T>::real> *sigma_out) const {
  Vector<typename Types<T>::real>* sigma = NULL;

  svdDecomposition(u, &sigma, v, workAroundFailures);
  sigmaPrinter.print(*sigma);
  // Control of the approximation
  int newK = findK(*sigma, epsilon);

  if(newK == 0) {
    delete *u;
    delete *v;
    delete sigma;
    *u = NULL;
    *v = NULL;
    return 0;
  }

  (*u)->resize(newK);
  sigma->rows = newK;
  (*v)->resize(newK);

  // We put the square root of singular values in sigma
  for (int i = 0; i < newK; i++)
    (*sigma)[i] = sqrt((*sigma)[i]);

  // Apply sigma 'symmetrically' on u and v
  (*u)->multiplyWithDiag(sigma);
  (*v)->multiplyWithDiag(sigma);
  if(sigma_out)
  {
    //printf("\nsqrt(Sigma_0) = %f, (MID)\n", (*sigma)[0]);
    *sigma_out = *sigma;
  }
  else{

  delete sigma;
  }

  return newK;
}

template <typename T>
int ScalarArray<T>::svdDecomposition(ScalarArray<T> **u,
                                     Vector<typename Types<T>::real> **sigma,
                                     ScalarArray<T> **v,
                                     bool workAroundFailures) const {
  DECLARE_CONTEXT;
  Timeline::Task t(Timeline::SVD, &rows, &cols);
  (void)t;
  // Allocate free space for U, S, V
  int p = std::min(rows, cols);

  *u = new ScalarArray<T>(rows, p, false);
  *sigma = new Vector<typename Types<T>::real>(p);
  *v = new ScalarArray<T>(p, cols, false); // We create v in transposed shape (as expected by lapack zgesvd)

  // To be prepared for working around a failure in SVD, I must do a copy of 'this'
  ScalarArray<T> *a = workAroundFailures ? copy() : NULL;

  assert(lda >= rows);

  char jobz = 'S';
  int info=0;

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

  try {
    if(envSA.gessd)
      info = sddCall(jobz, rows, cols, ptr(), lda, (*sigma)->ptr(), (*u)->ptr(),
                     (*u)->lda, (*v)->ptr(), (*v)->lda);
    else
      info = svdCall(jobz, jobz, rows, cols, ptr(), lda, (*sigma)->ptr(), (*u)->ptr(),
                     (*u)->lda, (*v)->ptr(), (*v)->lda);
    (*v)->transpose();
    (*u)->setOrtho(1);
    (*v)->setOrtho(1);
    delete a;
  } catch(LapackException & e) {
    // if SVD fails, and if workAroundFailures is set to true, I return a "fake" result that allows
    // computation to proceed. Otherwise, I stop here.
    if (!workAroundFailures) throw;

    printf("%s overridden...\n", e.what());
    // If rows<cols, then p==rows, 'u' is square, 'v' has the dimensions of 'this'.
    // fake 'u' is identity, fake 'v' is 'this^T'
    if (rows<cols) {
      for (int i=0 ; i<rows ; i++)
        for (int j=0 ; j<p ; j++)
          (*u)->get(i,j) = i==j ? 1 : 0 ;
      (*u)->setOrtho(1);
      delete *v;
      *v = a ;
      (*v)->transpose();
    } else {
      // Otherwise rows>=cols, then p==cols, 'u' has the dimensions of 'this', 'v' is square.
      // fake 'u' is 'this', fake 'v' is identity
      delete *u;
      *u = a ;
      for (int i=0 ; i<p ; i++)
        for (int j=0 ; j<cols ; j++)
          (*v)->get(i,j) = i==j ? 1 : 0 ;
      (*v)->setOrtho(1);
    }
    // Fake 'sigma' is all 1
    for (int i=0 ; i<p ; i++)
      (**sigma)[i] = 1;
  }

  return info;
}

template<typename T> void ScalarArray<T>::orthoColumns(ScalarArray<T> *resultR, int initialPivot) {
  DECLARE_CONTEXT;

  // We use the fact that the initial 'initialPivot' are orthogonal
  // We just normalize them and update the columns >= initialPivot using MGS-like approach
  ScalarArray<T> bK(*this, 0, rows, initialPivot, cols-initialPivot); // All the columns of 'this' after column 'initialPivot'
  for(int j = 0; j < initialPivot; ++j) {
    // Normalisation of column j
    Vector<T> aj(*this, j);
    resultR->get(j,j) = aj.norm();
    T coef = T(1) / resultR->get(j,j);
    aj.scale(coef);
  }
  // Remove the qj-component from vectors bk (k=initialPivot,...,n-1)
  if (initialPivot<cols) {
    if (envSA.mgsBlas3) {
      ScalarArray<T> aJ(*this, 0, rows, 0, initialPivot); // All the columns of 'this' from 0 to 'initialPivot-1'
      ScalarArray<T> aJ_bK(*resultR, 0, initialPivot, initialPivot, cols-initialPivot); // In 'r': row '0' to 'initialPivot-1', all the columns after column 'initialPivot-1'
      // Compute in 1 operation all the scalar products between a_0,...,a_init-1 and a_init, ..., a_n-1
      aJ_bK.gemm('C', 'N', 1, &aJ, &bK, 0);
      // Update a_init, ..., a_n-1
      bK.gemm('N', 'N', -1, &aJ, &aJ_bK, 1);
    } else {
      for(int j = 0; j < initialPivot; ++j) {
        Vector<T> aj(*this, j);
        ScalarArray<T> aj_bK(*resultR, j, 1, initialPivot, cols-initialPivot); // In 'r': row 'j', all the columns after column 'initialPivot'
        // Compute in 1 operation all the scalar products between aj and a_firstcol, ..., a_n
        aj_bK.gemm('C', 'N', 1, &aj, &bK, 0);
        // Update a_firstcol, ..., a_n
        bK.rankOneUpdateT(-1, aj, aj_bK);
      }
    }
  } // if (initialPivot<cols)
}

template<typename T> void ScalarArray<T>::qrDecomposition(ScalarArray<T> *resultR, int initialPivot) {
  DECLARE_CONTEXT;
  Timeline::Task t(Timeline::QR, &rows, &cols, &initialPivot);
  (void)t;

  if (!envSA.initPivot) initialPivot=0;
  assert(initialPivot>=0 && initialPivot<=cols);

  ScalarArray<T> *bK = NULL, *restR = NULL, *a = this; // we need this because if initialPivot>0, we will run the end of the computation on a subset of 'this'

  // Initial pivot
  if (initialPivot) {
    // modify the columns [initialPivot, cols[ to make them orthogonals to columns [0, initialPivot[
    orthoColumns(resultR, initialPivot);
    // then we do qrDecomposition on the remaining columns (without initial pivot)
    bK = new ScalarArray<T> (*this, 0, rows, initialPivot, cols-initialPivot); // All the columns of 'this' after column 'initialPivot'
    restR = new ScalarArray<T> (*resultR, initialPivot, cols-initialPivot, initialPivot, cols-initialPivot); // In 'r': all the rows and columns after row/column 'initialPivot'
    a = bK;
    resultR = restR;
  }

  //  SUBROUTINE DGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
  T* tau = (T*) calloc(std::min(a->rows, a->cols), sizeof(T));
  {
    size_t mm = std::max(a->rows, a->cols);
    size_t n = std::min(a->rows, a->cols);
    size_t multiplications = mm * n * n - (n * n * n) / 3 + mm * n + (n * n) / 2 + (29 * n) / 6;
    size_t additions = mm * n * n + (n * n * n) / 3 + 2 * mm * n - (n * n) / 2 + (5 * n) / 6;
    increment_flops(Multipliers<T>::mul * multiplications + Multipliers<T>::add * additions);
  }
  int info = proxy_lapack::geqrf(a->rows, a->cols, a->ptr(), a->rows, tau);
  HMAT_ASSERT(!info);

  // Copy the 'r' factor in the upper part of resultR
  for (int col = 0; col < a->cols; col++) {
    for (int row = 0; row <= col; row++) {
      resultR->get(row, col) = a->get(row, col);
    }
  }

  // Copy tau in the last column of 'this'
  memcpy(a->ptr(0, a->cols-1), tau, sizeof(T)*std::min(a->rows, a->cols));
  free(tau);

  // temporary data created if initialPivot>0
  if (bK) delete(bK);
  if (restR) delete(restR);

  return;
}

template<typename T>
  void ScalarArray<T>::trmm(Side side, Uplo uplo,
    char transA, Diag diag, T alpha, const ScalarArray<T> * a) {
  DECLARE_CONTEXT;
  {
    size_t multiplications = (size_t)rows * cols * (cols + 1) / 2;
    size_t additions = (size_t)rows * cols  * (cols - 1) / 2;
    increment_flops(Multipliers<T>::mul * multiplications + Multipliers<T>::add * additions);
  }
  proxy_cblas::trmm(to_blas(side), to_blas(uplo), transA, to_blas(diag),
    rows, cols, alpha, a->m, a->lda, this->m, this->lda);
}

template<typename T>
  void ScalarArray<T>::trsm(Side side, Uplo uplo,
    char transA, Diag diag, T alpha, const ScalarArray<T> * a) {
  if (rows == 0 || cols == 0) return;
  DECLARE_CONTEXT;
  {
    const size_t adds = cols * rows * (rows - 1) / 2;
    const size_t muls = cols * rows * (rows + 1) / 2;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  proxy_cblas::trsm(to_blas(side), to_blas(uplo), transA, to_blas(diag),
    rows, cols, alpha, a->m, a->lda, this->m, this->lda);
}

template<typename T>
int ScalarArray<T>::productQ(char side, char trans, ScalarArray<T>* c) const {
  DECLARE_CONTEXT;
  Timeline::Task t(Timeline::PRODUCTQ, &cols, &c->rows, &c->cols);
  (void)t;
  assert((side == 'L') ? rows == c->rows : rows == c->cols);
  {
    size_t _m = c->rows, _n = c->cols, _k = cols;
    size_t muls = 2 * _m * _n * _k - _n * _k * _k + 2 * _n * _k;
    size_t adds = 2 * _m * _n * _k - _n * _k * _k + _n * _k;
    increment_flops(Multipliers<T>::mul * muls + Multipliers<T>::add * adds);
  }

  // In qrDecomposition(), tau is stored in the last column of 'this'
  // it is not valid to work with 'tau' inside the array 'a' because zunmqr modifies 'a'
  // during computation. So we work on a copy of tau.
  std::vector<T> tau(std::min(rows, cols));
  memcpy(tau.data(), const_ptr(0, cols-1), sizeof(T)*std::min(rows, cols));

  // We don't use c->ptr() on purpose, because c->is_ortho is preserved here (Q is orthogonal)
  int info = proxy_lapack_convenience::or_un_mqr(side, trans, c->rows, c->cols, cols, const_ptr(), lda, tau.data(), c->m, c->lda);
  HMAT_ASSERT(!info);
  return 0;
}

template <typename T>
void ScalarArray<T>::reflect(Vector<T> &v_house, double beta, char transA)
{
    assert(abs(beta) >= abs(std::numeric_limits<T>::epsilon()));
    ScalarArray<T> w_house(1,cols);
    w_house.gemm(transA , 'N' , beta ,  &v_house ,this, 0);
    rankOneUpdateT(1,v_house, w_house);
}

template <typename T> void ScalarArray<T>::cpqrDecomposition(int * &sigma, double * &tau ,int *rank, double epsilon)
{ 
  double normSqr=0;
  int iter=0;
  int min_dim=std::min(cols , rows);
  sigma=(int*)malloc(min_dim*sizeof(int));
  tau=(double*)malloc(min_dim*sizeof(double));
  //sigma will be modifief each iteration to keep track of the columns transpositions
  for (int i =0 ; i < cols ; i++)
  {
    sigma[i]=i;
  }
  char transA;
  if(std::is_same<Z_t, T>::value || std::is_same<C_t, T>::value) transA='C';
  else transA='T';
  std::vector<double> normSqrCol(cols);
  int pivot=0;
  double max_norm=0;
  for (int i = 0 ; i<cols ; i++)
  {
    Vector<T> col_i(*this , i);
    normSqrCol[i]=col_i.normSqr();
    if (max_norm < normSqrCol[i])
    {
      max_norm=normSqrCol[i];
      pivot=i;
    }
    normSqr+=normSqrCol[i];
  }
  double norm_init=sqrt(normSqr);
  while (sqrt(normSqr) > epsilon*norm_init && iter < min_dim)
  {
    T x1=get(iter , pivot); 
    //Swap the columns, the coeficients of normSqrCol and update sigma
    T *tmp=(T*)malloc(sizeof(T)*rows);
    memcpy(tmp, &get(0,iter), sizeof(T)*rows);
    memcpy (&get(0,iter), &get(0,pivot), sizeof(T)*rows);
    memcpy (&get(0,pivot), tmp, sizeof(T)*rows);
    free(tmp);
    double tmp_d=normSqrCol[iter];
    normSqrCol[iter]=normSqrCol[pivot];
    normSqrCol[pivot]=tmp_d;
    int tmpInt=sigma[iter];
    sigma[iter]=sigma[pivot];
    sigma[pivot]=tmpInt;

    //Construction of Householder vector

    ScalarArray<T> remainder(*this , iter , rows-iter , iter , cols-iter);
    Vector<T> v_house(rows-iter);
    T mu=sqrt(normSqrCol[iter]);
    T alpha=std::abs(x1)!=0 ? x1+(x1/std::abs(x1))*mu : mu;
    v_house[0]=std::abs(x1)!=0 ? 1 : 0;
    for (int i = 1 ; i<rows-iter ; i++)
    {
      v_house[i]=remainder.get(i,0)/alpha;
    }

    //Householder update (only changes the bottom right part of m[iter : , iter :]=remainder)
    double beta=(-2/v_house.normSqr());
    tau[iter]=beta;//needed to later re-construct Q from this
    ScalarArray <T> w_house(1,cols-iter);
    w_house.gemm(transA , 'N' , beta ,&v_house , &remainder ,  0);
    remainder.rankOneUpdateT(1 , v_house , w_house);

    //norm udpate and determination of the next pivot

    max_norm=0;
    for(int i = 1 ; i < cols-iter ; i++)
    {
      normSqrCol[i+iter]-=std::pow(std::abs(remainder.get(0 , i)),2);
      normSqr-=std::pow(std::abs(remainder.get(0 , i)),2);
      if(normSqrCol[i]>max_norm)
      {
        max_norm=normSqrCol[i+iter];
        pivot=i+iter;
      }
    }
    normSqr-=std::pow(std::abs(remainder.get(0,0)),2);
    memcpy(&m[(iter)*rows+iter+1], &v_house.m[1] , (rows-iter-1)*sizeof(T));//storing v_house in the low part of this
    iter++;
  }
  *rank=iter;
  tau=(double*)realloc(tau, sizeof(double)*iter);
}
template<typename T> int ScalarArray<T>::modifiedGramSchmidt(ScalarArray<T> *result, double prec, int initialPivot ) {
  DECLARE_CONTEXT;
  Timeline::Task t(Timeline::MGS, &rows, &cols, &initialPivot);
  (void)t;

  if (!envSA.initPivot) initialPivot=0;
  assert(initialPivot>=0 && initialPivot<=cols);


  {
    size_t mm = rows;
    size_t n = cols;
    size_t multiplications = mm*n*n;
    size_t additions = mm*n*n;
    increment_flops(Multipliers<T>::mul * multiplications + Multipliers<T>::add * additions);
  }
  int rank=0;
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

  // Apply Modified Gram-Schmidt process with column pivoting to the 'initialPivot' first columns,
  // where the j-th pivot is column j
  // No stopping criterion here. Since these columns come from another recompression, we assume
  // they are all kept.
  // With this approach, we can use blas3 (more computationnaly efficient) but we do a classical
  // (not a "modified") Gram Schmidt algorithm, which is known to be numerically unstable
  // So expect to lose on final accuracy of hmat

  // Initial pivot
  if (initialPivot) {
    // modify the columns [initialPivot, cols[ to make them orthogonals to columns [0, initialPivot[
    orthoColumns(&r, initialPivot);
    rank = initialPivot;
  }

  // Init.
  Vector<double> norm2(cols); // Norm^2 of the columns of 'this' during computation
  Vector<double> norm2_orig(cols); // Norm^2 of the original columns of 'this'
  Vector<double> norm2_update(cols); // Norm^2 of the columns of 'this' normalized (will be 1. at the beginning, then decrease)
  // The actual norm^2 of column j during computation (before it is finally normalized) will be norm2[j] = norm2_orig[j] * norm2_update[j]
  // The choice of the pivot will be based on norm2_update[] or norm2[]
  relative_epsilon = 0.0;
  for(int j=rank; j < cols; ++j) {
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
  for(int j = rank; j < cols; ++j) {
    // Find the largest pivot
    int pivot = norm2.absoluteMaxIndex(j);
    double pivmax = norm2[pivot];

    if (envSA.mgsAltPivot) {
      pivot = norm2_update.absoluteMaxIndex(j);
      pivmax = norm2_update[pivot];
      relative_epsilon = prec * prec;
    }
    if (j<initialPivot) {
      pivot = j;
      pivmax = 1.;
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
      T coef = T(1) / r.get(j,j);
      aj.scale(coef);

      // Remove the qj-component from vectors bk (k=j+1,...,n-1)
      if (j<cols-1) {
        int firstcol=std::max(j+1, initialPivot) ;
        ScalarArray<T> bK(*this, 0, rows, firstcol, cols-firstcol); // All the columns of 'this' after column 'firstcol'
        ScalarArray<T> aj_bK(r, j, 1, firstcol, cols-firstcol); // In 'r': row 'j', all the columns after column 'firstcol'
        // Compute in 1 operation all the scalar products between aj and a_firstcol, ..., a_n
        aj_bK.gemm('C', 'N', 1, &aj, &bK, 0);
        // Update a_firstcol, ..., a_n
        bK.rankOneUpdateT(-1, aj, aj_bK);
        // Update the norms
        for(int k = firstcol; k < cols; ++k) {
          double rjk = std::abs(r.get(j,k));
          norm2[k] -= rjk * rjk;
          norm2_update[k] -= rjk * rjk / norm2_orig[k];
        }
      }
    } else
      break;
  } // for(int j = rank;

  // Update matrix dimensions
  cols = rank;
  result->rows = rank;

  // Apply perm to result
  for(int j = 0; j < result->cols; ++j) {
    // Copy the column j of r into the column perm[j] of result (only 'rank' rows)
    memcpy(result->ptr(0, perm[j]), r.const_ptr(0, j), result->rows*sizeof(T));
  }
  setOrtho(1);
  // Clean up
  delete[] perm;
  /* end of modified Gram-Schmidt */
  return rank;
}

template<typename T>
void ScalarArray<T>::multiplyWithDiagOrDiagInv(const ScalarArray<T>* d, bool inverse, Side side) {
  assert(d);
  assert(side == Side::LEFT  || (cols == d->rows));
  assert(side == Side::RIGHT || (rows == d->rows));
  assert(d->cols==1);

  {
    const size_t _rows = rows, _cols = cols;
    increment_flops(Multipliers<T>::mul * _rows * _cols);
  }
  if (side == Side::LEFT) { // line i is multiplied by d[i] or 1/d[i]
    // TODO: Test with scale to see if it is better.
    if (inverse) {
      ScalarArray<T> *d2 = new ScalarArray<T>(rows,1);
      for (int i = 0; i < rows; i++)
        d2->get(i) = T(1) / d->get(i);
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
      T diag_val = inverse ? T(1) / d->get(j,0) : d->get(j);
      proxy_cblas::scal(rows, diag_val, ptr(0,j), 1);
    }
  }
}

template<typename T>
void ScalarArray<T>::multiplyWithDiag(const ScalarArray<typename hmat::Types<T>::real>* d) {
  assert(d);
  assert(cols <= d->rows); // d can be larger than needed
  assert(d->cols==1);

  {
    const size_t _rows = rows, _cols = cols;
    increment_flops(Multipliers<T>::mul * _rows * _cols);
  }
  for (int j = 0; j < cols; j++) {
    // We don't use ptr() on purpose, because is_ortho is preserved here
    proxy_cblas::scal(rows, d->get(j), m+j*lda, 1);
  }
}

template<typename T> void ScalarArray<T>::addIdentity(T alpha) {
  int n = std::min(this->rows, this->cols);
  for(int i = 0; i < n; i++) {
    get(i, i) += alpha;
  }
}

template<typename T> typename Types<T>::dp ScalarArray<T>::diagonalProduct() const {
  assert(rows == cols);
  typename Types<T>::dp r = get(0,0);
  for(int i = 1; i < rows; i++) {
    r *= get(i,i);
  }
  return r;
}

template<typename T>
bool ScalarArray<T>::testOrtho() const {
  // code % 2 == 0 means we are in simple precision (real or complex)
  static double machine_accuracy = Constants<T>::code % 2 == 0 ? 1.19e-7 : 1.11e-16 ;
  static double test_accuracy = Constants<T>::code % 2 == 0 ? 1.e-3 : 1.e-7 ;
  static double ratioMax=0.;
  double ref = norm();
  if (ref==0.) return true;
  ScalarArray<T> *sp = new ScalarArray<T>(cols, cols);
  // Compute the scalar product sp = X^H.X
  sp->gemm('C', 'N', 1, this, this, 0);
  // Nullify the diagonal elements
  for (int i=0 ; i<cols ; i++)
    sp->get(i,i) = 0;
  // The norm of the rest should be below 'epsilon x norm of this' to have orthogonality and return true
  double res = sp->norm();
  delete sp;
  if (envSA.testOrtho) {
    double ratio = res/ref/machine_accuracy/sqrt((double)rows);
    if (ratio > ratioMax) {
      ratioMax = ratio;
      printf("testOrtho[%dx%d] test=%d get=%d        res=%g ref=%g res/ref=%g ratio=%g ratioMax=%g\n",
             rows, cols, (res < ref * test_accuracy), getOrtho(), res, ref, res/ref, ratio, ratioMax);
    }
  }
  return (res < ref * test_accuracy);
}

template<typename T>
T Vector<T>::dot(const Vector<T>* x, const Vector<T>* y) {
  assert(x->cols == 1);
  assert(y->cols == 1);
  assert(x->rows == y->rows);
  // TODO: Beware of large vectors (>2 billion elements) !
  return proxy_cblas_convenience::dot_c(x->rows, x->const_ptr(), 1, y->const_ptr(), 1);
}

template <typename T>
double Vector<T>::maxAbsolute() const
{
  double res = std::abs(this->get(0));

  for (int i = 1; i < this->rows; i++)
  {
    double val = std::abs(this->get(i));
    if(val > res)
    {
      res = val;
    }
  }
    return res;
}

template <typename T>
int Vector<T>::absoluteMaxIndex(int startIndex) const
{
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
