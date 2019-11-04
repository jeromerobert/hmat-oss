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
#include <complex>

#include <sys/stat.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include <stdlib.h>

namespace hmat {

/** FullMatrix */

  template<typename T>
FullMatrix<T>::FullMatrix(T* _m, const IndexSet*  _rows, const IndexSet*  _cols, int _lda)
  : data(_m, _rows->size(), _cols->size(), _lda), triUpper_(false), triLower_(false),
    rows_(_rows), cols_(_cols), pivots(NULL), diagonal(NULL) {
  assert(rows_);
  assert(cols_);
}

template<typename T>
FullMatrix<T>::FullMatrix(ScalarArray<T> *s, const IndexSet*  _rows, const IndexSet*  _cols)
  : data(*s), triUpper_(false), triLower_(false),
    rows_(_rows), cols_(_cols), pivots(NULL), diagonal(NULL) {
  assert(rows_);
  assert(cols_);
  // check the coherency between IndexSet and ScalarArray
  assert(rows_->size() == s->rows);
  assert(cols_->size() == s->cols);
}

template<typename T>
FullMatrix<T>::FullMatrix(const IndexSet*  _rows, const IndexSet*  _cols, bool zeroinit)
  : data(_rows->size(), _cols->size(), zeroinit), triUpper_(false), triLower_(false),
    rows_(_rows), cols_(_cols), pivots(NULL), diagonal(NULL) {
  assert(rows_);
  assert(cols_);
}

template<typename T> FullMatrix<T>::~FullMatrix() {
  if (pivots) {
    free(pivots);
  }
  if (diagonal) {
    delete diagonal;
  }
}

template<typename T> void FullMatrix<T>::clear() {
  data.clear();
  if (diagonal)
    diagonal->clear();
}

template<typename T> size_t FullMatrix<T>::storedZeros() const {
  return data.storedZeros();
}

template<typename T> void FullMatrix<T>::scale(T alpha) {
  data.scale(alpha);
  if (diagonal)
    diagonal->scale(alpha);
}

template<typename T> void FullMatrix<T>::transpose() {
  data.transpose();
  std::swap(rows_, cols_);
  // std::swap(triUpper_, triLower_) won't work because you can't swap bitfields
  if (triUpper_) {
    triUpper_ = false;
    triLower_ = true;
  } else if (triLower_) {
    triLower_ = false;
    triUpper_ = true;
  }
}

template<typename T> FullMatrix<T>* FullMatrix<T>::copy(FullMatrix<T>* result) const {
  if(result == NULL)
    result = new FullMatrix<T>(rows_, cols_, false);

  data.copy(&result->data);
  if (diagonal) {
    if(!result->diagonal)
      result->diagonal = new Vector<T>(rows());
    diagonal->copy(result->diagonal);
  }

  result->rows_ = rows_;
  result->cols_ = cols_;
  result->triLower_ = triLower_;
  result->triUpper_ = triUpper_;
  return result;
}

template<typename T> FullMatrix<T>* FullMatrix<T>::copyAndTranspose() const {
  assert(cols_);
  assert(rows_);
  FullMatrix<T>* result = new FullMatrix<T>(cols_, rows_);
  data.copyAndTranspose(&result->data);
  return result;
}

template<typename T> const FullMatrix<T>* FullMatrix<T>::subset(const IndexSet* subRows,
                                                                const IndexSet* subCols) const {
  assert(subRows->isSubset(*rows_));
  assert(subCols->isSubset(*cols_));
  // The offset in the matrix, and not in all the indices
  int rowsOffset = subRows->offset() - rows_->offset();
  int colsOffset = subCols->offset() - cols_->offset();
  ScalarArray<T> sub(data, rowsOffset, subRows->size(), colsOffset, subCols->size());
  return new FullMatrix<T>(&sub, subRows, subCols);
}


template<typename T>
void FullMatrix<T>::gemm(char transA, char transB, T alpha,
                         const FullMatrix<T>* a, const FullMatrix<T>* b,
                         T beta) {
  //assert(b->storedZeros() < b->rows() * b->cols());
  //assert(a->storedZeros() < a->rows() * a->cols());
  data.gemm(transA, transB, alpha, &a->data, &b->data, beta);
  //assert(storedZeros() < rows() * cols());
}

template<typename T>
void FullMatrix<T>::multiplyWithDiagOrDiagInv(const Vector<T>* d, bool inverse, bool left) {
  data.multiplyWithDiagOrDiagInv(d, inverse, left);
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

    virtual const char* what() const throw() {
        return invalidDiagonalMessage_.c_str();
    }

    virtual ~InvalidDiagonalException() throw() {}
};

template<typename T>
void FullMatrix<T>::ldltDecomposition() {
  // Void matrix
  if (rows() == 0 || cols() == 0) return;

  int n = this->rows();
  diagonal = new Vector<T>(n);
  HMAT_ASSERT(diagonal);
  assert(this->rows() == this->cols()); // We expect a square matrix
  //TODO : add flops counter

  // Standard LDLt factorization algorithm is:
  //  diag[j] = A(j,j) - sum_{k < j} L(j,k)^2 diag[k]
  //  L(i,j) = (A(i,j) - sum_{k < j} (L(i,k)L(j,k)diag[k])) / diag[j]
  // See for instance http://en.wikipedia.org/wiki/Cholesky_decomposition
  // An auxiliary array is introduced in order to perform less multiplications,
  // see  algorithm 1 in http://icl.cs.utk.edu/projectsfiles/plasma/pubs/ICL-UT-11-03.pdf
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
      if (v[j] == Constants<T>::zero)
        throw InvalidDiagonalException<T>(v[j], j, "ldltDecomposition");
      get(k,j) /= v[j];
    }
  }

  for(int i = 0; i < n; i++) {
    (*diagonal)[i] = get(i,i);
    get(i,i) = Constants<T>::pone;
    for (int j = i + 1; j < n; j++)
      get(i,j) = Constants<T>::zero;
  }

  triLower_ = true;
  assert(!isTriUpper());
  delete[] v;
}

template<typename T> void assertPositive(const T v, const int j, const char * const where) {
    if(v == Constants<T>::zero)
      throw InvalidDiagonalException<T>(v, j, where);
}

template<> void assertPositive(const S_t v, const int j, const char * const where) {
    if(!(v > 0))
      throw InvalidDiagonalException<S_t>(v, j, where);
}

template<> void assertPositive(D_t v, int j, const char * const where) {
    if(!(v > 0))
      throw InvalidDiagonalException<D_t>(v, j, where);
}

template<typename T> void FullMatrix<T>::lltDecomposition() {
    // Void matrix
    if (rows() == 0 || cols() == 0) return;

  int n = this->rows();
  assert(this->rows() == this->cols()); // We expect a square matrix

  // Standard LLt factorization algorithm is:
  //  L(j,j) = sqrt( A(j,j) - sum_{k < j} L(j,k)^2)
  //  L(i,j) = ( A(i,j) - sum_{k < j} L(i,k)L(j,k) ) / L(j,j)

    // from http://www.netlib.org/lapack/lawnspdf/lawn41.pdf page 120
  const size_t n2 = (size_t) n*n;
  const size_t n3 = n2 * n;
  const size_t muls = n3 / 6 + n2 / 2 + n / 3;
  const size_t adds = n3 / 6 - n / 6;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);

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

  // For real matrices, we could use the lapack version, could be faster
  // (There is no L.Lt factorisation for complex matrices in lapack)
  //    int info = proxy_lapack::potrf('L', this->rows(), this->data.m, this->data.lda);
  //    if(info != 0)
  //        // throw a pointer to be compliant with the Status class
  //        throw hmat::LapackException("potrf", info);

  for (int j = 0; j < n; j++) {
        for(int i = 0; i < j; i++) {
            get(i,j) = Constants<T>::zero;
        }
    }

  triLower_ = true;
  assert(!isTriUpper());
}

template<typename T>
void FullMatrix<T>::luDecomposition() {
  // Void matrix
  if (rows() == 0 || cols() == 0) return;

  pivots = (int*) calloc(rows(), sizeof(int));
  HMAT_ASSERT(pivots);
  data.luDecomposition(pivots);
}

template<typename T>
void FullMatrix<T>::solveLowerTriangularLeft(ScalarArray<T>* x, bool unitriangular) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;
  data.solveLowerTriangularLeft(x, pivots, unitriangular);
}


// The resolution of the upper triangular system does not need to
//  change the order of columns.
//  The pivots are not necessary here, but this helps to check
//  the matrix was factorized before.

template<typename T>
void FullMatrix<T>::solveUpperTriangularRight(ScalarArray<T>* x, bool unitriangular, bool lowerStored) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;
  data.solveUpperTriangularRight(x, unitriangular, lowerStored);
}

template<typename T>
void FullMatrix<T>::solveUpperTriangularLeft(ScalarArray<T>* x, bool unitriangular, bool lowerStored) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;
  data.solveUpperTriangularLeft(x, unitriangular, lowerStored);
}

template<typename T>
void FullMatrix<T>::solve(ScalarArray<T>* x) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;
  assert(pivots);
  data.solve(x, pivots);
}


template<typename T>
void FullMatrix<T>::inverse() {
  assert(rows() == cols());
  data.inverse();
}


template<typename T>
void FullMatrix<T>::copyMatrixAtOffset(const FullMatrix<T>* a,
                                       int rowOffset, int colOffset) {
  data.copyMatrixAtOffset(&a->data, rowOffset, colOffset);
}

template<typename T>
void FullMatrix<T>::axpy(T alpha, const FullMatrix<T>* a) {
  assert(a->storedZeros() < a->rows() * a->cols());
  size_t zb = storedZeros();
  (void)zb;
  data.axpy(alpha, &a->data);
  // assert(storedZeros() < rows() * cols());
}

template<typename T>
double FullMatrix<T>::normSqr() const {
  return data.normSqr();
}

template<typename T> double FullMatrix<T>::norm() const {
  return data.norm();
}

template<typename T> void FullMatrix<T>::addRand(double epsilon) {
  DECLARE_CONTEXT;
  data.addRand(epsilon);
}

template<typename T> void FullMatrix<T>::fromFile(const char * filename) {
  data.fromFile(filename);
}

template<typename T> void FullMatrix<T>::toFile(const char *filename) const {
  data.toFile(filename);
}

template<typename T> size_t FullMatrix<T>::memorySize() const {
   return data.memorySize();
}

template<typename T> void FullMatrix<T>::checkNan() const {
  data.checkNan();
  if (diagonal)
    diagonal->checkNan();
}

template<typename T> bool FullMatrix<T>::isZero() const {
  bool res = data.isZero();
  if (diagonal)
    res = res & diagonal->isZero();
  return res;
}

template<typename T> void FullMatrix<T>::conjugate() {
  data.conjugate();
  if (diagonal)
    diagonal->conjugate();
}

template<typename T> std::string FullMatrix<T>::description() const {
    std::ostringstream convert;
    convert << "FullMatrix " << this->rows_->description() << "x" << this->cols_->description() ;
    convert << "norm=" << norm();
    return convert.str();
}

// the classes declaration
template class FullMatrix<S_t>;
template class FullMatrix<D_t>;
template class FullMatrix<C_t>;
template class FullMatrix<Z_t>;

}  // end namespace hmat
