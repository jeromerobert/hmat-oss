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
FullMatrix<T>::FullMatrix(const IndexSet*  _rows, const IndexSet*  _cols)
  : data(_rows->size(), _cols->size()), triUpper_(false), triLower_(false),
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

template<typename T> size_t FullMatrix<T>::storedZeros() {
  return data.storedZeros();
}

template<typename T> size_t FullMatrix<T>::info(hmat_info_t & result, size_t& rowsMin, size_t& colsMin, size_t& rowsMax, size_t& colsMax) {
  size_t colsCount = 0, rowsCount = 0;
  colsMax = 0;
  colsMin = cols();
  rowsMax = 0;
  rowsMin = rows();
  bool* notNullRows = (bool*) calloc(rows(), sizeof(bool));
  bool* notNullCols = (bool*) calloc(cols(), sizeof(bool));
  for (int col = 0; col < cols(); col++) {
    for (int row = 0; row < rows(); row++) {
      if (std::abs(get(row, col)) > 1e-16) {
        rowsMin = (rowsMin < row) ? rowsMin : row;
        rowsMax = (rowsMax > row) ? rowsMax : row;
        colsMin = (colsMin < col) ? colsMin : col;
        colsMax = (colsMax > col) ? colsMax : col;
        if (!notNullRows[row]) rowsCount++;
        if (!notNullCols[col]) colsCount++;
        notNullRows[row] = true;
        notNullCols[col] = true;
      }
    }
  }
  return colsCount * rowsCount;
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
    result = new FullMatrix<T>(rows_, cols_);

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
  ScalarArray<T> sub(data.m + rowsOffset + colsOffset * data.lda,
                     subRows->size(), subCols->size(), data.lda);
  return new FullMatrix<T>(&sub, subRows, subCols);
}


template<typename T>
void FullMatrix<T>::gemm(char transA, char transB, T alpha,
                         const FullMatrix<T>* a, const FullMatrix<T>* b,
                         T beta) {
  data.gemm(transA, transB, alpha, &a->data, &b->data, beta);
}

template<typename T>
void FullMatrix<T>::multiplyWithDiagOrDiagInv(const Vector<T>* d, bool inverse, bool left) {
  assert(d);
  assert(left || (cols() == d->rows));
  assert(!left || (rows() == d->rows));

  T* diag = d->m;
  {
    const size_t _rows = rows(), _cols = cols();
    increment_flops(Multipliers<T>::mul * _rows * _cols);
  }
  if (left) {
    if (inverse) {
      // In this case, copying is a good idea since it avoids repeated
      // computations of 1 / diag[i].
      diag = (T*) malloc(d->rows * sizeof(T));
      HMAT_ASSERT(diag);
      memcpy(diag, d->m, d->rows * sizeof(T));
      for (int i = 0; i < d->rows; i++) {
        diag[i] = Constants<T>::pone / diag[i];
      }
    }
    // TODO: Test with scale to see if it is better.
    for (int j = 0; j < cols(); j++) {
      for (int i = 0; i < rows(); i++) {
        get(i, j) *= diag[i];
      }
    }
    if (inverse) {
      free(diag);
    }
  } else {
    for (int j = 0; j < cols(); j++) {
      T diag_val = inverse ? Constants<T>::pone / diag[j] : diag[j];
      proxy_cblas::scal(rows(), diag_val, data.m + j * ((size_t) data.lda), 1);
    }
  }
}

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
      HMAT_ASSERT_MSG(v[j] != Constants<T>::zero, "Division by 0 in LDLt");
      get(k,j) /= v[j];
    }
  }

  for(int i = 0; i < n; i++) {
    diagonal->m[i] = get(i,i);
    get(i,i) = Constants<T>::pone;
    for (int j = i + 1; j < n; j++)
      get(i,j) = Constants<T>::zero;
  }

  triLower_ = true;
  assert(!isTriUpper());
  delete[] v;
}

template<typename T> void assertPositive(T v) {
    HMAT_ASSERT_MSG(v != Constants<T>::zero, "Non positive diagonal value in LLt");
}

template<> void assertPositive(S_t v) {
    HMAT_ASSERT_MSG(v > 0, "Non positive diagonal value in LLt");
}

template<> void assertPositive(D_t v) {
    HMAT_ASSERT_MSG(v > 0, "Non positive diagonal value in LLt");
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
    assertPositive(get(j, j));

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
  int info;
  {
    const size_t _m = rows(), _n = cols();
    const size_t muls = _m * _n *_n / 2 - _n *_n*_n / 6 + _m * _n / 2 - _n*_n / 2 + 2 * _n / 3;
    const size_t adds = _m * _n *_n / 2 - _n *_n*_n / 6 + _m * _n / 2 + _n / 6;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  info = proxy_lapack::getrf(rows(), cols(), data.m, data.lda, pivots);
  HMAT_ASSERT(!info);
}

// The following code is very close to that of ZGETRS in LAPACK.
// However, the resolution here is divided in the two parties.

// Warning! The matrix has been obtained with ZGETRF therefore it is
// permuted! We have the factorization A = P L U  with P the
// permutation matrix. So to solve L X = B, we must
// solve LX = (P ^ -1 B), which is done by ZLASWP with
// the permutation. we used it just like in ZGETRS.
template<typename T>
void FullMatrix<T>::solveLowerTriangularLeft(ScalarArray<T>* x, bool unitriangular) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;

  {
    const size_t _m = rows(), _n = x->cols;
    const size_t adds = _n * _m * (_m - 1) / 2;
    const size_t muls = _n * _m * (_m + 1) / 2;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  if (pivots)
    proxy_lapack::laswp(x->cols, x->m, x->lda, 1, rows(), pivots, 1);
  proxy_cblas::trsm('L', 'L', 'N', unitriangular ? 'U' : 'N', rows(), x->cols, Constants<T>::pone, data.m, data.lda, x->m, x->lda);
}


// The resolution of the upper triangular system does not need to
//  change the order of columns.
//  The pivots are not necessary here, but this helps to check
//  the matrix was factorized before.

template<typename T>
void FullMatrix<T>::solveUpperTriangularRight(ScalarArray<T>* x, bool unitriangular, bool lowerStored) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;

  {
    const size_t _m = rows(), _n = x->cols;
    const size_t adds = _n * _m * (_m - 1) / 2;
    const size_t muls = _n * _m * (_m + 1) / 2;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  proxy_cblas::trsm('R', lowerStored ? 'L' : 'U', lowerStored ? 'T' : 'N', unitriangular ? 'U' : 'N',
    x->rows, x->cols, Constants<T>::pone, data.m, data.lda, x->m, x->lda);
}

template<typename T>
void FullMatrix<T>::solveUpperTriangularLeft(ScalarArray<T>* x, bool unitriangular, bool lowerStored) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;

  {
    const size_t _m = rows(), _n = x->cols;
    const size_t adds = _n * _m * (_n - 1) / 2;
    const size_t muls = _n * _m * (_n + 1) / 2;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  proxy_cblas::trsm('L', lowerStored ? 'L' : 'U', lowerStored ? 'T' : 'N', unitriangular ? 'U' : 'N',
    x->rows, x->cols, Constants<T>::pone, data.m, data.lda, x->m, x->lda);
}

template<typename T>
void FullMatrix<T>::solve(ScalarArray<T>* x) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;

  assert(pivots);
  int ierr = 0;
  {
    const size_t nrhs = x->cols;
    const size_t n = rows();
    const size_t adds = n * n * nrhs;
    const size_t muls = (n * n - n) * nrhs;
    increment_flops(Multipliers<T>::add * adds + Multipliers<T>::mul * muls);
  }
  ierr = proxy_lapack::getrs('N', rows(), x->cols, data.m, data.lda, pivots, x->m, x->rows);
  HMAT_ASSERT(!ierr);
}


template<typename T>
void FullMatrix<T>::inverse() {

  // The inversion is done in two steps with dgetrf for LU decomposition and
  // dgetri for inversion of triangular matrices

  assert(rows() == cols());

  int *ipiv = new int[rows()];
  int info;
  {
    size_t vn = cols(), vm = cols();
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
  info = proxy_lapack::getrf(rows(), cols(), data.m, data.lda, ipiv);
  HMAT_ASSERT(!info);
  // We call it twice: the first time to know the optimal size of
  // temporary arrays, and the second time for real calculation.
  int workSize;
  T workSize_req;
  info = proxy_lapack::getri(rows(), data.m, data.lda, ipiv, &workSize_req, -1);
  workSize = (int) hmat::real(workSize_req) + 1;
  T* work = new T[workSize];
  HMAT_ASSERT(work);
  info = proxy_lapack::getri(rows(), data.m, data.lda, ipiv, work, workSize);
  delete[] work;
  HMAT_ASSERT(!info);
  delete[] ipiv;
}


template<typename T>
void FullMatrix<T>::copyMatrixAtOffset(const FullMatrix<T>* a,
                                       int rowOffset, int colOffset) {
  data.copyMatrixAtOffset(&a->data, rowOffset, colOffset);
}

template<typename T>
void FullMatrix<T>::copyMatrixAtOffset(const FullMatrix<T>* a,
                                       int rowOffset, int colOffset,
                                       int rowsToCopy, int colsToCopy) {
  data.copyMatrixAtOffset(&a->data, rowOffset, colOffset, rowsToCopy, colsToCopy);
}

template<typename T>
void FullMatrix<T>::axpy(T alpha, const FullMatrix<T>* a) {
  data.axpy(alpha, &a->data);
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
  FILE * f = fopen(filename, "rb");
  int code;
  int r = fread(&code, sizeof(int), 1, f);
  HMAT_ASSERT(r == 1);
  HMAT_ASSERT(code == Constants<T>::code);
  r = fread(&data.rows, sizeof(int), 1, f);
  data.lda = data.rows;
  HMAT_ASSERT(r == 1);
  r = fread(&data.cols, sizeof(int), 1, f);
  HMAT_ASSERT(r == 1);
  r = fseek(f, 2 * sizeof(int), SEEK_CUR);
  HMAT_ASSERT(r == 0);
  if(data.m)
      free(data.m);
  size_t size = ((size_t) data.rows) * data.cols * sizeof(T);
  data.m = (T*) calloc(size, 1);
  r = fread(data.m, size, 1, f);
  fclose(f);
  HMAT_ASSERT(r == 1);
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
