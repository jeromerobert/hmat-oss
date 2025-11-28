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
#include <chrono>

namespace hmat {

template <typename T>
inline FPSimpleCompressor<T>::FPSimpleCompressor(hmat_FPcompress_t method)
{
  compressor = initCompressor<T>(method);
  compressionRatio = 1;
  compressionTime = 0;
  decompressionTime = 0;
}

template <typename T>
FPSimpleCompressor<T>::~FPSimpleCompressor()
{
  if(compressor)
    {
        delete compressor;
        compressor = nullptr;
    }
}

/** FullMatrix */

  template<typename T>
FullMatrix<T>::FullMatrix(T* _m, const IndexSet*  _rows, const IndexSet*  _cols, int _lda)
  : data(_m, _rows->size(), _cols->size(), _lda), triUpper_(false), triLower_(false),
    rows_(_rows), cols_(_cols), pivots(NULL), diagonal(NULL) {
  assert(rows_);
  assert(cols_);
  _compressor = nullptr;
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
  _compressor = nullptr;
}

template<typename T>
FullMatrix<T>::FullMatrix(const IndexSet*  _rows, const IndexSet*  _cols, bool zeroinit)
  : data(_rows->size(), _cols->size(), zeroinit), triUpper_(false), triLower_(false),
    rows_(_rows), cols_(_cols), pivots(NULL), diagonal(NULL) {
  assert(rows_);
  assert(cols_);
  _compressor = nullptr;
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
  //result->_compressor = (_compressor ? _compressor->copy() : NULL);

  if (diagonal) {
    if(!result->diagonal)
      result->diagonal = new Vector<T>(rows());
    diagonal->copy(result->diagonal);
  }

  result->rows_ = new IndexSet(rows_->offset(), rows_->size());
  result->cols_ = new IndexSet(cols_->offset(), cols_->size());
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
  data.gemm(transA, transB, alpha, &a->data, &b->data, beta);
}

template<typename T>
void FullMatrix<T>::multiplyWithDiagOrDiagInv(const Vector<T>* d, bool inverse, Side side) {
  data.multiplyWithDiagOrDiagInv(d, inverse, side);
}

template<typename T>
void FullMatrix<T>::ldltDecomposition() {
  // Void matrix
  if (rows() == 0 || cols() == 0) return;

  HMAT_ASSERT(rows() == cols());
  diagonal = new Vector<T>(rows());
  HMAT_ASSERT(diagonal);
  data.ldltDecomposition(*diagonal);

  triLower_ = true;
  assert(!isTriUpper());
}

template<typename T>
void FullMatrix<T>::lltDecomposition() {
  // Void matrix
  if (rows() == 0 || cols() == 0) return;

  data.lltDecomposition();

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
FactorizationData<T> FullMatrix<T>::getFactorizationData(Factorization algo) const {
  FactorizationData<T> result = { algo, {} };
  if (algo == Factorization::LU) {
    HMAT_ASSERT(pivots);
    result.data.pivots = pivots;
  } else if (algo == Factorization::LDLT) {
    HMAT_ASSERT(diagonal);
    result.data.diagonal = diagonal;
  }
  return result;
}

template<typename T>
void FullMatrix<T>::solveLowerTriangularLeft(ScalarArray<T>* x, Factorization algo, Diag diag, Uplo uplo) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;
  FactorizationData<T> context = getFactorizationData(algo);
  data.solveLowerTriangularLeft(x, context, diag, uplo);
}


// The resolution of the upper triangular system does not need to
//  change the order of columns.
//  The pivots are not necessary here, but this helps to check
//  the matrix was factorized before.

template<typename T>
void FullMatrix<T>::solveUpperTriangularRight(ScalarArray<T>* x, Factorization algo, Diag diag, Uplo uplo) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;
  FactorizationData<T> context = getFactorizationData(algo);
  data.solveUpperTriangularRight(x, context, diag, uplo);
}

template<typename T>
void FullMatrix<T>::solveUpperTriangularLeft(ScalarArray<T>* x, Factorization algo, Diag diag, Uplo uplo) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;
  FactorizationData<T> context = getFactorizationData(algo);
  data.solveUpperTriangularLeft(x, context, diag, uplo);
}

template<typename T>
void FullMatrix<T>::solve(ScalarArray<T>* x) const {
  // Void matrix
  if (x->rows == 0 || x->cols == 0) return;
  FactorizationData<T> context = getFactorizationData(Factorization::LU);
  data.solve(x, context);
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

template <typename T>
void FullMatrix<T>::FPcompress(double epsilon, hmat_FPcompress_t method)
{
  if(isFPcompressed()) //Already compressed !
  {
    return;
  }
  //printf("Begin Compression of Full block. Parameters : epsilon = %e , method = %d \n\n", epsilon, method);

  auto start = std::chrono::high_resolution_clock::now();

  _compressor = new FPSimpleCompressor<T>(method);

  _compressor->n_rows = data.rows;
  _compressor->n_cols = data.cols;

  std::vector<T> tmp(data.ptr(), data.ptr()+ _compressor->n_rows*_compressor->n_cols);

  _compressor->compressor->compress(tmp, data.rows * data.cols, epsilon);
  _compressor->compressionRatio = _compressor->compressor->get_ratio();

  data.resize(0);


  auto end = std::chrono::high_resolution_clock::now();
  _compressor->compressionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
}


template <typename T>
float FullMatrix<T>::FPdecompress()
{
  //printf("Begin Decompression of full block !\n");

  if(!isFPcompressed())
  {
     //printf("Compressors not instanciated\n");
    return 0;
  }
  if(!(_compressor->compressor))
  {
    return 0;
  }

  auto start = std::chrono::high_resolution_clock::now();

  data.resize(_compressor->n_cols);
  std::vector<T> tmp_out = _compressor->compressor->decompress();

  ScalarArray<T> tmp(tmp_out.data(), _compressor->n_rows, _compressor->n_cols);

  data.copyMatrixAtOffset(&tmp, 0, 0);

  
  auto end = std::chrono::high_resolution_clock::now();

  delete _compressor;
  _compressor = nullptr;

 return std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
}

template <typename T>
FullMatrix<T> *FullMatrix<T>::FPdecompressCopy(FullMatrix<T> *result) const
{
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

   if(!isFPcompressed())
  {
     //printf("Compressors not instanciated\n");
    return result;
  }
  if(!(_compressor->compressor))
  {
    return result;
  }

  auto start = std::chrono::high_resolution_clock::now();

  result->data.resize(_compressor->n_cols);
  
  std::vector<T> tmp_out = _compressor->compressor->decompressCopy();

  ScalarArray<T> tmp(tmp_out.data(), _compressor->n_rows, _compressor->n_cols);

  result->data.copyMatrixAtOffset(&tmp, 0, 0);

  
  auto end = std::chrono::high_resolution_clock::now();

  _compressor->decompressionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  


  return result;
}



template <typename T>
bool FullMatrix<T>::isFPcompressed() const {
    return _compressor != nullptr;
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
