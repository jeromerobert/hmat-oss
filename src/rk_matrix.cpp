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

#include "rk_matrix.hpp"
#include "h_matrix.hpp"
#include "cluster_tree.hpp"
#include <cstring> // memcpy
#include <cfloat> // DBL_MAX
#include "data_types.hpp"
#include "lapack_operations.hpp"
#include "blas_overloads.hpp"
#include "lapack_overloads.hpp"
#include "common/context.hpp"
#include "common/my_assert.h"

using namespace std;

namespace hmat {

/** RkApproximationControl */
template<typename T> RkApproximationControl RkMatrix<T>::approx;
int RkApproximationControl::findK(double *sigma, int maxK, double epsilon) {
  // Control of approximation for fixed approx.k != 0
  int newK = k;
  if (newK != 0) {
    newK = std::min(newK, maxK);
  } else {
    assert(epsilon >= 0.);
    double sumSingularValues = 0.;
    for (int i = 0; i < maxK; i++) {
      sumSingularValues += sigma[i];
    }
    int i = 0;
    for (i = 0; i < maxK; i++) {
      if (sigma[i] <= epsilon * sumSingularValues) {
        break;
      }
    }
    newK = i;
  }
  return newK;
}


/** RkMatrix */
template<typename T> RkMatrix<T>::RkMatrix(FullMatrix<T>* _a, const IndexSet* _rows,
                                           FullMatrix<T>* _b, const IndexSet* _cols,
                                           CompressionMethod _method)
  : rows(_rows),
    cols(_cols),
    a(_a),
    b(_b),
    method(_method)
{

  // We make a special case for empty matrices.
  if ((!a) && (!b)) {
    return;
  }
  assert(a->rows == rows->size());
  assert(b->rows == cols->size());
}

template<typename T> RkMatrix<T>::~RkMatrix() {
  clear();
}

template<typename T> FullMatrix<T>* RkMatrix<T>::eval() const {
  // Special case of the empty matrix, assimilated to the zero matrix.
  if (rank() == 0) {
    return FullMatrix<T>::Zero(rows->size(), cols->size());
  }
  FullMatrix<T>* result = new FullMatrix<T>(a->rows, b->rows);
  result->gemm('N', 'T', Constants<T>::pone, a, b, Constants<T>::zero);
  return result;
}

// Compute squared Frobenius norm
template<typename T> double RkMatrix<T>::normSqr() const {
  double result = 0;
  const int k = rank();
  for (int i = 1; i < k; ++i) {
    for (int j = 0; j < i; ++j) {
      result += hmat::real(proxy_cblas_convenience::dot_c(a->rows, a->m + i*a->lda, 1, a->m + j*a->lda, 1) *
                           proxy_cblas_convenience::dot_c(b->rows, b->m + i*b->lda, 1, b->m + j*b->lda, 1));
    }
  }
  result *= 2.0;
  for (int i = 0; i < k; ++i) {
    result += hmat::real(proxy_cblas_convenience::dot_c(a->rows, a->m + i*a->lda, 1, a->m + i*a->lda, 1) *
                         proxy_cblas_convenience::dot_c(b->rows, b->m + i*b->lda, 1, b->m + i*b->lda, 1));
  }
  return result;
}

template<typename T> void RkMatrix<T>::scale(T alpha) {
  // We need just to scale the first matrix, A.
  if (a) {
    a->scale(alpha);
  }
}

template<typename T> void RkMatrix<T>::clear() {
  delete a;
  delete b;
  a = NULL;
  b = NULL;
}

template<typename T>
void RkMatrix<T>::gemv(char trans, T alpha, const FullMatrix<T>* x, T beta, FullMatrix<T>* y) const {
  if (rank() == 0) {
    if (beta != Constants<T>::pone) {
      y->scale(beta);
    }
    return;
  }
  if (trans == 'N') {
    FullMatrix<T> z(b->cols, x->cols);
    z.gemm('T', 'N', Constants<T>::pone, b, x, Constants<T>::zero);
    y->gemm('N', 'N', alpha, a, &z, beta);
  } else {
    assert(trans == 'T');
    FullMatrix<T> z(a->cols, x->cols);
    z.gemm('T', 'N', Constants<T>::pone, a, x, Constants<T>::zero);
    y->gemm('N', 'N', alpha, b, &z, beta);
  }
}


template<typename T> const RkMatrix<T>* RkMatrix<T>::subset(const IndexSet* subRows,
                                                            const IndexSet* subCols) const {
  assert(subRows->isSubset(*rows));
  assert(subCols->isSubset(*cols));
  FullMatrix<T>* subA = NULL;
  FullMatrix<T>* subB = NULL;
  if(rank() > 0) {
    // The offset in the matrix, and not in all the indices
    int rowsOffset = subRows->offset() - rows->offset();
    int colsOffset = subCols->offset() - cols->offset();
    subA = new FullMatrix<T>(a->m + rowsOffset, subRows->size(), rank(), a->lda);
    subB = new FullMatrix<T>(b->m + colsOffset, subCols->size(), rank(), b->lda);
  }
  return new RkMatrix<T>(subA, subRows, subB, subCols, method);
}

template<typename T> size_t RkMatrix<T>::compressedSize() {
    return ((size_t)rows->size()) * rank() + ((size_t)cols->size()) * rank();
}

template<typename T> size_t RkMatrix<T>::uncompressedSize() {
    return ((size_t)rows->size()) * cols->size();
}

template<typename T> size_t RkMatrix<T>::storedZeros() {
  size_t result = 0;
  if (a)
    result += a->storedZeros() ;
  if (b)
    result += b->storedZeros() ;
  return result;
}

template<typename T> void RkMatrix<T>::truncate(double epsilon) {
  DECLARE_CONTEXT;

  if (rank() == 0) {
    assert(!(a || b));
    return;
  }

  assert(rows->size() >= rank());
  // Case: more columns than one dimension of the matrix.
  // In this case, the calculation of the SVD of the matrix "R_a R_b^t" is more
  // expensive than computing the full SVD matrix. We make then a full matrix conversion,
  // and compress it with RkMatrix::fromMatrix().
  // TODO: in this case, the epsilon of recompression is not respected
  if (rank() > std::min(rows->size(), cols->size())) {
    FullMatrix<T>* tmp = eval();
    RkMatrix<T>* rk = compressMatrix(tmp, rows, cols);
    delete tmp;
    // "Move" rk into this, and delete the old "this".
    swap(*rk);
    delete rk;
  }

  /* To recompress an Rk-matrix to Rk-matrix, we need :
      - A = Q_a R_A (QR decomposition)
      - B = Q_b R_b (QR decomposition)
      - Calculate the SVD of R_a R_b^t  = U S V^t
      - Make truncation U, S, and V in the same way as for
      compression of a full rank matrix, ie:
      - Restrict U to its newK first columns U_tilde
      - Restrict S to its newK first values (diagonal matrix) S_tilde
      - Restrict V to its newK first columns V_tilde
      - Put A = Q_a U_tilde SQRT (S_tilde)
      B = Q_b V_tilde SQRT(S_tilde)

     The sizes of the matrices are:
      - Qa : rows x k
      - Ra : k x k
      - Qb : cols x k
      - Rb : k x k
     So:
      - Ra Rb^t: k x k
      - U  : k * k
      - S  : k (diagonal)
      - V^t: k * k
     Hence:
      - newA: rows x newK
      - newB: cols x newK

  */
  int ierr;
  // QR decomposition of A and B
  T* tauA = qrDecomposition<T>(a); // A contient QaRa
  HMAT_ASSERT(tauA);
  T* tauB = qrDecomposition<T>(b); // B contient QbRb
  HMAT_ASSERT(tauB);

  // Matrices created by the SVD
  FullMatrix<T> *u = NULL, *vt = NULL;
  Vector<double> *sigma = NULL;
  {
    // The scope is to automatically delete rAFull.
    // For computing Ra*Rb^t, we would like to use directly a BLAS function
    // because Ra and Rb are triangular matrices.
    //
    // However, there is no multiplication of two triangular matrices in BLAS,
    // left matrix must be converted into full matrix first.
    FullMatrix<T> rAFull(rank(), rank());
    for (int col = 0; col < rank(); col++) {
      for (int row = 0; row <= col; row++) {
        rAFull.get(row, col) = a->get(row, col);
      }
    }

    // Ra Rb^t
    myTrmm<T>(&rAFull, b);
    // SVD of Ra Rb^t
    ierr = truncatedSvd<T>(&rAFull, &u, &sigma, &vt);
    HMAT_ASSERT(!ierr);
  }

  // Control of approximation
  int newK = approx.findK(sigma->v, rank(), epsilon);
  if (newK == 0)
  {
    delete a;
    a = NULL;
    delete b;
    b = NULL;
    return;
  }

  // We put the root of singular values in sigma
  for (int i = 0; i < rank(); i++) {
    sigma->v[i] = sqrt(sigma->v[i]);
  }

  // We need to calculate Qa * Utilde * SQRT (SigmaTilde)
  // For that we first calculated Utilde * SQRT (SigmaTilde)
  FullMatrix<T>* newA = FullMatrix<T>::Zero(rows->size(), newK);
  for (int col = 0; col < newK; col++) {
    T alpha = sigma->v[col];
    for (int row = 0; row < rank(); row++) {
      newA->get(row, col) = u->get(row, col) * alpha;
    }
  }
  delete u;
  u = NULL;
  // newA <- Qa * newA (et newA = Utilde * SQRT(SigmaTilde))
  productQ<T>('L', 'N', a, tauA, newA);
  free(tauA);

  // newB = Qb * VTilde * SQRT(SigmaTilde)
  FullMatrix<T>* newB = FullMatrix<T>::Zero(cols->size(), newK);
  // Copy with transposing
  for (int col = 0; col < newK; col++) {
    T alpha = sigma->v[col];
    for (int row = 0; row < rank(); row++) {
      newB->get(row, col) = vt->get(col, row) * alpha;
    }
  }
  delete vt;
  delete sigma;
  productQ<T>('L', 'N', b, tauB, newB);
  free(tauB);

  delete a;
  a = newA;
  delete b;
  b = newB;
}

// Swap members with members from another instance.
// Since rows and cols are constant, they must not be swapped.
template<typename T> void RkMatrix<T>::swap(RkMatrix<T>& other)
{
  assert(rows == other.rows);
  assert(cols == other.cols);
  std::swap(a, other.a);
  std::swap(b, other.b);
  std::swap(method, other.method);
}

template<typename T> void RkMatrix<T>::axpy(T alpha, const FullMatrix<T>* mat) {
  RkMatrix<T>* tmp = formattedAddParts(&alpha, &mat, &rows, &cols, 1);
  swap(*tmp);
  delete tmp;
}

template<typename T> void RkMatrix<T>::axpy(T alpha, const RkMatrix<T>* mat) {
  RkMatrix<T>* tmp = formattedAddParts(&alpha, &mat, 1);
  swap(*tmp);
  delete tmp;
}

template<typename T> RkMatrix<T>* RkMatrix<T>::formattedAdd(const RkMatrix<T>* o) const {
  const RkMatrix<T>* parts[1] = {o};
  T alpha[1] = {Constants<T>::pone};
  return formattedAddParts(alpha, parts, 1);
}

template<typename T> RkMatrix<T>* RkMatrix<T>::formattedAdd(const FullMatrix<T>* o, T alpha) const {
  const FullMatrix<T>* parts[1] = {o};
  const IndexSet* rowsList[1] = {rows};
  const IndexSet* colsList[1] = {cols};
  T alphaArray[1] = {alpha};
  return formattedAddParts(alphaArray, parts, rowsList, colsList, 1);
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::formattedAddParts(T* alpha, const RkMatrix<T>** parts,
                                            int n) const {
  DECLARE_CONTEXT;
  // If only one of the parts is non-zero, then the recompression is not necessary to
  // get exactly the same result.
  int notNullParts = (rank() == 0 ? 0 : 1);
  int kTotal = rank();
  CompressionMethod minMethod = method;
  for (int i = 0; i < n; i++) {
    // Check that partial RkMatrix indices are subsets of their global indices set.
    // According to the indices organization, it is necessary to check that the indices
    // of the matrix are such that:
    //   - parts[i].rows->offset >= rows->offset
    //   - offset + n <= this.offset + this.n
    //   - same for cols

    assert(parts[i]->rows->isSubset(*rows));
    assert(parts[i]->cols->isSubset(*cols));
    kTotal += parts[i]->rank();
    minMethod = std::min(minMethod, parts[i]->method);
    if (parts[i]->rank() != 0) {
      notNullParts += 1;
    }
  }
  // In case the sum of the ranks of the sub-matrices is greater than
  // the matrix size, it is more efficient to put everything in a
  // full matrix.
  if (kTotal >= std::min(rows->size(), cols->size())) {
    const FullMatrix<T>** fullParts = new const FullMatrix<T>*[n];
    const IndexSet** rowsParts = new const IndexSet*[n];
    const IndexSet** colsParts = new const IndexSet*[n];
    for (int i = 0; i < n; i++) {
      fullParts[i] = parts[i]->eval();
      rowsParts[i] = parts[i]->rows;
      colsParts[i] = parts[i]->cols;
    }
    RkMatrix<T>* result = formattedAddParts(alpha, fullParts, rowsParts, colsParts, n);
    for (int i = 0; i < n; i++) {
      delete fullParts[i];
    }
    delete[] fullParts;
    delete[] rowsParts;
    delete[] colsParts;
    return result;
  }

  FullMatrix<T>* resultA = new FullMatrix<T>(rows->size(), kTotal);
  resultA->clear();
  FullMatrix<T>* resultB = new FullMatrix<T>(cols->size(), kTotal);
  resultB->clear();
  // Special case if the original matrix is empty.
  if (rank() > 0) {
    resultA->copyMatrixAtOffset(a, 0, 0);
    resultB->copyMatrixAtOffset(b, 0, 0);
  }
  // According to the indices organization, the sub-matrices are
  // contiguous blocks in the "big" matrix whose columns offset is
  //      this-> k + parts [0] -> k + ... + parts [i - 1] -> k
  // and rows offset is
  //   parts [i] -> rows-> offset - rows-> offset
  // for rows, and size
  //      parts [i] -> rows-> nx parts [i] -> k (rows x columns)
  // Same for columns.
  int kOffset = rank();
  for (int i = 0; i < n; i++) {
    int rowOffset = parts[i]->rows->offset() - rows->offset();
    int rowCount = parts[i]->rows->size();
    int colCount = parts[i]->rank();
    if ((rowCount == 0) || (colCount == 0)) {
      continue;
    }
    resultA->copyMatrixAtOffset(parts[i]->a, rowOffset, kOffset);
    // Scaling the matrix already in place
    if (alpha[i] != Constants<T>::pone) {
      FullMatrix<T> tmp(resultA->m + rowOffset + ((size_t) kOffset) * resultA->lda,
                        parts[i]->a->rows, parts[i]->a->cols, resultA->lda);
      tmp.scale(alpha[i]);
    }
    rowOffset = parts[i]->cols->offset() - cols->offset();
    resultB->copyMatrixAtOffset(parts[i]->b, rowOffset, kOffset);
    kOffset += parts[i]->rank();
  }
  RkMatrix<T>* rk = new RkMatrix<T>(resultA, rows, resultB, cols, minMethod);
  if (notNullParts > 1) {
    rk->truncate(approx.recompressionEpsilon);
  }
  return rk;
}
template<typename T>
RkMatrix<T>* RkMatrix<T>::formattedAddParts(T* alpha, const FullMatrix<T>** parts,
                                            const IndexSet **rowsList,
                                            const IndexSet **colsList, int n) const {
  DECLARE_CONTEXT;
  FullMatrix<T>* me = eval();
  HMAT_ASSERT(me);

  for (int i = 0; i < n; i++) {
    assert(rowsList[i]->isSubset(*rows));
    assert(colsList[i]->isSubset(*cols));
    int rowOffset = rowsList[i]->offset() - rows->offset();
    int colOffset = colsList[i]->offset() - cols->offset();
    int maxCol = colsList[i]->size();
    int maxRow = rowsList[i]->size();
    for (int col = 0; col < maxCol; col++) {
      for (int row = 0; row < maxRow; row++) {
        me->get(row + rowOffset, col + colOffset) += alpha[i] * parts[i]->get(row, col);
      }
    }
  }
  RkMatrix<T>* result = compressMatrix(me, rows, cols);
  delete me;
  return result;
}


template<typename T> RkMatrix<T>* RkMatrix<T>::multiplyRkFull(char transR, char transM,
                                                              const RkMatrix<T>* rk,
                                                              const FullMatrix<T>* m,
                                                              const IndexSet* mCols) {
  DECLARE_CONTEXT;

  assert((transR == 'N') || (transM == 'N'));// we do not manage the case R^T*M^T
  assert(((transR == 'N') ? rk->cols->size() : rk->rows->size()) == ((transM == 'N') ? m->rows : m->cols));

  if(rk->rank() == 0) {
      return new RkMatrix<T>(NULL, transR ? rk->cols : rk->rows,
                             NULL, mCols, NoCompression);
  }
  RkMatrix<T>* rkCopy = (transR == 'N' ? new RkMatrix<T>(rk->a, rk->rows, rk->b, rk->cols, rk->method)
                         : new RkMatrix<T>(rk->b, rk->cols, rk->a, rk->rows, rk->method));

  FullMatrix<T>* newB = new FullMatrix<T>((transM == 'N')? m->cols : m->rows, rkCopy->b->cols);
  if (transM == 'N') {
    assert(m->rows == rkCopy->b->rows);
    assert(newB->rows == m->cols);
    assert(newB->cols == rkCopy->b->cols);
    newB->gemm('T', 'N', Constants<T>::pone, m, rkCopy->b, Constants<T>::zero);
  } else {
    assert(m->cols == rkCopy->b->rows);
    assert(newB->rows == m->rows);
    assert(newB->cols == rkCopy->b->cols);
    newB->gemm('N','N',Constants<T>::pone, m, rkCopy->b, Constants<T>::zero);
  }

  FullMatrix<T>* newA = rkCopy->a->copy();
  RkMatrix<T>* result =  new RkMatrix<T>(newA, rkCopy->rows, newB, mCols, rkCopy->method);
  rkCopy->a = NULL;
  rkCopy->b = NULL;
  delete rkCopy;

  /* R M = A B^t M = A (B^t M) = A (M^t B)^t */
  // MatrixXd b((rk->b.transpose() * (*m)).transpose());
  return result;
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::multiplyFullRk(char transM, char transR,
                                         const FullMatrix<T>* m,
                                         const RkMatrix<T>* rk,
                                         const IndexSet* mRows) {
  DECLARE_CONTEXT;
  assert((transR == 'N') || (transM == 'N')); // we do not manage the case R^T*M^T
  FullMatrix<T>* a = rk->a;
  FullMatrix<T>* b = rk->b;
  if (transR == 'T') { // permutation to transpose the matrix Rk
    std::swap(a, b);
  }
  const IndexSet *rkCols = ((transR == 'N')? rk->cols : rk->rows);

  /* M R = M (A B^t) = (MA) B^t */
  assert(((transM == 'N') ? m->rows : m->cols) == mRows->size());
  FullMatrix<T>* newA = new FullMatrix<T>((transM == 'N')? m->rows:m->cols,(transR == 'N')? a->cols:b->cols);
  if (transM == 'N') {
    newA->gemm('N', 'N', Constants<T>::pone, m, a, Constants<T>::zero);
  } else {
    newA->gemm('T', 'N',Constants<T>::pone, m, a, Constants<T>::zero);
  }
  FullMatrix<T>* newB = b->copy();
  return new RkMatrix<T>(newA, mRows, newB, rkCols, rk->method);
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::multiplyRkH(char transRk, char transH,
                                      const RkMatrix<T>* rk, const HMatrix<T>* h) {
  DECLARE_CONTEXT;
  assert((transRk == 'N') || (transH == 'N')); // we do not manage the case R^T*M^T
  assert(((transRk == 'N') ? *rk->cols : *rk->rows) == ((transH == 'N')? *h->rows() : *h->cols()));

  FullMatrix<T>* a = (transRk == 'N')? rk->a : rk->b;
  FullMatrix<T>* b = (transRk == 'N')? rk->b : rk->a;
  const IndexSet* rkRows = ((transRk == 'N')? rk->rows : rk->cols);

  // R M = A (M^t B)^t
  // Size of the HMatrix is n x m, with
  //   n = h.data.bt->data.first->data.n
  //   m =h.data.bt->data.second->data.n
  // So H^t size is m x n and the product is m x cols(B)
  // and the number of columns of B is k.
  int p = rk->rank();

  assert(b->cols == p);
  FullMatrix<T>* resB = new FullMatrix<T>(transH == 'N' ? h->cols()->size() : h->rows()->size(), p);
  resB->clear();
  h->gemv(transH == 'N' ? 'T' : 'N', Constants<T>::pone, b, Constants<T>::zero, resB);
  FullMatrix<T>* newA = a->copy();
  const IndexSet *newCols = ((transH == 'N' )? h->cols() : h->rows());
  return new RkMatrix<T>(newA, rkRows, resB, newCols, rk->method);
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::multiplyHRk(char transH, char transR,
                                      const HMatrix<T>* h, const RkMatrix<T>* rk) {

  DECLARE_CONTEXT;
  if (rk->rank() == 0) {
    const IndexSet* newRows = ((transH == 'N') ? h-> rows() : h->cols());
    const IndexSet* newCols = ((transR == 'N') ? rk->cols : rk->rows);
    return new RkMatrix<T>(NULL, newRows, NULL, newCols, rk->method);
  }
  // M R = (M A) B^t
  // The size of the HMatrix is n x m

  // m = h.data.bt->data.second->data.n
  // Therefore the product is n x cols(A)
  // and the number of columns of A is k.
  assert((transR == 'N') || (transH == 'N')); // we do not manage the case of product transposee*transposee
  FullMatrix<T>* a = rk->a;
  FullMatrix<T>* b = rk->b;
  if (transR == 'T') { // permutation of a and b to transpose the matrix Rk
    std::swap(a, b);
  }
  const IndexSet *rkCols = ((transR == 'N' )? rk->cols : rk->rows);
  int n = ((transH == 'N')? h->rows()->size() : h->cols()->size());
  int p = rk->rank();
  FullMatrix<T>* resA = new FullMatrix<T>(n, p);
  resA->clear();
  h->gemv(transH, Constants<T>::pone, a, Constants<T>::zero, resA);
  FullMatrix<T>* newB = b->copy();
  const IndexSet* newRows = ((transH == 'N')? h-> rows() : h->cols());
  // If this base been transposed earlier, back in the right direction.

  if (transR == 'T') {
    std::swap(a, b);
  }
  return new RkMatrix<T>(resA, newRows, newB, rkCols, rk->method);
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::multiplyRkRk(char transA, char transB,
                                       const RkMatrix<T>* a, const RkMatrix<T>* b) {
  DECLARE_CONTEXT;
  assert(((transA == 'N') ? *a->cols : *a->rows) == ((transB == 'N') ? *b->rows : *b->cols));
  // It is possible to do the computation differently, yielding a
  // different rank and a different amount of computation.
  // TODO: choose the best order.
  FullMatrix<T>* Aa = (transA == 'N' ? a->a : a->b);
  FullMatrix<T>* Ab = (transA == 'N' ? a->b : a->a);
  FullMatrix<T>* Ba = (transB == 'N' ? b->a : b->b);
  FullMatrix<T>* Bb = (transB == 'N' ? b->b : b->a);

  assert(Ab->rows == Ba->rows); // compatibility of the multiplication

  // We want to compute the matrix Aa.t^Ab.Ba.t^Bb and return an Rk matrix
  // Usually, the best way is to start with tmp=t^Ab.Ba which produces a 'small' matrix rank_a x rank_b
  // Then we can either :
  // - compute Aa.tmp : the cost is rank_a.rank_b.row_a, the resulting Rk has rank rank_b
  // - compute tmp.Bb : the cost is rank_a.rank_b.col_b, the resulting Rk has rank rank_a
  // the best choice depends on the ranks & dimensions, and also on our priority (flops or resulting rank)

  FullMatrix<T>* tmp = new FullMatrix<T>(a->rank(), b->rank());
  FullMatrix<T>* newA = new FullMatrix<T>(Aa->rows, b->rank());

  assert(tmp->rows == Ab->cols);
  assert(tmp->cols == Ba->cols);
  assert(Ab->rows == Ba->rows);
  tmp->gemm('T', 'N', Constants<T>::pone, Ab, Ba, Constants<T>::zero);
  assert(Ab->cols == tmp->rows);
  newA->gemm('N', 'N', Constants<T>::pone, Aa, tmp, Constants<T>::zero);
  delete tmp;
  FullMatrix<T>* newB = Bb->copy();

  CompressionMethod combined = std::min(a->method, b->method);
  return new RkMatrix<T>(newA, ((transA == 'N') ? a->rows : a->cols), newB, ((transB == 'N') ? b->cols : b->rows), combined);
}

template<typename T>
size_t RkMatrix<T>::computeRkRkMemorySize(char transA, char transB,
                                                const RkMatrix<T>* a, const RkMatrix<T>* b)
{
    FullMatrix<T>* Bb = (transB == 'N' ? b->b : b->a);
    FullMatrix<T>* Aa = (transA == 'N' ? a->a : a->b);
    return Bb == NULL ? 0 : Bb->memorySize() +
           Aa == NULL ? 0 : Aa->rows * b->rank() * sizeof(T);
}

template<typename T>
void RkMatrix<T>::multiplyWithDiagOrDiagInv(const HMatrix<T> * d, bool inverse, bool left) {
  assert(*d->rows() == *d->cols());
  assert(!left || (*rows == *d->cols()));
  assert(left  || (*cols == *d->rows()));

  // extracting the diagonal
  T* diag = new T[d->cols()->size()];
  d->extractDiagonal(diag);
  if (inverse) {
    for (int i = 0; i < d->cols()->size(); i++) {
      diag[i] = Constants<T>::pone / diag[i];
    }
  }

  // left multiplication by d of b (if M<-M*D : left = false) or a (if M<-D*M : left = true)
  FullMatrix<T>* aOrB = (left ? a : b);
  for (int j = 0; j < rank(); j++) {
    for (int i = 0; i < aOrB->rows; i++) {
      aOrB->get(i, j) *= diag[i];
    }
  }
  delete[] diag;
}

template<typename T> void RkMatrix<T>::gemmRk(char transHA, char transHB,
                                              T alpha, const HMatrix<T>* ha, const HMatrix<T>* hb, T beta) {
  DECLARE_CONTEXT;
  // TODO: remove this limitation, if needed.
  assert(beta == Constants<T>::pone);

  // This is ugly!  When ha node is void, we replace ha by its non-void child
  // so that further computations are similar to the non-void case.
  // Indeed, this is ugly...
  while (!ha->isLeaf())
  {
    if (ha->nrChild() >= 4) {
      if (ha->get(0, 0)->rows()->size() == 0 && ha->get(0, 0)->cols()->size() == 0)
      {
        ha = ha->get(1, 1);
        continue;
      }
      if (ha->get(1, 1) && ha->get(1, 1)->rows()->size() == 0 && ha->get(1, 1)->cols()->size() == 0)
      {
        ha = ha->get(0, 0);
        continue;
      }
    }
    break;
  }
  while (!hb->isLeaf())
  {
    if (hb->nrChild() >= 4) {
      if (hb->get(0, 0)->rows()->size() == 0 && hb->get(0, 0)->cols()->size() == 0)
      {
        hb = hb->get(1, 1);
        continue;
      }
      if (hb->get(1, 1) && hb->get(1, 1)->rows()->size() == 0 && hb->get(1, 1)->cols()->size() == 0)
      {
        hb = hb->get(0, 0);
        continue;
      }
    }
    break;
  }
  // void matrix
  if (ha->rows()->size() == 0 || ha->cols()->size() == 0 || hb->rows()->size() == 0 || hb->cols()->size() == 0) return;

  if (!(ha->isLeaf() || hb->isLeaf())) {
    int nbColsA = transHA == 'N' ? ha->nrChildCol() : ha->nrChildRow() ; /* Col blocks of A */
    int nbRowsB = transHA == 'N' ? hb->nrChildRow() : ha->nrChildCol() ; /* Row blocks of B */
    int nbRows = transHA == 'N' ? ha->nrChildRow() : ha->nrChildCol() ; /* Row blocks of the product */
    int nbCols = transHB == 'N' ? hb->nrChildCol() : hb->nrChildRow() ; /* Col blocks of the product */
    int nbCom  = transHA == 'N' ? ha->nrChildCol() : ha->nrChildRow() ; /* Common dimension between A and B */
    RkMatrix<T>* subRks[nbRows * nbCols];
    for (int i = 0; i < nbRows; i++) {
      for (int j = 0; j < nbCols; j++) {
        int ib = min(i, nbRowsB - 1);
        int ja = min(j, nbColsA - 1);
        const IndexSet* subRows = (transHA == 'N' ? ha->get(i, ja)->rows() : ha->get(ja, i)->cols());
        const IndexSet* subCols = (transHB == 'N' ? hb->get(ib, j)->cols() : hb->get(j, ib)->rows());
        subRks[i + j * nbRows] = new RkMatrix<T>(NULL, subRows, NULL, subCols, NoCompression);
        for (int k = 0; k < nbCom; k++) {
          // C_ij = A_ik * B_kj
          const HMatrix<T>* a_ik = (transHA == 'N' ? ha->get(i, k) : ha->get(k, i));
          const HMatrix<T>* b_kj = (transHB == 'N' ? hb->get(k, j) : hb->get(j, k));
          subRks[i + j * nbRows]->gemmRk(transHA, transHB, alpha, a_ik, b_kj, beta);
        }
      }
    }
    // Reconstruction of C by adding the parts
    std::vector<T> alpha(nbRows * nbCols, Constants<T>::pone);
    RkMatrix<T>* rk = formattedAddParts(&alpha[0], (const RkMatrix<T>**) subRks, nbRows * nbCols);
    swap(*rk);
    for (int i = 0; i < nbRows * nbCols; i++) {
      delete subRks[i];
    }
    delete rk;
  } else {
    RkMatrix<T>* rk = NULL;
    // One of the product matrix is a leaf
    if ((ha->isLeaf() && ha->isNull()) || (hb->isLeaf() && hb->isNull())) {
      // Nothing to do
    } else if (ha->isRkMatrix() || hb->isRkMatrix()) {
      rk = HMatrix<T>::multiplyRkMatrix(transHA, transHB, ha, hb);
    } else {
      assert(ha->isFullMatrix() || hb->isFullMatrix());
      FullMatrix<T>* fullMat = HMatrix<T>::multiplyFullMatrix(transHA, transHB, ha, hb);
      if(fullMat) {
        rk = compressMatrix(fullMat, (transHA == 'N' ? ha->rows() : ha->cols()),
                           (transHB == 'N' ? hb->cols() : hb->rows()));
        delete fullMat;
      }
    }
    if(rk) {
      axpy(alpha, rk);
      delete rk;
    }
  }
}

template<typename T> void RkMatrix<T>::copy(RkMatrix<T>* o) {
  delete a;
  delete b;
  rows = o->rows;
  cols = o->cols;
  a = (o->a ? o->a->copy() : NULL);
  b = (o->b ? o->b->copy() : NULL);
}


template<typename T> void RkMatrix<T>::checkNan() const {
  if (rank() == 0) {
    return;
  }
  a->checkNan();
  b->checkNan();
}

// Templates declaration
template class RkMatrix<S_t>;
template class RkMatrix<D_t>;
template class RkMatrix<C_t>;
template class RkMatrix<Z_t>;

}  // end namespace hmat
