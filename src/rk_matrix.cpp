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

namespace hmat {

/** RkApproximationControl */
template<typename T> RkApproximationControl RkMatrix<T>::approx;
int RkApproximationControl::findK(Vector<double> &sigma, double epsilon) {
  // Control of approximation for fixed approx.k != 0
  int newK = k;
  if (newK != 0) {
    newK = std::min(newK, sigma.rows);
  } else {
    assert(epsilon >= 0.);
    static char *useL2Criterion = getenv("HMAT_L2_CRITERION");
    double threshold_eigenvalue = 0.0;
    if (useL2Criterion == NULL) {
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
    newK = i;
  }
  return newK;
}


/** RkMatrix */
template<typename T> RkMatrix<T>::RkMatrix(ScalarArray<T>* _a, const IndexSet* _rows,
                                           ScalarArray<T>* _b, const IndexSet* _cols,
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


template<typename T> ScalarArray<T>* RkMatrix<T>::evalArray(ScalarArray<T>* result) const {
  if(result==NULL)
    result = new ScalarArray<T>(rows->size(), cols->size());
  if (rank())
    result->gemm('N', 'T', Constants<T>::pone, a, b, Constants<T>::zero);
  else
    result->clear();
  return result;
}

template<typename T> FullMatrix<T>* RkMatrix<T>::eval() const {
  FullMatrix<T>* result = new FullMatrix<T>(rows, cols);
  evalArray(&result->data);
  return result;
}

// Compute squared Frobenius norm
template<typename T> double RkMatrix<T>::normSqr() const {
  return a->norm_abt_Sqr(*b);
}

template<typename T> void RkMatrix<T>::scale(T alpha) {
  // We need just to scale the first matrix, A.
  if (a) {
    a->scale(alpha);
  }
}

template<typename T> void RkMatrix<T>::transpose() {
  std::swap(a, b);
  std::swap(rows, cols);
}

template<typename T> void RkMatrix<T>::clear() {
  delete a;
  delete b;
  a = NULL;
  b = NULL;
}

template<typename T>
void RkMatrix<T>::gemv(char trans, T alpha, const FullMatrix<T>* x, T beta, FullMatrix<T>* y) const {
  gemv(trans, alpha, &x->data, beta, &y->data);
}

template<typename T>
void RkMatrix<T>::gemv(char trans, T alpha, const ScalarArray<T>* x, T beta, ScalarArray<T>* y) const {
  if (rank() == 0) {
    if (beta != Constants<T>::pone) {
      y->scale(beta);
    }
    return;
  }
  if (trans == 'N') {
    // Compute Y <- Y + alpha * A * B^T * X
    ScalarArray<T> z(b->cols, x->cols);
    z.gemm('T', 'N', Constants<T>::pone, b, x, Constants<T>::zero);
    y->gemm('N', 'N', alpha, a, &z, beta);
  } else if (trans == 'T') {
    // Compute Y <- Y + alpha * (A*B^T)^T * X = Y + alpha * B * A^T * X
    ScalarArray<T> z(a->cols, x->cols);
    z.gemm('T', 'N', Constants<T>::pone, a, x, Constants<T>::zero);
    y->gemm('N', 'N', alpha, b, &z, beta);
  } else {
    assert(trans == 'C');
    // Compute Y <- Y + alpha * (A*B^T)^H * X = Y + alpha * conj(B) * A^H * X
    ScalarArray<T> z(a->cols, x->cols);
    z.gemm('C', 'N', Constants<T>::pone, a, x, Constants<T>::zero);
    ScalarArray<T> * newB = b->copy();
    newB->conjugate();
    y->gemm('N', 'N', alpha, newB, &z, beta);
    delete newB;
  }
}

template<typename T> const RkMatrix<T>* RkMatrix<T>::subset(const IndexSet* subRows,
                                                            const IndexSet* subCols) const {
  assert(subRows->isSubset(*rows));
  assert(subCols->isSubset(*cols));
  ScalarArray<T>* subA = NULL;
  ScalarArray<T>* subB = NULL;
  if(rank() > 0) {
    // The offset in the matrix, and not in all the indices
    int rowsOffset = subRows->offset() - rows->offset();
    int colsOffset = subCols->offset() - cols->offset();
    subA = new ScalarArray<T>(*a, rowsOffset, subRows->size(), 0, rank());
    subB = new ScalarArray<T>(*b, colsOffset, subCols->size(), 0, rank());
  }
  return new RkMatrix<T>(subA, subRows, subB, subCols, method);
}

template<typename T> size_t RkMatrix<T>::compressedSize() {
    return ((size_t)rows->size()) * rank() + ((size_t)cols->size()) * rank();
}

template<typename T> size_t RkMatrix<T>::uncompressedSize() {
    return ((size_t)rows->size()) * cols->size();
}

template<typename T> void RkMatrix<T>::addRand(double epsilon) {
  DECLARE_CONTEXT;
  a->addRand(epsilon);
  b->addRand(epsilon);
  return;
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
    RkMatrix<T>* rk = truncatedSvd(tmp);
    delete tmp;
    // "Move" rk into this, and delete the old "this".
    swap(*rk);
    delete rk;
    // TODO: shouldn't we 'return' here ? the rest of the code below is an expensive way to apply the epsilon of recompression...
  }

  static char *useCUSTOM = getenv("HMAT_CUSTOM_RECOMPRESS");
  if (useCUSTOM){
    mGSTruncate(epsilon);
    return;
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
  T* tauA = a->qrDecomposition(); // A contains Qa and Ra
  HMAT_ASSERT(tauA);
  T* tauB = b->qrDecomposition(); // B contains Qb and Rb
  HMAT_ASSERT(tauB);

  // Matrices created by the SVD
  ScalarArray<T> *u = NULL, *v = NULL;
  Vector<double> *sigma = NULL;
  {
    // The scope is to automatically delete rAFull.
    // For computing Ra*Rb^t, we would like to use directly a BLAS function
    // because Ra and Rb are triangular matrices.
    //
    // However, there is no multiplication of two triangular matrices in BLAS,
    // left matrix must be converted into full matrix first.
    ScalarArray<T> rAFull(rank(), rank());
    for (int col = 0; col < rank(); col++) {
      for (int row = 0; row <= col; row++) {
        rAFull.get(row, col) = a->get(row, col);
      }
    }

    // Ra <- Ra Rb^t with b upper triangular matrix
    rAFull.myTrmm(b);
    // SVD of Ra Rb^t
    ierr = rAFull.svdDecomposition(&u, (ScalarArray<double> **)&sigma, &v); // TODO use something else than SVD ?
    HMAT_ASSERT(!ierr);
  }

  // Control of approximation
  int newK = approx.findK(*sigma, epsilon);
  if (newK == 0)
  {
    delete u;
    delete v;
    delete sigma;
    free(tauA);
    free(tauB);
    delete a;
    a = NULL;
    delete b;
    b = NULL;
    return;
  }

  // Resize u, sigma, v (not very clean...)
  u->cols =newK;
  sigma->rows = newK;
  v->cols =newK;

  // We put the square root of singular values in sigma
  for (int i = 0; i < newK; i++) {
    (*sigma)[i] = sqrt((*sigma)[i]);
  }
  // TODO why not rather apply SigmaTilde to only a or b, and avoid computing square roots ?

  // We need to calculate Qa * Utilde * SQRT (SigmaTilde)
  // For that we first calculate Utilde * SQRT (SigmaTilde)
  u->multiplyWithDiag(sigma);

  ScalarArray<T>* newA = new ScalarArray<T>(rows->size(), newK);
  newA->copyMatrixAtOffset(u, 0, 0);
  delete u;
  u = NULL;
  // newA <- Qa * newA (et newA = Utilde * SQRT(SigmaTilde))
  a->productQ('L', 'N', tauA, newA);
  free(tauA);

  // newB = Qb * VTilde * SQRT(SigmaTilde)
  v->multiplyWithDiag(sigma);
  ScalarArray<T>* newB = new ScalarArray<T>(cols->size(), newK);
  newB->copyMatrixAtOffset(v, 0, 0);
  delete v;
  delete sigma;
  // newB <- Qb * newB
  b->productQ('L', 'N', tauB, newB);
  free(tauB);

  delete a;
  a = newA;
  delete b;
  b = newB;
}

template<typename T> void RkMatrix<T>::mGSTruncate(double epsilon) {
  DECLARE_CONTEXT;

  if (rank() == 0) {
    assert(!(a || b));
    return;
  }

  ScalarArray<T>* ur = NULL;
  Vector<double>* sr = NULL;
  ScalarArray<T>* vr = NULL;
  int kA, kB, newK;

  int krank = rank();

  // Limit scope to automatically destroy ra, rb and matR
  {
    // Gram-Schmidt on a
    ScalarArray<T> ra(krank, krank);
    kA = a->modifiedGramSchmidt( &ra, epsilon );
    // On input, a0(m,k)
    // On output, a(m,kA), ra(kA,k) such that a0 = a * ra

    // Gram-Schmidt on b
    ScalarArray<T> rb(krank, krank);
    kB = b->modifiedGramSchmidt( &rb, epsilon );
    // On input, b0(p,k)
    // On output, b(p,kB), rb(kB,k) such that b0 = b * rb

    // M = a0*b0^T = a*(ra*rb^T)*b^T
    // We perform an SVD on ra*rb^T:
    //  (ra*rb^T) = U*S*S*Vt
    // and M = (a*U*S)*(S*Vt*b^T) = (a*U*S)*(b*(S*Vt)^T)^T
    ScalarArray<T> matR(kA, kB);
    matR.gemm('N','T', Constants<T>::pone, &ra, &rb , Constants<T>::zero);

    // SVD
    int ierr = matR.svdDecomposition(&ur, (ScalarArray<double> **)&sr, &vr);
    // On output, ur->rows = kA, vr->rows = kB
    HMAT_ASSERT(!ierr);
  }

  // Remove small singular values and compute square root of sr
  newK = approx.findK(*sr, epsilon);
  assert(newK>0);
  for(int i = 0; i < newK; ++i) {
    (*sr)[i] = sqrt((*sr)[i]);
  }
  ur->cols = newK;
  vr->cols = newK;
  /* Scaling of ur and vr */
  ur->multiplyWithDiag(sr);
  vr->multiplyWithDiag(sr);

  delete sr;

  /* Multiplication by orthogonal matrix Q: no or/un-mqr as
    this comes from Gram-Schmidt procedure not Householder
  */
  ScalarArray<T> *newA = new ScalarArray<T>(a->rows, newK);
  newA->gemm('N', 'N', Constants<T>::pone, a, ur, Constants<T>::zero);

  ScalarArray<T> *newB = new ScalarArray<T>(b->rows, newK);
  newB->gemm('N', 'N', Constants<T>::pone, b, vr, Constants<T>::zero);

  delete ur;
  delete vr;

  delete a;
  a = newA;
  delete b;
  b = newB;

  if (rank() == 0) {
    assert(!(b || a));
  }
}

// Swap members with members from another instance.
template<typename T> void RkMatrix<T>::swap(RkMatrix<T>& other)
{
  assert(rows == other.rows);
  assert(cols == other.cols);
  std::swap(a, other.a);
  std::swap(b, other.b);
  std::swap(method, other.method);
}

template<typename T> void RkMatrix<T>::axpy(T alpha, const FullMatrix<T>* mat) {
  RkMatrix<T>* tmp = formattedAddParts(&alpha, &mat, 1);
  swap(*tmp);
  delete tmp;
}

template<typename T> void RkMatrix<T>::axpy(T alpha, const RkMatrix<T>* mat) {
  RkMatrix<T>* tmp = formattedAddParts(&alpha, &mat, 1);
  swap(*tmp);
  delete tmp;
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::formattedAddParts(const T* alpha, const RkMatrix<T>* const * parts,
                                            int n, bool dotruncate) const {
  // TODO check if formattedAddParts() actually uses sometimes this 'alpha' parameter (or is it always 1 ?)
  DECLARE_CONTEXT;
  // If only one of the parts is non-zero, then the recompression is not necessary to
  // get exactly the same result.
  int notNullParts = (rank() == 0 ? 0 : 1);
  int kTotal = rank();
  CompressionMethod minMethod = method;
  for (int i = 0; i < n; i++) {
    if (!parts[i])
      continue;
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

  if(notNullParts == 0)
    return new RkMatrix<T>(NULL, rows, NULL, cols, minMethod);

  // In case the sum of the ranks of the sub-matrices is greater than
  // the matrix size, it is more efficient to put everything in a
  // full matrix.
  if (kTotal >= std::min(rows->size(), cols->size())) {
    const FullMatrix<T>** fullParts = new const FullMatrix<T>*[n];
    for (int i = 0; i < n; i++) {
      if (!parts[i])
        continue;
      fullParts[i] = parts[i]->eval();
    }
    RkMatrix<T>* result = formattedAddParts(alpha, fullParts, n);
    for (int i = 0; i < n; i++) {
      delete fullParts[i];
    }
    delete[] fullParts;
    return result;
  }

  ScalarArray<T>* resultA = new ScalarArray<T>(rows->size(), kTotal);
  ScalarArray<T>* resultB = new ScalarArray<T>(cols->size(), kTotal);
  // Special case if the original matrix is not empty.
  if (rank() > 0) {
    resultA->copyMatrixAtOffset(a, 0, 0);
    resultB->copyMatrixAtOffset(b, 0, 0);
  }
  // According to the indices organization, the sub-matrices are
  // contiguous blocks in the "big" matrix whose columns offset is
  //      kOffset = this->k + parts[0]->k + ... + parts[i-1]->k
  // rows offset is
  //   parts[i]->rows->offset - rows->offset
  // rows size
  //      parts[i]->rows->size x parts[i]->k (rows x columns)
  // Same for columns.
  int kOffset = rank();
  for (int i = 0; i < n; i++) {
    if (!parts[i])
      continue;
    int rowOffset = parts[i]->rows->offset() - rows->offset();
    int rowCount = parts[i]->rows->size();
    int colCount = parts[i]->rank();
    if ((rowCount == 0) || (colCount == 0)) {
      continue;
    }
    resultA->copyMatrixAtOffset(parts[i]->a, rowOffset, kOffset);
    // Scaling the matrix already in place inside resultA
    if (alpha[i] != Constants<T>::pone) {
      ScalarArray<T> tmp(*resultA, rowOffset, parts[i]->a->rows, kOffset, parts[i]->a->cols);
      tmp.scale(alpha[i]);
    }
    rowOffset = parts[i]->cols->offset() - cols->offset();
    resultB->copyMatrixAtOffset(parts[i]->b, rowOffset, kOffset);
    kOffset += parts[i]->rank();
  }
  RkMatrix<T>* rk = new RkMatrix<T>(resultA, rows, resultB, cols, minMethod);
  if (notNullParts > 1 && dotruncate) {
    rk->truncate(approx.recompressionEpsilon);
  }
  return rk;
}
template<typename T>
RkMatrix<T>* RkMatrix<T>::formattedAddParts(const T* alpha, const FullMatrix<T>* const * parts, int n) const {
  DECLARE_CONTEXT;
  FullMatrix<T>* me = eval();
  HMAT_ASSERT(me);

  // TODO: here, we convert Rk->Full, Update the Full with parts[], and Full->Rk. We could also
  // create a new empty Full, update, convert to Rk and add it to 'this'.
  // If the parts[] are smaller than 'this', convert them to Rk and add them could be less expensive
  for (int i = 0; i < n; i++) {
    if (!parts[i])
      continue;
    const IndexSet *rows_full = parts[i]->rows_;
    const IndexSet *cols_full = parts[i]->cols_;
    assert(rows_full->isSubset(*rows));
    assert(cols_full->isSubset(*cols));
    int rowOffset = rows_full->offset() - rows->offset();
    int colOffset = cols_full->offset() - cols->offset();
    int maxCol = cols_full->size();
    int maxRow = rows_full->size();
    ScalarArray<T> sub(me->data, rowOffset, maxRow, colOffset, maxCol);
    sub.axpy(alpha[i], &parts[i]->data);
  }
  RkMatrix<T>* result = truncatedSvd(me); // TODO compress with something else than SVD
  delete me;
  return result;
}


template<typename T> RkMatrix<T>* RkMatrix<T>::multiplyRkFull(char transR, char transM,
                                                              const RkMatrix<T>* rk,
                                                              const FullMatrix<T>* m) {
  DECLARE_CONTEXT;

  assert(((transR == 'N') ? rk->cols->size() : rk->rows->size()) == ((transM == 'N') ? m->rows() : m->cols()));
  const IndexSet *rkRows = ((transR == 'N')? rk->rows : rk->cols);
  const IndexSet *mCols = ((transM == 'N')? m->cols_ : m->rows_);

  if(rk->rank() == 0) {
      return new RkMatrix<T>(NULL, rkRows, NULL, mCols, NoCompression);
  }
  // If transM is 'N' and transR is 'N', we compute
  //  A * B^T * M ==> newA = A, newB = M^T * B
  // We can deduce all other cases from this one:
  //   * if transR is 'T', all we have to do is to swap A and B
  //   * if transR is 'C', we swap A and B, but they must also
  //     be conjugate; let us look at the different cases:
  //     + if transM is 'N', newB = M^T * conj(B) = conj(M^H * B)
  //     + if transM is 'T', newB = M * conj(B)
  //     + if transM is 'C', newB = conj(M) * conj(B) = conj(M * B)

  ScalarArray<T> *newA, *newB;
  ScalarArray<T>* a = rk->a;
  ScalarArray<T>* b = rk->b;
  if (transR != 'N') {
    // if transR == 'T', we permute a and b; if transR == 'C', they will
    // also have to be conjugated, but this cannot be done here because rk
    // is const, this will be performed below.
    std::swap(a, b);
  }
  newA = a->copy();
  newB = new ScalarArray<T>(transM == 'N' ? m->cols() : m->rows(), b->cols);
  if (transR == 'C') {
    newA->conjugate();
    if (transM == 'N') {
      newB->gemm('C', 'N', Constants<T>::pone, &m->data, b, Constants<T>::zero);
      newB->conjugate();
    } else if (transM == 'T') {
      ScalarArray<T> *conjB = b->copy();
      conjB->conjugate();
      newB->gemm('N', 'N', Constants<T>::pone, &m->data, conjB, Constants<T>::zero);
      delete conjB;
    } else {
      assert(transM == 'C');
      newB->gemm('N', 'N', Constants<T>::pone, &m->data, b, Constants<T>::zero);
      newB->conjugate();
    }
  } else {
    if (transM == 'N') {
      newB->gemm('T', 'N', Constants<T>::pone, &m->data, b, Constants<T>::zero);
    } else if (transM == 'T') {
      newB->gemm('N', 'N', Constants<T>::pone, &m->data, b, Constants<T>::zero);
    } else {
      assert(transM == 'C');
      ScalarArray<T> *conjB = b->copy();
      conjB->conjugate();
      newB->gemm('N', 'N', Constants<T>::pone, &m->data, conjB, Constants<T>::zero);
      newB->conjugate();
      delete conjB;
    }
  }
  RkMatrix<T>* result = new RkMatrix<T>(newA, rkRows, newB, mCols, rk->method);
  return result;
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::multiplyFullRk(char transM, char transR,
                                         const FullMatrix<T>* m,
                                         const RkMatrix<T>* rk) {
  DECLARE_CONTEXT;
  // If transM is 'N' and transR is 'N', we compute
  //  M * A * B^T  ==> newA = M * A, newB = B
  // We can deduce all other cases from this one:
  //   * if transR is 'T', all we have to do is to swap A and B
  //   * if transR is 'C', we swap A and B, but they must also
  //     be conjugate; let us look at the different cases:
  //     + if transM is 'N', newA = M * conj(A)
  //     + if transM is 'T', newA = M^T * conj(A) = conj(M^H * A)
  //     + if transM is 'C', newA = M^H * conj(A) = conj(M^T * A)
  ScalarArray<T> *newA, *newB;
  ScalarArray<T>* a = rk->a;
  ScalarArray<T>* b = rk->b;
  if (transR != 'N') { // permutation to transpose the matrix Rk
    std::swap(a, b);
  }
  const IndexSet *rkCols = ((transR == 'N')? rk->cols : rk->rows);
  const IndexSet *mRows = ((transM == 'N')? m->rows_ : m->cols_);

  newA = new ScalarArray<T>(mRows->size(), b->cols);
  newB = b->copy();
  if (transR == 'C') {
    newB->conjugate();
    if (transM == 'N') {
      ScalarArray<T> *conjA = a->copy();
      conjA->conjugate();
      newA->gemm('N', 'N', Constants<T>::pone, &m->data, conjA, Constants<T>::zero);
      delete conjA;
    } else if (transM == 'T') {
      newA->gemm('C', 'N', Constants<T>::pone, &m->data, a, Constants<T>::zero);
      newA->conjugate();
    } else {
      assert(transM == 'C');
      newA->gemm('T', 'N', Constants<T>::pone, &m->data, a, Constants<T>::zero);
      newA->conjugate();
    }
  } else {
    newA->gemm(transM, 'N', Constants<T>::pone, &m->data, a, Constants<T>::zero);
  }
  RkMatrix<T>* result = new RkMatrix<T>(newA, mRows, newB, rkCols, rk->method);
  return result;
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::multiplyRkH(char transR, char transH,
                                      const RkMatrix<T>* rk, const HMatrix<T>* h) {
  DECLARE_CONTEXT;
  assert(((transR == 'N') ? *rk->cols : *rk->rows) == ((transH == 'N')? *h->rows() : *h->cols()));

  const IndexSet* rkRows = ((transR == 'N')? rk->rows : rk->cols);

  // If transR == 'N'
  //    transM == 'N': (A*B^T)*M = A*(M^T*B)^T
  //    transM == 'T': (A*B^T)*M^T = A*(M*B)^T
  //    transM == 'C': (A*B^T)*M^H = A*(conj(M)*B)^T = A*conj(M*conj(B))^T
  // If transR == 'T', we only have to swap A and B
  // If transR == 'C', we swap A and B, then
  //    transM == 'N': R^H*M = conj(A)*(M^T*conj(B))^T = conj(A)*conj(M^H*B)^T
  //    transM == 'T': R^H*M^T = conj(A)*(M*conj(B))^T
  //    transM == 'C': R^H*M^H = conj(A)*conj(M*B)^T
  //
  // Size of the HMatrix is n x m,
  // So H^t size is m x n and the product is m x cols(B)
  // and the number of columns of B is k.
  ScalarArray<T> *newA, *newB;
  ScalarArray<T>* a = rk->a;
  ScalarArray<T>* b = rk->b;
  if (transR != 'N') { // permutation to transpose the matrix Rk
    std::swap(a, b);
  }

  const IndexSet *newCols = ((transH == 'N' )? h->cols() : h->rows());

  newA = a->copy();
  newB = new ScalarArray<T>(transH == 'N' ? h->cols()->size() : h->rows()->size(), b->cols);
  if (transR == 'C') {
    newA->conjugate();
    if (transH == 'N') {
      h->gemv('C', Constants<T>::pone, b, Constants<T>::zero, newB);
      newB->conjugate();
    } else if (transH == 'T') {
      ScalarArray<T> *conjB = b->copy();
      conjB->conjugate();
      h->gemv('N', Constants<T>::pone, conjB, Constants<T>::zero, newB);
      delete conjB;
    } else {
      assert(transH == 'C');
      h->gemv('N', Constants<T>::pone, b, Constants<T>::zero, newB);
      newB->conjugate();
    }
  } else {
    if (transH == 'N') {
      h->gemv('T', Constants<T>::pone, b, Constants<T>::zero, newB);
    } else if (transH == 'T') {
      h->gemv('N', Constants<T>::pone, b, Constants<T>::zero, newB);
    } else {
      assert(transH == 'C');
      ScalarArray<T> *conjB = b->copy();
      conjB->conjugate();
      h->gemv('N', Constants<T>::pone, conjB, Constants<T>::zero, newB);
      delete conjB;
      newB->conjugate();
    }
  }
  RkMatrix<T>* result = new RkMatrix<T>(newA, rkRows, newB, newCols, rk->method);
  return result;
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

  // If transH is 'N' and transR is 'N', we compute
  //  M * A * B^T  ==> newA = M * A, newB = B
  // We can deduce all other cases from this one:
  //   * if transR is 'T', all we have to do is to swap A and B
  //   * if transR is 'C', we swap A and B, but they must also
  //     be conjugate; let us look at the different cases:
  //     + if transH is 'N', newA = M * conj(A)
  //     + if transH is 'T', newA = M^T * conj(A) = conj(M^H * A)
  //     + if transH is 'C', newA = M^H * conj(A) = conj(M^T * A)
  ScalarArray<T> *newA, *newB;
  ScalarArray<T>* a = rk->a;
  ScalarArray<T>* b = rk->b;
  if (transR != 'N') { // permutation to transpose the matrix Rk
    std::swap(a, b);
  }
  const IndexSet *rkCols = ((transR == 'N')? rk->cols : rk->rows);
  const IndexSet* newRows = ((transH == 'N')? h-> rows() : h->cols());

  newA = new ScalarArray<T>(transH == 'N' ? h->rows()->size() : h->cols()->size(), b->cols);
  newB = b->copy();
  if (transR == 'C') {
    newB->conjugate();
    if (transH == 'N') {
      ScalarArray<T> *conjA = a->copy();
      conjA->conjugate();
      h->gemv('N', Constants<T>::pone, conjA, Constants<T>::zero, newA);
      delete conjA;
    } else if (transH == 'T') {
      h->gemv('C', Constants<T>::pone, a, Constants<T>::zero, newA);
      newA->conjugate();
    } else {
      assert(transH == 'C');
      h->gemv('T', Constants<T>::pone, a, Constants<T>::zero, newA);
      newA->conjugate();
    }
  } else {
    h->gemv(transH, Constants<T>::pone, a, Constants<T>::zero, newA);
  }
  RkMatrix<T>* result = new RkMatrix<T>(newA, newRows, newB, rkCols, rk->method);
  return result;
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::multiplyRkRk(char trans1, char trans2,
                                       const RkMatrix<T>* r1, const RkMatrix<T>* r2) {
  DECLARE_CONTEXT;
  assert(((trans1 == 'N') ? *r1->cols : *r1->rows) == ((trans2 == 'N') ? *r2->rows : *r2->cols));
  // It is possible to do the computation differently, yielding a
  // different rank and a different amount of computation.
  // TODO: choose the best order.
  ScalarArray<T>* a1 = (trans1 == 'N' ? r1->a : r1->b);
  ScalarArray<T>* b1 = (trans1 == 'N' ? r1->b : r1->a);
  ScalarArray<T>* a2 = (trans2 == 'N' ? r2->a : r2->b);
  ScalarArray<T>* b2 = (trans2 == 'N' ? r2->b : r2->a);

  assert(b1->rows == a2->rows); // compatibility of the multiplication

  // We want to compute the matrix a1.t^b1.a2.t^b2 and return an Rk matrix
  // Usually, the best way is to start with tmp=t^b1.a2 which produces a 'small' matrix rank1 x rank2
  // Then we can either :
  // - compute a1.tmp : the cost is rank1.rank2.row_a, the resulting Rk has rank rank2
  // - compute tmp.b2 : the cost is rank1.rank2.col_b, the resulting Rk has rank rank1
  // We use the solution which gives the lowest rank.

  // TODO also, once we have the small matrix tmp=t^b1.a2, we could do a recompression on it for low cost
  // using SVD + truncation. This also removes the choice above, since tmp=U.S.V is then applied on both sides

  ScalarArray<T>* tmp = new ScalarArray<T>(r1->rank(), r2->rank());
  if (trans1 == 'C' && trans2 == 'C') {
    tmp->gemm('T', 'N', Constants<T>::pone, b1, a2, Constants<T>::zero);
    tmp->conjugate();
  } else if (trans1 == 'C') {
    ScalarArray<T> *conj_b1 = b1->copy();
    conj_b1->conjugate();
    tmp->gemm('T', 'N', Constants<T>::pone, conj_b1, a2, Constants<T>::zero);
    delete conj_b1;
  } else if (trans2 == 'C') {
    ScalarArray<T> *conj_a2 = a2->copy();
    conj_a2->conjugate();
    tmp->gemm('T', 'N', Constants<T>::pone, b1, conj_a2, Constants<T>::zero);
    delete conj_a2;
  } else {
    tmp->gemm('T', 'N', Constants<T>::pone, b1, a2, Constants<T>::zero);
  }

  ScalarArray<T> *newA, *newB;
  if (r1->rank() < r2->rank()) {
    newA = a1->copy();
    if (trans1 == 'C') {
      newA->conjugate();
    }
    newB = new ScalarArray<T>(b2->rows, r1->rank());
    if (trans2 == 'C') {
      ScalarArray<T> *conj_b2 = b2->copy();
      conj_b2->conjugate();
      newB->gemm('N', 'T', Constants<T>::pone, conj_b2, tmp, Constants<T>::zero);
      delete conj_b2;
    } else {
      newB->gemm('N', 'T', Constants<T>::pone, b2, tmp, Constants<T>::zero);
    }
  } else {
    newA = new ScalarArray<T>(a1->rows, r2->rank());
    if (trans1 == 'C') {
      ScalarArray<T> *conj_a1 = a1->copy();
      conj_a1->conjugate();
      newA->gemm('N', 'N', Constants<T>::pone, conj_a1, tmp, Constants<T>::zero);
      delete conj_a1;
    } else {
      newA->gemm('N', 'N', Constants<T>::pone, a1, tmp, Constants<T>::zero);
    }
    newB = b2->copy();
    if (trans2 == 'C') {
      newB->conjugate();
    }
  }
  delete tmp;

  CompressionMethod combined = std::min(r1->method, r2->method);
  return new RkMatrix<T>(newA, ((trans1 == 'N') ? r1->rows : r1->cols), newB, ((trans2 == 'N') ? r2->cols : r2->rows), combined);
}

template<typename T>
size_t RkMatrix<T>::computeRkRkMemorySize(char trans1, char trans2,
                                                const RkMatrix<T>* r1, const RkMatrix<T>* r2)
{
    ScalarArray<T>* b2 = (trans2 == 'N' ? r2->b : r2->a);
    ScalarArray<T>* a1 = (trans1 == 'N' ? r1->a : r1->b);
    return b2 == NULL ? 0 : b2->memorySize() +
           a1 == NULL ? 0 : a1->rows * r2->rank() * sizeof(T);
}

template<typename T>
void RkMatrix<T>::multiplyWithDiagOrDiagInv(const HMatrix<T> * d, bool inverse, bool left) {
  assert(*d->rows() == *d->cols());
  assert(!left || (*rows == *d->cols()));
  assert(left  || (*cols == *d->rows()));

  // extracting the diagonal
  Vector<T>* diag = new Vector<T>(d->cols()->size());
  d->extractDiagonal(diag->ptr());

  // left multiplication by d of b (if M<-M*D : left = false) or a (if M<-D*M : left = true)
  ScalarArray<T>* aOrB = (left ? a : b);
  aOrB->multiplyWithDiagOrDiagInv(diag, inverse, true);

  delete diag;
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
    if (ha->nrChildRow() >= 2 && ha->nrChildCol() >= 2) {
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
    if (hb->nrChildRow() >= 2 && hb->nrChildCol() >= 2) {
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
    // Recursion case
    int nbRows = transHA == 'N' ? ha->nrChildRow() : ha->nrChildCol() ; /* Row blocks of the product */
    int nbCols = transHB == 'N' ? hb->nrChildCol() : hb->nrChildRow() ; /* Col blocks of the product */
    int nbCom  = transHA == 'N' ? ha->nrChildCol() : ha->nrChildRow() ; /* Common dimension between A and B */
    RkMatrix<T>* subRks[nbRows * nbCols];
    for (int i = 0; i < nbRows; i++) {
      for (int j = 0; j < nbCols; j++) {
        subRks[i + j * nbRows]=(RkMatrix<T>*)NULL;
        for (int k = 0; k < nbCom; k++) {
          // C_ij = A_ik * B_kj
          const HMatrix<T>* a_ik = (transHA == 'N' ? ha->get(i, k) : ha->get(k, i));
          const HMatrix<T>* b_kj = (transHB == 'N' ? hb->get(k, j) : hb->get(j, k));
          if (a_ik && b_kj) {
            if (subRks[i + j * nbRows]==NULL) {
              const IndexSet* subRows = (transHA == 'N' ? a_ik->rows() : a_ik->cols());
              const IndexSet* subCols = (transHB == 'N' ? b_kj->cols() : b_kj->rows());
              subRks[i + j * nbRows] = new RkMatrix<T>(NULL, subRows, NULL, subCols, NoCompression);
            }
            subRks[i + j * nbRows]->gemmRk(transHA, transHB, alpha, a_ik, b_kj, beta);
          }
        } // k loop
      } // j loop
    } // i loop
    // Reconstruction of C by adding the parts
    std::vector<T> alphaV(nbRows * nbCols, Constants<T>::pone);
    RkMatrix<T>* rk = formattedAddParts(&alphaV[0], (const RkMatrix<T>**) subRks, nbRows * nbCols);
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
        rk = truncatedSvd(fullMat); // TODO compress with something else than SVD
        delete fullMat;
      }
    }
    if(rk) {
      axpy(alpha, rk);
      delete rk;
    }
  }
}

template<typename T> void RkMatrix<T>::copy(const RkMatrix<T>* o) {
  delete a;
  delete b;
  rows = o->rows;
  cols = o->cols;
  a = (o->a ? o->a->copy() : NULL);
  b = (o->b ? o->b->copy() : NULL);
}

template<typename T> RkMatrix<T>* RkMatrix<T>::copy() const {
  RkMatrix<T> *result = new RkMatrix<T>(NULL, rows, NULL, cols, this->method);
  result->copy(this);
  return result;
}


template<typename T> void RkMatrix<T>::checkNan() const {
  if (rank() == 0) {
    return;
  }
  a->checkNan();
  b->checkNan();
}

template<typename T> void RkMatrix<T>::conjugate() {
  if (a) a->conjugate();
  if (b) b->conjugate();
}

template<typename T> T RkMatrix<T>::get(int i, int j) const {
  return a->dot_aibj(i, *b, j);
}

template<typename T> void RkMatrix<T>::writeArray(hmat_iostream writeFunc, void * userData) const{
  a->writeArray(writeFunc, userData);
  b->writeArray(writeFunc, userData);
}

// Templates declaration
template class RkMatrix<S_t>;
template class RkMatrix<D_t>;
template class RkMatrix<C_t>;
template class RkMatrix<Z_t>;

}  // end namespace hmat
