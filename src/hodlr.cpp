/*
  HMat-OSS (HMatrix library, open source software)

  Copyright (C) 2021 Airbus SAS

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
#include "hodlr.hpp"
#include "h_matrix.hpp"

namespace {
using namespace hmat;
template<typename T> void desymmetrize(HMatrix<T> * m) {
  auto m10 = m->get(1,0);
  auto m01 = m->internalCopy(m10->colsTree(), m10->rowsTree());
  m01->rk(nullptr);
  m01->copyAndTranspose(m10);
  m->insertChild(0, 1, m01);
}

template<typename T>
void solve(HMatrix<T> * const m, ScalarArray<T> *x, int xOffset, const HODLRNode<T> * node) {
  if(m->isLeaf()) {
    ScalarArray<T> xc(*x, m->cols()->offset() - xOffset, m->cols()->size(), 0, x->cols);
    m->solveLlt(&xc);
    return;
  }
  solve(m->get(0, 0), x, xOffset, node->child0);
  solve(m->get(1, 1), x, xOffset, node->child1);
  // Compute (I-U.KK.tV).X that is X = X - U.KK.tV.X where
  // U and V are so that tU.V is equal to the 2 anti-diagonal blocks of m which
  // are by (HODLR) definition Rk. U and V are not explicitly created to avoid useless copy.
  // U is made from m10 and m01 rk->a arrays and V is made from m10 and m01 rk->b arrays.
  auto m10 = m->get(1,0);
  auto m01 = m->get(0,1);
  ScalarArray<T> tvx(node->kk.rows, x->cols);
  ScalarArray<T> tvx0(tvx, 0, m01->rank(), 0, x->cols);
  ScalarArray<T> tvx1(tvx, m01->rank(), m10->rank(), 0, x->cols);
  ScalarArray<T> x0(*x, m10->cols()->offset() - xOffset, m10->cols()->size(), 0, x->cols);
  ScalarArray<T> x1(*x, m01->cols()->offset() - xOffset, m01->cols()->size(), 0, x->cols);
  // Compute tV.X in tvx0 and tvx1
  tvx0.gemm('T', 'N', 1, m01->rk()->b, &x1, 0);
  tvx1.gemm('T', 'N', 1, m10->rk()->b, &x0, 0);
  // compute KK.(tV.Xa)
  FactorizationData<T> fd = { Factorization::LU, { node->pivot }};
  node->kk.solve(&tvx, fd);
  // Compute Xa = Xa - U.(KK.tV.Xa)
  x0.gemm('N', 'N', -1, m01->rk()->a, &tvx0, 1);
  x1.gemm('N', 'N', -1, m10->rk()->a, &tvx1, 1);
}

template<typename T>
void solve(HMatrix<T> * const m, HMatrix<T> *x, const HODLRNode<T> * node) {
  solve(m, x->rk()->a, x->rows()->offset(), node);
}

template<typename T>
void factorize(HMatrix<T> * m, HODLRNode<T> * node) {
  HMAT_ASSERT_MSG(!m->isLeaf(), "Not HODLR matrix");
  // | m00     |   | m00^-1        |   | I    Rk1 |
  // | m01 m11 | = |        m11^-1 | * | Rk0  I   |
  // The second factor can be written as U.V^t+I where U and V
  // are tall and skinny. (U.V^t+I)^1 = I - U.(I+V^t.U)^-1.V^t (Woodbury).
  auto m00 = m->get(0,0);
  auto m11 = m->get(1,1);
  auto m10 = m->get(1,0);
  HMAT_ASSERT_MSG(m10->isRkMatrix(), "Not HODLR matrix");
  HMAT_ASSERT_MSG(m->get(0,1) == nullptr, "Not lowered stored matrix");
  if(m00->isLeaf()) {
    m00->lltDecomposition(nullptr);
  } else {
    factorize(m00, node->child0);
  }
  if(m11->isLeaf()) {
    m11->lltDecomposition(nullptr);
  } else {
    factorize(m11, node->child1);
  }
  desymmetrize(m);
  auto m01 = m->get(0,1);
  solve(m00, m01, node->child0);
  solve(m11, m10, node->child1);
  int r0 = m10->rk()->rank();
  int r1 = m01->rk()->rank();
  // Compute kk=(I+V^t.U)^-1
  ScalarArray<T> tb0a1(node->kk, r1, r0, 0, r1);
  ScalarArray<T> tb1a0(node->kk, 0, r1, r1, r0);
  tb0a1.gemm('T', 'N', 1, m10->rk()->b, m01->rk()->a, 0);
  tb1a0.gemm('T', 'N', 1, m01->rk()->b, m10->rk()->a, 0);
  node->kk.addIdentity(1);
  node->kk.luDecomposition(node->pivot);
}

template<typename T>
void solveUpperTriangularLeft(HMatrix<T> * const m, ScalarArray<T> *x, int xOffset, const HODLRNode<T> * node) {
  if(m->isLeaf()) {
    ScalarArray<T> xc(*x, m->cols()->offset() - xOffset, m->cols()->size(), 0, x->cols);
    m->solveUpperTriangularLeft(&xc, Factorization::LLT, Diag::NONUNIT, Uplo::LOWER);
    return;
  }
  auto m10 = m->get(1,0);
  ScalarArray<T> x0(*x, m10->cols()->offset() - xOffset, m10->cols()->size(), 0, x->cols);
  ScalarArray<T> x1(*x, m10->rows()->offset() - xOffset, m10->rows()->size(), 0, x->cols);
  ScalarArray<T> tax1(m10->rank(), x->cols, false);
  tax1.gemm('T', 'N', 1, m10->rk()->a, &x1, 0);
  ScalarArray<T> tkktax1(m10->rank(), x->cols, false);
  tkktax1.gemm('T', 'N', 1, &node->kk, &tax1, 0);
  x1.gemm('N', 'N', -1, m10->rk()->a, &tkktax1, 1);
  m10->gemv('T', -1, &x1, 1, &x0);
  solveUpperTriangularLeft(m->get(0, 0), x, xOffset, node->child0);
  solveUpperTriangularLeft(m->get(1, 1), x, xOffset, node->child1);
}

template<typename T>
void solveLowerTriangularLeft(HMatrix<T> * const m, ScalarArray<T> *x, int xOffset, const HODLRNode<T> * node) {
  if(m->isLeaf()) {
    ScalarArray<T> xc(*x, m->cols()->offset() - xOffset, m->cols()->size(), 0, x->cols);
    m->solveLowerTriangularLeft(&xc, Factorization::LLT, Diag::NONUNIT, Uplo::LOWER);
    return;
  }
  solveLowerTriangularLeft(m->get(0, 0), x, xOffset, node->child0);
  solveLowerTriangularLeft(m->get(1, 1), x, xOffset, node->child1);
  auto m10 = m->get(1,0);
  ScalarArray<T> x0(*x, m10->cols()->offset() - xOffset, m10->cols()->size(), 0, x->cols);
  ScalarArray<T> x1(*x, m10->rows()->offset() - xOffset, m10->rows()->size(), 0, x->cols);
  m10->gemv('N', -1, &x0, 1, &x1);
  ScalarArray<T> tax1(m10->rank(), x->cols, false);
  tax1.gemm('T', 'N', 1, m10->rk()->a, &x1, 0);
  ScalarArray<T> kkax1(m10->rank(), x->cols, false);
  kkax1.gemm('N', 'N', 1, &node->kk, &tax1, 0);
  x1.gemm('N', 'N', -1, m10->rk()->a, &kkax1, 1);
}

template<typename T>
T logdet(HMatrix<T> * a, HODLRNode<T> * node) {
  HMAT_ASSERT_MSG(!a->isLeaf(), "Not HODLR matrix");
  auto a00 = a->get(0,0);
  auto a11 = a->get(1,1);
  T result = a00->isLeaf() ? a00->logdet() : logdet(a00, node->child0);
  result += a11->isLeaf() ? a11->logdet() : logdet(a11, node->child1);
  result += std::log(node->detm11_);
  return result;
}

template<typename T>
void factorizeSym(HMatrix<T> * a, HODLRNode<T> * node) {
  HMAT_ASSERT_MSG(!a->isLeaf(), "Not HODLR matrix");
  // |a00    |   |P00   |   |I   Rk1|   |P00^t     |
  // |a01 a11| = |   P11| * |Rk0 I  | * |     P11^t|
  // Let's write the second factor as I+U.K.U^t, with
  // K00=K11=0 and K01=K10=I and U a tall and skinny matrix
  auto a00 = a->get(0,0);
  auto a11 = a->get(1,1);
  auto a10 = a->get(1,0);
  HMAT_ASSERT_MSG(a10->isRkMatrix(), "Not HODLR matrix");
  HMAT_ASSERT_MSG(a->get(0,1) == nullptr, "Not lowered stored matrix");
  int r = a10->rank();
  if(a00->isLeaf()) {
    a00->lltDecomposition(nullptr);
  } else {
    factorizeSym(a00, node->child0);
  }
  if(a11->isLeaf()) {
    a11->lltDecomposition(nullptr);
  } else {
    factorizeSym(a11, node->child1);
  }
  solveLowerTriangularLeft(a00, a10->rk()->b, a10->cols()->offset(), node->child0);
  solveLowerTriangularLeft(a11, a10->rk()->a, a10->rows()->offset(), node->child1);
  // Cholesky like factorization of I+U.K.U^t:
  // I+U.K.U^t = (I + U.X.U^t).(I + U.X.U^t)^t = Q.Q^t where
  // X=L^-t.(M − I).L^-1 where M.M^t = I + L^t.K.L and L.L^t = U^t.U
  // We have X00=0, X01=0, X10=I. We just need to compute X11.
  // Compute U^t.U
  ScalarArray<T> ata(r, r, false);
  ata.gemm('T', 'N', 1, a10->rk()->a, a10->rk()->a, 0);
  ScalarArray<T> la(r, r, false);
  la.copyMatrixAtOffset(&ata, 0, 0);
  // Compute L11 (we don't need L00)
  la.lltDecomposition();
  // Compute I + L^t.K.L in node->x11
  node->x11.gemm('T', 'N', -1, a10->rk()->b, a10->rk()->b, 0);
  node->x11.trmm(Side::RIGHT, Uplo::LOWER, 'N', Diag::NONUNIT, 1, &la);
  node->x11.trmm(Side::LEFT, Uplo::LOWER, 'T', Diag::NONUNIT, 1, &la);
  node->x11.addIdentity(1);
  // Compute M11 in node->x11
  node->x11.lltDecomposition();
  // We have det(I+U.K.U^t) = det(M11)^2
  node->detm11_ = node->x11.diagonalProduct();
  node->x11.addIdentity(-1);
  // Compute X11
  node->x11.trsm(Side::RIGHT, Uplo::LOWER, 'N', Diag::NONUNIT, 1, &la);
  node->x11.trsm(Side::LEFT, Uplo::LOWER, 'T', Diag::NONUNIT, 1, &la);
  // Q^-1 = I - U.(I+X.U^t.U)^-1.X.tV (Woodbury identity)
  // We store (I+X.U^t.U)^-1.X to node->kk to be able to solve later on.
  // Q00 = I so we just need to store what is needed to compute Q11^-1
  ScalarArray<T> & ixutu = la; // reuse la (aka L11) storage
  ixutu.gemm('N', 'N', 1, &node->x11, &ata, 0);
  ixutu.addIdentity(1);
  int * pivots = new int[r];
  ixutu.luDecomposition(pivots);
  node->kk.copyMatrixAtOffset(&node->x11, 0, 0);
  FactorizationData<T> fd = { Factorization::LU, { pivots }};
  ixutu.solve(&node->kk, fd);
  delete[] pivots;
}

template<typename T> void gemv(char trans, T alpha, HMatrix<T> * const ma, const ScalarArray<T> & x,
                               T beta, ScalarArray<T> & y, HODLRNode<T> * node, int offset) {
  if(ma->isLeaf()) {
    ma->gemv(trans, alpha, &x, beta, &y);
    return;
  }
  auto ma10 = ma->get(1,0);
  int r = ma10->rank();
  const ScalarArray<T> & a = *ma->get(1,0)->rk()->a;
  const ScalarArray<T> & b = *ma->get(1,0)->rk()->b;
  int offset0 = ma10->cols()->offset();
  int offset1 = ma10->rows()->offset();
  const ScalarArray<T> x0(x, offset0 - offset, ma10->cols()->size(), 0, x.cols);
  const ScalarArray<T> x1(x, offset1 - offset, ma10->rows()->size(), 0, x.cols);
  ScalarArray<T> y0(y, offset0 - offset, ma10->cols()->size(), 0, y.cols);
  ScalarArray<T> y1(y, offset1 - offset, ma10->rows()->size(), 0, y.cols);
  if(trans == 'N') {
    // |y0|    |L0  |   |I       |   |x0|
    // |y1| += |  L1| x |a.bT Q11| x |x1|
    // Q11 = I + a.X11.aT
    // y1 += L1.(x1 + a.(bT.x0 + X11.aT.x1))
    ScalarArray<T> * atx1 = new ScalarArray<T>(r, x.cols, false);
    ScalarArray<T> x11atx1(r, x.cols, false);
    atx1->gemm('T', 'N', 1, &a, &x1, 0);
    x11atx1.gemm('N', 'N', 1, &node->x11, atx1, 0);
    delete atx1;
    // x11atx1 <- bT.x0 + X11.aT.x1
    x11atx1.gemm('T', 'N', 1, &b, &x0, 1);
    ScalarArray<T> x1bis(x1.rows, x1.cols, false);
    x1bis.copyMatrixAtOffset(&x1, 0, 0);
    // x1bis = x1 + a.(bT.x0 + X11.aT.x1)
    x1bis.gemm('N', 'N', 1, &a, &x11atx1, 1);
    gemv(trans, alpha, ma->get(1,1), x1bis, beta, y1, node->child1, offset1);
    gemv(trans, alpha, ma->get(0,0), x0, beta, y0, node->child0, offset0);
  } else {
    // |y0|    |I b.a^T|   |L0^T    |   |x0|
    // |y1| += |  Q11^T| x |    L1^T| x |x1|
    ScalarArray<T> l1tx1(x1.rows, x1.cols, false);
    // y0 <- beta*y0 + alpha*L0^T*x0
    gemv(trans, alpha, ma->get(0,0), x0, beta, y0, node->child0, offset0);
    gemv<T>(trans, alpha, ma->get(1,1), x1, 0, l1tx1, node->child1, offset1);
    ScalarArray<T> atl1tx1(r, x1.cols, false);
    ScalarArray<T> x11atl1tx1(r, x1.cols, false);
    atl1tx1.gemm('T', 'N', 1, &a, &l1tx1, 0);
    // y0 += alpha.b.aT.L1^T.x1
    y0.gemm('N', 'N', alpha, &b, &atl1tx1, 1);
    x11atl1tx1.gemm('T', 'N', 1, &node->x11, &atl1tx1, 0);
    // y1 = beta*y1 + alpha*a.X11^t.aT.L1^t.x1
    y1.gemm('N', 'N', alpha, &a, &x11atl1tx1, beta);
    // y1 += alpha*L1^t*x1
    y1.axpy(alpha, &l1tx1);
  }
}

}
namespace hmat {

/**
 * The Woodbury matrix identity gives (I+U.V^t)^-1 = I - U.(I+V^t.U)^-1.V^t
 * and we need to store kk = (I+tV.U)^-1. This struct is node tree to
 * store kk at each level of the HODLR tree.
 */
template<typename T> struct HODLRNode {
  ScalarArray<T> x11, kk;
  /**
   * The pivot after the LU factorization of kk.
   * This is only relevent for non-symmetric factorization. In the case
   * of symmetric factorization this is always nullptr.
   */
  int * pivot;
  HODLRNode *child0, *child1;
  /**
   * The determinant of M11 which is also the determinant of Q in the
   * symmetric decomposition (A=L.Q.Q^t.L^t)
   */
  T detm11_;

  HODLRNode(int n, int x11n): x11(x11n, x11n),
    kk(n, n), pivot(x11n == 0 ? new int[n] : nullptr), detm11_(0) {}

  ~HODLRNode() {
    delete[] pivot;
    delete child0;
    delete child1;
  }

  bool isSymmetric() {
    assert((x11.rows > 0) == (pivot == nullptr));
    return x11.rows > 0;
  }

  static HODLRNode<T> * create(HMatrix<T> * m, bool sym) {
    if(m->isLeaf()) {
      return nullptr;
    } else {
      int n = m->get(1,0)->rank();
      HODLRNode * r = sym ? new HODLRNode(n, n) : new HODLRNode(2 * n, 0);
      r->child0 = create(m->get(0,0), sym);
      r->child1 = create(m->get(1,1), sym);
      return r;
    }
  }
};

template<typename T>
void HODLR<T>::solve(HMatrix<T> * const m, HMatrix<T> *x) const {
  ::solve(m, x, root);
}

template<typename T>
void HODLR<T>::solve(HMatrix<T> * const m, ScalarArray<T> &x) const {
  ::solve(m, &x, 0, root);
}

template<typename T>
void HODLR<T>::factorize(HMatrix<T> * m, hmat_progress_t* p) {
  root = HODLRNode<T>::create(m, false);
  ::factorize(m, root);
}

template<typename T>
void HODLR<T>::factorizeSym(HMatrix<T> * m, hmat_progress_t* p) {
  root = HODLRNode<T>::create(m, true);
  ::factorizeSym(m, root);
}

template<typename T>
void HODLR<T>::solveSym(HMatrix<T> * const m, ScalarArray<T> &x) const {
  assert(x.rows == m->rows()->size());
  assert(m->cols()->size() == m->rows()->size());
  ::solveLowerTriangularLeft(m, &x, 0, root);
  ::solveUpperTriangularLeft(m, &x, 0, root);
}

template<typename T> void HODLR<T>::gemv(char trans, T alpha, HMatrix<T> * const a, ScalarArray<T> & x, T beta, ScalarArray<T> & y) const {
  HMAT_ASSERT_MSG(root != nullptr && root->isSymmetric(), "gemv is only supported for symmetrically factorized HODLR matrices");
  HMAT_ASSERT(trans == 'N' || trans == 'T');
  HMAT_ASSERT(x.cols == y.cols);
  HMAT_ASSERT(x.rows == y.rows);
  ::gemv(trans, alpha, a, x, beta, y, root, 0);
}

template<typename T> bool HODLR<T>::isFactorized() const {
  return root != nullptr;
}

template<typename T> T HODLR<T>::logdet(HMatrix<T> * const m) const {
  HMAT_ASSERT_MSG(root != nullptr && root->isSymmetric(), "logdet is only supported for symmetrically factorized HODLR matrices");
  return ::logdet(m, root);
}

template<typename T> HODLR<T>::~HODLR() {
  delete root;
}

template class HODLR<S_t>;
template class HODLR<D_t>;
template class HODLR<C_t>;
template class HODLR<Z_t>;
}
