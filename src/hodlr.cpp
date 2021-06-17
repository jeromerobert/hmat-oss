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
  auto m00 = m->get(0,0);
  auto m11 = m->get(1,1);
  auto m10 = m->get(1,0);
  if(m00->isLeaf()) {
    m00->lltDecomposition(nullptr);
    m11->lltDecomposition(nullptr);
  } else {
    factorize(m00, node->child0);
    factorize(m11, node->child1);
  }
  desymmetrize(m);
  auto m01 = m->get(0,1);
  solve(m00, m01, node->child0);
  solve(m11, m10, node->child1);
  int r0 = m10->rk()->rank();
  int r1 = m01->rk()->rank();
  ScalarArray<T> tb0a1(node->kk, r1, r0, 0, r1);
  ScalarArray<T> tb1a0(node->kk, 0, r1, r1, r0);
  tb0a1.gemm('T', 'N', 1, m10->rk()->b, m01->rk()->a, 0);
  tb1a0.gemm('T', 'N', 1, m01->rk()->b, m10->rk()->a, 0);
  for(int i = 0; i < node->kk.rows; i++) {
    node->kk.get(i, i) = 1;
  }
  node->kk.luDecomposition(node->pivot);
}
}
namespace hmat {

/**
 * The Woodbury matrix identity gives (I+tU.V)^-1 = I - U.(I+tV.U)^-1.tV
 * and we need to store kk = (I+tV.U)^-1. This struct is node tree to
 * store kk at each level of the HODLR tree.
 */
template<typename T> struct HODLRNode {
  ScalarArray<T> kk;
  /** The pivot after the LU factorization of kk */
  int * pivot;
  HODLRNode *child0, *child1;

  HODLRNode(int n): kk(n, n), pivot(new int[n]) {}

  ~HODLRNode() { delete[] pivot; }

  static HODLRNode<T> * create(HMatrix<T> * m) {
    if(m->isLeaf()) {
      return nullptr;
    } else {
      int n = 2 * m->get(1,0)->rank();
      HODLRNode * r = new HODLRNode(n);
      r->child0 = create(m->get(0,0));
      r->child1 = create(m->get(1,1));
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
  root = HODLRNode<T>::create(m);
  ::factorize(m, root);
}

template class HODLR<S_t>;
template class HODLR<D_t>;
template class HODLR<C_t>;
template class HODLR<Z_t>;
}
