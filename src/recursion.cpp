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

#include "config.h"

/*! \file
  \ingroup HMatrix
  \brief Templated Matrix class for recursive algorithms used by HMatrix.
*/

#include "recursion.hpp"
#include "h_matrix.hpp"
#include "common/context.hpp"

namespace hmat {

  template<typename T, typename Mat>
  void RecursionMatrix<T, Mat>::recursiveLdltDecomposition(hmat_progress_t * progress) {

    //  Recursive LDLT factorization:
    //
    //  [ h11 |th21 ]    [ L11 |  0  ]   [ D1 | 0  ]   [ tL11 | tL21 ]
    //  [ ----+---- ]  = [ ----+---- ] * [----+----] * [------+------]
    //  [ h21 | h22 ]    [ L21 | L22 ]   [ 0  | D2 ]   [   0  | tL22 ]
    //
    //  h11 = L11 * D1 * tL11 (gives L11 and D1 by LDLT decomposition)
    //  h21 = L21 * D1 * tL11 (gives L21 when L11 and D1 are known)
    //  h22 = L21 * D1 * tL21 + L22 * D2 * tL22 (gives L22 and D2 by LDLT decomposition of h22-L21*D1*tL21)

    // for all i, j<=i : hij = sum_k Lik Dk t{Ljk} k<=i,j
    // The algorithm loops over 3 steps: for all k=1, 2, ..., n
    //   - We factorize the element (k,k)
    //   - We use "solve" to compute the rest of the column 'k'
    //   - We update the rest of the matrix [k+1, .., n]x[k+1, .., n] (below diag)

    HMAT_ASSERT_MSG(me()->nrChildRow()==me()->nrChildCol(),
                    "RecursionMatrix<T, Mat>::recursiveLdltDecomposition: case not allowed "
                    "Nr Child A[%d, %d] Dimensions A=%s ",
                    me()->nrChildRow(), me()->nrChildCol(), me()->description().c_str());

    for (int k=0 ; k<me()->nrChildRow() ; k++) {
      // Hkk <- Lkk * Dk * tLkk
      me()->get(k,k)->ldltDecomposition(progress);
      // Solve the rest of column k: solve Lik Dk tLkk = Hik and get Lik
      for (int i=k+1 ; i<me()->nrChildRow() ; i++) {
        if (!me()->get(i,k))
          continue;
        me()->get(k,k)->solveUpperTriangularRight(me()->get(i,k), false, true);
        me()->get(i,k)->multiplyWithDiag(me()->get(k,k), false, true);
      }
      // update the rest of the matrix [k+1, .., n]x[k+1, .., n] (below diag)
      for (int i=k+1 ; i<me()->nrChildRow() ; i++) {
        if (!me()->get(i,k))
          continue;
        // Hij <- Hij - Lik Dk tLjk
        // if i=j, we can use mdmtProduct, otherwise we use mdntProduct
        for (int j=k+1 ; j<i ; j++)
          if (me()->get(i,k) && me()->get(j,k))
            me()->get(i,j)->mdntProduct(me()->get(i,k), me()->get(k,k), me()->get(j,k)); // hij -= Lik.Dk.tLjk
        me()->get(i,i)->mdmtProduct(me()->get(i,k), me()->get(k,k)); //  hii -= Lik.Dk.tLik
      }
    }

  }

  template<typename T, typename Mat>
  void RecursionMatrix<T, Mat>::recursiveSolveUpperTriangularRight(Mat* b, bool unitriangular, bool lowerStored) const {

    //  Backward substitution:
    //  [ X11 | X12 ]    [ U11 | U12 ]   [ b11 | b12 ]
    //  [ ----+---- ] *  [-----+-----] = [ ----+---- ]
    //  [ X21 | X22 ]    [  0  | U22 ]   [ b21 | b22 ]
    //
    //  X11 * U11 = b11 (by recursive backward substitution)
    //  X21 * U11 = b21 (by recursive backward substitution)
    //  X11 * U12 + X12 * U22 = b12 (backward substitution of X12*U22=b12-X11*U12)
    //  X21 * U12 + X22 * U22 = b22 (backward substitution of X22*U22=b22-X21*U12)
    // X and b are not necessarily square

    // First we handle the general case, where dimensions (in terms of number of children) are compatible
    if (me()->nrChildRow() == b->nrChildCol()) {

      for (int k=0 ; k<b->nrChildRow() ; k++) // loop on the lines of b
        for (int i=0 ; i<me()->nrChildRow() ; i++) {
          // Update b[k,i] with the contribution of the solutions already computed b[k,j] j<i
          for (int j=0 ; j<i ; j++)
            if (b->get(k, j) && (lowerStored ? me()->get(i,j) : me()->get(j,i)))
              b->get(k, i)->gemm('N', lowerStored ? 'T' : 'N', Constants<T>::mone, b->get(k, j), lowerStored ? me()->get(i,j) : me()->get(j,i), Constants<T>::pone);
          // Solve the i-th diagonal system
          if (b->get(k,i))
          me()->get(i, i)->solveUpperTriangularRight(b->get(k,i), unitriangular, lowerStored);
        }

    } else if (me()->nrChildRow()>1 && b->nrChildCol()==1 && b->nrChildRow()>1) {
      // Then we handle the specific case where me() is divided in rows, and b is not divided in cols: we recurse on b's children only
      //  [    X11    ]   [ U11 | U12 ]   [    b11    ]
      //  [ --------- ] * [ ----+---- ] = [ --------- ]
      //  [    X21    ]   [  0  | U22 ]   [    b21    ]
      for (int k=0 ; k<b->nrChildRow() ; k++) // loop on the rows of b
        me()->recursiveSolveUpperTriangularRight(b->get(k,0), unitriangular, lowerStored);

    } else {
      HMAT_ASSERT_MSG(false, "RecursionMatrix<T, Mat>::recursiveSolveUpperTriangularRight: case not yet handled "
                             "Nr Child A[%d, %d] b[%d, %d] "
                             "Dimensions A=%s b=%s",
                      me()->nrChildRow(), me()->nrChildCol(), b->nrChildRow(), b->nrChildCol(),
                      me()->description().c_str(), b->description().c_str());

    }

  }

  template<typename T, typename Mat>
  void RecursionMatrix<T, Mat>::recursiveMdmtProduct(const Mat* m, const Mat* d) {

    //
    //  [ h11 |th21 ]    [ M11 | M12 ]   [ D1 | 0  ]   [ tM11 | tM21 ]
    //  [ ----+---- ] -= [ ----+---- ] * [----+----] * [------+------]
    //  [ h21 | h22 ]    [ M21 | M22 ]   [ 0  | D2 ]   [ tM12 | tM22 ]
    //
    //  h11 -= M11.D1.tM11 + M12.D2.tM12
    //  h21 -= M21.D1.tM11 + M22.D2.tM12
    //  h22 -= M21.D1.tM21 + M22.D2.tM22
    //
    //  hij -= sum_k Mik.Dk.tMjk

    // First we handle the general case, where dimensions (in terms of number of children) are compatible.
    // recursiveMdmtProduct is called when neither this and m are leaves, but d may be a leaf.
    const int block_row_d = d->isLeaf() ? 1 : d->nrChildRow();
    const int block_col_d = d->isLeaf() ? 1 : d->nrChildCol();
    if (me()->nrChildRow()==me()->nrChildCol() && block_row_d==block_col_d && me()->nrChildRow()==m->nrChildRow() && m->nrChildCol()==block_row_d) {
      if (d->isLeaf()) {
        //  hij -= Mik.Dk.tMjk : if i=j, we use mdmtProduct. Otherwise, mdntProduct
        for(int i=0 ; i<me()->nrChildRow() ; i++) {
          if (!m->get(i,0))
            continue;
          for (int j=0 ; j<i ; j++)
            if (m->get(j,0))
              me()->get(i,j)->mdntProduct(m->get(i,0), d, m->get(j,0)); // hij -= Mi0.D.tMj0
          me()->get(i,i)->mdmtProduct(m->get(i,0),  d);               // hii -= Mi0.D.tMi0
        }
      } else {
        //  hij -= Mik.Dk.tMjk : if i=j, we use mdmtProduct. Otherwise, mdntProduct
        for(int i=0 ; i<me()->nrChildRow() ; i++)
          for(int k=0 ; k<m->nrChildCol() ; k++) {
            const Mat* m_ik = m->get(i,k);
            if (!m_ik)
              continue;
            const Mat* d_k = d->get(k,k);
            for (int j=0 ; j<i ; j++)
              if (m->get(j,k))
                me()->get(i,j)->mdntProduct(m_ik, d_k, m->get(j,k)); // hij -= Mik.Dk.tMjk
            me()->get(i,i)->mdmtProduct(m_ik, d_k); //  hii -= Mik.Dk.tMik
          }
      }
    } else {
      HMAT_ASSERT_MSG(false, "RecursionMatrix<T, Mat>::recursiveMdmtProduct: case not yet handled "
                             "Nr Child this[%d, %d] m[%d, %d] d[%d, %d]"
                             "Dimensions this=%s m=%s d=%s",
                      me()->nrChildRow(), me()->nrChildCol(), m->nrChildRow(), m->nrChildCol(),
                      d->nrChildRow(), d->nrChildCol(),
                      me()->description().c_str(), m->description().c_str(), d->description().c_str());

    }
  }

  template<typename T, typename Mat>
  void RecursionMatrix<T, Mat>::recursiveSolveLowerTriangularLeft(Mat* b, bool unitriangular, MainOp mainOp) const {

    //  Forward substitution:
    //  [ L11 |  0  ]    [ X11 | X12 ]   [ b11 | b12 ]
    //  [ ----+---- ] *  [-----+-----] = [ ----+---- ]
    //  [ L21 | L22 ]    [ X21 | X22 ]   [ b21 | b22 ]
    //
    //  L11 * X11 = b11 (by recursive forward substitution)
    //  L11 * X12 = b12 (by recursive forward substitution)
    //  L21 * X11 + L22 * X21 = b21 (forward substitution of L22*X21=b21-L21*X11)
    //  L21 * X12 + L22 * X22 = b22 (forward substitution of L22*X22=b22-L21*X12)
    // X and b are not necessarily square

    // First we handle the general case, where dimensions (in terms of number of children) are compatible
    if (me()->nrChildCol() == b->nrChildRow()) {

      for (int k=0 ; k<b->nrChildCol() ; k++) // loop on the column of b
        for (int i=0 ; i<me()->nrChildRow() ; i++) {
          // Update b[i,k] with the contribution of the solutions already computed b[j,k] j<i
          if (!b->get(i, k)) continue;
          for (int j=0 ; j<i ; j++)
            if (me()->get(i,j) && b->get(j,k))
              b->get(i, k)->gemm('N', 'N', Constants<T>::mone, me()->get(i, j), b->get(j,k),
                                 Constants<T>::pone, mainOp);
          // Solve the i-th diagonal system
          me()->get(i, i)->solveLowerTriangularLeft(b->get(i,k), unitriangular, mainOp);
        }

    } else if (me()->nrChildCol()>1 && b->nrChildRow()==1 && b->nrChildCol()>1) {
      // Then we handle the specific case where me() is divided in columns, and b is not divided in rows: we recurse on b's children only
      //  [ L11 |  0  ]    [     |     ]   [     |     ]
      //  [ ----+---- ] *  [ X11 | X12 ] = [ b11 | b12 ]
      //  [ L21 | U22 ]    [     |     ]   [     |     ]
      for (int k=0 ; k<b->nrChildCol() ; k++) // loop on the column of b
        me()->recursiveSolveLowerTriangularLeft(b->get(0,k), unitriangular, mainOp);

    } else {
      HMAT_ASSERT_MSG(false, "RecursionMatrix<T, Mat>::recursiveSolveLowerTriangularLeft: case not yet handled "
                             "Nr Child A[%d, %d] b[%d, %d] "
                             "Dimensions A=%s b=%s",
                      me()->nrChildRow(), me()->nrChildCol(), b->nrChildRow(), b->nrChildCol(),
                      me()->description().c_str(), b->description().c_str());

    }
  }

  template<typename T, typename Mat>
  void RecursionMatrix<T, Mat>::recursiveLuDecomposition(hmat_progress_t * progress) {

    // |     |     |    |     |     |   |     |     |
    // | h11 | h12 |    | L11 |     |   | U11 | U12 |
    // |-----|-----| =  |-----|-----| * |-----|-----|
    // | h21 | h22 |    | L21 | L22 |   |     | U22 |
    // |     |     |    |     |     |   |     |     |
    //
    // h11 = L11 * U11 => (L11,U11) = h11.luDecomposition
    // h12 = L11 * U12 => trsm L
    // h21 = L21 * U11 => trsm R
    // h22 = L21 * U12 + L22 * U22 => (L22,U22) = (h22 - L21*U12).luDecomposition()
    //
    // hij = sum_k Lik Ukj k<=i,j
    // The algorithm loops over 3 steps: for all k=1, 2, ..., n
    //   - We factorize the element (k,k)
    //   - We use "solve" to compute the rest of the columns and rows 'k'
    //   - We update the rest of the matrix [k+1, .., n]x[k+1, .., n]

    HMAT_ASSERT_MSG(me()->nrChildRow()==me()->nrChildCol(),
                    "RecursionMatrix<T, Mat>::recursiveLuDecomposition: case not allowed "
                    "Nr Child A[%d, %d] Dimensions A=%s ",
                    me()->nrChildRow(), me()->nrChildCol(), me()->description().c_str());

    for (int k=0 ; k<me()->nrChildRow() ; k++) {
      // Hkk <- Lkk * Ukk
      me()->get(k,k)->luDecomposition(progress);
      // Solve the rest of line k: solve Lkk Uki = Hki and get Uki
      for (int i=k+1 ; i<me()->nrChildRow() ; i++)
        if (me()->get(k,k) && me()->get(k,i))
          me()->get(k,k)->solveLowerTriangularLeft(me()->get(k,i), true);
      // Solve the rest of column k: solve Lik Ukk = Hik and get Lik
      for (int i=k+1 ; i<me()->nrChildRow() ; i++)
        if (me()->get(k,k) && me()->get(i,k))
          me()->get(k,k)->solveUpperTriangularRight(me()->get(i,k), false, false);
      // update the rest of the matrix starting at (k+1, k+1)
      for (int i=k+1 ; i<me()->nrChildRow() ; i++)
        for (int j=k+1 ; j<me()->nrChildRow() ; j++)
          // Hij <- Hij - Lik Ukj
          if (me()->get(i,k) && me()->get(k,j))
          me()->get(i,j)->gemm('N', 'N', Constants<T>::mone, me()->get(i,k), me()->get(k,j), Constants<T>::pone);
    }

  }

  template<typename T, typename Mat>
  void RecursionMatrix<T, Mat>::recursiveInverseNosym() {

    //  Matrix inversion:
    //  The idea to inverse M is to consider the extended matrix obtained by putting Identity next to M :
    //
    //  [ M11 | M12 |  I  |  0  ]
    //  [ ----+-----+-----+---- ]
    //  [ M21 | M22 |  0  |  I  ]
    //
    //  We then apply operations on the line of this matrix (matrix multiplication of an entire line,
    // linear combination of lines)
    // to transform the 'M' part into identity. Doing so, the identity part will at the end contain M-1.
    // We loop on the column of M.
    // At the end of loop 'k', the 'k' first columns of 'M' are now identity,
    // and the 'k' first columns of Identity have changed (it's no longer identity, it's not yet M-1).
    // The matrix 'this' stores the first 'k' block of the identity part of the extended matrix, and the last n-k blocks of the M part
    // At the end, 'this' contains M-1

    HMAT_ASSERT_MSG(me()->nrChildRow()==me()->nrChildCol(),
                    "RecursionMatrix<T, Mat>::recursiveInverseNosym: case not allowed "
                    "Nr Child A[%d, %d] Dimensions A=%s ",
                    me()->nrChildRow(), me()->nrChildCol(), me()->description().c_str());

    for (int k=0 ; k<me()->nrChildRow() ; k++){
      // Inverse M_kk
      me()->get(k,k)->inverse();
      // Update line 'k' = left-multiplied by M_kk-1
      for (int j=0 ; j<me()->nrChildCol() ; j++)
        if (j!=k) {
          // Mkj <- Mkk^-1 Mkj we use a temp matrix X because this type of product is not allowed with gemm (beta=0 erases Mkj before using it !)
          Mat* x = me()->get(k,j)->copy();
          me()->get(k,j)->gemm('N', 'N', Constants<T>::pone, me()->get(k,k), x, Constants<T>::zero);
          x->destroy();
        }
      // Update the rest of matrix M
      for (int i=0 ; i<me()->nrChildRow() ; i++)
        // line 'i' -= Mik x line 'k' (which has just been multiplied by Mkk-1)
        for (int j=0 ; j<me()->nrChildCol() ; j++)
          if (i!=k && j!=k)
            // Mij <- Mij - Mik Mkk^-1 Mkj (with Mkk-1.Mkj allready stored in Mkj)
            me()->get(i,j)->gemm('N', 'N', Constants<T>::mone, me()->get(i,k), me()->get(k,j), Constants<T>::pone);
      // Update column 'k' = right-multiplied by -M_kk-1
      for (int i=0 ; i<me()->nrChildRow() ; i++)
        if (i!=k) {
          // Mik <- - Mik Mkk^-1
          Mat* x = me()->get(i,k)->copy();
          me()->get(i,k)->gemm('N', 'N', Constants<T>::mone, x, me()->get(k,k), Constants<T>::zero);
          x->destroy();
        }
    }

  }

  template<typename T, typename Mat>
  void RecursionMatrix<T, Mat>::recursiveLltDecomposition(hmat_progress_t * progress) {

    // |     |     |    |     |     |   |     |     |
    // | h11 | h21 |    | L1  |     |   | L1t | Lt  |
    // |-----|-----| =  |-----|-----| * |-----|-----|
    // | h21 | h22 |    | L   | L2  |   |     | L2t |
    // |     |     |    |     |     |   |     |     |
    //
    // h11 = L1 * L1t => L1 = h11.lltDecomposition
    // h21 = L*L1t => L = L1t.solve(h21) => trsm R with L1 lower stored
    // h22 = L*Lt + L2 * L2t => L2 = (h22 - L*Lt).lltDecomposition()
    //
    //
    // for all i, j<=i : hij = sum_k Lik t{Ljk} k<=i,j
    // The algorithm loops over 3 steps: for all k=1, 2, ..., n
    //   - We factorize the element (k,k)
    //   - We use "solve" to compute the rest of the column 'k'
    //   - We update the rest of the matrix [k+1, .., n]x[k+1, .., n] (below diag)

    HMAT_ASSERT_MSG(me()->nrChildRow()==me()->nrChildCol(),
                    "RecursionMatrix<T, Mat>::recursiveLltDecomposition: case not allowed "
                    "Nr Child A[%d, %d] Dimensions A=%s ",
                    me()->nrChildRow(), me()->nrChildCol(), me()->description().c_str());

    for (int k=0 ; k<me()->nrChildRow() ; k++) {
      // Hkk <- Lkk * tLkk
      me()->get(k,k)->lltDecomposition(progress);
      // Solve the rest of column k: solve Lik tLkk = Hik and get Lik
      for (int i=k+1 ; i<me()->nrChildRow() ; i++)
        me()->get(k,k)->solveUpperTriangularRight(me()->get(i,k), false, true);
      // update the rest of the matrix [k+1, .., n]x[k+1, .., n] (below diag)
      for (int i=k+1 ; i<me()->nrChildRow() ; i++)
        for (int j=k+1 ; j<=i ; j++)
          // Hij <- Hij - Lik tLjk
          me()->get(i,j)->gemm('N', 'T', Constants<T>::mone, me()->get(i,k), me()->get(j,k), Constants<T>::pone);
    }
  }

  template<typename T, typename Mat>
  void RecursionMatrix<T, Mat>::recursiveSolveUpperTriangularLeft(Mat* b,
     bool unitriangular, bool lowerStored, MainOp mainOp) const {

    //  Backward substitution:
    //  [ U11 | U12 ]    [ X11 | X12 ]   [ b11 | b12 ]
    //  [ ----+---- ] *  [-----+-----] = [ ----+---- ]
    //  [  0  | U22 ]    [ X21 | X22 ]   [ b21 | b22 ]
    //
    //  U22 * X21 = b21 (by recursive backward substitution)
    //  U22 * X22 = b22 (by recursive backward substitution)
    //  U11 * X12 + U12 * X22 = b12 (backward substitution of U11*X12=b12-U12*X22)
    //  U11 * X11 + U12 * X21 = b11 (backward substitution of U11*X11=b11-U12*X21)
    // X and b are not necessarily square

    // First we handle the general case, where dimensions (in terms of number of children) are compatible
    if (me()->nrChildCol() == b->nrChildRow()) {

      for (int k=0 ; k<b->nrChildCol() ; k++) { // Loop on the column of the RHS
        for (int i=me()->nrChildRow()-1 ; i>=0 ; i--) {
          // Solve the i-th diagonal system
          me()->get(i, i)->solveUpperTriangularLeft(b->get(i,k), unitriangular, lowerStored, mainOp);
          // Update b[j,k] j<i with the contribution of the solutions just computed b[i,k]
          for (int j=0 ; j<i ; j++) {
            const Mat* u_ji = (lowerStored ? me()->get(i, j) : me()->get(j, i));
            b->get(j,k)->gemm(lowerStored ? 'T' : 'N', 'N', Constants<T>::mone, u_ji, b->get(i,k),
                              Constants<T>::pone, mainOp);
          }
        }
      }

    } else if (me()->nrChildCol()>1 && b->nrChildRow()==1 && b->nrChildCol()>1) {
      // Then we handle the specific case where me() is divided in columns, and b is not divided in rows: we recurse on b's children only
      //  [ U11 | U12 ]    [     |     ]   [     |     ]
      //  [ ----+---- ] *  [ X11 | X12 ] = [ b11 | b12 ]
      //  [  0  | U22 ]    [     |     ]   [     |     ]
      for (int k=0 ; k<b->nrChildCol() ; k++) // loop on the column of b
        me()->recursiveSolveUpperTriangularLeft(b->get(0,k), unitriangular, lowerStored);

    } else {
      HMAT_ASSERT_MSG(false, "RecursionMatrix<T, Mat>::recursiveSolveUpperTriangularLeft: case not yet handled "
                             "Nr Child A[%d, %d] b[%d, %d] "
                             "Dimensions A=%s b=%s",
                      me()->nrChildRow(), me()->nrChildCol(), b->nrChildRow(), b->nrChildCol(),
                      me()->description().c_str(), b->description().c_str());

    }

  }

  template <typename T, typename Mat> void RecursionMatrix<T, Mat>::transposeMeta() {
      if (!me()->isLeaf()) {
          // We cannot not, in general, transpose in-place, so we need a backup of 'children'
          std::vector<Mat*> children_bak(me()->nrChild());
          for(int i = 0; i < me()->nrChild(); i++)
              children_bak[i] = me()->getChild(i);
          // and finally we fill 'children'
          int k = 0;
          for (int j = 0; j < me()->nrChildCol(); j++)
              for (int i = 0; i < me()->nrChildRow(); i++)
                  me()->getChild(j + i * me()->nrChildCol()) = children_bak[k++];
          for (int i = 0; i < me()->nrChild(); i++)
              if (me()->getChild(i))
                  me()->getChild(i)->transposeMeta();
      }
  }
}  // end namespace hmat
