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

#ifdef __INTEL_COMPILER
#include <mathimf.h>
#else
#include <cmath>
#endif

#include "compression.hpp"

#include <vector>
#include <cfloat>
#include <cstring>

#include "cluster_tree.hpp"
#include "interaction.hpp"
#include "rk_matrix.hpp"
#include "lapack_operations.hpp"
#include "lapack_overloads.hpp"
#include "full_matrix.hpp"
#include "common/context.hpp"
#include "common/my_assert.h"

#ifdef _MSC_VER
// Intel compiler defines isnan in global namespace
// MSVC defines _isnan
# ifndef __INTEL_COMPILER
#  define isnan _isnan
# endif
#elif defined(__MINGW32__)
using std::isnan;
#endif

using std::vector;
using std::min;
using std::max;


/** Convenience class to lighten the getRow() / getCol() calls.
*/
template<typename T>
class ClusterAssemblyFunction {
  const AssemblyFunction<T>& f;
  void* handle;
  const ClusterData* rows;
  const ClusterData* cols;

public:
  ClusterAssemblyFunction(const AssemblyFunction<T>& _f,
                        const ClusterData* _rows, const ClusterData* _cols)
    : f(_f), handle(NULL), rows(_rows), cols(_cols) {
    f.prepareBlock(rows, cols, &handle);
  }
  ~ClusterAssemblyFunction() {
    f.releaseBlock(handle);
  }
  void getRow(int index, Vector<typename Types<T>::dp>& result) const {
    f.getRow(rows, cols, index, handle, &result);
  }
  void getCol(int index, Vector<typename Types<T>::dp>& result) const {
    f.getCol(rows, cols, index, handle, &result);
  }
private:
  ClusterAssemblyFunction(ClusterAssemblyFunction&o) {}; // No copy
};

template<typename T> static double squaredNorm(const T x);
template<> double squaredNorm(const C_t x) {
  return norm(x);
}
template<> double squaredNorm(const Z_t x) {
  return norm(x);
}
template<typename T> double squaredNorm(const T x) {
  return x * x;
}

template<typename T> static bool isZero(const Vector<T>& v) {
  int index = v.absoluteMaxIndex();
  return v.v[index] == Constants<T>::zero;
}


/** \brief Updates a row to reflect its current value in the matrix.

    \param rowVec the row to update
    \param row the index of this row
    \param rows the rows that have already been computed
    \param cols the cols that have already been computed
    \params k the number of rows that have already been computed
 */
template<typename T>
static void updateRow(Vector<T>& rowVec, int row, const vector<Vector<T>*>& rows,
                      const vector<Vector<T>*>& cols, int k) {
  for (int l = 0; l < k; l++) {
    rowVec.axpy(Constants<T>::mone * cols[l]->v[row], rows[l]);
  }
}

template<typename T>
static void updateCol(Vector<T>& colVec, int col, const vector<Vector<T>*>& cols,
                      const vector<Vector<T>*>& rows, int k) {
  updateRow<T>(colVec, col, cols, rows, k);
}


template<typename T> static void findMax(FullMatrix<T>* m, int& i, int& j) {
  i = 0;
  j = 0;
  double maxNorm = squaredNorm<T>(m->get(i, j));
  const int cols = m->cols;
  const int rows = m->rows;
  for (int col = 0; col < cols; col++) {
    for (int row = 0; row < rows; row++) {
      double norm = squaredNorm<T>(m->get(row, col));
      if (norm > maxNorm) {
        i = row;
        j = col;
        maxNorm = norm;
      }
    }
  }
}


/*! \brief Find a column that is free and not null, or return an error.

  \param block The block assembly function
  \param colFree an array of the columns that are free to choose
                 from. This array is updated by this function.
  \param col the column to be returned. It doesn't need to be zeroed beforehand.'
  \return the index of the chosen column, or -1 if no column can be found.
 */
template<typename T>
static int findCol(ClusterAssemblyFunction<T>& block, vector<bool>& colFree,
                   Vector<typename Types<T>::dp>& col) {
  int colCount = colFree.size();
  bool found = false;
  int i;
  for (i = 0; i < colCount; i++) {
    if (colFree[i]) {
      col.clear();
      block.getCol(i, col);
      colFree[i] = false;
      if (!isZero(col)) {
        found = true;
        break;
      }
    }
  }
  return (found ? i : -1);
}


template<typename T>
static int findMinRow(ClusterAssemblyFunction<T>& block,
                      vector<bool>& rowFree,
                      const vector<Vector<typename Types<T>::dp>*>& aCols,
                      const vector<Vector<typename Types<T>::dp>*>& bCols,
                      const Vector<typename Types<T>::dp>& aRef,
                      Vector<typename Types<T>::dp>& row) {

  int rowCount = aRef.rows;
  double minNorm2;
  int i_ref;
  bool found = false;

  while (!found) {
    i_ref = -1;
    minNorm2 = DBL_MAX;
    for (int i = 0; i < rowCount; i++) {
      if (rowFree[i]) {
        double norm2 = squaredNorm<typename Types<T>::dp>(aRef.v[i]);
        if (norm2 < minNorm2) {
          i_ref = i;
          minNorm2 = norm2;
        }
      }
    }
    if (i_ref == -1) {
      return i_ref;
    }
    row.clear();
    block.getRow(i_ref, row);
    updateRow<typename Types<T>::dp>(row, i_ref, bCols, aCols, aCols.size());
    found = !isZero(row);
    rowFree[i_ref] = false;
  }
  return i_ref;
}

template<typename T>
static int findMinCol(ClusterAssemblyFunction<T>& block,
                      vector<bool>& colFree,
                      const vector<Vector<typename Types<T>::dp>*>& aCols,
                      const vector<Vector<typename Types<T>::dp>*>& bCols,
                      const Vector<typename Types<T>::dp>& bRef,
                      Vector<typename Types<T>::dp>& col) {
  int colCount = bRef.rows;
  double minNorm2;
  int j_ref;
  bool found = false;

  while (!found) {
    j_ref = -1;
    minNorm2 = DBL_MAX;
    for (int j = 0; j < colCount; j++) {
      if (colFree[j]) {
        double norm2 = squaredNorm<typename Types<T>::dp>(bRef.v[j]);
        if (norm2 < minNorm2) {
          j_ref = j;
          minNorm2 = norm2;
        }
      }
    }
    if (j_ref == -1) {
      return j_ref;
    }
    col.clear();
    block.getCol(j_ref, col);
    updateCol<typename Types<T>::dp>(col, j_ref, aCols, bCols, bCols.size());
    found = !isZero(col);
    colFree[j_ref] = false;
  }
  return j_ref;
}


template<typename T>
RkMatrix<T>* compressMatrix(FullMatrix<T>* m, const ClusterData* rows,
                            const ClusterData* cols) {
  DECLARE_CONTEXT;
  myAssert((size_t) m->rows == rows->n);
  myAssert((size_t) m->cols == cols->n);
  myAssert(m->lda >= m->rows);

  bool zeroMatrix = true;
  for (size_t col = 0; col < m->cols; col++) {
    Vector<T> v(m->m + m->rows * col, m->rows);
    zeroMatrix = zeroMatrix && isZero(v);
    if (!zeroMatrix) break;
  }
  if (zeroMatrix) {
    return new RkMatrix<T>(NULL, rows, NULL, cols, NoCompression);
  }
  // In the case of non-square matrix, we don't calculated singular vectors
  // bigger than the minimum dimension of the matrix. However this is not
  // necessary here, since k < min (n, p) for M matrix (nxp).
  int rowCount = m->rows;
  int colCount = m->cols;
  FullMatrix<T> *u = NULL, *vt = NULL;
  Vector<double>* sigma = NULL;

  int info = truncatedSvd<T>(m, &u, &sigma, &vt);
  strongAssert(info == 0);
  // Control of the approximation

  int maxK = min(rowCount, colCount);
  int k = RkMatrix<T>::approx.findK(sigma->v, maxK, RkMatrix<T>::approx.assemblyEpsilon);

  for (int col = 0; col < k; col++) {
    for (int row = 0; row < rowCount; row++) {
      u->get(row, col) *= sigma->v[col];
    }
  }

  FullMatrix<T> matU(u->m, rowCount, k);
  FullMatrix<T>* uTilde = matU.copy();
  FullMatrix<T> matV(vt->m, k, colCount, maxK);
  FullMatrix<T>* vTilde = matV.copyAndTranspose();

  delete u;
  delete vt;
  delete sigma;
  FullMatrix<T>* a = uTilde;
  return new RkMatrix<T>(a, rows, vTilde, cols, Svd);
}


template<typename T>
static RkMatrix<typename Types<T>::dp>*
compressSvd(const AssemblyFunction<T>& f,
            const ClusterData* rows, const ClusterData* cols) {
  DECLARE_CONTEXT;
  typedef typename Types<T>::dp dp_t;
  FullMatrix<dp_t>* m = f.assemble(rows, cols);
  RkMatrix<dp_t>* result = compressMatrix(m, rows, cols);
  delete m;
  return result;
}


template<typename T>
static RkMatrix<typename Types<T>::dp>*
compressAcaFull(const AssemblyFunction<T>& f,
                const ClusterData* rows, const ClusterData* cols) {
  DECLARE_CONTEXT;
  typedef typename Types<T>::dp dp_t;
  FullMatrix<dp_t>* m = f.assemble(rows, cols);

  double estimateSquaredNorm = 0;
  int maxK = min(m->rows, m->cols);
  if (RkMatrix<dp_t>::approx.k > 0) {
    maxK = min(maxK, RkMatrix<dp_t>::approx.k);
  }

  FullMatrix<dp_t> tmpA(m->rows, maxK);
  tmpA.clear();
  FullMatrix<dp_t> tmpB(m->cols, maxK);
  tmpB.clear();
  int nu;

  for (nu = 0; nu < maxK; nu++) {
    int i_nu, j_nu;
    findMax(m, i_nu, j_nu);
    dp_t delta = m->get(i_nu, j_nu);
    if (squaredNorm(delta) == 0.) {
      break;
    }

    // Creation of the vectors A_i_nu and B_j_nu
    memcpy(tmpA.m + nu * tmpA.rows, m->m + j_nu * m->rows,
           sizeof(dp_t) * tmpA.rows);
    for (int j = 0; j < m->cols; j++) {
      tmpB.get(j, nu) = m->get(i_nu, j) / delta;
    }

    FullMatrix<dp_t> a_nu(tmpA.m + nu * tmpA.rows, tmpA.rows, 1);
    FullMatrix<dp_t> b_nu(tmpB.m + nu * tmpB.rows, tmpB.rows, 1);
    m->gemm('N', 'T', Constants<dp_t>::mone, &a_nu, &b_nu, Constants<dp_t>::pone);

    // Update the estimate norm
    // Let S_{k-1} be the previous estimate. We have (for the Frobenius norm):
    //  ||S_k||^2 = ||S_{k-1}||^2 + \sum_{l = 0}^{nu-1} (<a_k, a_l> <b_k, b_l> + <a_l, a_k> <b_l, b_k>))
    //              + ||a_k||^2 ||b_k||^2
    {
      Vector<dp_t> va_nu(tmpA.m + nu * tmpA.rows, tmpA.rows);
      Vector<dp_t> vb_nu(tmpB.m + nu * tmpB.rows, tmpB.rows);
      Vector<dp_t> a_l(tmpA.m, tmpA.rows);
      Vector<dp_t> b_l(tmpB.m, tmpB.rows);
      // The sum
      double newEstimate = 0.0;
      for (int l = 0; l < nu - 1; l++) {
        a_l.v = tmpA.m + l * tmpA.rows;
        b_l.v = tmpB.m + l * tmpB.rows;
        newEstimate += hmat::real(Vector<dp_t>::dot(&va_nu, &a_l) * Vector<dp_t>::dot(&vb_nu, &b_l));
      }
      estimateSquaredNorm += 2.0 * newEstimate;
      double a_nu_norm = va_nu.norm();
      double b_nu_norm = vb_nu.norm();
      double ab_norm_2 = (a_nu_norm * a_nu_norm) * (b_nu_norm * b_nu_norm);
      estimateSquaredNorm += ab_norm_2;

      // Evaluate the stopping criterion
      // ||a_nu|| ||b_nu|| < epsilon * ||S_nu||
      // <=> ||a_nu||^2 ||b_nu||^2 < epsilon^2 ||S_nu||^2
      double epsilon = RkMatrix<dp_t>::approx.assemblyEpsilon;
      if (ab_norm_2 < epsilon * epsilon * estimateSquaredNorm) {
        break;
      }
    }
  }
  delete m;

  if (nu == 0) {
    return new RkMatrix<dp_t>(NULL, rows, NULL, cols, AcaFull);
  }

  FullMatrix<dp_t>* newA = new FullMatrix<dp_t>(tmpA.rows, nu);
  newA->clear();
  memcpy(newA->m, tmpA.m, sizeof(dp_t) * tmpA.rows * nu);
  FullMatrix<dp_t>* newB = new FullMatrix<dp_t>(tmpB.rows, nu);
  newB->clear();
  memcpy(newB->m, tmpB.m, sizeof(dp_t) * tmpB.rows * nu);

  return new RkMatrix<dp_t>(newA, rows, newB, cols, AcaFull);
}


template<typename T>
static RkMatrix<typename Types<T>::dp>*
compressAcaPartial(const AssemblyFunction<T>& f,
                   const ClusterData* rows, const ClusterData* cols) {
  typedef typename Types<T>::dp dp_t;
  double estimateSquaredNorm = 0;
  ClusterAssemblyFunction<T> block(f, rows, cols);

  const int rowCount = rows->n;
  const int colCount = cols->n;
  int maxK = min(rowCount, colCount);
  // Contains false for the rows that were already used as pivot
  vector<bool> rowFree(rowCount, true);
  int rowPivotCount = 0;
  // idem for columns
  vector<bool> colFree(colCount, true);
  vector<Vector<dp_t>*> aCols;
  vector<Vector<dp_t>*> bCols;

  int I = 0;
  int J = 0;
  int k = 0;

  do {
    Vector<dp_t>* bCol = new Vector<dp_t>(cols->n);
    // Calculation of row I and its residue
    block.getRow(I, *bCol);
    updateRow(*bCol, I, bCols, aCols, k);
    rowFree[I] = false;
    rowPivotCount++;

    // Find max and argmax of the residue
    double maxNorm2 = 0.;
    for (int j = 0; j < colCount; j++) {
      double norm2 = squaredNorm<dp_t>(bCol->v[j]);
      if (colFree[j] && norm2 > maxNorm2) {
        maxNorm2 = norm2;
        J = j;
      }
    }

    if (bCol->v[J] == Constants<dp_t>::zero) {
      delete bCol;
      // We look for another row which has not already been used.
      I = 0;
      while (!rowFree[I]) {
        I++;
      }
    } else {
      // Find pivot and scale column B
      dp_t pivot = Constants<dp_t>::pone / bCol->v[J];
      bCol->scale(pivot);
      bCols.push_back(bCol);

      // Compute column J and residue
      Vector<dp_t>* aCol = new Vector<dp_t>(rows->n);
      block.getCol(J, *aCol);
      updateCol(*aCol, J, aCols, bCols, k);
      colFree[J] = false;
      aCols.push_back(aCol);

      // Find max and argmax of the residue
      maxNorm2 = 0.;
      for (int i = 0; i < rowCount; i++) {
        double norm2 = squaredNorm<dp_t>(aCol->v[i]);
        if (rowFree[i] && norm2 > maxNorm2) {
          maxNorm2 = norm2;
          I = i;
        }
      }

      // Update the estimated norm
      // Let S_{k-1} be the previous estimate. We have (for the Frobenius norm):
      //  ||S_k||^2 = ||S_{k-1}||^2 + \sum_{l = 0}^{nu-1} (<a_k, a_l> <b_k, b_l> + <a_l, a_k> <b_l, b_k>))
      //              + ||a_k||^2 ||b_k||^2
      double newEstimate = 0.0;
      for (int l = 0; l < k; l++) {
        newEstimate += hmat::real(Vector<dp_t>::dot(aCol, aCols[l]) * Vector<dp_t>::dot(bCol, bCols[l]));
      }
      estimateSquaredNorm += 2.0 * newEstimate;
      double aColNorm = aCol->norm();
      double bColNorm = bCol->norm();
      double ab_norm_2 = (aColNorm * aColNorm) * (bColNorm * bColNorm);
      estimateSquaredNorm += ab_norm_2;
      k++;

      // Evaluate the stopping criterion
      // ||a_nu|| ||b_nu|| < epsilon * ||S_nu||
      // <=> ||a_nu||^2 ||b_nu||^2 < epsilon^2 ||S_nu||^2
      double epsilon = RkMatrix<dp_t>::approx.assemblyEpsilon;
      if (ab_norm_2 < epsilon * epsilon * estimateSquaredNorm) {
        break;
      }
    }
  } while (rowPivotCount < maxK);

  FullMatrix<dp_t> *newA, *newB;
  if (k != 0) {
    newA = new FullMatrix<dp_t>(rows->n, k);
    for (int i = 0; i < k; i++) {
      memcpy(newA->m + (i * newA->rows), aCols[i]->v, sizeof(dp_t) * newA->rows);
      delete aCols[i];
      aCols[i] = NULL;
    }
    newB = new FullMatrix<dp_t>(cols->n, k);
    for (int i = 0; i < k; i++) {
      memcpy(newB->m + (i * newB->rows), bCols[i]->v, sizeof(dp_t) * newB->rows);
      delete bCols[i];
      bCols[i] = NULL;
    }
  } else {
    // If k == 0, block is only made of zeros.
    return new RkMatrix<dp_t>(NULL, rows, NULL, cols, AcaPartial);
  }

  return new RkMatrix<dp_t>(newA, rows, newB, cols, AcaPartial);
}


template<typename T>
static RkMatrix<typename Types<T>::dp>*
compressAcaPlus(const AssemblyFunction<T>& f,
                const ClusterData* rows, const ClusterData* cols) {
  typedef typename Types<T>::dp dp_t;
  double estimateSquaredNorm = 0;
  int i_ref, j_ref;
  int rowCount = rows->n, colCount = cols->n;
  int maxK = min(rowCount, colCount);
  Vector<dp_t> bRef(colCount), aRef(rowCount);
  vector<bool> rowFree(rowCount, true), colFree(colCount, true);
  vector<Vector<dp_t>*> aCols, bCols;
  ClusterAssemblyFunction<T> block(f, rows, cols);

  j_ref = findCol(block, colFree, aRef);
  if (j_ref == -1) {
	// The block is completely zero.
    return new RkMatrix<dp_t>(NULL, rows, NULL, cols, AcaPlus);
  }

  // The reference row is chosen such that it intersects the reference
  // column at its argmin index.
  i_ref = findMinRow(block, rowFree, aCols, bCols, aRef, bRef);

  int k = 0;
  do {
    Vector<dp_t>* bVec = new Vector<dp_t>(colCount);
    Vector<dp_t>* aVec = new Vector<dp_t>(rowCount);
    int i_star, j_star;
    dp_t i_star_value, j_star_value;

    i_star = aRef.absoluteMaxIndex();
    i_star_value = aRef.v[i_star];

    j_star = bRef.absoluteMaxIndex();
    j_star_value = bRef.v[j_star];

    if (squaredNorm<dp_t>(i_star_value) > squaredNorm<dp_t>(j_star_value)) {
      // i_star is fixed, we look for j_star
      block.getRow(i_star, *bVec);
      // Calculate the residue
      updateRow<dp_t>(*bVec, i_star, bCols, aCols, k);
      j_star = bVec->absoluteMaxIndex();
      dp_t pivot = bVec->v[j_star];
      strongAssert(pivot != Constants<dp_t>::zero);
      // Calculate a
      block.getCol(j_star, *aVec);
      updateCol<dp_t>(*aVec, j_star, aCols, bCols, k);
      aVec->scale(Constants<dp_t>::pone / pivot);
    } else {
      // j_star is fixed, we look for i_star
      block.getCol(j_star, *aVec);
      updateCol<dp_t>(*aVec, j_star, aCols, bCols, k);
      i_star = aVec->absoluteMaxIndex();
      dp_t pivot = aVec->v[i_star];
      strongAssert(pivot != Constants<dp_t>::zero);
      // Calculate b
      block.getRow(i_star, *bVec);
      updateRow<dp_t>(*bVec, i_star, bCols, aCols, k);
      bVec->scale(Constants<dp_t>::pone / pivot);
    }

    rowFree[i_star] = false;
    colFree[j_star] = false;

    aCols.push_back(aVec);
    bCols.push_back(bVec);
    // Update the estimate norm
    // Let S_{k-1} be the previous estimate. We have (for the Frobenius norm):
    //  ||S_k||^2 = ||S_{k-1}||^2 + \sum_{l = 0}^{nu-1} (<a_k, a_l> <b_k, b_l> + <u_l, u_k> <b_l, b_k>))
    //              + ||a_k||^2 ||b_k||^2
    double newEstimate = 0.0;
    for (int l = 0; l < k; l++) {
      newEstimate += hmat::real(Vector<dp_t>::dot(aVec, aCols[l]) * Vector<dp_t>::dot(bVec, bCols[l]));
    }
    estimateSquaredNorm += 2.0 * newEstimate;
    double aVecNorm = aVec->norm();
    double bVecNorm = bVec->norm();
    double ab_norm_2 = (aVecNorm * aVecNorm) * (bVecNorm * bVecNorm);
    estimateSquaredNorm += ab_norm_2;
    k++;

    // Evaluate the stopping criterion
    // ||a_nu|| ||b_nu|| < epsilon * ||S_nu||
    // <=> ||a_nu||^2 ||b_nu||^2 < epsilon^2 ||S_nu||^2
    double epsilon = RkMatrix<dp_t>::approx.assemblyEpsilon;
    if (ab_norm_2 < epsilon * epsilon * estimateSquaredNorm) {
      break;
    }

    // Update of a_ref and b_ref
    aRef.axpy(Constants<dp_t>::mone * bCols[k - 1]->v[j_ref], aCols[k - 1]);
    bRef.axpy(Constants<dp_t>::mone * aCols[k - 1]->v[i_ref], bCols[k - 1]);
    bool needNewA = isZero(aRef) || (j_star == j_ref);
    bool needNewB = isZero(bRef) || (i_star == i_ref);

    // If the row or the column of reference have been already chosen as pivot,
    // we can not keep it and then we take one or two others.
    if (needNewA && needNewB) {
      bool found = false;
      while (!found) {
        aRef.clear();
        j_ref = findCol(block, colFree, aRef);
         // We can not find non-zero column anymore, done!
        if (j_ref == -1) {
          break;
        }
        updateCol<dp_t>(aRef, j_ref, aCols, bCols, k);
        found = !isZero(aRef);
      }
      if (!found) {
        break;
      }
      bRef.clear();
      i_ref = findMinRow(block, rowFree, aCols, bCols, aRef, bRef);
      // We can not find non-zero line anymore, done!
      if (i_ref == -1) {
        break;
      }
    } else if (needNewB) {
      bRef.clear();
      i_ref = findMinRow(block, rowFree, aCols, bCols, aRef, bRef);
      // We can not find non-zero line anymore, done!
      if (i_ref == -1) {
        break;
      }
    } else if (needNewA) {
      aRef.clear();
      j_ref = findMinCol(block, colFree, aCols, bCols, bRef, aRef);
      // We can not find non-zero column anymore, done!
      if (j_ref == -1) {
        break;
      }
    }
  } while (k < maxK);

  myAssert(k > 0);
  FullMatrix<dp_t>* newA = new FullMatrix<dp_t>(rows->n, k);
  for (int i = 0; i < k; i++) {
    memcpy(newA->m + (i * newA->rows), aCols[i]->v, sizeof(dp_t) * newA->rows);
    delete aCols[i];
    aCols[i] = NULL;
  }
  FullMatrix<dp_t>* newB = new FullMatrix<dp_t>(cols->n, k);
  for (int i = 0; i < k; i++) {
    memcpy(newB->m + (i * newB->rows), bCols[i]->v, sizeof(dp_t) * newB->rows);
    delete bCols[i];
    bCols[i] = NULL;
  }
  return new RkMatrix<dp_t>(newA, rows, newB, cols, AcaPlus);
}

#include <iostream>
/* Appele par HMatrix<T>::assemble() */
template<typename T>
RkMatrix<typename Types<T>::dp>* compress(CompressionMethod method,
                                          const AssemblyFunction<T>& f,
                                          const ClusterData* rows,
                                          const ClusterData* cols) {
  typedef typename Types<T>::dp dp_t;
  RkMatrix<dp_t>* rk = NULL;

  // Always compress the smallest blocks using an SVD. Small blocks tend to have
  // a bad compression ratio anyways, and the SVD is not very costly in this
  // case.
  // TODO: allow the setting of the size limit ?
  if (max(rows->n, cols->n) < 100) {
    rk = compressSvd(f, rows, cols);
  } else {
    switch (method) {
    case Svd:
      rk = compressSvd(f, rows, cols);
      break;
    case AcaFull:
      rk = compressAcaFull(f, rows, cols);
      break;
    case AcaPartial:
      rk = compressAcaPartial(f, rows, cols);
      break;
    case AcaPlus:
      rk = compressAcaPlus(f, rows, cols);
      break;
    case NoCompression:
      // Should not happen
      break;
    }
  }

  if (HMatrix<T>::validateCompression) {
    FullMatrix<dp_t>* full = f.assemble(rows, cols);
    if (rk->a) rk->a->checkNan();
    if (rk->b) rk->b->checkNan();
    FullMatrix<dp_t>* rkFull = rk->eval();
    double approxNorm = rkFull->norm();
    double fullNorm = full->norm();

    // If I meet a NaN, I save & leave
    // TODO : improve this behaviour
    if (isnan(approxNorm)) {
      rkFull->toFile("Rk");
      full->toFile("Full");
      strongAssert(false);
    }

    rkFull->axpy(Constants<T>::mone, full);
    double diffNorm = rkFull->norm();
    if (diffNorm > HMatrix<T>::validationErrorThreshold * fullNorm ) {
      std::cout << "["<< rows->offset<<","<<rows->offset+rows->n -1 <<"]x["<< cols->offset<<","<<cols->offset+cols->n-1 <<"]"<< std::endl
           << std::scientific
           << "|M|  = " << fullNorm << std::endl
           << "|Rk| = " << approxNorm << std::endl
           << "|M - Rk| / |Rk| = " << diffNorm / fullNorm << std::endl
           << "Rank = " << rk->k << " / " << min(full->rows, full->cols) << std::endl << std::endl;

      if (HMatrix<T>::validationReRun) {
        // Call compression a 2nd time, for debugging with gdb the work of the compression algorithm...
        RkMatrix<dp_t>* rk_bis = NULL;

        if (max(rows->n, cols->n) < 100) {
          rk_bis = compressSvd(f, rows, cols);
        } else {
          switch (method) {
            case Svd:
              rk_bis = compressSvd(f, rows, cols);
              break;
            case AcaFull:
              rk_bis = compressAcaFull(f, rows, cols);
              break;
            case AcaPartial:
              rk_bis = compressAcaPartial(f, rows, cols);
              break;
            case AcaPlus:
              rk_bis = compressAcaPlus(f, rows, cols);
              break;
            case NoCompression:
              // Should not happen
              break;
          }
        }
        delete rk_bis ;
      }

      if (HMatrix<T>::validationDump) {
        std::string filename;
        std::ostringstream convert;   // stream used for the conversion
        convert << "["<< rows->offset<<","<<rows->offset+rows->n -1 <<"]x["<< cols->offset<<","<<cols->offset+cols->n-1 <<"]" ;

        filename = "Rk_";
        filename += convert.str(); // set 'Result' to the contents of the stream
        rkFull->toFile(filename.c_str());
        filename = "Full_"+convert.str(); // set 'Result' to the contents of the stream
        full->toFile(filename.c_str());
      }
    }

    delete rkFull;
    delete full;
  }
  return rk;
}

// Declaration of the used templates
template RkMatrix<S_t>* compressMatrix(FullMatrix<S_t>* m, const ClusterData* rows, const ClusterData* cols);
template RkMatrix<D_t>* compressMatrix(FullMatrix<D_t>* m, const ClusterData* rows, const ClusterData* cols);
template RkMatrix<C_t>* compressMatrix(FullMatrix<C_t>* m, const ClusterData* rows, const ClusterData* cols);
template RkMatrix<Z_t>* compressMatrix(FullMatrix<Z_t>* m, const ClusterData* rows, const ClusterData* cols);

template RkMatrix<Types<S_t>::dp>* compress<S_t>(CompressionMethod method, const AssemblyFunction<S_t>& f, const ClusterData* rows, const ClusterData* cols);
template RkMatrix<Types<D_t>::dp>* compress<D_t>(CompressionMethod method, const AssemblyFunction<D_t>& f, const ClusterData* rows, const ClusterData* cols);
template RkMatrix<Types<C_t>::dp>* compress<C_t>(CompressionMethod method, const AssemblyFunction<C_t>& f, const ClusterData* rows, const ClusterData* cols);
template RkMatrix<Types<Z_t>::dp>* compress<Z_t>(CompressionMethod method, const AssemblyFunction<Z_t>& f, const ClusterData* rows, const ClusterData* cols);
