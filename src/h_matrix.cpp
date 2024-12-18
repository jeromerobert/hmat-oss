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
  \brief HMatrix type.
*/
#include <algorithm>
#include <list>
#include <vector>
#include <cstring>

#include "h_matrix.hpp"
#include "cluster_tree.hpp"
#include "admissibility.hpp"
#include "data_types.hpp"
#include "compression.hpp"
#include "recursion.hpp"
#include "common/context.hpp"
#include "common/my_assert.h"
#include "json.hpp"

using namespace std;

namespace hmat {

// The default values below will be overwritten in default_engine.cpp by HMatSettings values
template<typename T> bool HMatrix<T>::coarsening = false;
template<typename T> bool HMatrix<T>::recompress = false;
template<typename T> bool HMatrix<T>::validateNullRowCol = false;
template<typename T> bool HMatrix<T>::validateCompression = false;
template<typename T> bool HMatrix<T>::validateRecompression=false;
template<typename T> bool HMatrix<T>::validationReRun = false;
template<typename T> bool HMatrix<T>::validationDump = false;
template<typename T> double HMatrix<T>::validationErrorThreshold = 0;

template<typename T> HMatrix<T>::~HMatrix() {
  if (isRkMatrix() && rk_) {
    delete rk_;
    rk_ = NULL;
  }
  if (full_) {
    delete full_;
    full_ = NULL;
  }
  if(ownRowsClusterTree_)
      delete rows_;
  if(ownColsClusterTree_)
      delete cols_;
}


template<typename T>
void reorderVector(ScalarArray<T>* v, int* indices, int axis) {
  DECLARE_CONTEXT;
  if (!indices) return;
  const int n = axis == 0 ? v->rows : v->cols;
  // If permutation is identity, do nothing
  bool identity = true;
  for (int i = 0; i < n; i++) {
    if (indices[i] != i) {
      identity = false;
      break;
    }
  }
  if (identity) return;

  if (axis == 0) {
    Vector<T> tmp(n);
    for (int col = 0; col < v->cols; col++) {
      Vector<T> column(*v, col);
      for (int i = 0; i < n; i++)
        tmp[i] = column[indices[i]];
      tmp.copy(&column);
    }
  } else {
    ScalarArray<T> tmp(1, n);
    for (int row = 0; row < v->rows; row++) {
      ScalarArray<T> column(*v, row, 1, 0, n);
      for (int i = 0; i < n; i++)
        tmp.get(0, i) = column.get(0, indices[i]);
      tmp.copy(&column);
    }
  }
}

template<typename T>
void restoreVectorOrder(ScalarArray<T>* v, int* indices, int axis) {
  DECLARE_CONTEXT;
  if (!indices) return;
  const int n = axis == 0 ? v->rows : v->cols;
  // If permutation is identity, do nothing
  bool identity = true;
  for (int i = 0; i < n; i++) {
    if (indices[i] != i) {
      identity = false;
      break;
    }
  }
  if (identity) return;

  if (axis == 0) {
    Vector<T> tmp(n);
    for (int col = 0; col < v->cols; col++) {
      Vector<T> column(*v, col);
      for (int i = 0; i < n; i++) {
        tmp[indices[i]] = column[i];
      }
      tmp.copy(&column);
    }
  } else {
    ScalarArray<T> tmp(1, n);
    for (int row = 0; row < v->rows; row++) {
      ScalarArray<T> column(*v, row, 1, 0, n);
      for (int i = 0; i < n; i++)
        tmp.get(0, indices[i]) = column.get(0, i);
      tmp.copy(&column);
    }
  }
}

template<typename T>
HMatrix<T>::HMatrix(const ClusterTree* _rows, const ClusterTree* _cols, const hmat::MatrixSettings * settings,
                    int _depth, SymmetryFlag symFlag, AdmissibilityCondition * admissibilityCondition)
  : Tree<HMatrix<T> >(NULL, _depth), RecursionMatrix<T, HMatrix<T> >(),
    rows_(_rows), cols_(_cols), rk_(NULL),
    rank_(UNINITIALIZED_BLOCK), approximateRank_(UNINITIALIZED_BLOCK),
    isUpper(false), isLower(false),
    isTriUpper(false), isTriLower(false), keepSameRows(true), keepSameCols(true), temporary_(false),
    ownRowsClusterTree_(false), ownColsClusterTree_(false), localSettings(settings, 1e-4)
{
  if (isVoid())
    return;
  const bool lowRank = admissibilityCondition->isLowRank(*rows_, *cols_);
  if (!split(admissibilityCondition, lowRank, symFlag)) {
    // If we cannot split, we are on a leaf
    const bool forceFull = admissibilityCondition->forceFull(*rows_, *cols_);
    const bool forceRk   = admissibilityCondition->forceRk(*rows_, *cols_);
    assert(!(forceFull && forceRk));
    if (forceRk || (lowRank && !forceFull))
      rk(NULL);
    else
      full(NULL);
    approximateRank_ = admissibilityCondition->getApproximateRank(*(rows_), *(cols_));
  }
  assert(!this->isLeaf() || isAssembled());
}

template<typename T>
bool HMatrix<T>::split(AdmissibilityCondition * admissibilityCondition, bool lowRank,
                      SymmetryFlag symFlag) {
  assert(rank_ == NONLEAF_BLOCK || rank_ == UNINITIALIZED_BLOCK || (this->isLeaf() && isNull()));
  // We would like to create a block of matrix in one of the following case:
  // - rows_->isLeaf() && cols_->isLeaf() : both rows and cols are leaves.
  // - Block is too small to recurse and compress (for performance)
  // - Block is compressible and in-place compression is possible
  // In the other cases, we subdivide.
  //
  // FIXME: But in practice this does not work yet, so we stop recursion as soon as either rows
  // or cols is a leaf.
  bool stopRecursion = admissibilityCondition->stopRecursion(*rows_, *cols_);
  bool forceRecursion = admissibilityCondition->forceRecursion(*rows_, *cols_, sizeof(T));
  assert(!(forceRecursion && stopRecursion));
  // check we can actually split
  if ((rows_->isLeaf() && cols_->isLeaf()) || stopRecursion || (lowRank && !forceRecursion))
    return false;
  pair<bool, bool> splitRC = admissibilityCondition->splitRowsCols(*rows_, *cols_);
  assert(splitRC.first || splitRC.second);
  keepSameRows = !splitRC.first;
  keepSameCols = !splitRC.second;
  isLower = (symFlag == kLowerSymmetric ? true : false);
  for (int i = 0; i < nrChildRow(); ++i) {
    // Don't recurse on rows if splitRowsCols() told us not to.
    ClusterTree* rowChild = const_cast<ClusterTree*>((keepSameRows ? rows_ : rows_->getChild(i)));
    for (int j = 0; j < nrChildCol(); ++j) {
      // Don't recurse on cols if splitRowsCols() told us not to.
      ClusterTree* colChild = const_cast<ClusterTree*>((keepSameCols ? cols_ : cols_->getChild(j)));
      if ((symFlag == kNotSymmetric) || (isUpper && (i <= j)) || (isLower && (i >= j))) {
        if (!admissibilityCondition->isInert(*rowChild, *colChild)) {
          // Create child only if not 'inert' (inert = will always be null)
          this->insertChild(i, j,
                            new HMatrix<T>(rowChild, colChild, localSettings.global,
                                           this->depth + 1,
                                           i == j ? symFlag : kNotSymmetric,
                                           admissibilityCondition->at(this->depth + 1)));
        } else
          // If 'inert', the child is NULL
          this->insertChild(i, j, NULL);
      }
    }
  }
  if(nrChildRow() > 0 && nrChildCol() > 0)
    rank_ = NONLEAF_BLOCK;
  return true;
}

template<typename T>
HMatrix<T>::HMatrix(const hmat::MatrixSettings * settings) :
    Tree<HMatrix<T> >(NULL), RecursionMatrix<T, HMatrix<T> >(), rows_(NULL), cols_(NULL),
    rk_(NULL), rank_(UNINITIALIZED_BLOCK), approximateRank_(UNINITIALIZED_BLOCK),
    isUpper(false), isLower(false), isTriUpper(false), isTriLower(false),
    keepSameRows(true), keepSameCols(true), temporary_(false), ownRowsClusterTree_(false),
    ownColsClusterTree_(false), localSettings(settings, -1.0)
    {}

template<typename T> HMatrix<T> * HMatrix<T>::internalCopy(bool temporary, bool withRowChild, bool withColChild) const {
    HMatrix<T> * r = new HMatrix<T>(localSettings.global);
    r->rows_ = rows_;
    r->cols_ = cols_;
    r->temporary_ = temporary;
    r->localSettings.epsilon_ = localSettings.epsilon_;
    if(withRowChild || withColChild) {
        // Here, we come from HMatrixHandle<T>::createGemmTemporaryRk()
        // we want to go 1 level below data (which is an Rk)
        // so we don't use get(i,j) since data has no children
        // we dont use this->nrChildRow and this->nrChildCol either, they would return 1
        // (since 'this' is rows- and cols-admissible, unlike 'r')
        r->keepSameRows = !withRowChild;
        r->keepSameCols = !withColChild;
        for(int i = 0; i < r->nrChildRow(); i++) {
            for(int j = 0; j < r->nrChildCol(); j++) {
                HMatrix<T>* child = new HMatrix<T>(localSettings.global);
                child->temporary_ = temporary;
                child->rows_ = withRowChild ? rows_->getChild(i) : rows_;
                child->cols_ = withColChild ? cols_->getChild(j) : cols_;
                child->localSettings.epsilon_ = localSettings.epsilon_;
                assert(child->rows_ != NULL);
                assert(child->cols_ != NULL);
                assert(child->localSettings.epsilon_ > 0);
                child->rk(NULL);
                r->insertChild(i, j, child);
            }
        }
    }
    return r;
}

template<typename T> HMatrix<T>* HMatrix<T>::internalCopy(
        const ClusterTree * rows, const ClusterTree * cols) const {
    HMatrix<T> * r = new HMatrix<T>(localSettings.global);
    r->temporary_ = true;
    r->rows_ = rows;
    r->cols_ = cols;
    r->localSettings.epsilon_ = localSettings.epsilon_;
    return r;
}

template<typename T>
HMatrix<T>* HMatrix<T>::copyStructure() const {
  HMatrix<T>* h = internalCopy();
  h->isUpper = isUpper;
  h->isLower = isLower;
  h->isTriUpper = isTriUpper;
  h->isTriLower = isTriLower;
  h->keepSameRows = keepSameRows;
  h->keepSameCols = keepSameCols;
  h->rank_ = rank_ >= 0 ? 0 : rank_;
  h->approximateRank_ = approximateRank_;
  if(!this->isLeaf()){
    for (int i = 0; i < this->nrChild(); ++i) {
      if (this->getChild(i)) {
        h->insertChild(i, this->getChild(i)->copyStructure());
      }
      else
        h->insertChild(i, NULL);
    }
  }
  return h;
}

template<typename T>
HMatrix<T>* HMatrix<T>::Zero(const HMatrix<T>* o) {
  // leaves are filled by 0
  HMatrix<T> *h = o->internalCopy();
  h->isLower = o->isLower;
  h->isUpper = o->isUpper;
  h->isTriUpper = o->isTriUpper;
  h->isTriLower = o->isTriLower;
  h->keepSameRows = o->keepSameRows;
  h->keepSameCols = o->keepSameCols;
  h->rank_ = o->rank_ >= 0 ? 0 : o->rank_;
  if (h->rank_==0)
    h->rk(new RkMatrix<T>(NULL, h->rows(), NULL, h->cols()));
  h->approximateRank_ = o->approximateRank_;
  if(!o->isLeaf()){
    for (int i = 0; i < o->nrChild(); ++i) {
      if (o->getChild(i)) {
        h->insertChild(i, HMatrix<T>::Zero(o->getChild(i)));
        } else
        h->insertChild(i, NULL);
    }
  }
  return h;
}

template<typename T>
void HMatrix<T>::setClusterTrees(const ClusterTree* rows, const ClusterTree* cols) {
    rows_ = rows;
    cols_ = cols;
    if(isRkMatrix() && rk()) {
        rk()->rows = &(rows->data);
        rk()->cols = &(cols->data);
    } else if(isFullMatrix()) {
        full()->rows_ = &(rows->data);
        full()->cols_ = &(cols->data);
    } else if(!this->isLeaf()) {
      for (int i = 0; i < nrChildRow(); ++i) {
        // if rows not admissible, don't recurse on them
        const ClusterTree* rowChild = (keepSameRows ? rows : rows->me()->getChild(i));
        for (int j = 0; j < nrChildCol(); ++j) {
          // if cols not admissible, don't recurse on them
          const ClusterTree* colChild = (keepSameCols ? cols : cols->me()->getChild(j));
          if ((isLower || isTriLower) &&
              (rowChild->data.offset() < colChild->data.offset() || (rowChild->data.offset() == colChild->data.offset() && i < j)))
            continue;
          if ((isUpper || isTriUpper) &&
              (rowChild->data.offset() > colChild->data.offset() || (rowChild->data.offset() == colChild->data.offset() && i > j)))
            continue;
          if(get(i, j))
            get(i, j)->setClusterTrees(rowChild, colChild);
        }
      }
    }
}

template<typename T>
void HMatrix<T>::assemble(Assembly<T>& f, const AllocationObserver & ao) {
  if (this->isLeaf()) {
    // If the leaf is admissible, matrix assembly and compression.
    // if not we keep the matrix.
    FullMatrix<T> * m = NULL;
    RkMatrix<T>* assembledRk = NULL;
    f.assemble(localSettings, *rows_, *cols_, isRkMatrix(), m, assembledRk, lowRankEpsilon(), ao);
    HMAT_ASSERT(m == NULL || assembledRk == NULL);
    if(assembledRk) {
        assert(isRkMatrix());
        if(rk_)
            delete rk_;
        rk(assembledRk);
    } else {
        assert(!isRkMatrix());
        if(full_)
            delete full_;
        full(m);
    }
  } else {
    full_ = NULL;
    rk_ = NULL;
    for (int i = 0; i < this->nrChild(); i++) {
      if (this->getChild(i))
        this->getChild(i)->assemble(f, ao);
    }
    assembledRecurse();
    if (coarsening)
      coarsen(RkMatrix<T>::approx.coarseningEpsilon);
  }
}

template<typename T>
void HMatrix<T>::assembleSymmetric(Assembly<T>& f,
   HMatrix<T>* upper, bool onlyLower, const AllocationObserver & ao) {
  if (!onlyLower) {
    if (!upper){
      upper = this;
    }
    assert(*this->rows() == *upper->cols());
    assert(*this->cols() == *upper->rows());
  }

  if (this->isLeaf()) {
    // If the leaf is admissible, matrix assembly and compression.
    // if not we keep the matrix.
    this->assemble(f, ao);
    if (isRkMatrix()) {
      if ((!onlyLower) && (upper != this)) {
        // Admissible leaf: a matrix represented by AB^t is transposed by exchanging A and B.
        RkMatrix<T>* newRk = rk()->copy();
        newRk->transpose();
        if(upper->isRkMatrix() && upper->rk() != NULL)
            delete upper->rk();
        upper->rk(newRk);
      }
    } else {
      if ((!onlyLower) && ( upper != this)) {
        if(isFullMatrix())
            upper->full(full()->copyAndTranspose());
        else
            upper->full(NULL);
      }
    }
  } else {
    if (onlyLower) {
      for (int i = 0; i < nrChildRow(); i++) {
        for (int j = 0; j < nrChildCol(); j++) {
          if ((*rows() == *cols()) && (j > i)) {
            continue;
          }
          if (get(i,j))
            get(i,j)->assembleSymmetric(f, NULL, true, ao);
        }
      }
    } else {
      if (this == upper) {
        for (int i = 0; i < nrChildRow(); i++) {
          for (int j = 0; j <= i; j++) {
            HMatrix<T> *child = get(i, j);
            HMatrix<T> *upperChild = get(j, i);
            assert((child != NULL) == (upperChild != NULL));
            if (child)
              child->assembleSymmetric(f, upperChild, false, ao);
          }
        }
      } else {
        for (int i = 0; i < nrChildRow(); i++) {
          for (int j = 0; j < nrChildCol(); j++) {
            HMatrix<T> *child = get(i, j);
            HMatrix<T> *upperChild = upper->get(j, i);
            assert((child != NULL) == (upperChild != NULL));
            if (child)
              child->assembleSymmetric(f, upperChild, false, ao);
          }
        }
        upper->assembledRecurse();
        if (coarsening)
          coarsen(RkMatrix<T>::approx.coarseningEpsilon, upper);
      }
    }
    assembledRecurse();
  }
}

template<typename T> void HMatrix<T>::info(hmat_info_t & result) {
    result.nr_block_clusters++;
    int r = rows()->size();
    int c = cols()->size();
    if(r == 0 || c == 0) {
        return;
    } else if(this->isLeaf()) {
        size_t s = ((size_t)r) * c;
        result.uncompressed_size += s;
        if(isRkMatrix()) {
            size_t mem = rank() * (((size_t)r) + c);
            result.compressed_size += mem;
            int dim = result.largest_rk_dim_cols + result.largest_rk_dim_rows;
            if(rows()->size() + cols()->size() > dim) {
                result.largest_rk_dim_cols = c;
                result.largest_rk_dim_rows = r;
            }

            size_t old_s = ((size_t)result.largest_rk_mem_cols + result.largest_rk_mem_rows) * result.largest_rk_mem_rank;
            if(mem > old_s) {
                result.largest_rk_mem_cols = c;
                result.largest_rk_mem_rows = r;
                result.largest_rk_mem_rank = rank();
            }
            result.rk_count++;
            result.rk_size += s;
        } else {
            result.compressed_size += s;
            result.full_count ++;
            result.full_size += s;
        }
    } else {
        for (int i = 0; i < this->nrChild(); i++) {
            HMatrix<T> *child = this->getChild(i);
            if (child)
                child->info(result);
        }
    }
}

template<typename T>
void HMatrix<T>::eval(FullMatrix<T>* result, bool renumber) const {
  if (this->isLeaf()) {
    if (this->isNull()) return;
    FullMatrix<T> *mat = isRkMatrix() ? rk()->eval() : full();
    int *rowIndices = rows()->indices() + rows()->offset();
    int rowCount = rows()->size();
    int *colIndices = cols()->indices() + cols()->offset();
    int colCount = cols()->size();
    if(renumber) {
      for (int j = 0; j < colCount; j++)
        for (int i = 0; i < rowCount; i++)
          result->get(rowIndices[i], colIndices[j]) = mat->get(i, j);
    } else {
      for (int j = 0; j < colCount; j++)
        memcpy(&result->get(rows()->offset(), cols()->offset() + j), &mat->get(0, j), rowCount * sizeof(T));
    }
    if (isRkMatrix()) {
      delete mat;
    }
  } else {
    for (int i = 0; i < this->nrChild(); i++) {
      if (this->getChild(i)) {
        this->getChild(i)->eval(result, renumber);
      }
    }
  }
}

template<typename T>
void HMatrix<T>::evalPart(FullMatrix<T>* result, const IndexSet* _rows,
                          const IndexSet* _cols) const {
  if (this->isLeaf()) {
    if (this->isNull()) return;
    FullMatrix<T> *mat = isRkMatrix() ? rk()->eval() : full();
    const int rowOffset = rows()->offset() - _rows->offset();
    const int rowCount = rows()->size();
    const int colOffset = cols()->offset() - _cols->offset();
    const int colCount = cols()->size();
    for (int j = 0; j < colCount; j++) {
      memcpy(&result->get(rowOffset, j + colOffset), &mat->get(0, j), rowCount * sizeof(T));
    }
    if (isRkMatrix()) {
      delete mat;
    }
  } else {
    for (int i = 0; i < this->nrChild(); i++) {
      if (this->getChild(i)) {
        this->getChild(i)->evalPart(result, _rows, _cols);
      }
    }
  }
}

template<typename T> double HMatrix<T>::normSqr() const {
  double result = 0.;
  if (rows()->size() == 0 || cols()->size() == 0) {
    return result;
  }
  if (this->isLeaf() && isAssembled() && !isNull()) {
    if (isRkMatrix()) {
      result = rk()->normSqr();
    } else {
      result = full()->normSqr();
    }
  } else if(!this->isLeaf()){
    for (int i = 0; i < this->nrChild(); i++) {
      const HMatrix<T> *res=this->getChild(i);
      if (res) {
        // When computing the norm of symmetric matrices, extra-diagonal blocks count twice
        double coeff = (isUpper || isLower) && ! (*res->rows() == *res->cols()) ? 2. : 1. ;
        result += coeff * res->normSqr();
      }
    }
  }
  return result;
}

// Return an approximation of the largest eigenvalue via the power method.
// If needed, we could also return the corresponding eigenvector.
template<typename T> T HMatrix<T>::approximateLargestEigenvalue(int max_iter, double epsilon) const {
  if (max_iter <= 0) return 0.0;
  if (rows()->size() == 0 || cols()->size() == 0) {
    return 0.0;
  }
  const int nrow = rows()->size();
  Vector<T>  xv(nrow);
  Vector<T>  xv1(nrow);
  Vector<T>* x  = &xv;
  Vector<T>* x1 = &xv1;
  T ev = 0;
  for (int i = 0; i < nrow; i++)
    xv[i] = static_cast<T>(rand()/(double)RAND_MAX);
  double normx = x->norm();
  if (normx == 0.0)
    return approximateLargestEigenvalue(max_iter - 1, epsilon);
  x->scale(static_cast<T>(1.0/normx));
  int iter = 0;
  double aev = 0.0;
  double aev_p = 0.0;
  do {
    // old eigenvalue
    aev_p = aev;
    // Compute x(k+1) = A x(k)
    //        ev(k+1) = <x(k+1),x(k)>
    //         x(k+1) = x(k+1) / ||x(k+1)||
    gemv('N', 1, x, 0, x1);
    ev = Vector<T>::dot(x,x1);
    // new abs(ev)
    aev = std::abs(ev);
    normx = x1->norm();
    // If x1 is null, restart so that starting point is different.
    // Decrease max_iter to prevent infinite recursion.
    if (normx == 0.0)
      return approximateLargestEigenvalue(max_iter - 1, epsilon);
    x1->scale(static_cast<T>(1.0/normx));
    std::swap(x,x1);
    iter++;
  } while(iter < max_iter && std::abs(aev - aev_p) > epsilon * aev);
  return ev;
}


template<typename T>
void HMatrix<T>::scale(T alpha) {
  if(alpha == T(0)) {
    this->clear();
  } else if(alpha == T(1)) {
    return;
  } else if (this->isLeaf()) {
    if (isNull()) {
      // nothing to do
    } else if (isRkMatrix()) {
      rk()->scale(alpha);
    } else {
      assert(isFullMatrix());
      full()->scale(alpha);
    }
  } else {
    for (int i = 0; i < this->nrChild(); i++) {
      if (this->getChild(i)) {
        this->getChild(i)->scale(alpha);
      }
    }
  }
}

template<typename T>
bool HMatrix<T>::coarsen(double epsilon, HMatrix<T>* upper, bool force) {
  // If all children are Rk leaves, then we try to merge them into a single Rk-leaf.
  // This is done if the memory of the resulting leaf is less than the sum of the initial
  // leaves. Note that this operation could be used hierarchically.

  bool allRkLeaves = true;
  std::vector< RkMatrix<T> const* > childrenArray(this->nrChild());
  size_t childrenElements = 0;
  for (int i = 0; i < this->nrChild(); i++) {
    childrenArray[i] = nullptr;
    HMatrix<T> *child = this->getChild(i);
    if (!child) continue;
    if (!child->isRkMatrix()) {
      allRkLeaves = false;
      break;
    } else {
      childrenArray[i] = child->rk();
      if(childrenArray[i])
        childrenElements += (childrenArray[i]->rows->size()
                           + childrenArray[i]->cols->size()) * childrenArray[i]->rank();
    }
  }
  if (allRkLeaves) {
    std::vector<T> alpha(this->nrChild(), 1);
    RkMatrix<T> * candidate = new RkMatrix<T>(NULL, rows(), NULL, cols());
    candidate->formattedAddParts(epsilon, &alpha[0], childrenArray.data(), this->nrChild());
    size_t elements = (((size_t) candidate->rows->size()) + candidate->cols->size()) * candidate->rank();
    if (force || elements < childrenElements) {
      // Replace 'this' by the new Rk matrix
      for (int i = 0; i < this->nrChild(); i++)
        this->removeChild(i);
      this->children.clear();
      rk(candidate);
      assert(this->isLeaf());
      assert(isRkMatrix());
      // If necessary, replace 'upper' by the new Rk matrix transposed (exchange a and b)
      if (upper) {
        for (int i = 0; i < this->nrChild(); i++)
          upper->removeChild(i);
        upper->children.clear();
        RkMatrix<T>* newRk = candidate->copy();
        newRk->transpose();
        upper->rk(newRk);
        assert(upper->isLeaf());
        assert(upper->isRkMatrix());
      }
    } else {
      delete candidate;
    }
  }

  return allRkLeaves;
}

template<typename T> const HMatrix<T> * HMatrix<T>::getChildForGEMM(char & t, int i, int j) const {
  // At most 1 of these flags must be 'true'
  assert(isUpper + isLower + isTriUpper + isTriLower >= -1);
  assert(!this->isLeaf());

  const HMatrix<T>* res;
  if(t != 'N')
    std::swap(i,j);
  if( (isLower && j > i) ||
      (isUpper && i > j) ) {
    res = get(j, i);
    t = t == 'N' ? 'T' : 'N';
  } else {
    res = get(i, j);
  }
  return res;
}

// y <- alpha * op(this) * x + beta * y or y <- alpha * x * op(this) + beta * y.
template<typename T>
void HMatrix<T>::gemv(char matTrans, T alpha, const ScalarArray<T>* x, T beta, ScalarArray<T>* y, Side side) const {
  if (rows()->size() == 0 || cols()->size() == 0) return;
  // The dimensions of the H-matrix and the 2 ScalarArrays must match exactly
  if(side == Side::LEFT) {
    // Y + H * X
    assert(x->cols == y->cols);
    assert((matTrans != 'N' ? cols()->size() : rows()->size()) == y->rows);
    assert((matTrans != 'N' ? rows()->size() : cols()->size()) == x->rows);
  } else {
    // Y + X * H
    assert(x->rows == y->rows);
    assert((matTrans != 'N' ? cols()->size() : rows()->size()) == x->cols);
    assert((matTrans != 'N' ? rows()->size() : cols()->size()) == y->cols);
  }
  if (beta != T(1)) {
    y->scale(beta);
  }

  if (!this->isLeaf()) {
    for (int i = 0, iend = (matTrans=='N' ? nrChildRow() : nrChildCol()); i < iend; i++)
      for (int j = 0, jend = (matTrans=='N' ? nrChildCol() : nrChildRow()); j < jend; j++) {
        char trans = matTrans;
        // trans(child) = the child (i,j) of matTrans(this)
        const HMatrix<T>* child = getChildForGEMM(trans, i, j);
        if (child) {

          // I get the rows and cols info of 'child'
          int colsOffset = child->cols()->offset() - cols()->offset();
          int colsSize   = child->cols()->size();
          int rowsOffset = child->rows()->offset() - rows()->offset();
          int rowsSize   = child->rows()->size();

          // swap if needed to get the info for trans(child)
          if (trans != 'N') {
            std::swap(colsOffset, rowsOffset);
            std::swap(colsSize,   rowsSize);
          }

          if (side == Side::LEFT) {
            // get the rows subset of X aligned with 'trans(child)' cols and Y aligned with 'trans(child)' rows
            const ScalarArray<T> subX(*x, colsOffset, colsSize, 0, x->cols);
            ScalarArray<T> subY(*y, rowsOffset, rowsSize, 0, y->cols);
            child->gemv(trans, alpha, &subX, 1, &subY, side);
          } else {
            // get the columns subset of X aligned with 'trans(child)' rows and Y aligned with 'trans(child)' columns
            const ScalarArray<T> subX(*x, 0, x->rows, rowsOffset, rowsSize);
            ScalarArray<T> subY(*y, 0, y->rows, colsOffset, colsSize);
            child->gemv(trans, alpha, &subX, 1, &subY, side);
          }
        }
        else continue;
      }

  } else {
    // We are on a leaf of the matrix 'this'
    if (isFullMatrix()) {
      if (side == Side::LEFT) {
        y->gemm(matTrans, 'N', alpha, &full()->data, x, 1);
      } else {
        y->gemm('N', matTrans, alpha, x, &full()->data, 1);
      }
    } else if(!isNull()){
      rk()->gemv(matTrans, alpha, x, 1, y, side);
    }
  }
}

template<typename T>
void HMatrix<T>::gemv(char matTrans, T alpha, const FullMatrix<T>* x, T beta, FullMatrix<T>* y, Side side) const {
  gemv(matTrans, alpha, &x->data, beta, &y->data, side);
}

/**
 * @brief Compress any HMatrix to a Rk
 *
 * This is different from HMatrix::coarsen because we don't check that we
 * only act on Rk. This is only used in AXPY.
 */
template <typename T> RkMatrix<T> * toRk(const HMatrix<T> *m) {
  assert(!m->isRkMatrix()); // Avoid useless copy
  RkMatrix<T> * r;
  if(m->isLeaf()) {
    // must copy because acaFull (as truncatedSvd) modify the input
    FullMatrix<T> * mc = m->full()->copy();
    r = acaFull(mc, m->lowRankEpsilon());
    delete mc;
  } else {
    r = new RkMatrix<T>(NULL, m->rows(), NULL, m->cols());
    vector<const RkMatrix<T> *> rkLeaves;
    vector<RkMatrix<T> *> tmpRkLeaves;
    vector<T> alphas;
    for(int i = 0; i < m->nrChild(); i++) {
      HMatrix<T> * c = m->getChild(i);
      if(c == nullptr || (c->isLeaf() && c->isNull())) {
        // do nothing
      } else if(c->isRkMatrix()) {
        rkLeaves.push_back(c->rk());
        alphas.push_back(1);
      } else {
        RkMatrix<T> * crk = toRk(c);
        tmpRkLeaves.push_back(crk);
        rkLeaves.push_back(crk);
        alphas.push_back(1);
      }
    }
    if(!rkLeaves.empty()) {
      r->formattedAddParts(m->lowRankEpsilon(), &alphas[0], &rkLeaves[0], rkLeaves.size());
    }
    for(auto t: tmpRkLeaves)
      delete t;
  }
  return r;
}

/**
 * @brief List all Rk matrice in the m matrice.
 * @return true if the matrix contains only rk matrices, fall if it contains
 * both rk and full matrices
 */
template<typename T> bool listAllRk(const HMatrix<T> * m, vector<const RkMatrix<T>*> & result) {
    if(m == NULL) {
        // do nothing
    } else if(m->isRkMatrix())
        result.push_back(m->rk());
    else if(m->isLeaf())
        return false;
    else {
        for(int i = 0; i < m->nrChild(); i++) {
            if(m->getChild(i) && !listAllRk(m->getChild(i), result))
                return false;
        }
    }
    return true;
}

/**
 * @brief generic AXPY implementation that dispatch to others or recurse
 */
template <typename T> void HMatrix<T>::axpy(T alpha, const HMatrix<T> *x) {
  if (x->isLeaf()) {
    if (x->isNull()) {
      // nothing to do
    } else if (x->isFullMatrix())
      axpy(alpha, x->full());
    else if (x->isRkMatrix())
      axpy(alpha, x->rk());
  } else {
    HMAT_ASSERT(*rows() == *x->rows());
    HMAT_ASSERT(*cols() == *x->cols());
    if (this->isLeaf()) {
      if (isRkMatrix()) {
        if (!rk())
          rk(new RkMatrix<T>(NULL, rows(), NULL, cols()));
        vector<const RkMatrix<T> *> rkLeaves;
        if (listAllRk(x, rkLeaves)) {
          vector<T> alphas(rkLeaves.size(), alpha);
          rk()->formattedAddParts(lowRankEpsilon(), &alphas[0], &rkLeaves[0], rkLeaves.size());
        } else {
          RkMatrix<T> *tmpRk = toRk<T>(x);
          rk()->axpy(lowRankEpsilon(), alpha, tmpRk);
          delete tmpRk;
        }
        rank_ = rk()->rank();
      } else {
        if (full() == NULL)
          full(new FullMatrix<T>(rows(), cols()));
        FullMatrix<T> xFull(x->rows(), x->cols());
        x->evalPart(&xFull, x->rows(), x->cols());
        full()->axpy(alpha, &xFull);
      }
    } else
      for (int i = 0; i < this->nrChild(); i++) {
        HMatrix<T> *child = this->getChild(i);
        const HMatrix<T> *bChild = x->isLeaf() ? x : x->getChild(i);
        if (bChild != NULL) {
          HMAT_ASSERT(child != NULL); // This may happen but this is not supported yet
          child->axpy(alpha, bChild);
        }
      }
  }
}

/** @brief AXPY between 'this' an H matrix and a subset of B with B a RkMatrix */
template<typename T>
void HMatrix<T>::axpy(T alpha, const RkMatrix<T>* b) {
  DECLARE_CONTEXT;
  // this += alpha * b
  assert(b);
  assert(b->rows->intersects(*rows()));
  assert(b->cols->intersects(*cols()));

  if (b->rank() == 0 || rows()->size() == 0 || cols()->size() == 0) {
    return;
  }

  // If 'this' is not a leaf, we recurse with the same 'b'
  if (!this->isLeaf()) {
    for (int i = 0; i < this->nrChild(); i++) {
      HMatrix<T> * c = this->getChild(i);
      if (c) {
        if(b->rank() < std::min(c->rows()->size(), c->cols()->size()) && b->rank() > 10) {
          RkMatrix<T> * bc = b->truncatedSubset(c->rows(), c->cols(), c->lowRankEpsilon());
          c->axpy(alpha, bc);
          delete bc;
        }
        else c->axpy(alpha, b);
      }
    }
  } else {
    // To add-up a leaf to a RkMatrix, resizing may be necessary.
    bool needResizing = b->rows->isStrictSuperSet(*rows())
      || b->cols->isStrictSuperSet(*cols());
    const RkMatrix<T>* newRk = b;
    if (needResizing) {
      newRk = b->subset(rows(), cols());
    }
    if (isRkMatrix()) {
      if(!rk())
          rk(new RkMatrix<T>(NULL, rows(), NULL, cols()));
      rk()->axpy(lowRankEpsilon(), alpha, newRk);
      rank_ = rk()->rank();
    } else {
      // In this case, the matrix has small size
      // then evaluating the Rk-matrix is cheaper
      FullMatrix<T>* rkMat = newRk->eval();
      bool thisSuperSetb = this->rows()->isStrictSuperSet(*b->rows) || this->cols()->isStrictSuperSet(*b->cols);
      if(full() == NULL && !thisSuperSetb) {
        rkMat->scale(alpha);
        full(rkMat);
      } else {
        this->axpy(alpha, rkMat);
        delete rkMat;
      }
    }
    if (needResizing) {
      delete newRk;
    }
  }
}

/** @brief AXPY between 'this' an H matrix and a subset of B with B a FullMatrix */
template<typename T>
void HMatrix<T>::axpy(T alpha, const FullMatrix<T>* b) {
  DECLARE_CONTEXT;
  bool bSuperSetThis = b->rows_->isStrictSuperSet(*this->rows()) || b->cols_->isStrictSuperSet(*this->cols());
  bool thisSuperSetb = this->rows()->isStrictSuperSet(*b->rows_) || this->cols()->isStrictSuperSet(*b->cols_);
  // this += alpha * b
  // If 'this' is not a leaf, we recurse with the same 'b'
  if (!this->isLeaf()) {
    for (int i = 0; i < this->nrChild(); i++) {
      HMatrix<T>* child = this->getChild(i);
      if (child)
        child->axpy(alpha, b);
    }
  } else {
    const FullMatrix<T>* subMat = bSuperSetThis ? b->subset(rows(), cols()) : b;
    if (isRkMatrix()) {
      assert(b->rows_->isSuperSet(*this->rows()) && b->cols_->isSuperSet(*this->cols()));
      if(!rk())
        rk(new RkMatrix<T>(NULL, rows(), NULL, cols()));
      rk()->axpy(lowRankEpsilon(), alpha, subMat);
      rank_ = rk()->rank();
    } else {
      if(isFullMatrix() || thisSuperSetb){
        if(thisSuperSetb && full() == NULL) {
          full(new FullMatrix<T>(this->rows(), this->cols()));
        }
        auto subThis = thisSuperSetb ? this->subset(b->rows_, b->cols_) : this;
        subThis->full()->axpy(alpha, subMat);
        if(thisSuperSetb)
          delete subThis;
      } else {
        assert(!isAssembled() || full() == NULL);
        full(subMat->copy());
        if(alpha != T(1))
          full()->scale(alpha);
      }
    }
    if(bSuperSetThis)
      delete subMat;
  }
}

template<typename T>
void HMatrix<T>::addIdentity(T alpha)
{
  if (this->isLeaf()) {
    if (isNull()) {
      HMAT_ASSERT(!this->isRkMatrix());
      full(new FullMatrix<T>(rows(), cols()));
    }
    if (isFullMatrix()) {
      FullMatrix<T> * b = full();
      assert(b->rows() == b->cols());
      b->data.addIdentity(alpha);
    } else {
      HMAT_ASSERT(false);
    }
  } else {
    for (int i = 0; i < nrChildRow(); i++)
      if(get(i, i) != nullptr)
        get(i,i)->addIdentity(alpha);
  }
}

template<typename T>
void HMatrix<T>::addRand(double epsilon)
{
  if (this->isLeaf()) {
    if (isFullMatrix()) {
      full()->addRand(epsilon);
    } else {
      rk()->addRand(epsilon);
    }
  } else {
    for (int i = 0; i < nrChildRow(); i++) {
      for(int j = 0; j < nrChildCol(); j++) {
	if(get(i,j)) {
          get(i,j)->addRand(epsilon);
        }
      }
    }
  }
}

template<typename T> HMatrix<T> * HMatrix<T>::subset(
    const IndexSet * rows, const IndexSet * cols) const
{
    if((this->rows() == rows && this->cols() == cols) ||
       (*(this->rows()) == *rows && *(this->cols()) == *cols) ||
       (!rows->isSubset(*(this->rows())) || !cols->isSubset(*(this->cols())))) // TODO cette ligne me parait louche... si rows et cols sont pas bons, on renvoie 'this' sans meme se plaindre ???
        return const_cast<HMatrix<T>*>(this);

    // this could be implemented but if you need it you more
    // likely have something to fix at a higher level.
    assert(!this->isNull());

    if(this->isLeaf()) {
        HMatrix<T> * tmpMatrix = new HMatrix<T>(this->localSettings.global);
        tmpMatrix->temporary_=true;
        tmpMatrix->localSettings.epsilon_ = localSettings.epsilon_;
        ClusterTree * r = rows_->slice(rows->offset(), rows->size());
        ClusterTree * c = cols_->slice(cols->offset(), cols->size());

        // ensure the cluster tree are properly freed
        r->father = r;
        c->father = c;

        tmpMatrix->rows_ = r;
        tmpMatrix->cols_ = c;
        tmpMatrix->ownClusterTrees(true, true);
        if(this->isRkMatrix()) {
          tmpMatrix->rk(const_cast<RkMatrix<T>*>(rk()->subset(tmpMatrix->rows(), tmpMatrix->cols())));
        } else {
          tmpMatrix->full(const_cast<FullMatrix<T>*>(full()->subset(tmpMatrix->rows(), tmpMatrix->cols())));
        }
        return tmpMatrix;
    } else {
        // 'This' is not a leaf
        //TODO not yet implemented but should not happen
        HMAT_ASSERT(false);
    }
}

/**
 * @brief Ensure that matrices have compatible cluster trees.
 * @param row_a If true check the number of row of A is compatible else check columns
 * @param row_b If true check the number of row of B is compatible else check columns
 * @param in_a The input A matrix whose dimension must be checked
 * @param in_b The input B matrix whose dimension must be checked
 * @param out_a A subset of the A matrix which have compatible dimension with out_b.
 *  out_a is a view on in_a, no data are copied. It can possibly return in_a if matrices
 *  are already compatibles.
 * @param out_b A subset of the B matrix which have compatible dimension with out_a.
 *  out_b is a view on in_b, no data are copied. It can possibly return in_b if matrices
 *  are already compatibles.
 */
template<typename T> void
makeCompatible(bool row_a, bool row_b,
               const HMatrix<T> * in_a, const HMatrix<T> * in_b,
               HMatrix<T> * & out_a, HMatrix<T> * & out_b) {

    // suppose that A is bigger than B: in that case A will change, not B
    const IndexSet * cdb = row_b ? in_b->rows() : in_b->cols();
    if(row_a) // restrict the rows of in_a to cdb
        out_a = in_a->subset(cdb, in_a->cols());
    else // or the cols
        out_a = in_a->subset(in_a->rows(), cdb);

    // if A has changed, B won't change so we bypass this second step
    if(out_a == in_a) {
        // suppose than B is bigger than A: B will change, not A
        const IndexSet * cda = row_a ? in_a->rows() : in_a->cols();
        if(row_b)
            out_b = in_b->subset(cda, in_b->cols());
        else
            out_b = in_b->subset(in_b->rows(), cda);
    }
    else
        out_b = const_cast<HMatrix<T> *>(in_b);
}

/**
 * @brief A GEMM implementation which do not require matrices have compatible
 * cluster tree.
 *
 *  We compute the product alpha.f(a).f(b)+c -> c (with c=this)
 *  f(a)=transpose(a) if transA='T', f(a)=a if transA='N' (idem for b)
 */
template<typename T> void HMatrix<T>::uncompatibleGemm(char transA, char transB, T alpha,
                                                  const HMatrix<T>* a, const HMatrix<T>* b) {
    // Computing a(m,0) * b(0,n) here may give wrong results because of format conversions, exit early
    if(isVoid() || a->isVoid())
        return;
    HMatrix<T> * va = NULL;
    HMatrix<T> * vb = NULL;
    HMatrix<T> * vc = NULL;;
    HMatrix<T> * vva = NULL;
    HMatrix<T> * vvb = NULL;
    HMatrix<T> * vvc = NULL;

    // Create va & vb = the subsets of a & b that match each other for doing the product f(a).f(b)
    // We modify the columns of f(a) and the rows of f(b)
    makeCompatible<T>(transA != 'N', transB == 'N', a, b, va, vb);

    if(this->isLeaf() && !this->isRkMatrix() && this->full() == NULL) {
	  // C (this) is a null full block. We cannot get the subset of it and we
	  // don't know yet if we need to allocate it
	  fullHHGemm(this, transA, transB, alpha, va, vb);
      if(va != a)
        delete va;
      if(vb != b)
        delete vb;
      return;
    } else {
      // Create vva & vc = the subsets of va & c (=this) that match each other for doing the sum c+f(a).f(b)
      // We modify the rows of f(a) and the rows of c
      makeCompatible<T>(transA == 'N', true, va, this, vva, vc);

      // Create vvb & vvc = the subsets of vb & vc that match each other for doing the sum c+f(a).f(b)
      // We modify the columns of f(b) and the columns of c
      makeCompatible<T>(transB != 'N', false, vb, vc, vvb, vvc);
    }

    // Delete the intermediate matrices, except if subset() in makecompatible() has returned the original matrix
    if(va != vva && va != a)
        delete va;
    if(vb != vvb && vb != b)
        delete vb;
    if(vc != vvc && vc != this)
        delete vc;

    // writing on a subset of an RkMatrix is not possible without
    // modifying the whole matrix
    assert(!isRkMatrix() || vvc == this);
    // Do the product on the matrices that are now compatible
    vvc->leafGemm(transA, transB, alpha, vva, vvb);

    // Delete the temporary matrices
    if(vva != a)
        delete vva;
    if(vvb != b)
        delete vvb;
    if(vvc != this)
        delete vvc;
}

template<typename T>
unsigned char * compatibilityGridForGEMM(const HMatrix<T>* a, Axis axisA, char transA, const HMatrix<T>* b, Axis axisB, char transB) {
    // Let us first consider C = A^T * B where A, B and C are top-level matrices:
    //  [ C11 | C12 ]   [ A11^T | A21^T ]   [ B11 | B12 ]
    //  [ ----+---- ] = [ ------+------ ] * [ ----+---- ]
    //  [ C21 | C22 ]   [ A12^T | A22^T ]   [ B21 | B22 ]
    // This multiplication is possible only if columns of A^T and rows of B are the same,
    // rows of A^T and rows of C are the same, and columns of B and columns of C are the same.
    // Matrices are built from the same cluster trees, so we know that blocks are split at the
    // same place, and this function will return:
    //    compatibilityGridForGEMM(A, COL, 'T', B, ROW, 'N') = {1, 0, 0, 1}
    //    compatibilityGridForGEMM(A, ROW, 'T', C, ROW, 'N') = {1, 0, 0, 1}
    //    compatibilityGridForGEMM(B, COL, 'N', C, ROW, 'N') = {1, 0, 0, 1}
    // But blocks could be split in a single direction instead of 2, or could not
    // be split; if A had been split only by rows, we would have
    //  [ C11 | C12 ]                       [ B11 | B12 ]
    //  [ ----+---- ] = [ A11^T | A21^T ] * [ ----+---- ]
    //  [ C21 | C22 ]                       [ B21 | B22 ]
    //    compatibilityGridForGEMM(A, ROW, 'T', C, ROW, 'N') = {1, 1}
    //
    // Situation is much more complicated when considering inner nodes; for instance let us
    // have a look at A11^T * B11 with A11 and B11 being defined just above. Rows of A11^T
    // are equal to rows(C11)+rows(C21), and thus (considering that A11 and C11 are split
    // into 4 nodes)
    //    compatibilityGridForGEMM(A11, ROW, 'T', C11, ROW, 'N') = {1, 1, 0, 0}
    //
    // This function is generic, it works for all cases as long as matrices are built from
    // the same cluster trees.

    int row_a = transA == 'N' ? a->nrChildRow() : a->nrChildCol();
    int col_a = transA == 'N' ? a->nrChildCol() : a->nrChildRow();
    int row_b = transB == 'N' ? b->nrChildRow() : b->nrChildCol();
    int col_b = transB == 'N' ? b->nrChildCol() : b->nrChildRow();
    size_t nr_blocks = (axisA == Axis::ROW ? row_a : col_a) * (axisB == Axis::ROW ? row_b : col_b);
    unsigned char * result = new unsigned char[nr_blocks];
    memset(result, 0, nr_blocks);

    if (axisA == Axis::ROW) {
        for (int iA = 0; iA < row_a; iA++) {
            // All children on a row have the same row cluster tree, get it
            // from the first non null child.  We also consider the case where
            // 'a' is a leaf.
            const HMatrix<T> *childA = a->isLeaf() ? a : nullptr;
            char tA = transA;
            for (int jA = 0; !childA && jA < col_a; jA++) {
              tA = transA;
              childA = a->getChildForGEMM(tA, iA, jA);
            }
            // This row contains only null children, skip it
            if(!childA)
                continue;
            if (axisB == Axis::ROW) {
                for (int iB = 0; iB < row_b; iB++) {
                    for (int jB = 0; jB < col_b; jB++) {
                        char tB = transB;
                        const HMatrix<T> *childB = b->isLeaf() ? b : b->getChildForGEMM(tB, iB, jB);
                        if(childB) {
                            result[iA * row_b + iB] = (tA == 'N' ? childA->rows() : childA->cols())->intersects(*(tB == 'N' ? childB->rows() : childB->cols()));
                            break;
                        }
                    }
                }
            } else {
                for (int jB = 0; jB < col_b; jB++) {
                    for (int iB = 0; iB < row_b; iB++) {
                        char tB = transB;
                        const HMatrix<T> *childB = b->isLeaf() ? b : b->getChildForGEMM(tB, iB, jB);
                        if(childB) {
                            result[iA * col_b + jB] = (tA == 'N' ? childA->rows() : childA->cols())->intersects(*(tB == 'N' ? childB->cols() : childB->rows()));
                            break;
                        }
                    }
                }
            }
        }
    } else {
        for (int jA = 0; jA < col_a; jA++) {
            const HMatrix<T> *childA = a->isLeaf() ? a : nullptr;
            char tA = transA;
            for (int iA = 0; !childA && iA < row_a; iA++) {
              tA = transA;
              childA = a->getChildForGEMM(tA, iA, jA);
            }
            // This column contains only null children, skip it
            if(!childA)
                continue;
            if (axisB == Axis::ROW) {
                for (int iB = 0; iB < row_b; iB++) {
                    for (int jB = 0; jB < col_b; jB++) {
                        char tB = transB;
                        const HMatrix<T> *childB = b->isLeaf() ? b : b->getChildForGEMM(tB, iB, jB);
                        if(childB) {
                            result[jA * row_b + iB] = (tA == 'N' ? childA->cols() : childA->rows())->intersects(*(tB == 'N' ? childB->rows() : childB->cols()));
                            break;
                        }
                    }
                }
            } else {
                for (int jB = 0; jB < col_b; jB++) {
                    for (int iB = 0; iB < row_b; iB++) {
                        char tB = transB;
                        const HMatrix<T> *childB = b->isLeaf() ? b : b->getChildForGEMM(tB, iB, jB);
                        if(childB) {
                            result[jA * col_b + jB] = (tA == 'N' ? childA->cols() : childA->rows())->intersects(*(tB == 'N' ? childB->cols() : childB->rows()));
                            break;
                        }
                    }
                }
            }
        }
    }
    return result;
}

template<typename T> void
HMatrix<T>::recursiveGemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>*b) {
    // Computing a(m,0) * b(0,n) here may give wrong results because of format conversions, exit early
    if(isVoid() || a->isVoid())
        return;

    // None of the matrices is a leaf
    if (!this->isLeaf() && !a->isLeaf() && !b->isLeaf()) {
        int row_a = transA == 'N' ? a->nrChildRow() : a->nrChildCol();
        int col_a = transA == 'N' ? a->nrChildCol() : a->nrChildRow();
        int row_b = transB == 'N' ? b->nrChildRow() : b->nrChildCol();
        int col_b = transB == 'N' ? b->nrChildCol() : b->nrChildRow();
        int row_c = nrChildRow();
        int col_c = nrChildCol();

        // There are 6 nested loops, this may be an issue if there are more
        // than 2 children in each direction; precompute compatibility between
        // blocks to improve performance:
        //   + columns of a and rows of b
        unsigned char * is_compatible_a_b = compatibilityGridForGEMM(a, Axis::COL, transA, b, Axis::ROW, transB);
        //   + rows of a and rows of c
        unsigned char * is_compatible_a_c = compatibilityGridForGEMM(a, Axis::ROW, transA, this, Axis::ROW, 'N');
        //   + columns of b and columns of c
        unsigned char * is_compatible_b_c = compatibilityGridForGEMM(b, Axis::COL, transB, this, Axis::COL, 'N');
        //  With these arrays, we can exit early from loops on iA, jB and l
        //  when blocks are not compatible, and thus there are only 3 real
        //  loops (on i, j, k) and performance penalty should be negligible.
        for (int i = 0; i < row_c; i++) {
            for (int j = 0; j < col_c; j++) {
                HMatrix<T>* child = get(i, j);
                if (!child) { // symmetric/triangular case or empty block coming from symbolic factorisation of sparse matrices
                    continue;
                }

                for (int iA = 0; iA < row_a; iA++) {
                  if (!is_compatible_a_c[iA * row_c + i])
                    continue;
                  for (int jB = 0; jB < col_b; jB++) {
                    if (!is_compatible_b_c[jB * col_c + j])
                      continue;
                    for (int k = 0; k < col_a; k++) {
                      char tA = transA;
                      const HMatrix<T> * childA = a->getChildForGEMM(tA, iA, k);
                      if(!childA)
                        continue;
                      for (int l = 0; l < row_b; l++) {
                        if (!is_compatible_a_b[k * row_b + l])
                          continue;
                        char tB = transB;
                        const HMatrix<T> * childB = b->getChildForGEMM(tB, l, jB);
                        if(childB)
                          child->gemm(tA, tB, alpha, childA, childB, 1);
                      }
                    }
                  }
                }
            }
        }
        delete [] is_compatible_a_b;
        delete [] is_compatible_a_c;
        delete [] is_compatible_b_c;
    } // if (!this->isLeaf() && !a->isLeaf() && !b->isLeaf())
    else
        uncompatibleGemm(transA, transB, alpha, a, b);
}

/**
 * @brief product of 2 H-matrix to a full block.
 *
 * Full blocks may be null and it's that case it's not possible to take
 * a subset from them. We don't want to allocate them before the recursion
 * because the recursion may show that allocation is not needed. This function
 * allow to recurse and bypass the uncompatibleGemm method which would create
 * invalid subset.
 */
template<typename T> void fullHHGemm(HMatrix<T> *c, char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>*b) {
  assert(c->isLeaf());
  assert(!c->isRkMatrix());
  if(!a->isLeaf() && !b->isLeaf()) {
    for (int i = 0; i < (transA=='N' ? a->nrChildRow() : a->nrChildCol()) ; i++) {
      for (int j = 0; j < (transB=='N' ? b->nrChildCol() : b->nrChildRow()) ; j++) {
        const HMatrix<T> *childA, *childB;
        for (int k = 0; k < (transA=='N' ? a->nrChildCol() : a->nrChildRow()) ; k++) {
          char tA = transA;
          char tB = transB;
          childA = a->getChildForGEMM(tA, i, k);
          childB = b->getChildForGEMM(tB, k, j);
          if(childA && childB)
            fullHHGemm(c, tA, tB, alpha, childA, childB);
        }
      }
    }
  } else if(!a->isRecursivelyNull() && !b->isRecursivelyNull()) {
    if(c->full() == NULL)
      c->full(new FullMatrix<T>(c->rows(), c->cols()));
    c->gemm(transA, transB, alpha, a, b, 1);
  }
}

template<typename T> void
HMatrix<T>::leafGemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>*b) {
    assert((transA == 'N' ? *a->cols() : *a->rows()) == ( transB == 'N' ? *b->rows() : *b->cols())); // pour le produit A*B
    assert((transA == 'N' ? *a->rows() : *a->cols()) == *this->rows()); // compatibility of A*B + this : Rows
    assert((transB == 'N' ? *b->cols() : *b->rows()) == *this->cols()); // compatibility of A*B + this : columns

    // One of the matrices is a leaf
    assert(this->isLeaf() || a->isLeaf() || b->isLeaf());

    // the resulting matrix is not a leaf.
    if (!this->isLeaf()) {
        // If the resulting matrix is subdivided then at least one of the matrices of the product is a leaf.
        // One matrix is a RkMatrix
        if (a->isRkMatrix() || b->isRkMatrix()) {
            if ((a->isRkMatrix() && a->isNull())
                    || (b->isRkMatrix() && b->isNull())) {
                return;
            }
            RkMatrix<T>* rkMat = HMatrix<T>::multiplyRkMatrix(lowRankEpsilon(), transA, transB, a, b);
            axpy(alpha, rkMat);
            delete rkMat;
        } else {
            // None of the matrices of the product is a Rk-matrix so one of them is
            // a full matrix so as the result.
            assert(a->isFullMatrix() || b->isFullMatrix());
            FullMatrix<T>* fullMat = HMatrix<T>::multiplyFullMatrix(transA, transB, a, b);
            if(fullMat) {
                axpy(alpha, fullMat);
                delete fullMat;
            }
        }
        return;
    }

    if (isRkMatrix()) {
        // The resulting matrix is a RkMatrix leaf.
        // At least one of the matrix is not a leaf.
        // The different cases are :
        //  a. R += H * H
        //  b. R += H * R
        //  c. R += R * H
        //  d. R += H * M
        //  e. R += M * H
        //  f. R += M * M

        // Cases a, b and c give an Hmatrix which has to be hierarchically converted into a Rkmatrix.
        // Cases c, d, e and f give a RkMatrix
        assert(isRkMatrix());
        assert((transA == 'N' ? *a->cols() : *a->rows()) == (transB == 'N' ? *b->rows() : *b->cols()));
        assert(*rows() == (transA == 'N' ? *a->rows() : *a->cols()));
        assert(*cols() == (transB == 'N' ? *b->cols() : *b->rows()));
        if(rk() == NULL)
            rk(new RkMatrix<T>(NULL, rows(), NULL, cols()));
        rk()->gemmRk(lowRankEpsilon(), transA, transB, alpha, a, b);
        rank_ = rk()->rank();
        return;
    }

    // a, b are H matrices and 'this' is full
    if ( this->isLeaf() && ((!a->isLeaf() && !b->isLeaf()) || isNull()) ) {
      fullHHGemm(this, transA, transB, alpha, a, b);
      return;
    }

    // The resulting matrix is a full matrix
    FullMatrix<T>* fullMat;
    if (a->isRkMatrix() || b->isRkMatrix()) {
        assert(a->isRkMatrix() || b->isRkMatrix());
        if ((a->isRkMatrix() && a->isNull())
                || (b->isRkMatrix() && b->isNull())) {
            return;
        }
        RkMatrix<T>* rkMat = HMatrix<T>::multiplyRkMatrix(lowRankEpsilon(), transA, transB, a, b);
        fullMat = rkMat->eval();
        delete rkMat;
    } else if(a->isLeaf() && b->isLeaf() && isFullMatrix()){
        full()->gemm(transA, transB, alpha, a->full(), b->full(), 1);
        return;
    } else {
      // if a or b is a leaf, it is Full (since Rk have been treated before)
        fullMat = HMatrix<T>::multiplyFullMatrix(transA, transB, a, b);
    }

    // It's not optimal to concider that the result is a FullMatrix but
    // this is a H*F case and it almost never happen
    if(fullMat) {
      if (isFullMatrix()) {
        full()->axpy(alpha, fullMat);
        delete fullMat;
      } else {
        full(fullMat);
        fullMat->scale(alpha);
      }
    }
}

template<typename T>
void HMatrix<T>::gemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>* b, T beta, MainOp) {
  // Computing a(m,0) * b(0,n) here may give wrong results because of format conversions, exit early
  if(isVoid() || a->isVoid())
      return;

  // This and B are Rk matrices with the same panel 'b' -> the gemm is only applied on the panels 'a'
  if(isRkMatrix() && !isNull() && b->isRkMatrix() && !b->isNull() && rk()->b == b->rk()->b) {
    // Ca * CbT = beta * Ca * CbT + alpha * A * Ba * BbT
    // As Cb = Bb we get
    // Ca = beta * Ca + alpha A * Ba with only Ca and Ba scalar arrays
    // We support C and B not compatible (larger) with A so we first slice them
    assert(transB == 'N');
    const IndexSet * r = transA == 'N' ? a->rows() : a->cols();
    const IndexSet * c = transA == 'N' ? a->cols() : a->rows();
    ScalarArray<T> cSubset(rk()->a->rowsSubset( r->offset() -    rows()->offset(), r->size()));
    ScalarArray<T> bSubset(b->rk()->a->rowsSubset( c->offset() - b->rows()->offset(), c->size()));
    a->gemv(transA, alpha, &bSubset, beta, &cSubset);
    return;
  }

  // This and A are Rk matrices with the same panel 'a' -> the gemm is only applied on the panels 'b'
  if(isRkMatrix() && !isNull() && a->isRkMatrix() && !a->isNull() && rk()->a == a->rk()->a) {
    // Ca * CbT = beta * Ca * CbT + alpha * Aa * AbT * B
    // As Ca = Aa we get
    // CbT = beta * CbT + alpha AbT * B with only Cb and Ab scalar arrays
    // we transpose:
    // Cb = beta * Cb + alpha BT * Ab
    // We support C and B not compatible (larger) with A so we first slice them
    assert(transA == 'N');
    assert(transB != 'C');
    const IndexSet * r = transB == 'N' ? b->rows() : b->cols();
    const IndexSet * c = transB == 'N' ? b->cols() : b->rows();
    ScalarArray<T> cSubset(rk()->b->rowsSubset( c->offset() -    cols()->offset(), c->size()));
    ScalarArray<T> aSubset(a->rk()->b->rowsSubset( r->offset() - a->cols()->offset(), r->size()));
    b->gemv(transB == 'N' ? 'T' : 'N', alpha, &aSubset, beta, &cSubset);
    return;
  }

  this->scale(beta);

  if((a->isLeaf() && (!a->isAssembled() || a->isNull())) ||
     (b->isLeaf() && (!b->isAssembled() || b->isNull()))) {
      if(!isAssembled() && this->isLeaf())
          rk(new RkMatrix<T>(NULL, rows(), NULL, cols()));
      return;
  }

  // Once the scaling is done, beta is reset to 1
  // to avoid an other scaling.
  recursiveGemm(transA, transB, alpha, a, b);
}

template<typename T>
FullMatrix<T>* multiplyFullH(char transM, char transH,
                                         const FullMatrix<T>* mat,
                                         const HMatrix<T>* h) {
  assert(transH != 'C');
  FullMatrix<T>* resultT;
  if(transM == 'C') {
    // R = M* * H = (H^t * conj(M))^t
    FullMatrix<T>* matT = mat->copy();
    matT->conjugate();
    resultT = multiplyHFull(transH == 'N' ? 'T' : 'N',
                            'N', h, matT);
    delete matT;
  } else {
    // R = M * H = (H^t * M^t*)^t
    resultT = multiplyHFull(transH == 'N' ? 'T' : 'N',
                            transM == 'N' ? 'T' : 'N',
                            h, mat);
  }
  if(resultT != NULL)
    resultT->transpose();
  return resultT;
}

template<typename T> bool HMatrix<T>::isRecursivelyNull() const {
  if(this->isLeaf())
    return isNull();
  else for(int i = 0; i < this->nrChild(); i++) {
    if(this->getChild(i) && !this->getChild(i)->isRecursivelyNull())
      return false;
  }
  return true;
}

template<typename T>
FullMatrix<T>* multiplyHFull(char transH, char transM,
                                         const HMatrix<T>* h,
                                         const FullMatrix<T>* mat) {
  assert((transH == 'N' ? h->cols()->size() : h->rows()->size())
           == (transM == 'N' ? mat->rows() : mat->cols()));
  if(h->isRecursivelyNull())
    return NULL;
  FullMatrix<T>* result =
    new FullMatrix<T>((transH == 'N' ? h->rows() : h->cols()),
                      (transM == 'N' ? mat->cols_ : mat->rows_));
  if (transM == 'N') {
    h->gemv(transH, 1, mat, 0, result);
  } else {
    FullMatrix<T>* matT = mat->copyAndTranspose();
    if (transM == 'C') {
      matT->conjugate();
    }
    h->gemv(transH, 1, matT, 0, result);
    delete matT;
  }
  return result;
}

template<typename T>
RkMatrix<T>* HMatrix<T>::multiplyRkMatrix(double epsilon, char transA, char transB, const HMatrix<T>* a, const HMatrix<T>* b){
  // We know that one of the matrices is a RkMatrix
  assert(a->isRkMatrix() || b->isRkMatrix());
  RkMatrix<T> *rk = NULL;
  // Matrices range compatibility
  if((transA == 'N') && (transB == 'N'))
    assert(a->cols()->size() == b->rows()->size());
  if((transA != 'N') && (transB == 'N'))
    assert(a->rows()->size() == b->rows()->size());
  if((transA == 'N') && (transB != 'N'))
    assert(a->cols()->size() == b->cols()->size());

  // The cases are:
  //  - A Rk, B H
  //  - A H,  B Rk
  //  - A Rk, B Rk
  //  - A Rk, B F
  //  - A F,  B Rk
  if (a->isRkMatrix() && !b->isLeaf()) {
    rk = RkMatrix<T>::multiplyRkH(transA, transB, a->rk(), b);
    HMAT_ASSERT(rk);
  }
  else if (!a->isLeaf() && b->isRkMatrix()) {
    rk = RkMatrix<T>::multiplyHRk(transA, transB, a, b->rk());
    HMAT_ASSERT(rk);
  }
  else if (a->isRkMatrix() && b->isRkMatrix()) {
    rk = RkMatrix<T>::multiplyRkRk(transA, transB, a->rk(), b->rk(), epsilon);
    HMAT_ASSERT(rk);
  }
  else if (a->isRkMatrix() && b->isFullMatrix()) {
    rk = RkMatrix<T>::multiplyRkFull(transA, transB, a->rk(), b->full());
    HMAT_ASSERT(rk);
  }
  else if (a->isFullMatrix() && b->isRkMatrix()) {
    rk = RkMatrix<T>::multiplyFullRk(transA, transB, a->full(), b->rk());
    HMAT_ASSERT(rk);
  } else if(a->isNull() || b->isNull()) {
    return new RkMatrix<T>(NULL, transA ? a->cols() : a->rows(),
                           NULL, transB ? b->rows() : b->cols());
  } else {
    // None of the above cases, impossible.
    HMAT_ASSERT(false);
  }
  return rk;
}

template<typename T>
FullMatrix<T>* HMatrix<T>::multiplyFullMatrix(char transA, char transB,
                                              const HMatrix<T>* a,
                                              const HMatrix<T>* b) {
  // At least one full matrix, and not RkMatrix.
  assert(a->isFullMatrix() || b->isFullMatrix());
  assert(!(a->isRkMatrix() || b->isRkMatrix()));
  FullMatrix<T> *result = NULL;
  // The cases are:
  //  - A H, B F
  //  - A F, B H
  //  - A F, B F
  if (!a->isLeaf() && b->isFullMatrix()) {
    result = multiplyHFull(transA, transB, a, b->full());
  } else if (a->isFullMatrix() && !b->isLeaf()) {
    result = multiplyFullH(transA, transB, a->full(), b);
  } else if (a->isFullMatrix() && b->isFullMatrix()) {
    const IndexSet* aRows = (transA == 'N')? a->rows() : a->cols();
    const IndexSet* bCols = (transB == 'N')? b->cols() : b->rows();
    result = new FullMatrix<T>(aRows, bCols);
    result->gemm(transA, transB, 1, a->full(), b->full(),
                 0);
  } else if(a->isNull() || b->isNull()) {
    return NULL;
  } else {
    // None of above, impossible
    HMAT_ASSERT(false);
  }
  return result;
}

template<typename T>
void HMatrix<T>::multiplyWithDiag(const HMatrix<T>* d, Side side, bool inverse) const {
  assert(*d->rows() == *d->cols());
  assert(side == Side::LEFT  || (*cols() == *d->rows()));
  assert(side == Side::RIGHT || (*rows() == *d->cols()));

  if (isVoid()) return;

  // The symmetric matrix must be taken into account: lower or upper
  if (!this->isLeaf()) {
    if (d->isLeaf()) {
      for (int i=0 ; i<std::min(nrChildRow(), nrChildCol()) ; i++)
        if(get(i,i))
          get(i,i)->multiplyWithDiag(d, side, inverse);
      for (int i=0 ; i<nrChildRow() ; i++)
        for (int j=0 ; j<nrChildCol() ; j++)
          if (i!=j && get(i,j)) {
            get(i,j)->multiplyWithDiag(d, side, inverse);
          }
      return;
    }

    // First the diagonal, then the rest...
    for (int i=0 ; i<std::min(nrChildRow(), nrChildCol()) ; i++)
      if(get(i,i))
        get(i,i)->multiplyWithDiag(d->get(i,i), side, inverse);
    for (int i=0 ; i<nrChildRow() ; i++)
      for (int j=0 ; j<nrChildCol() ; j++)
        if (i!=j && get(i,j)) {
        int k = side == Side::LEFT ? i : j;
        get(i,j)->multiplyWithDiag(d->get(k,k), side, inverse);
        // TODO couldn't we handle this case with the previous one, using getChildForGEMM(d,i,i) that returns 'd' itself when 'd' is a leaf ?
    }
  } else if (isRkMatrix() && !isNull()) {
    rk()->multiplyWithDiagOrDiagInv(d, inverse, side);
  } else if(isFullMatrix()){
    if (d->isFullMatrix()) {
      full()->multiplyWithDiagOrDiagInv(d->full()->diagonal, inverse, side);
    } else {
      Vector<T> diag(d->rows()->size());
      d->extractDiagonal(diag.ptr());
      full()->multiplyWithDiagOrDiagInv(&diag, inverse, side);
    }
  } else {
    // this is a null matrix (either full of Rk) so nothing to do
  }
}

template<typename T> void HMatrix<T>::transposeMeta(bool temporaryOnly) {
    if(temporaryOnly && !temporary_)
      return;
    // called by HMatrix<T>::transpose() and HMatrixHandle<T>::transpose()
    // if the matrix is symmetric, inverting it(Upper/Lower)
    if (isLower || isUpper) {
        isLower = !isLower;
        isUpper = !isUpper;
    }
    // if the matrix is triangular, on inverting it (isTriUpper/isTriLower)
    if (isTriLower || isTriUpper) {
        isTriLower = !isTriLower;
        isTriUpper = !isTriUpper;
    }
    // Warning: nrChildRow() uses keepSameRows and rows_
    bool tmp = keepSameCols; // can't use swap on bitfield so manual swap...
    keepSameCols = keepSameRows;
    keepSameRows = tmp;
    swap(rows_, cols_);
    RecursionMatrix<T, HMatrix<T> >::transposeMeta(temporaryOnly);
}

template <typename T> void HMatrix<T>::transposeData() {
    if (this->isLeaf()) {
        if (isRkMatrix() && rk()) {
            rk()->transpose();
        } else if (isFullMatrix()) {
            full()->transpose();
        }
    } else {
        for (int i = 0; i < this->nrChild(); i++)
            if (this->getChild(i))
                this->getChild(i)->transposeData();
    }
}

template<typename T> void HMatrix<T>::transpose() {
    transposeData();
    transposeMeta();
}

template<> void HMatrix<S_t>::conjugate() {}
template<> void HMatrix<D_t>::conjugate() {}
template<typename T> void HMatrix<T>::conjugate() {
  std::vector<const HMatrix<T> *> stack;
  stack.push_back(this);
  while(!stack.empty()) {
    const HMatrix<T> * m = stack.back();
    stack.pop_back();
    if(!m->isLeaf()) {
      for(int i = 0; i < m->nrChild(); i++) {
        if(m->getChild(i) != NULL)
          stack.push_back(m->getChild(i));
      }
    } else if(m->isNull()) {
      // nothing to do
    } else if(m->isRkMatrix()) {
      m->rk()->conjugate();
    } else {
      m->full()->conjugate();
    }
  }
}

template<typename T>
void HMatrix<T>::copyAndTranspose(const HMatrix<T>* o) {
  assert(o);
  assert(*this->rows() == *o->cols());
  assert(*this->cols() == *o->rows());
  assert(this->isLeaf() == o->isLeaf());

  if (this->isLeaf()) {
    if (o->isRkMatrix()) {
      assert(!isFullMatrix());
      if (rk()) {
        delete rk();
      }
      RkMatrix<T>* newRk = o->rk()->copy();
      newRk->transpose();
      rk(newRk);
    } else {
      if (isFullMatrix()) {
        delete full();
      }
      const FullMatrix<T>* oF = o->full();
      if(oF == NULL) {
        full(NULL);
      } else {
        full(oF->copyAndTranspose());
        if (oF->diagonal) {
          if (!full()->diagonal) {
            full()->diagonal = new Vector<T>(oF->rows());
            HMAT_ASSERT(full()->diagonal);
          }
          oF->diagonal->copy(full()->diagonal);
        }
      }
    }
  } else {
    for (int i=0 ; i<nrChildRow() ; i++)
      for (int j=0 ; j<nrChildCol() ; j++)
        if (get(i,j) && o->get(j, i))
          get(i, j)->copyAndTranspose(o->get(j, i));
  }
}

template<typename T>
void HMatrix<T>::truncate() {
  if (this->isLeaf()) {
    if (this->isRkMatrix()) {
      if (rk()) {
        rk()->truncate(localSettings.epsilon_);
        rank_ = rk()->rank();
      }
    }
  } else {
    for (int i = 0; i < this->nrChild(); i++) {
      HMatrix<T>* child = this->getChild(i);
      if (child) {
        child->truncate();
      }
    }
  }
}

template<typename T>
const ClusterData* HMatrix<T>::rows() const {
  return &(rows_->data);
}

template<typename T>
const ClusterData* HMatrix<T>::cols() const {
  return &(cols_->data);
}

template<typename T> HMatrix<T>* HMatrix<T>::copy() const {
  HMatrix<T>* M=Zero(this);
  M->copy(this);
  return M;
}

// Copy the data of 'o' into 'this'
// The structure of both H-matrix is supposed to be allready similar
template<typename T>
void HMatrix<T>::copy(const HMatrix<T>* o) {
  DECLARE_CONTEXT;

  assert(*rows() == *o->rows());
  assert(*cols() == *o->cols());

  isLower = o->isLower;
  isUpper = o->isUpper;
  isTriUpper = o->isTriUpper;
  isTriLower = o->isTriLower;
  approximateRank_ = o->approximateRank_;
  if (this->isLeaf()) {
    assert(o->isLeaf());
    if (isAssembled() && isNull() && o->isNull()) {
      return;
    }
    // When the matrix was not allocated but only the structure
    if (o->isFullMatrix() && isFullMatrix()) {
      o->full()->copy(full());
    } else if(o->isFullMatrix()) {
      assert(!isAssembled() || isNull());
      full(o->full()->copy());
    } else if (o->isRkMatrix() && !rk()) {
      rk(new RkMatrix<T>(NULL, o->rk()->rows, NULL, o->rk()->cols));
    }
    assert((isRkMatrix() == o->isRkMatrix())
           && (isFullMatrix() == o->isFullMatrix()));
    if (o->isRkMatrix()) {
      rk()->copy(o->rk());
      rank_ = rk()->rank();
    }
  } else {
    assert(o->rank_==NONLEAF_BLOCK);
    rank_ = o->rank_;
    for (int i = 0; i < o->nrChild(); i++) {
        if (o->getChild(i)) {
          assert(this->getChild(i));
          this->getChild(i)->copy(o->getChild(i));
        } else {
          assert(!this->getChild(i));
      }
    }
  }
}

template<typename T>
void HMatrix<T>::lowRankEpsilon(double epsilon, bool recursive) {
  localSettings.epsilon_ = epsilon;
  if(recursive && !this->isLeaf()) {
    for (int i = 0; i < this->nrChild(); i++) {
      HMatrix<T>* child = this->getChild(i);
      if (child)
        child->lowRankEpsilon(epsilon);
    }
  }
}

template<typename T> void HMatrix<T>::clear() {
  if(!this->isLeaf()) {
    for (int i = 0; i < this->nrChild(); i++) {
      HMatrix<T>* child = this->getChild(i);
      if (child)
        child->clear();
    }
  } else if(isRkMatrix()) {
    if(rk())
      delete rk();
    rk(NULL);
  } else if(isFullMatrix()) {
    delete full();
    full(NULL);
  }
}

template<typename T>
void HMatrix<T>::inverse() {
  DECLARE_CONTEXT;

  HMAT_ASSERT_MSG(!isLower, "HMatrix::inverse not available for symmetric matrices");

  if (this->isLeaf()) {
    assert(isFullMatrix());
    full()->inverse();
  } else {

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

    if (isLower) {

      vector<HMatrix<T>*> TM(nrChildCol());
    for (int k=0 ; k<nrChildRow() ; k++){
      // Inverse M_kk
      get(k,k)->inverse();
      // Update line 'k' = left-multiplied by M_kk-1
        for (int j=0 ; j<nrChildCol() ; j++) {

            // Mkj <- Mkk^-1 Mkj we use a temp matrix X because this type of product is not allowed with gemm (beta=0 erases Mkj before using it !)
          if (j<k) { // under the diag we store TMj=Mkj
            TM[j] = get(k,j)->copy();
            get(k,j)->gemm('N', 'N', 1, get(k,k), TM[j], 0);
          } else if (j>k) { // above the diag : Mkj = t Mjk, we store TMj=-Mjk.tMkk-1 = -Mjk.Mkk-1 (Mkk est sym)
            TM[j] = Zero(get(j,k));
            TM[j]->gemm('N', 'T', -1, get(j,k), get(k,k), 0);
          }
        }
      // Update the rest of matrix M
      for (int i=0 ; i<nrChildRow() ; i++)
          // line 'i' -= Mik x line 'k' (which has just been multiplied by Mkk-1)
        for (int j=0 ; j<nrChildCol() ; j++)
            if (i!=k && j!=k && j<=i) {
              // Mij <- Mij - Mik (Mkk^-1 Mkj) (with Mkk-1.Mkj allready stored in Mkj and TMj=Mjk.tMkk-1)
              // cas k < j <     i        Mkj n'existe pas, on prend t{TMj} = -Mkk-1Mkj
              if (k<j)
                get(i,j)->gemm('N', 'T', 1, get(i,k), TM[j], 1);
              // cas     j < k < i        Toutes les matrices existent sous la diag
              else if (k<i)
            get(i,j)->gemm('N', 'N', -1, get(i,k), get(k,j), 1);
              // cas     j <     i < k    Mik n'existe pas, on prend TM[i] = Mki
              else
                get(i,j)->gemm('T', 'N', -1, TM[i], get(k,j), 1);
            }
      // Update column 'k' = right-multiplied by -M_kk-1
      for (int i=0 ; i<nrChildRow() ; i++)
          if (i>k) {
          // Mik <- - Mik Mkk^-1
            get(i,k)->copy(TM[i]);
  }
        for (int j=0 ; j<nrChildCol() ; j++) {
          delete TM[j];
          TM[j]=NULL;
        }
      }
    } else {
      this->recursiveInverseNosym();
  }
  }
}

template<typename T>
void HMatrix<T>::solveLowerTriangularLeft(HMatrix<T>* b, Factorization algo, Diag diag, Uplo uplo, MainOp) const {
  DECLARE_CONTEXT;
  if (isVoid()) return;
  // At first, the recursion one (simple case)
  if (!this->isLeaf() && !b->isLeaf()) {
    this->recursiveSolveLowerTriangularLeft(b, algo, diag, uplo);
  } else if(!b->isLeaf()) {
    // B isn't a leaf, then 'this' is one
    assert(this->isLeaf());
    // Evaluate B as a full matrix, solve, and restore in the matrix
    // TODO: check if it's not too bad
    FullMatrix<T> bFull(b->rows(), b->cols());
    b->evalPart(&bFull, b->rows(), b->cols());
    this->solveLowerTriangularLeft(&bFull, algo, diag, uplo);
    b->clear();
    b->axpy(1, &bFull);
  } else if(b->isNull()) {
    // nothing to do
  } else {
    if (b->isFullMatrix()) {
      this->solveLowerTriangularLeft(b->full(), algo, diag, uplo);
    } else {
      assert(b->isRkMatrix());
      HMatrix<T> * bSubset = b->subset(uplo == Uplo::LOWER ? this->cols() : this->rows(), b->cols());
      this->solveLowerTriangularLeft(bSubset->rk()->a, algo, diag, uplo);
      if(bSubset != b)
          delete bSubset;
    }
  }
}

template<typename T>
void HMatrix<T>::solveLowerTriangularLeft(ScalarArray<T>* b, Factorization algo, Diag diag, Uplo uplo) const {
  DECLARE_CONTEXT;
  assert(*rows() == *cols());
  assert(cols()->size() == b->rows);
  if (isVoid()) return;
  if (this->isLeaf()) {
    assert(this->isFullMatrix());
    full()->solveLowerTriangularLeft(b, algo, diag, uplo);
  } else {
    //  Forward substitution:
    //  [ L11 |  0  ]   [ X1 ]   [ b1 ]
    //  [ ----+---- ] * [----] = [ -- ]
    //  [ L21 | L22 ]   [ X2 ]   [ b2 ]
    //
    //  L11 * X1 = b1 (by recursive forward substitution)
    //  L21 * X1 + L22 * X2 = b2 (forward substitution of L22*X2=b2-L21*X1)
    //

    int offset(0);
    vector<ScalarArray<T> > sub;
    for (int i=0 ; i<nrChildRow() ; i++) {
      // Create sub[i] = a ScalarArray (without copy of data) for the rows in front of the i-th matrix block
      sub.push_back(ScalarArray<T>(*b, offset, get(i, i)->cols()->size(), 0, b->cols));
      offset += get(i, i)->cols()->size();
      // Update sub[i] with the contribution of the solutions already computed sub[j] j<i
      for (int j=0 ; j<i ; j++) {
        const HMatrix<T>* u_ji = (uplo == Uplo::LOWER ? get(i, j) : get(j, i));
        if (u_ji)
          u_ji->gemv(uplo == Uplo::LOWER ? 'N' : 'T', -1, &sub[j], 1, &sub[i]);
      }
      // Solve the i-th diagonal system
      get(i, i)->solveLowerTriangularLeft(&sub[i], algo, diag, uplo);
    }
  }
}

template<typename T>
void HMatrix<T>::solveLowerTriangularLeft(FullMatrix<T>* b, Factorization algo, Diag diag, Uplo uplo) const {
  solveLowerTriangularLeft(&b->data, algo, diag, uplo);
}

template<typename T>
void HMatrix<T>::solveUpperTriangularRight(HMatrix<T>* b, Factorization algo, Diag diag, Uplo uplo) const {
  DECLARE_CONTEXT;
  if (rows()->size() == 0 || cols()->size() == 0) return;
  // The recursion one (simple case)
  if (!this->isLeaf() && !b->isLeaf()) {
    this->recursiveSolveUpperTriangularRight(b, algo, diag, uplo);
  } else if(!b->isLeaf()) {
    // B isn't a leaf, then 'this' is one
    assert(this->isLeaf());
    assert(isFullMatrix());
    // Evaluate B, solve by column and restore all in the matrix
    // TODO: check if it's not too bad
    FullMatrix<T> bFull(b->rows(), b->cols());
    b->evalPart(&bFull, b->rows(), b->cols());
    this->solveUpperTriangularRight(&bFull, algo, diag, uplo);
    b->clear();
    b->axpy(1, &bFull);
  } else if(b->isNull()) {
    // nothing to do
  } else {
    if (b->isFullMatrix()) {
      this->solveUpperTriangularRight(b->full(), algo, diag, uplo);
    } else {
      assert(b->isRkMatrix());
      // Xa Xb^t U = Ba Bb^t
      //   - Xa = Ba
      //   - Xb^t U = Bb^t
      // Xb is stored without being transposed, thus we solve
      // U^t Xb = Bb instead
      HMatrix<T> * tmp = b->subset(b->rows(), uplo == Uplo::LOWER ? this->cols() : this->rows());
      this->solveLowerTriangularLeft(tmp->rk()->b, algo, diag, uplo);
      if(tmp != b)
          delete tmp;
    }
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangularRight(ScalarArray<T>* b, Factorization algo, Diag diag, Uplo uplo) const {
  DECLARE_CONTEXT;
  assert(*rows() == *cols());
  assert(rows()->size() == b->cols);
  if (isVoid()) return;
  if (this->isLeaf()) {
    assert(this->isFullMatrix());
    full()->solveUpperTriangularRight(b, algo, diag, uplo);
  } else {
    //  Forward substitution:
    //                [ U11 | U12 ]
    //  [ X1 | X2 ] * [ ----+---- ] = [ b1 | b2 ]
    //                [  0  | U22 ]
    //
    //  X1 * U11 = b1 (by recursive forward substitution)
    //  X1 * U12 + X2 * U22 = b2 (forward substitution of X2*U22=b2-X1*U12)
    //

    int offset(0);
    vector<ScalarArray<T> > sub;
    for (int i=0 ; i<nrChildCol() ; i++) {
      // Create sub[i] = a ScalarArray (without copy of data) for the columns in front of the i-th matrix block
      sub.push_back(ScalarArray<T>(*b, 0, b->rows, offset, get(i, i)->rows()->size()));
      offset += get(i, i)->rows()->size();
      // Update sub[i] with the contribution of the solutions already computed sub[j]
      for (int j=0 ; j<i ; j++) {
        const HMatrix<T>* u_ji = (uplo == Uplo::LOWER ? get(i, j) : get(j, i));
        if (u_ji)
          u_ji->gemv(uplo == Uplo::LOWER ? 'T' : 'N', -1, &sub[j], 1, &sub[i], Side::RIGHT);
      }
      // Solve the i-th diagonal system
      get(i, i)->solveUpperTriangularRight(&sub[i], algo, diag, uplo);
    }
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangularRight(FullMatrix<T>* b, Factorization algo, Diag diag, Uplo uplo) const {
  solveUpperTriangularRight(&b->data, algo, diag, uplo);
}

/* Resolve U.X=B, solution saved in B, with B Hmat
   Only called by luDecomposition
 */
template<typename T>
void HMatrix<T>::solveUpperTriangularLeft(HMatrix<T>* b, Factorization algo, Diag diag, Uplo uplo, MainOp) const {
  DECLARE_CONTEXT;
  if (rows()->size() == 0 || cols()->size() == 0) return;
  // At first, the recursion one (simple case)
  if (!this->isLeaf() && !b->isLeaf()) {
    this->recursiveSolveUpperTriangularLeft(b, algo, diag, uplo);
  } else if(!b->isLeaf()) {
    // B isn't a leaf, then 'this' is one
    assert(this->isLeaf());
    // Evaluate B, solve by column, and restore in the matrix
    // TODO: check if it's not too bad
    FullMatrix<T> bFull(b->rows(), b->cols());
    b->evalPart(&bFull, b->rows(), b->cols());
    this->solveUpperTriangularLeft(&bFull, algo, diag, uplo);
    b->clear();
    b->axpy(1, &bFull);
  } else if(b->isNull()) {
    // nothing to do
  } else {
    if (b->isFullMatrix()) {
      this->solveUpperTriangularLeft(b->full(), algo, diag, uplo);
    } else {
      assert(b->isRkMatrix());
      HMatrix * bSubset = b->subset(uplo == Uplo::LOWER ? this->rows() : this->cols(), b->cols());
      this->solveUpperTriangularLeft(bSubset->rk()->a, algo, diag, uplo);
      if(bSubset != b)
          delete bSubset;
    }
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangularLeft(ScalarArray<T>* b, Factorization algo, Diag diag, Uplo uplo) const {
  DECLARE_CONTEXT;
  assert(*rows() == *cols());
  assert(rows()->size() == b->rows || uplo == Uplo::UPPER);
  assert(cols()->size() == b->rows || uplo == Uplo::LOWER);
  if (rows()->size() == 0 || cols()->size() == 0) return;
  if (this->isLeaf()) {
    full()->solveUpperTriangularLeft(b, algo, diag, uplo);
  } else {
    //  Backward substitution:
    //  [ U11 | U12 ]   [ X1 ]   [ b1 ]
    //  [ ----+---- ] * [----] = [ -- ]
    //  [  0  | U22 ]   [ X2 ]   [ b2 ]
    //
    //  U22 * X2 = b12(by recursive backward substitution)
    //  U11 * X1 + U12 * X2 = b1 (backward substitution of U11*X1=b1-U12*X2)
    //

    int offset(0);
    vector<ScalarArray<T> > sub;
    for (int i=0 ; i<nrChildRow() ; i++) {
      // Create sub[i] = a ScalarArray (without copy of data) for the rows in front of the i-th matrix block
      sub.push_back(b->rowsSubset(offset, get(i, i)->cols()->size()));
      offset += get(i, i)->cols()->size();
    }
    for (int i=nrChildRow()-1 ; i>=0 ; i--) {
      // Solve the i-th diagonal system
      get(i, i)->solveUpperTriangularLeft(&sub[i], algo, diag, uplo);
      // Update sub[j] j<i with the contribution of the solutions just computed sub[i]
      for (int j=0 ; j<i ; j++) {
        const HMatrix<T>* u_ji = (uplo == Uplo::LOWER ? get(i, j) : get(j, i));
        if (u_ji)
          u_ji->gemv(uplo == Uplo::LOWER ? 'T' : 'N', -1, &sub[i], 1, &sub[j]);
      }
    }
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangularLeft(FullMatrix<T>* b, Factorization algo, Diag diag, Uplo uplo) const {
  solveUpperTriangularLeft(&b->data, algo, diag, uplo);
}

template<typename T> void HMatrix<T>::lltDecomposition(hmat_progress_t * progress) {

    assertLower(this);
    if (isVoid()) {
        // nothing to do
    } else if(this->isLeaf()) {
        full()->lltDecomposition();
        if(progress != NULL) {
            progress->current= rows()->offset() + rows()->size();
            progress->update(progress);
        }
    } else {
        HMAT_ASSERT(isLower);
      this->recursiveLltDecomposition(progress);
    }
    isTriLower = true;
    isLower = false;
}

template<typename T>
void HMatrix<T>::luDecomposition(hmat_progress_t * progress) {
  DECLARE_CONTEXT;

  if (rows()->size() == 0 || cols()->size() == 0) return;
  if (this->isLeaf()) {
    assert(isFullMatrix());
    full()->luDecomposition();
    full()->checkNan();
    if(progress != NULL) {
      progress->current= rows()->offset() + rows()->size();
      progress->update(progress);
    }
  } else {
    this->recursiveLuDecomposition(progress);
  }
}

template<typename T>
void HMatrix<T>::mdntProduct(const HMatrix<T>* m, const HMatrix<T>* d, const HMatrix<T>* n) {
  DECLARE_CONTEXT;

  HMatrix<T>* x = m->copy();
  x->multiplyWithDiag(d); // x=M.D
  this->gemm('N', 'T', -1, x, n, 1); // this -= M.D.tN
  delete x;
}

template<typename T>
void HMatrix<T>::mdmtProduct(const HMatrix<T>* m, const HMatrix<T>* d) {
  DECLARE_CONTEXT;
  if (isVoid() || d->isVoid() || m->isVoid()) return;
  // this <- this - M * D * M^T
  //
  // D is stored separately in full matrix of diagonal leaves (see full_matrix.hpp).
  // this is symmetric and stored as lower triangular.
  // Warning: d must be the result of an ldlt factorization
  assertLower(this);
  assert(*d->rows() == *d->cols());       // D is square
  assert(*this->rows() == *this->cols()); // this is square
  assert(*m->cols() == *d->rows());       // Check if we can have the produit M*D and D*M^T
  assert(*this->rows() == *m->rows());

  if(!this->isLeaf()) {
    if (!m->isLeaf()) {
      this->recursiveMdmtProduct(m, d);
    } else if (m->isRkMatrix() && !m->isNull()) {
      HMatrix<T>* m_copy = m->copy();

      assert(*m->cols() == *d->rows());
      assert(*m_copy->rk()->cols == *d->rows());
      m_copy->multiplyWithDiag(d); // right multiplication by D
      RkMatrix<T>* rkMat = RkMatrix<T>::multiplyRkRk('N', 'T', m_copy->rk(), m->rk(), m->lowRankEpsilon());
      delete m_copy;

      this->axpy(-1, rkMat);
      delete rkMat;
    } else if(m->isFullMatrix()){
      HMatrix<T>* copy_m = m->copy();
      HMAT_ASSERT(copy_m);
      copy_m->multiplyWithDiag(d); // right multiplication by D

      FullMatrix<T>* fullMat = HMatrix<T>::multiplyFullMatrix('N', 'T', copy_m, m);
      HMAT_ASSERT(fullMat);
      delete copy_m;

      this->axpy(-1, fullMat);
      delete fullMat;
    } else {
      // m is a null matrix (either Rk or Full) so nothing to do.
    }
  } else {
    assert(isFullMatrix());
    if (m->isRkMatrix()) {
      // this : full
      // m    : rk
      // Strategy: compute mdm^T as FullMatrix and then do this<-this - mdm^T

      // 1) copy  m = AB^T : m_copy
      // 2) m_copy <- m_copy * D    (multiplyWithDiag)
      // 3) rkMat <- multiplyRkRk ( m_copy , m^T)
      // 4) fullMat <- evaluation as a FullMatrix of the product rkMat = (A*(D*B)^T) * (A*B^T)^T
      // 5) this <- this - fullMat
      if (!m->isNull()) {
        HMatrix<T>* m_copy = m->copy();
        m_copy->multiplyWithDiag(d);

        RkMatrix<T>* rkMat = RkMatrix<T>::multiplyRkRk('N', 'T', m_copy->rk(), m->rk(), m->lowRankEpsilon());
        FullMatrix<T>* fullMat = rkMat->eval();
        delete m_copy;
        delete rkMat;
        full()->axpy(-1, fullMat);
        delete fullMat;
      }
    } else if (m->isFullMatrix()) {
      // S <- S - M*D*M^T
      assert(!full()->isTriUpper());
      assert(!full()->isTriLower());
      assert(!m->full()->isTriUpper());
      assert(!m->full()->isTriLower());
      FullMatrix<T> mTmp(m->rows(), m->cols());
      mTmp.copyMatrixAtOffset(m->full(), 0, 0);
      if (d->isFullMatrix()) {
        mTmp.multiplyWithDiagOrDiagInv(d->full()->diagonal, false, Side::RIGHT);
      } else {
        Vector<T> diag(d->cols()->size());
        d->extractDiagonal(diag.ptr());
        mTmp.multiplyWithDiagOrDiagInv(&diag, false, Side::RIGHT);
      }
      full()->gemm('N', 'T', -1, &mTmp, m->full(), 1);
    } else if (!m->isLeaf()){
      FullMatrix<T> mTmp(m->rows(), m->cols());
      m->evalPart(&mTmp, m->rows(), m->cols());
      FullMatrix<T> mTmpCopy(m->rows(), m->cols());
      mTmpCopy.copyMatrixAtOffset(&mTmp, 0, 0);
      if (d->isFullMatrix()) {
        mTmp.multiplyWithDiagOrDiagInv(d->full()->diagonal, false, Side::RIGHT);
      } else {
        Vector<T> diag(d->cols()->size());
        d->extractDiagonal(diag.ptr());
        mTmp.multiplyWithDiagOrDiagInv(&diag, false, Side::RIGHT);
      }
      full()->gemm('N', 'T', -1, &mTmp, &mTmpCopy, 1);
    }
  }
}

template<typename T> void assertLdlt(const HMatrix<T> * me) {
    // Void block (row & col)
    if (me->rows()->size() == 0 && me->cols()->size() == 0) return;
#ifdef DEBUG_LDLT
    assert(me->isTriLower);
    if (me->isLeaf()) {
        assert(me->isFullMatrix());
        assert(me->full()->diagonal);
    } else {
      for (int i=0 ; i<me->nrChildRow() ; i++)
        assertLdlt(me->get(i,i));
    }
#else
    ignore_unused_arg(me);
#endif
}

template<typename T> void assertLower(const HMatrix<T> * me) {
#ifdef DEBUG_LDLT
    if (me->isLeaf()) {
        return;
    } else {
        assert(me->isLower);
        for (int i=0 ; i<me->nrChildRow() ; i++)
          for (int j=0 ; j<me->nrChildCol() ; j++) {
            if (i<j) /* NULL above diag */
              assert(!me->get(i,j));
            if (i==j) /* Lower on diag */
              assertLower(me->get(i,i));
          }
    }
#else
    ignore_unused_arg(me);
#endif
}

template<typename T> void assertUpper(const HMatrix<T> * me) {
#ifdef DEBUG_LDLT
    if (me->isLeaf()) {
        return;
    } else {
        assert(me->isUpper);
        for (int i=0 ; i<me->nrChildRow() ; i++)
          for (int j=0 ; j<me->nrChildCol() ; j++) {
            if (i==j) /* Upper on diag */
              assertUpper(me->get(i,i));
          }
    }
#else
    ignore_unused_arg(me);
#endif
}

template<typename T>
void HMatrix<T>::ldltDecomposition(hmat_progress_t * progress) {
  DECLARE_CONTEXT;
  assertLower(this);

  if (isVoid()) {
    // nothing to do
  } else if (this->isLeaf()) {
    //The basic case of the recursion is necessarily a full matrix leaf
    //since the recursion is done with *rows() == *cols().

    assert(isFullMatrix());
    full()->ldltDecomposition();
    if(progress != NULL) {
        progress->current= rows()->offset() + rows()->size();
        progress->update(progress);
    }
    assert(full()->diagonal);
  } else {
    this->recursiveLdltDecomposition(progress);
  }
  isTriLower = true;
  isLower = false;
}

template<typename T>
void HMatrix<T>::solve(ScalarArray<T>* b) const {
  DECLARE_CONTEXT;
  // Solve (LU) X = b
  // First compute L Y = b
  this->solveLowerTriangularLeft(b, Factorization::LU, Diag::UNIT, Uplo::LOWER);
  // Then compute U X = Y
  this->solveUpperTriangularLeft(b, Factorization::LU, Diag::NONUNIT, Uplo::UPPER);
}

template<typename T>
void HMatrix<T>::solve(FullMatrix<T>* b) const {
  solve(&b->data);
}

template<typename T>
void HMatrix<T>::trsm( char side, char uplo, char trans, char diag,
		       T alpha, HMatrix<T>* B ) const {

    bool upper   = (uplo == 'u') || (uplo == 'U');
    bool left    = (side == 'l') || (side == 'L');
    Diag unit    = (diag == 'u' || diag == 'U') ? Diag::UNIT : Diag::NONUNIT;
    bool notrans = (trans == 'n') || (trans == 'N');

    /* Upper case */
    if ( upper  ) {
	if ( left ) {
	    if ( notrans ) {
		/* LUN */
		solveUpperTriangularLeft( B, Factorization::LU, unit, Uplo::UPPER );
	    }
	    else {
		/* LUT */
		HMAT_ASSERT_MSG( 0, "ERROR: TRSM LUT case is for now missing !!!" );
	    }
	}
	else {
	    if ( notrans ) {
		/* RUN */
		solveUpperTriangularRight( B, Factorization::LU, unit, Uplo::UPPER );
	    }
	    else {
		/* RUT */
		HMAT_ASSERT_MSG( 0, "ERROR: TRSM RUT case is for now missing !!!" );
	    }
	}
    }
    else {
	if ( left ) {
	    if ( notrans ) {
		/* LLN */
		solveLowerTriangularLeft( B, Factorization::LU, unit, Uplo::LOWER );
	    }
	    else {
		/* LLT */
		solveUpperTriangularLeft( B, Factorization::LU, unit, Uplo::LOWER );
	    }
	}
	else {
	    if ( notrans ) {
		/* RLN */
		HMAT_ASSERT_MSG( 0, "ERROR: TRSM RLN case is for now missing !!!" );
	    }
	    else {
		/* RLT */
		solveUpperTriangularRight( B, Factorization::LU, unit, Uplo::LOWER );
	    }
	}
    }
}

template<typename T>
void HMatrix<T>::trsm( char side, char uplo, char trans, char diag,
		       T alpha, ScalarArray<T>* B ) const {

    bool upper   = (uplo == 'u') || (uplo == 'U');
    bool left    = (side == 'l') || (side == 'L');
    Diag unit    = (diag == 'u' || diag == 'U') ? Diag::UNIT : Diag::NONUNIT;
    bool notrans = (trans == 'n') || (trans == 'N');

    /* Upper case */
    if ( upper  ) {
	if ( left ) {
	    if ( notrans ) {
		/* LUN */
		solveUpperTriangularLeft( B, Factorization::LU, unit, Uplo::UPPER );
	    }
	    else {
		/* LUT */
		HMAT_ASSERT_MSG( 0, "ERROR: TRSM LUT case is for now missing !!!" );
	    }
	}
	else {
	    if ( notrans ) {
		/* RUN */
		solveUpperTriangularRight( B, Factorization::LU, unit, Uplo::UPPER );
	    }
	    else {
		/* RUT */
		HMAT_ASSERT_MSG( 0, "ERROR: TRSM RUT case is for now missing !!!" );
	    }
	}
    }
    else {
	if ( left ) {
	    if ( notrans ) {
		/* LLN */
		solveLowerTriangularLeft( B, Factorization::LU, unit, Uplo::LOWER );
	    }
	    else {
		/* LLT */
		solveUpperTriangularLeft( B, Factorization::LU, unit, Uplo::LOWER );
	    }
	}
	else {
	    if ( notrans ) {
		/* RLN */
		HMAT_ASSERT_MSG( 0, "ERROR: TRSM RLN case is for now missing !!!" );
	    }
	    else {
		/* RLT */
		solveUpperTriangularRight( B, Factorization::LU, unit, Uplo::LOWER );
	    }
	}
    }
}

template<typename T>
void HMatrix<T>::extractDiagonal(T* diag, int components) const {
  DECLARE_CONTEXT;
  if (rows()->size() == 0 || cols()->size() == 0) return;
  if(this->isLeaf()) {
    assert(isFullMatrix());
    if(full()->diagonal && components == 1) {
      // LDLt
      memcpy(diag, full()->diagonal->const_ptr(), full()->rows() * sizeof(T));
    } else if (components == 1) {
      for (int i = 0; i < full()->rows(); ++i)
        diag[i] = full()->get(i,i);
    } else {
      HMAT_ASSERT(full()->rows() % components == 0);
      for (int i = 0; i < full()->rows() / components; ++i)
        for (int k = 0; k < components; ++k)
          for (int l = 0; l < components; ++l)
            diag[(i * components + k) * components + l] = full()->get(i * components + k, i * components + l);
    }
  } else {
    for (int i=0 ; i<nrChildRow() ; i++) {
      get(i,i)->extractDiagonal(diag, components);
      diag += get(i,i)->rows()->size() * components;
    }
  }
}

template<typename T> typename Types<T>::dp HMatrix<T>::logdet() const {
  if(this->isLeaf()) {
    HMAT_ASSERT(this->isFullMatrix() && (this->isTriLower || this->isTriUpper));
    return std::log(this->full()->data.diagonalProduct());
  } else {
    T r = 0;
    for (int i=0 ; i<nrChildRow() ; i++) {
      r += get(i,i)->logdet();
    }
    return r;
  }
}

/* Solve M.X=B with M hmat LU factorized*/
template<typename T> void HMatrix<T>::solve(
        HMatrix<T>* b,
        Factorization algo) const {
    DECLARE_CONTEXT;
    switch(algo) {
    case Factorization::LU:
        /* Solve LX=B, result in B */
        this->solveLowerTriangularLeft(b, algo, Diag::UNIT, Uplo::LOWER);
        /* Solve UX=B, result in B */
        this->solveUpperTriangularLeft(b, algo, Diag::NONUNIT, Uplo::UPPER);
        break;
    case Factorization::LDLT:
        /* Solve LX=B, result in B */
        this->solveLowerTriangularLeft(b, algo, Diag::UNIT, Uplo::LOWER);
        /* Solve DX=B, result in B */
        b->multiplyWithDiag(this, Side::LEFT, true);
        /* Solve L^tX=B, result in B */
        this->solveUpperTriangularLeft(b, algo, Diag::UNIT, Uplo::LOWER);
        break;
    case Factorization::LLT:
        /* Solve LX=B, result in B */
        this->solveLowerTriangularLeft(b, algo, Diag::NONUNIT, Uplo::LOWER);
        /* Solve L^tX=B, result in B */
        this->solveUpperTriangularLeft(b, algo, Diag::NONUNIT, Uplo::LOWER);
        break;
    default:
        HMAT_ASSERT(false);
    }
}

template<typename T> void HMatrix<T>::solveDiagonal(ScalarArray<T>* b) const {
    // Solve D*X = B and store result into B
    // Diagonal extraction
    if (rows()->size() == 0 || cols()->size() == 0) return;
    if(isFullMatrix() && full()->diagonal) {
      // LDLt
      b->multiplyWithDiagOrDiagInv(full()->diagonal, true, Side::LEFT); // multiply to the left by the inverse
    } else {
      // LLt
      Vector<T>* diag = new Vector<T>(cols()->size());
      extractDiagonal(diag->ptr());
      b->multiplyWithDiagOrDiagInv(diag, true, Side::LEFT); // multiply to the left by the inverse
      delete diag;
    }
}

template<typename T> void HMatrix<T>::solveDiagonal(FullMatrix<T>* b) const {
  solveDiagonal(&b->data);
}

template<typename T>
void HMatrix<T>::solveLdlt(ScalarArray<T>* b) const {
  DECLARE_CONTEXT;
  assertLdlt(this);
  // L*D*L^T * X = B
  // B <- solution of L * Y = B : Y = D*L^T * X
  this->solveLowerTriangularLeft(b, Factorization::LDLT, Diag::UNIT, Uplo::LOWER);

  // B <- D^{-1} Y : solution of D*Y = B : Y = L^T * X
  this->solveDiagonal(b);

  // B <- solution of L^T X = B :  the solution X we are looking for is stored in B
  this->solveUpperTriangularLeft(b, Factorization::LDLT, Diag::UNIT, Uplo::LOWER);
}

template<typename T>
void HMatrix<T>::solveLdlt(FullMatrix<T>* b) const {
  solveLdlt(&b->data);
}

template<typename T>
void HMatrix<T>::solveLlt(ScalarArray<T>* b) const {
  DECLARE_CONTEXT;
  // L*L^T * X = B
  // B <- solution of L * Y = B : Y = L^T * X
  this->solveLowerTriangularLeft(b, Factorization::LLT, Diag::NONUNIT, Uplo::LOWER);

  // B <- solution of L^T X = B :  the solution X we are looking for is stored in B
  this->solveUpperTriangularLeft(b, Factorization::LLT, Diag::NONUNIT, Uplo::LOWER);
}

template<typename T>
void HMatrix<T>::solveLlt(FullMatrix<T>* b) const {
  solveLlt(&b->data);
}

template<typename T>
void HMatrix<T>::checkStructure() const {
#if 0
  if (this->isLeaf()) {
    return;
  }
  for (int i = 0; i < this->nrChild(); i++) {
      HMatrix<T>* child = this->getChild(i);
      if (child) {
        assert(child->rows()->isSubset(*(this->rows())) && child->cols()->isSubset(*(this->cols())));
        child->checkStructure();
    }
  }
#endif
}

template<typename T>
void HMatrix<T>::checkNan() const {
#if 0
  if (this->isLeaf()) {
    if (isFullMatrix()) {
      full()->checkNan();
    }
    if (isRkMatrix()) {
      rk()->checkNan();
    }
  } else {
    for (int i = 0; i < this->nrChild(); i++) {
        if (this->getChild(i)) {
          this->getChild(i)->checkNan();
      }
    }
  }
#endif
}

template<typename T> void HMatrix<T>::setTriLower(bool value)
{
    isTriLower = value;
    if(!this->isLeaf())
    {
      for (int i = 0; i < nrChildRow(); i++)
        get(i, i)->setTriLower(value);
    }
}

template<typename T> void HMatrix<T>::setLower(bool value)
{
    isLower = value;
    if(!this->isLeaf())
    {
      for (int i = 0; i < nrChildRow(); i++)
        get(i, i)->setLower(value);
    }
}

template<typename T>  void HMatrix<T>::rk(const ScalarArray<T> * a, const ScalarArray<T> * b) {
    if(!isAssembled())
        rk(NULL);
    assert(isRkMatrix());
    if(a == NULL && isNull())
        return;
    delete rk_;
    rk(new RkMatrix<T>(a == NULL ? NULL : a->copy(), rows(),
                       b == NULL ? NULL : b->copy(), cols()));
}

template<typename T> void HMatrix<T>::listAllLeaves(std::deque<const HMatrix<T> *> & out) const {
  std::vector<const HMatrix<T> *> stack;
  stack.push_back(this);
  while(!stack.empty()) {
    const HMatrix<T> * m = stack.back();
    stack.pop_back();
    if(m->isLeaf()) {
      out.push_back(m);
    } else {
      for(int i = 0; i < m->nrChild(); i++) {
        if(m->getChild(i) != nullptr)
          stack.push_back(m->getChild(i));
      }
    }
  }
}

// No way to avoid copy/past of the const version
template<typename T> void HMatrix<T>::listAllLeaves(std::deque<HMatrix<T> *> & out) {
  std::vector<HMatrix<T> *> stack;
  stack.push_back(this);
  while(!stack.empty()) {
    HMatrix<T> * m = stack.back();
    stack.pop_back();
    if(m->isLeaf()) {
      out.push_back(m);
    } else {
      for(int i = 0; i < m->nrChild(); i++) {
        if(m->getChild(i) != nullptr)
          stack.push_back(m->getChild(i));
      }
    }
  }
}

template<typename T> std::string HMatrix<T>::toString() const {
    std::deque<const HMatrix<T> *> leaves;
    this->listAllLeaves(leaves);
    int nbAssembled = 0;
    int nbNullFull = 0;
    int nbNullRk = 0;
    double diagNorm = 0;
    for(unsigned int i = 0; i < leaves.size(); i++) {
        const HMatrix<T> * l = leaves[i];
        if(l->isAssembled()) {
            nbAssembled++;
            if(l->isNull()) {
                if(l->isRkMatrix())
                    nbNullRk++;
                else
                    nbNullFull++;
            }
            else if(l->isFullMatrix() && l->full()->diagonal) {
                diagNorm += l->full()->diagonal->normSqr();
            }
        }
    }
    diagNorm = sqrt(diagNorm);
    std::stringstream sstm;
    sstm << "HMatrix(rows=[" << rows()->offset() << ", " << rows()->size() <<
            "], cols=[" << cols()->offset() << ", " << cols()->size() <<
            "], pointer=" << (void*)this << ", leaves=" << leaves.size() <<
            ", assembled=" << isAssembled() << ", assembledLeaves=" << nbAssembled <<
            ", nullFull=" << nbNullFull << ", nullRk=" << nbNullRk <<
            ", rank=" << rank_ << ", diagNorm=" << diagNorm << ")";
    return sstm.str();
}

template<typename T>
HMatrix<T> * HMatrix<T>::unmarshall(const MatrixSettings * settings, int rank, int approxRank, char bitfield, double epsilon) {
    HMatrix<T> * m = new HMatrix<T>(settings);
    m->rank_ = rank;
    m->isUpper = (bitfield & 1 << 0 ? true : false);
    m->isLower = (bitfield & 1 << 1 ? true : false);
    m->isTriUpper = (bitfield & 1 << 2 ? true : false);
    m->isTriLower = (bitfield & 1 << 3 ? true : false);
    m->keepSameRows = (bitfield & 1 << 4 ? true : false);
    m->keepSameCols = (bitfield & 1 << 5 ? true : false);
    m->approximateRank_ = approxRank;
    m->lowRankEpsilon(epsilon, false);
    return m;
}

/** Create a temporary block from a list of children */
template<typename T>
HMatrix<T>::HMatrix(const ClusterTree * rows, const ClusterTree * cols,
                    std::vector<HMatrix*> & _children):
    Tree<HMatrix<T> >(NULL, 0), rows_(rows), cols_(cols),
    rk_(NULL), rank_(UNINITIALIZED_BLOCK),
    approximateRank_(UNINITIALIZED_BLOCK), isUpper(false), isLower(false),
    keepSameRows(false), keepSameCols(false), temporary_(true), ownRowsClusterTree_(false),
    ownColsClusterTree_(false), localSettings(_children[0]->localSettings.global, -1.0) {
    this->children = _children;
}

template<typename T> void HMatrix<T>::rank(int rank) {
    HMAT_ASSERT_MSG(rank_ >= 0, "HMatrix::rank can only be used on Rk blocks");
    HMAT_ASSERT_MSG(!rk() || rk()->a == NULL || rk()->rank() == rank,
        "HMatrix::rank can only be used on evicted blocks");
    rank_ = rank;
}


template<typename T> void HMatrix<T>::temporary(bool b) {
  temporary_ = b;
  for (int i=0; i<this->nrChild(); i++) {
    if (this->getChild(i))
      this->getChild(i)->temporary(b);
  }
}

// Templates declaration
template class HMatrix<S_t>;
template class HMatrix<D_t>;
template class HMatrix<C_t>;
template class HMatrix<Z_t>;

template void reorderVector(ScalarArray<S_t>* v, int* indices, int axis);
template void reorderVector(ScalarArray<D_t>* v, int* indices, int axis);
template void reorderVector(ScalarArray<C_t>* v, int* indices, int axis);
template void reorderVector(ScalarArray<Z_t>* v, int* indices, int axis);

template void restoreVectorOrder(ScalarArray<S_t>* v, int* indices, int axis);
template void restoreVectorOrder(ScalarArray<D_t>* v, int* indices, int axis);
template void restoreVectorOrder(ScalarArray<C_t>* v, int* indices, int axis);
template void restoreVectorOrder(ScalarArray<Z_t>* v, int* indices, int axis);

template unsigned char * compatibilityGridForGEMM(const HMatrix<S_t>* a, Axis axisA, char transA, const HMatrix<S_t>* b, Axis axisB, char transB);
template unsigned char * compatibilityGridForGEMM(const HMatrix<D_t>* a, Axis axisA, char transA, const HMatrix<D_t>* b, Axis axisB, char transB);
template unsigned char * compatibilityGridForGEMM(const HMatrix<C_t>* a, Axis axisA, char transA, const HMatrix<C_t>* b, Axis axisB, char transB);
template unsigned char * compatibilityGridForGEMM(const HMatrix<Z_t>* a, Axis axisA, char transA, const HMatrix<Z_t>* b, Axis axisB, char transB);

}  // end namespace hmat

#include "recursion.cpp"

namespace hmat {

  // Explicit template instantiation
  template class RecursionMatrix<S_t, HMatrix<S_t> >;
  template class RecursionMatrix<C_t, HMatrix<C_t> >;
  template class RecursionMatrix<D_t, HMatrix<D_t> >;
  template class RecursionMatrix<Z_t, HMatrix<Z_t> >;

}  // end namespace hmat
