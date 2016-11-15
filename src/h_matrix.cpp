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
#include "rk_matrix.hpp"
#include "data_types.hpp"
#include "compression.hpp"
#include "postscript.hpp"
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
void reorderVector(ScalarArray<T>* v, int* indices) {
  DECLARE_CONTEXT;
  const int n = v->rows;
  Vector<T> tmp(n);
  for (int col = 0; col < v->cols; col++) {
    T* column = v->m + ((size_t) n) * col;
    for (int i = 0; i < n; i++) {
      tmp.m[i] = column[indices[i]];
    }
    memcpy(column, tmp.m, sizeof(T) * n);
  }
}


template<typename T>
void restoreVectorOrder(ScalarArray<T>* v, int* indices) {
  DECLARE_CONTEXT;
  const int n = v->rows;
  Vector<T> tmp(n);

  for (int col = 0; col < v->cols; col++) {
    T* column = v->m + ((size_t) n) * col;
    for (int i = 0; i < n; i++) {
      tmp.m[indices[i]] = column[i];
    }
    memcpy(column, tmp.m, sizeof(T) * n);
  }
}


template<typename T>
HMatrix<T>::HMatrix(ClusterTree* _rows, ClusterTree* _cols, const hmat::MatrixSettings * settings,
                    int _depth, SymmetryFlag symFlag, AdmissibilityCondition * admissibilityCondition)
  : Tree<HMatrix<T> >(NULL, _depth), RecursionMatrix<T, HMatrix<T> >(),
    rows_(_rows), cols_(_cols), rk_(NULL),
    rank_(UNINITIALIZED_BLOCK), approximateRank_(UNINITIALIZED_BLOCK),
    isUpper(false), isLower(false),
    isTriUpper(false), isTriLower(false), keepSameRows(true), keepSameCols(true), temporary_(false),
    ownRowsClusterTree_(false), ownColsClusterTree_(false), localSettings(settings)
{
  if (isVoid())
    return;
  // We would like to create a block of matrix in one of the following case:
  // - rows_->isLeaf() && cols_->isLeaf() : both rows and cols are leaves.
  // - Block is too small to recurse and compress (for performance)
  // - Block is compressible and in-place compression is possible
  // In the other cases, we subdivide.
  //
  // FIXME: But in practice this does not work yet, so we stop recursion as soon as either rows
  // or cols is a leaf.
  const bool lowRank = admissibilityCondition->isLowRank(*rows_, *cols_);
  const bool stopRecursion = admissibilityCondition->stopRecursion(*rows_, *cols_);
  const bool forceRecursion = admissibilityCondition->forceRecursion(*rows_, *cols_);
  assert(!(forceRecursion && stopRecursion));
  if ((rows_->isLeaf() && cols_->isLeaf()) || stopRecursion || (lowRank && !forceRecursion)) {
    const bool forceFull = admissibilityCondition->forceFull(*rows_, *cols_);
    const bool forceRk   = admissibilityCondition->forceRk(*rows_, *cols_);
    assert(!(forceFull && forceRk));
    if (forceRk || (lowRank && !forceFull))
      rk(NULL);
    approximateRank_ = admissibilityCondition->getApproximateRank(*(rows_), *(cols_));
  } else {
    pair<bool, bool> split = admissibilityCondition->splitRowsCols(*rows_, *cols_);
    assert(split.first || split.second);
    keepSameRows = !split.first;
    keepSameCols = !split.second;
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
            this->insertChild(i, j, new HMatrix<T>(rowChild, colChild, settings, _depth+1, (i == j ? symFlag : kNotSymmetric), admissibilityCondition));
          } else
            // If 'inert', the child is NULL
            this->insertChild(i, j, NULL);
        }
      }
    }
  }
  admissibilityCondition->clean(*(rows_));
  admissibilityCondition->clean(*(cols_));
}

template<typename T>
HMatrix<T>::HMatrix(const hmat::MatrixSettings * settings) :
    Tree<HMatrix<T> >(NULL), RecursionMatrix<T, HMatrix<T> >(), rows_(NULL), cols_(NULL),
    rk_(NULL), rank_(UNINITIALIZED_BLOCK), approximateRank_(UNINITIALIZED_BLOCK),
    isUpper(false), isLower(false), isTriUpper(false), isTriLower(false),
    keepSameRows(true), keepSameCols(true), temporary_(false), ownRowsClusterTree_(false),
    ownColsClusterTree_(false), localSettings(settings)
    {}

template<typename T> HMatrix<T> * HMatrix<T>::internalCopy(bool temporary, bool withRowChild, bool withColChild) const {
    HMatrix<T> * r = new HMatrix<T>(localSettings.global);
    r->rows_ = rows_;
    r->cols_ = cols_;
    r->temporary_ = temporary;
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
                assert(child->rows_ != NULL);
                assert(child->cols_ != NULL);
                child->rk(new RkMatrix<T>(NULL, &child->rows_->data,
                                          NULL, &child->cols_->data,
                                          NoCompression));
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
    h->rk(new RkMatrix<T>(NULL, h->rows(), NULL, h->cols(), NoCompression));
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
    f.assemble(localSettings, *rows_, *cols_, isRkMatrix(), m, assembledRk, ao);
    HMAT_ASSERT(m == NULL || assembledRk == NULL);
    if(assembledRk) {
        if(rk_)
            delete rk_;
        rk(assembledRk);
    } else {
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
      coarsen();
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
        RkMatrix<T>* newRk = new RkMatrix<T>(NULL, upper->rows(),
                                          NULL, upper->cols(), rk()->method);
        newRk->a = rk()->b ? rk()->b->copy() : NULL;
        newRk->b = rk()->a ? rk()->a->copy() : NULL;
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
          coarsen(upper);
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
        size_t rowsMin, colsMin, rowsMax, colsMax;
        static char * fitStudy = getenv("HMAT_FIT_STUDY");
        if(isRkMatrix()) {
            size_t mem = rank() * (((size_t)r) + c);
            result.compressed_size += mem;
            if(!rk())
                rk(new RkMatrix<T>(NULL, rows(), NULL, cols(), NoCompression));

            if (fitStudy) {
              FullMatrix<T>* rk_eval = rk()->eval();
              result.rk_fit_nnz += rk_eval->info(result, rowsMin, colsMin, rowsMax, colsMax);
              if (colsMax == 0) {
                colsMax = cols()->size();
                colsMin = 0;
                rowsMax = rows()->size();
                rowsMin = 0;
              } else
              result.rk_square_fit_nnz += (colsMax - colsMin) * (rowsMax - rowsMin);

              HMatrix<T> * tmpMatrix = new HMatrix<T>(this->localSettings.global);
              tmpMatrix->temporary_=true;
              tmpMatrix->ownClusterTrees(true, true);
              ClusterTree * rows_subset = rows_->slice(rows()->offset() + rowsMin, rowsMax - rowsMin);
              ClusterTree * cols_subset = cols_->slice(cols()->offset() + colsMin, colsMax - colsMin);
              rows_subset->father = rows_subset;
              cols_subset->father = cols_subset;
              tmpMatrix->rows_ = rows_subset;
              tmpMatrix->cols_ = cols_subset;
              RkMatrix<T>* newRk = const_cast<RkMatrix<T>*>(rk()->subset(tmpMatrix->rows(), tmpMatrix->cols()));
              result.rk_fit_compressed_size += newRk->rank() * (((size_t)newRk->rows->size()) + newRk->cols->size());
              size_t rk_eval_zeros = rk_eval->storedZeros();
              size_t rk_compressed_zeros = rk()->storedZeros();
              result.rk_as_full_size += rk()->uncompressedSize();
              result.rk_eval_zeros += rk_eval_zeros;
              result.rk_eval_nnz += s - rk_eval_zeros;
              result.rk_compressed_zeros += rk_compressed_zeros;
              result.rk_compressed_nnz += mem - rk_compressed_zeros;
            }
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
          if (isFullMatrix()) {
            result.full_zeros += full()->storedZeros();
            if (fitStudy) {
              result.full_nnz += s - full()->storedZeros();
              size_t fit = full()->info(result, rowsMin, colsMin, rowsMax, colsMax);
              result.full_fit_nnz += fit;
              result.full_square_fit_nnz += (colsMax+1 - colsMin) * (rowsMax+1 - rowsMin);
              RkMatrix<T>* rk = truncatedSvd(full()->copy());
              result.full_as_rk_size += rk->compressedSize();
              HMatrix<T> * tmpMatrix = new HMatrix<T>(this->localSettings.global);
              tmpMatrix->temporary_=true;
              tmpMatrix->ownClusterTrees(true, true);
              ClusterTree * r = rows_->slice(rows()->offset() + rowsMin, rowsMax - rowsMin);
              ClusterTree * c = cols_->slice(cols()->offset() + colsMin, colsMax - colsMin);
              r->father = r;
              c->father = c;
              tmpMatrix->rows_ = r;
              tmpMatrix->cols_ = c;
              RkMatrix<T>* newRk = const_cast<RkMatrix<T>*>(rk->subset(tmpMatrix->rows(), tmpMatrix->cols()));
              result.full_fit_compressed_size += newRk->rank() * (((size_t)newRk->rows->size()) + newRk->cols->size());
              delete newRk;
              delete tmpMatrix;
            }
            result.compressed_size += s;
            result.full_count ++;
            result.full_size += s;
          }
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
  if (this->isLeaf() && !isNull()) {
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
  T ev = Constants<T>::zero;
  for (int i = 0; i < nrow; i++)
    x->m[i] = static_cast<T>(rand()/(double)RAND_MAX);
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
    gemv('N', Constants<T>::pone, x, Constants<T>::zero, x1);
    ev = x->dot(x,x1);
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
  if(alpha == Constants<T>::zero) {
    this->clear();
  } else if(alpha == Constants<T>::pone) {
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
bool HMatrix<T>::coarsen(HMatrix<T>* upper) {
  // If all children are Rk leaves, then we try to merge them into a single Rk-leaf.
  // This is done if the memory of the resulting leaf is less than the sum of the initial
  // leaves. Note that this operation could be used hierarchically.

  bool allRkLeaves = true;
  const RkMatrix<T>* childrenArray[this->nrChild()];
  size_t childrenElements = 0;
  for (int i = 0; i < this->nrChild(); i++) {
    HMatrix<T> *child = this->getChild(i);
    if (!child) continue;
    if (!child->isRkMatrix()) {
      allRkLeaves = false;
      break;
    } else {
      childrenArray[i] = child->rk();
      childrenElements += (childrenArray[i]->rows->size()
                           + childrenArray[i]->cols->size()) * childrenArray[i]->rank();
    }
  }
  if (allRkLeaves) {
    std::vector<T> alpha(this->nrChild(), Constants<T>::pone);
    RkMatrix<T> dummy(NULL, rows(), NULL, cols(), NoCompression);
    RkMatrix<T>* candidate = dummy.formattedAddParts(&alpha[0], childrenArray, this->nrChild());
    size_t elements = (((size_t) candidate->rows->size()) + candidate->cols->size()) * candidate->rank();
    if (elements < childrenElements) {
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
        upper->rk(new RkMatrix<T>(candidate->b->copy(), upper->rows(),
                                  candidate->a->copy(), upper->cols(), candidate->method));
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

// y <- alpha * op(this) * x + beta * y.
template<typename T>
void HMatrix<T>::gemv(char matTrans, T alpha, const ScalarArray<T>* x, T beta, ScalarArray<T>* y) const {
  // The dimensions of the H-matrix and the 2 ScalarArrays must match exactly
  assert(x->cols == y->cols);
  if (rows()->size() == 0 || cols()->size() == 0) return;
  assert((matTrans != 'N' ? cols()->size() : rows()->size()) == y->rows);
  assert((matTrans != 'N' ? rows()->size() : cols()->size()) == x->rows);
  if (beta != Constants<T>::pone) {
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

          // get the rows subset of X aligned with 'trans(child)' cols and Y aligned with 'trans(child)' rows
          const ScalarArray<T> subX(x->rowsSubset(colsOffset, colsSize));
          ScalarArray<T> subY(y->rowsSubset(rowsOffset, rowsSize));
          child->gemv(trans, alpha, &subX, Constants<T>::pone, &subY);
        }
        else continue;
      }

  } else {
    // We are on a leaf of the matrix 'this'
    if (isFullMatrix()) {
      y->gemm(matTrans, 'N', alpha, &full()->data, x, Constants<T>::pone);
    } else if(!isNull()){
      rk()->gemv(matTrans, alpha, x, Constants<T>::pone, y);
    }
  }
}

template<typename T>
void HMatrix<T>::gemv(char matTrans, T alpha, const FullMatrix<T>* x, T beta, FullMatrix<T>* y) const {
  gemv(matTrans, alpha, &x->data, beta, &y->data);
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
template<typename T>
void HMatrix<T>::axpy(T alpha, const HMatrix<T>* x) {
    if(x->isLeaf() && x->isNull())
        return;

    if(*rows() == *x->rows() && *cols() == *x->cols()) {
        if (this->isLeaf()) {
            if (isRkMatrix()) {
                if(!rk())
                    rk(new RkMatrix<T>(NULL, rows(), NULL, cols(), NoCompression));
                if(x->isRkMatrix()) {
                    rk()->axpy(alpha, x->rk());
                    rank_ = rk()->rank();
                } else if(!x->isLeaf()){
                    vector<const RkMatrix<T>*> rkLeaves;
                    if(listAllRk(x, rkLeaves)) {
                        vector<T> alphas(rkLeaves.size(), alpha);
                        RkMatrix<T>* tmp = rk()->formattedAddParts(&alphas[0], &rkLeaves[0], rkLeaves.size());
                        delete rk();
                        rk(tmp);
                    } else {
                        // x has contains both full and Rk matrices, this is not
                        // supported yet.
                        HMAT_ASSERT(false);
                    }
                } else {
                    RkMatrix<T>* tmp = rk()->formattedAdd(x->full(), alpha);
                    delete rk();
                    rk(tmp);
                }
            } else {
                if(full() == NULL)
                    full(new FullMatrix<T>(rows(), cols()));
                if(!x->isLeaf()) {
                    FullMatrix<T> xFull(x->rows(), x->cols());
                    x->evalPart(&xFull, x->rows(), x->cols());
                    full()->axpy(alpha, &xFull);
                } else if(x->isFullMatrix()) {
                    full()->axpy(alpha, x->full());
                } else if(x->isRkMatrix()) {
                    FullMatrix<T> * f = x->rk()->eval();
                    full()->axpy(alpha, f);
                    delete f;
                } else {
                    HMAT_ASSERT(false);
                }
            }
        } else {
            for (int i = 0; i < this->nrChild(); i++) {
                    HMatrix<T>* child = this->getChild(i);
                    const HMatrix<T>* bChild = x->isLeaf() ? x : x->getChild(i);
                    if (child && bChild)
                        child->axpy(alpha, bChild);
                    // TODO what to do if only 1 of the 2 is NULL ?
                }
            }
    } else {
        if(x->isFullMatrix()) {
            axpy(alpha, x->full());
        }
        else if(x->isRkMatrix()) {
            axpy(alpha, x->rk());
        }
        else if(x->isLeaf()){
            // X is an empty leaf, so nothing to do
        } else {
            HMAT_ASSERT(false);
        }
    }
}

/** @brief AXPY between 'this' an H matrix and a subset of B with B a RkMatrix */
template<typename T>
void HMatrix<T>::axpy(T alpha, const RkMatrix<T>* b) {
  DECLARE_CONTEXT;
  // this += alpha * b
  assert(b);
  assert(b->rows->isSuperSet(*rows()));
  assert(b->cols->isSuperSet(*cols()));

  if (b->rank() == 0 || rows()->size() == 0 || cols()->size() == 0) {
    return;
  }

  // If 'this' is not a leaf, we recurse with the same 'b'
  if (!this->isLeaf()) {
    for (int i = 0; i < this->nrChild(); i++) {
      if (this->getChild(i)) {
        this->getChild(i)->axpy(alpha, b);
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
          rk(new RkMatrix<T>(NULL, rows(), NULL, cols(), NoCompression));
      rk()->axpy(alpha, newRk);
      rank_ = rk()->rank();
    } else {
      // In this case, the matrix has small size
      // then evaluating the Rk-matrix is cheaper
      FullMatrix<T>* rkMat = newRk->eval();
      if(!isFullMatrix())
        full(new FullMatrix<T>(rows(), cols()));

      full()->axpy(alpha, rkMat);
      delete rkMat;
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
  // this += alpha * b
  assert(b->rows_->isSuperSet(*this->rows()) && b->cols_->isSuperSet(*this->cols()));

  // If 'this' is not a leaf, we recurse with the same 'b'
  if (!this->isLeaf()) {
    for (int i = 0; i < this->nrChild(); i++) {
      HMatrix<T>* child = this->getChild(i);
      if (child)
        child->axpy(alpha, b);
    }
  } else {
    const FullMatrix<T>* subMat = b->subset(rows(), cols());
    if (isRkMatrix()) {
      if(!rk())
        rk(new RkMatrix<T>(NULL, rows(), NULL, cols(), NoCompression));
      rk()->axpy(alpha, subMat);
      rank_ = rk()->rank();
    } else if(isFullMatrix()){
       full()->axpy(alpha, subMat);
    } else {
       assert(full() == NULL);
       full(subMat->copy());
       if(alpha != Constants<T>::pone)
         full()->scale(alpha);
    }
    delete subMat;
  }
}

template<typename T>
void HMatrix<T>::addIdentity(T alpha)
{
  if (this->isLeaf()) {
    if (isFullMatrix()) {
      FullMatrix<T> * b = full();
      assert(b->rows() == b->cols());
      for (int i = 0; i < b->rows(); i++) {
          b->get(i, i) += alpha;
      }
    }
  } else {
    for (int i = 0; i < nrChildRow(); i++)
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

    if(this->isLeaf()) {
        HMatrix<T> * tmpMatrix = new HMatrix<T>(this->localSettings.global);
        tmpMatrix->temporary_=true;
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

    if (this->isLeaf() && !this->isRkMatrix() && !this->isFullMatrix())
    {
      this->full(new FullMatrix<T>(this->rows(), this->cols()));
    }
    // Create vva & vc = the subsets of va & c (=this) that match each other for doing the sum c+f(a).f(b)
    // We modify the rows of f(a) and the rows of c
    makeCompatible<T>(transA == 'N', true, va, this, vva, vc);


    // Create vvb & vvc = the subsets of vb & vc that match each other for doing the sum c+f(a).f(b)
    // We modify the columns of f(b) and the columns of c
    makeCompatible<T>(transB != 'N', false, vb, vc, vvb, vvc);

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

template<typename T> void
HMatrix<T>::recursiveGemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>*b) {
    // Computing a(m,0) * b(0,n) here may give wrong results because of format conversions, exit early
    if(isVoid() || a->isVoid())
        return;

    // None of the matrices is a leaf
    if (!this->isLeaf() && !a->isLeaf() && !b->isLeaf()) {
        for (int i = 0; i < nrChildRow(); i++) {
            for (int j = 0; j < nrChildCol(); j++) {
                HMatrix<T>* child = get(i, j);
                if (!child) { // symmetric/triangular case or empty block coming from symbolic factorisation of sparse matrices
                    continue;
                }
                // Void child
                if (child->isVoid()) continue;

                const HMatrix<T> *childA, *childB;
                for (int k = 0; k < (transA=='N' ? a->nrChildCol() : a->nrChildRow()) ; k++) {
                  char tA = transA;
                  char tB = transB;
                  childA = a->getChildForGEMM(tA, i, k);
                  childB = b->getChildForGEMM(tB, k, j);
                  if(childA && childB)
                    child->gemm(tA, tB, alpha, childA, childB, Constants<T>::pone);
                }
            }
        }
        return;
    } // if (!this->isLeaf() && !a->isLeaf() && !b->isLeaf())
    else
        uncompatibleGemm(transA, transB, alpha, a, b);
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
            RkMatrix<T>* rkMat = HMatrix<T>::multiplyRkMatrix(transA, transB, a, b);
            axpy(alpha, rkMat);
            delete rkMat;
        } else {
            // None of the matrices of the product is a Rk-matrix so one of them is
            // a full matrix so as the result.
            assert(a->isFullMatrix() || b->isFullMatrix());
            FullMatrix<T>* fullMat = HMatrix<T>::multiplyFullMatrix(transA, transB, a, b);
            axpy(alpha, fullMat);
            delete fullMat;
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
            rk(new RkMatrix<T>(NULL, rows(), NULL, cols(), NoCompression));
        rk()->gemmRk(transA, transB, alpha, a, b, Constants<T>::pone);
        rank_ = rk()->rank();
        return;
    }

    // a, b are H matrices and 'this' is full
    if ( this->isLeaf() && !a->isLeaf() && !b->isLeaf()) {
      for (int i = 0; i < (transA=='N' ? a->nrChildRow() : a->nrChildCol()) ; i++) {
        for (int j = 0; j < (transB=='N' ? b->nrChildCol() : b->nrChildRow()) ; j++) {
          const HMatrix<T> *childA, *childB;
          for (int k = 0; k < (transA=='N' ? a->nrChildCol() : a->nrChildRow()) ; k++) {
            char tA = transA;
            char tB = transB;
            childA = a->getChildForGEMM(tA, i, k);
            childB = b->getChildForGEMM(tB, k, j);
            if(childA && childB)
              gemm(tA, tB, alpha, childA, childB, Constants<T>::pone);
          }
        }
      }
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
        RkMatrix<T>* rkMat = HMatrix<T>::multiplyRkMatrix(transA, transB, a, b);
        fullMat = rkMat->eval();
        delete rkMat;
    } else if(a->isLeaf() || b->isLeaf()){
      // if a or b is a leaf, it is Full (since Rk have been treated before)
        fullMat = HMatrix<T>::multiplyFullMatrix(transA, transB, a, b);
    } else {
        // should not happen anymore
        HMAT_ASSERT(false);
    }

    // It's not optimal to concider that the result is a FullMatrix but
    // this is a H*F case and it almost never happen
    if (isFullMatrix()) {
      full()->axpy(alpha, fullMat);
      delete fullMat;
    } else {
      full(fullMat);
      fullMat->scale(alpha);
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
    // Ca = beta * Ca + alpha A * Ba with only Ca and Ba full matrices
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
    // CbT = beta * CbT + alpha AbT * B with only Cb and Ab full matrices
    // we transpose:
    // Cb = beta * Cb + alpha BT * Ab
    // We support C and B not compatible (larger) with A so we first slice them
    assert(transA == 'N');
    const IndexSet * r = transB == 'N' ? b->rows() : b->cols();
    const IndexSet * c = transB == 'N' ? b->cols() : b->rows();
    ScalarArray<T> cSubset(rk()->b->rowsSubset( c->offset() -    cols()->offset(), c->size()));
    ScalarArray<T> aSubset(a->rk()->b->rowsSubset( r->offset() - a->cols()->offset(), r->size()));
    b->gemv(transB == 'N' ? 'T' : 'N', alpha, &aSubset, beta, &cSubset);
    return;
  }

  this->scale(beta);

  if((a->isLeaf() && a->isNull()) || (b->isLeaf() && b->isNull())) {
      if(!isAssembled() && this->isLeaf())
          rk(new RkMatrix<T>(NULL, rows(), NULL, cols(), NoCompression));
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
  // R = M * H = (H^t * M^t*)^t
  FullMatrix<T>* resultT = multiplyHFull(transH == 'N' ? 'T' : 'N',
                                         transM == 'N' ? 'T' : 'N',
                                         h, mat);
  resultT->transpose();
  return resultT;
}

template<typename T>
FullMatrix<T>* multiplyHFull(char transH, char transM,
                                         const HMatrix<T>* h,
                                         const FullMatrix<T>* mat) {
  assert((transH == 'N' ? h->cols()->size() : h->rows()->size())
           == (transM == 'N' ? mat->rows() : mat->cols()));
  FullMatrix<T>* result =
    new FullMatrix<T>((transH == 'N' ? h->rows() : h->cols()),
                      (transM == 'N' ? mat->cols_ : mat->rows_));
  if (transM == 'N') {
    h->gemv(transH, Constants<T>::pone, mat, Constants<T>::zero, result);
  } else {
    FullMatrix<T>* matT = mat->copyAndTranspose();
    if (transM == 'C') {
      matT->data.conjugate();
    }
    h->gemv(transH, Constants<T>::pone, matT, Constants<T>::zero, result);
    delete matT;
  }
  return result;
}

template<typename T>
RkMatrix<T>* HMatrix<T>::multiplyRkMatrix(char transA, char transB, const HMatrix<T>* a, const HMatrix<T>* b){
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
    rk = RkMatrix<T>::multiplyRkRk(transA, transB, a->rk(), b->rk());
    HMAT_ASSERT(rk);
  }
  else if (a->isRkMatrix() && b->isFullMatrix()) {
    rk = RkMatrix<T>::multiplyRkFull(transA, transB, a->rk(), b->full(), (transB == 'N' ? b->cols() : b->rows()));
    HMAT_ASSERT(rk);
  }
  else if (a->isFullMatrix() && b->isRkMatrix()) {
    rk = RkMatrix<T>::multiplyFullRk(transA, transB, a->full(), b->rk(), (transA == 'N' ? a->rows() : a->cols()));
    HMAT_ASSERT(rk);
  } else if(a->isNull() || b->isNull()) {
    return new RkMatrix<T>(NULL, transA ? a->cols() : a->rows(),
                           NULL, transB ? b->rows() : b->cols(), NoCompression);
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
    HMAT_ASSERT(result);
  } else if (a->isFullMatrix() && !b->isLeaf()) {
    result = multiplyFullH(transA, transB, a->full(), b);
    HMAT_ASSERT(result);
  } else if (a->isFullMatrix() && b->isFullMatrix()) {
    const IndexSet* aRows = (transA == 'N')? a->rows() : a->cols();
    const IndexSet* bCols = (transB == 'N')? b->cols() : b->rows();
    result = new FullMatrix<T>(aRows, bCols);
    result->gemm(transA, transB, Constants<T>::pone, a->full(), b->full(),
                 Constants<T>::zero);
    HMAT_ASSERT(result);
  } else if(a->isNull() || b->isNull()) {
    return NULL;
  } else {
    // None of above, impossible
    HMAT_ASSERT(false);
  }
  return result;
}

template<typename T>
void HMatrix<T>::multiplyWithDiag(const HMatrix<T>* d, bool left, bool inverse) const {
  assert(*d->rows() == *d->cols());
  assert(left || (*cols() == *d->rows()));
  assert(!left || (*rows() == *d->cols()));

  if (isVoid()) return;

  // The symmetric matrix must be taken into account: lower or upper
  if (!this->isLeaf()) {
    if (d->isLeaf()) {
      for (int i=0 ; i<std::min(nrChildRow(), nrChildCol()) ; i++)
        if(get(i,i))
          get(i,i)->multiplyWithDiag(d, left, inverse);
      for (int i=0 ; i<nrChildRow() ; i++)
        for (int j=0 ; j<nrChildCol() ; j++)
          if (i!=j && get(i,j)) {
            get(i,j)->multiplyWithDiag(d, left, inverse);
          }
      return;
    }

    // First the diagonal, then the rest...
    for (int i=0 ; i<std::min(nrChildRow(), nrChildCol()) ; i++)
      if(get(i,i))
        get(i,i)->multiplyWithDiag(d->get(i,i), left, inverse);
    for (int i=0 ; i<nrChildRow() ; i++)
      for (int j=0 ; j<nrChildCol() ; j++)
        if (i!=j && get(i,j)) {
        int k = left ? i : j;
        get(i,j)->multiplyWithDiag(d->get(k,k), left, inverse);
        // TODO couldn't we handle this case with the previous one, using getChildForGEMM(d,i,i) that returns 'd' itself when 'd' is a leaf ?
    }
  } else if (isRkMatrix() && !isNull()) {
    rk()->multiplyWithDiagOrDiagInv(d, inverse, left);
  } else if(isFullMatrix()){
    if (d->isFullMatrix()) {
      full()->multiplyWithDiagOrDiagInv(d->full()->diagonal, inverse, left);
    } else {
      Vector<T> diag(d->rows()->size());
      d->extractDiagonal(diag.m);
      full()->multiplyWithDiagOrDiagInv(&diag, inverse, left);
    }
  } else {
    // this is a null matrix (either full of Rk) so nothing to do
  }
}

template<typename T> void HMatrix<T>::transposeMeta() {
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
    RecursionMatrix<T, HMatrix<T> >::transposeMeta();
    swap(rows_, cols_);
}

template <typename T> void HMatrix<T>::transposeData() {
    if (this->isLeaf()) {
        if (isRkMatrix() && rk()) {
            // To transpose an Rk-matrix, simple exchange A and B : (AB^T)^T = (BA^T)
            swap(rk()->a, rk()->b);
            swap(rk()->rows, rk()->cols);
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
      const RkMatrix<T>* oRk = o->rk();
      ScalarArray<T>* newA = oRk->b ? oRk->b->copy() : NULL;
      ScalarArray<T>* newB = oRk->a ? oRk->a->copy() : NULL;
      rk(new RkMatrix<T>(newA, oRk->cols, newB, oRk->rows, oRk->method));
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
          memcpy(full()->diagonal->m, oF->diagonal->m, oF->rows() * sizeof(T));
        }
      }
    }
  } else {
    for (int i=0 ; i<nrChildRow() ; i++)
      for (int j=0 ; j<nrChildCol() ; j++)
        if (get(i,j))
          get(i, j)->copyAndTranspose(o->get(j, i));
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

template<typename T>
void HMatrix<T>::createPostcriptFile(const std::string& filename) const {
    PostscriptDumper<T> dumper;
    dumper.write(this, filename);
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
      rk(new RkMatrix<T>(NULL, o->rk()->rows, NULL, o->rk()->cols, o->rk()->method));
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
void HMatrix<T>::clear() {
  if (isVoid()) {
    // nothing to do
  } else if (this->isLeaf()) {
    if(!isAssembled() || isNull()) {
      // nothing to do
    } else if(isRkMatrix()) {
      rk()->clear();
    } else {
      assert(isFullMatrix());
      delete full_;
      full_ = NULL;
    }
  } else {
    for (int i = 0; i < this->nrChild(); i++) {
      HMatrix<T>* child = this->getChild(i);
      if (child)
        child->clear();
    }
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
            get(k,j)->gemm('N', 'N', Constants<T>::pone, get(k,k), TM[j], Constants<T>::zero);
          } else if (j>k) { // above the diag : Mkj = t Mjk, we store TMj=-Mjk.tMkk-1 = -Mjk.Mkk-1 (Mkk est sym)
            TM[j] = Zero(get(j,k));
            TM[j]->gemm('N', 'T', Constants<T>::mone, get(j,k), get(k,k), Constants<T>::zero);
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
                get(i,j)->gemm('N', 'T', Constants<T>::pone, get(i,k), TM[j], Constants<T>::pone);
              // cas     j < k < i        Toutes les matrices existent sous la diag
              else if (k<i)
            get(i,j)->gemm('N', 'N', Constants<T>::mone, get(i,k), get(k,j), Constants<T>::pone);
              // cas     j <     i < k    Mik n'existe pas, on prend TM[i] = Mki
              else
                get(i,j)->gemm('T', 'N', Constants<T>::mone, TM[i], get(k,j), Constants<T>::pone);
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
void HMatrix<T>::solveLowerTriangularLeft(HMatrix<T>* b, bool unitriangular, MainOp) const {
  DECLARE_CONTEXT;
  if (isVoid()) return;
  // At first, the recursion one (simple case)
  if (!this->isLeaf() && !b->isLeaf()) {
    this->recursiveSolveLowerTriangularLeft(b, unitriangular);
  } else {
    // if B is a leaf, the resolve is done by column
    if (b->isLeaf()) {
      if (b->isFullMatrix()) {
        this->solveLowerTriangularLeft(b->full(), unitriangular);
      } else {
        if (b->isNull()) {
          return;
        }
        assert(b->isRkMatrix());
        HMatrix<T> * tmp = b->subset(this->cols(), b->cols());
        this->solveLowerTriangularLeft(tmp->rk()->a, unitriangular);
        if(tmp != b)
            delete tmp;
      }
    } else {
      // B isn't a leaf, then 'this' is one
      assert(this->isLeaf());
      // Evaluate B, solve by column, and restore in the matrix
      // TODO: check if it's not too bad
      FullMatrix<T> bFull(b->rows(), b->cols());
      b->evalPart(&bFull, b->rows(), b->cols());
      this->solveLowerTriangularLeft(&bFull, unitriangular);
      b->clear();
      b->axpy(Constants<T>::pone, &bFull);
    }
  }
}

template<typename T>
void HMatrix<T>::solveLowerTriangularLeft(ScalarArray<T>* b, bool unitriangular) const {
  DECLARE_CONTEXT;
  assert(*rows() == *cols());
  assert(cols()->size() == b->rows); // Here : the change : OK or not ??????? : cols <-> rows
  if (isVoid()) return;
  if (this->isLeaf()) {
    assert(this->isFullMatrix());
    // LAPACK resolution
    full()->solveLowerTriangularLeft(b, unitriangular);
  } else {
    //  Forward substitution:
    //  [ L11 |  0  ]    [ X1 ]   [ b1 ]
    //  [ ----+---- ] *  [----] = [ -- ]
    //  [ L21 | L22 ]    [ X2 ]   [ b2 ]
    //
    //  L11 * X1 = b1 (by recursive forward substitution)
    //  L21 * X1 + L22 * X2 = b2 (forward substitution of L22*X21=b21-L21*X11)
    //

    int offset(0);
    vector<ScalarArray<T> > sub;
    for (int i=0 ; i<nrChildRow() ; i++) {
      // Create sub[i] = a FullMatrix (without copy of data) for the lines in front of the i-th matrix block
      sub.push_back(b->rowsSubset(offset, get(i, i)->cols()->size()));
      offset += get(i, i)->cols()->size();
      // Update sub[i] with the contribution of the solutions already computed sub[j] j<i
      for (int j=0 ; j<i ; j++)
        if (get(i,j))
          get(i, j)->gemv('N', Constants<T>::mone, &sub[j], Constants<T>::pone, &sub[i]);
      // Solve the i-th diagonal system
      get(i, i)->solveLowerTriangularLeft(&sub[i], unitriangular);
    }
  }
}

template<typename T>
void HMatrix<T>::solveLowerTriangularLeft(FullMatrix<T>* b, bool unitriangular) const {
  solveLowerTriangularLeft(&b->data, unitriangular);
}

template<typename T>
void HMatrix<T>::solveUpperTriangularRight(HMatrix<T>* b, bool unitriangular, bool lowerStored) const {
  DECLARE_CONTEXT;
  if (rows()->size() == 0 || cols()->size() == 0) return;
  // The recursion one (simple case)
  if (!this->isLeaf() && !b->isLeaf()) {
    this->recursiveSolveUpperTriangularRight(b, unitriangular, lowerStored);
  } else {
    // if B is a leaf, the resolve is done by row
    if (b->isLeaf()) {
      if (b->isFullMatrix()) {
        b->full()->transpose();
        this->solveUpperTriangularRight(b->full(), unitriangular, lowerStored);
        b->full()->transpose();
      } else if(!b->isNull() && b->isRkMatrix()){
        // Xa Xb^t U = Ba Bb^t
        //   - Xa = Ba
        //   - Xb^t U = Bb^t
        // Xb is stored without been transposed
        // it become again a resolution by column of Bb
        HMatrix<T> * tmp;
        if(*rows() == *b->cols())
            tmp = b;
        else
            tmp = b->subset(b->rows(), this->rows());
        this->solveUpperTriangularRight(tmp->rk()->b, unitriangular, lowerStored);
        if(tmp != b)
            delete tmp;
      } else {
        // b is a null block so nothing to do
      }
    } else {
      // B is not a leaf, then so is L
      assert(this->isLeaf());
      assert(isFullMatrix());
      // Evaluate B, solve by column and restore all in the matrix
      // TODO: check if it's not too bad
      FullMatrix<T>* bFull = new FullMatrix<T>(b->rows(), b->cols());
      b->evalPart(bFull, b->rows(), b->cols());
      bFull->transpose();
      this->solveUpperTriangularRight(bFull, unitriangular, lowerStored);
      bFull->transpose();
      // int bRows = b->rows()->size();
      // int bCols = b->cols()->size();
      // Vector<T> bRow(bCols);
      // for (int row = 0; row < bRows; row++) {
      //   blasCopy<T>(bCols, bFull->data.m + row, bRows, bRow.v, 1);
      //   FullMatrix<T> tmp(bRow.v, bRow.rows, 1);
      //   this->solveUpperTriangularRight(&tmp);
      //   blasCopy<T>(bCols, bRow.v, 1, bFull->data.m + row, bRows);      // }
      // }
      b->clear();
      b->axpy(Constants<T>::pone, bFull);
      delete bFull;
    }
  }
}

/* Resolve U.X=B, solution saved in B, with B Hmat
   Only called by luDecomposition
 */
template<typename T>
void HMatrix<T>::solveUpperTriangularLeft(HMatrix<T>* b, bool unitriangular, bool lowerStored, MainOp) const {
  DECLARE_CONTEXT;
  if (rows()->size() == 0 || cols()->size() == 0) return;
  // At first, the recursion one (simple case)
  if (!this->isLeaf() && !b->isLeaf()) {
    this->recursiveSolveUpperTriangularLeft(b, unitriangular, lowerStored);
  } else {
    // if B is a leaf, the resolve is done by column
    if (b->isLeaf()) {
      HMatrix * bSubset = b->subset(lowerStored ? this->rows() : this->cols(), b->cols());
      if (bSubset->isFullMatrix()) {
        this->solveUpperTriangularLeft(bSubset->full(), unitriangular, lowerStored);
      } else if(!bSubset->isNull()){
        assert(b->isRkMatrix());
        this->solveUpperTriangularLeft(bSubset->rk()->a, unitriangular, lowerStored);
      }
      if(b != bSubset)
          delete bSubset;
    } else {
      // B isn't a leaf, then so is L
      assert(this->isLeaf());
      // Evaluate B, solve by column, and restore in the matrix
      // TODO: check if it's not too bad
      FullMatrix<T>* bFull = new FullMatrix<T>(b->rows(), b->cols());
      b->evalPart(bFull, b->rows(), b->cols());
      this->solveUpperTriangularLeft(bFull, unitriangular, lowerStored);
      b->clear();
      b->axpy(Constants<T>::pone, bFull);
      delete bFull;
    }
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangularRight(ScalarArray<T>* b, bool unitriangular, bool lowerStored) const {
  DECLARE_CONTEXT;
  assert(*rows() == *cols());
  if (rows()->size() == 0 || cols()->size() == 0) return;
  // B is supposed given in form of a row vector, but transposed
  // so we can deal it with a subset as usual.
  if (this->isLeaf()) {
    assert(this->isFullMatrix());
    ScalarArray<T>* bCopy = b->copyAndTranspose();
    full()->solveUpperTriangularRight(bCopy, unitriangular, lowerStored);
    bCopy->transpose();
    b->copyMatrixAtOffset(bCopy, 0, 0);
    delete bCopy;
  } else {

    int offset(0);
    vector<ScalarArray<T> > sub;
    for (int i=0 ; i<nrChildRow() ; i++) {
      // Create sub[i] = a FullMatrix (without copy of data) for the lines in front of the i-th matrix block
      sub.push_back(b->rowsSubset(offset, get(i, i)->rows()->size()));
      offset += get(i, i)->rows()->size();
    }
    for (int i=0 ; i<nrChildRow() ; i++) {
      // Update sub[i] with the contribution of the solutions already computed sub[j]
      for (int j=0 ; j<i ; j++) {
        const HMatrix<T>* u_ji = (lowerStored ? get(i, j) : get(j, i));
        if (u_ji)
          u_ji->gemv(lowerStored ? 'N' : 'T', Constants<T>::mone, &sub[j], Constants<T>::pone, &sub[i]);
      }
      // Solve the i-th diagonal system
      get(i, i)->solveUpperTriangularRight(&sub[i], unitriangular, lowerStored);
    }
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangularRight(FullMatrix<T>* b, bool unitriangular, bool lowerStored) const {
  solveUpperTriangularRight(&b->data, unitriangular, lowerStored);
}

template<typename T>
void HMatrix<T>::solveUpperTriangularLeft(ScalarArray<T>* b, bool unitriangular, bool lowerStored) const {
  DECLARE_CONTEXT;
  assert(*rows() == *cols());
  assert(rows()->size() == b->rows || !lowerStored);
  assert(cols()->size() == b->rows || lowerStored);
  if (rows()->size() == 0 || cols()->size() == 0) return;
  if (this->isLeaf()) {
    full()->solveUpperTriangularLeft(b, unitriangular, lowerStored);
  } else {

    int offset(0);
    vector<ScalarArray<T> > sub;
    for (int i=0 ; i<nrChildRow() ; i++) {
      // Create sub[i] = a FullMatrix (without copy of data) for the lines in front of the i-th matrix block
      sub.push_back(b->rowsSubset(offset, get(i, i)->cols()->size()));
      offset += get(i, i)->cols()->size();
    }
    for (int i=nrChildRow()-1 ; i>=0 ; i--) {
      // Solve the i-th diagonal system
      get(i, i)->solveUpperTriangularLeft(&sub[i], unitriangular, lowerStored);
      // Update sub[j] j<i with the contribution of the solutions just computed sub[i]
      for (int j=0 ; j<i ; j++) {
        const HMatrix<T>* u_ji = (lowerStored ? get(i, j) : get(j, i));
        if (u_ji)
          u_ji->gemv(lowerStored ? 'T' : 'N', Constants<T>::mone, &sub[i], Constants<T>::pone, &sub[j]);
      }
    }
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangularLeft(FullMatrix<T>* b, bool unitriangular, bool lowerStored) const {
  solveUpperTriangularLeft(&b->data, unitriangular, lowerStored);
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
  this->gemm('N', 'T', Constants<T>::mone, x, n, Constants<T>::pone); // this -= M.D.tN
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
      RkMatrix<T>* rkMat = RkMatrix<T>::multiplyRkRk('N', 'T', m_copy->rk(), m->rk());
      delete m_copy;

      this->axpy(Constants<T>::mone, rkMat);
      delete rkMat;
    } else if(m->isFullMatrix()){
      HMatrix<T>* copy_m = m->copy();
      HMAT_ASSERT(copy_m);
      copy_m->multiplyWithDiag(d); // right multiplication by D

      FullMatrix<T>* fullMat = HMatrix<T>::multiplyFullMatrix('N', 'T', copy_m, m);
      HMAT_ASSERT(fullMat);
      delete copy_m;

      this->axpy(Constants<T>::mone, fullMat);
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

        RkMatrix<T>* rkMat = RkMatrix<T>::multiplyRkRk('N', 'T', m_copy->rk(), m->rk());
        FullMatrix<T>* fullMat = rkMat->eval();
        delete m_copy;
        delete rkMat;
        full()->axpy(Constants<T>::mone, fullMat);
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
        mTmp.multiplyWithDiagOrDiagInv(d->full()->diagonal, false, false);
      } else {
        Vector<T> diag(d->cols()->size());
        d->extractDiagonal(diag.m);
        mTmp.multiplyWithDiagOrDiagInv(&diag, false, false);
      }
      full()->gemm('N', 'T', Constants<T>::mone, &mTmp, m->full(), Constants<T>::pone);
    } else if (!m->isLeaf()){
      FullMatrix<T> mTmp(m->rows(), m->cols());
      m->evalPart(&mTmp, m->rows(), m->cols());
      FullMatrix<T> mTmpCopy(m->rows(), m->cols());
      mTmpCopy.copyMatrixAtOffset(&mTmp, 0, 0);
      if (d->isFullMatrix()) {
        mTmp.multiplyWithDiagOrDiagInv(d->full()->diagonal, false, false);
      } else {
        Vector<T> diag(d->cols()->size());
        d->extractDiagonal(diag.m);
        mTmp.multiplyWithDiagOrDiagInv(&diag, false, false);
      }
      full()->gemm('N', 'T', Constants<T>::mone, &mTmp, &mTmpCopy, Constants<T>::pone);
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
  this->solveLowerTriangularLeft(b, true);
  // Then compute U X = Y
  this->solveUpperTriangularLeft(b, false, false);
}

template<typename T>
void HMatrix<T>::solve(FullMatrix<T>* b) const {
  solve(&b->data);
}

template<typename T>
void HMatrix<T>::extractDiagonal(T* diag) const {
  DECLARE_CONTEXT;
  if (rows()->size() == 0 || cols()->size() == 0) return;
  if(this->isLeaf()) {
    assert(isFullMatrix());
    if(full()->diagonal) {
      // LDLt
      memcpy(diag, full()->diagonal->m, full()->rows() * sizeof(T));
    } else {
      // LLt
      for (int i = 0; i < full()->rows(); ++i)
        diag[i] = full()->data.m[i*full()->rows() + i];
    }
  } else {
    for (int i=0 ; i<nrChildRow() ; i++) {
      get(i,i)->extractDiagonal(diag);
      diag += get(i,i)->rows()->size();
    }
  }
}

/* Solve M.X=B with M hmat LU factorized*/
template<typename T> void HMatrix<T>::solve(
        HMatrix<T>* b,
        hmat_factorization_t factorizationType) const {
    DECLARE_CONTEXT;
    switch(factorizationType) {
    case hmat_factorization_lu:
    /* Solve LX=B, result in B */
    this->solveLowerTriangularLeft(b, true);
    /* Solve UX=B, result in B */
        this->solveUpperTriangularLeft(b, false, false);
        break;
    case hmat_factorization_ldlt:
        /* Solve LX=B, result in B */
        this->solveLowerTriangularLeft(b, true);
        /* Solve DX=B, result in B */
        b->multiplyWithDiag(this, true, true);
        /* Solve L^tX=B, result in B */
        this->solveUpperTriangularLeft(b, true, true);
        break;
    case hmat_factorization_llt:
        /* Solve LX=B, result in B */
        this->solveLowerTriangularLeft(b, false);
        /* Solve L^tX=B, result in B */
        this->solveUpperTriangularLeft(b, false, true);
        break;
    default:
        HMAT_ASSERT(false);
    }
}

template<typename T> void HMatrix<T>::solveDiagonal(ScalarArray<T>* b) const {
    // Solve D*X = B and store result into B
    // Diagonal extraction
    T* diag;
    bool extracted = false;
    if (rows()->size() == 0 || cols()->size() == 0) return;
    if(isFullMatrix() && full()->diagonal) {
        // LDLt
        diag = full()->diagonal->m;
    } else {
        // LLt
        diag = new T[cols()->size()];
        extractDiagonal(diag);
        extracted = true;
    }
    for (int j = 0; j < b->cols; j++) {
        for (int i = 0; i < b->rows; i++) {
            b->get(i, j) = b->get(i, j) / diag[i];
        }
    }
    if(extracted)
        delete[] diag;
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
  this->solveLowerTriangularLeft(b, true);

  // B <- D^{-1} Y : solution of D*Y = B : Y = L^T * X
  this->solveDiagonal(b);

  // B <- solution of L^T X = B :  the solution X we are looking for is stored in B
  this->solveUpperTriangularLeft(b, true, true);
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
  this->solveLowerTriangularLeft(b, false);

  // B <- solution of L^T X = B :  the solution X we are looking for is stored in B
  this->solveUpperTriangularLeft(b, false, true);
}

template<typename T>
void HMatrix<T>::solveLlt(FullMatrix<T>* b) const {
  solveLlt(&b->data);
}

template<typename T>
void HMatrix<T>::checkStructure() const {
  return;
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
}

template<typename T>
void HMatrix<T>::checkNan() const {
  return;
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

template<typename T>  void HMatrix<T>::rk(const ScalarArray<T> * a, const ScalarArray<T> * b, bool updateRank) {
    assert(isRkMatrix());
    if(a == NULL && isNull())
        return;
    if(rk_ == NULL)
        rk(new RkMatrix<T>(NULL, rows(), NULL, cols(), Svd));
    // TODO: if the matrices exist and are of the right size (same rank),
    // reuse them.
    if (rk_->a) {
      delete rk_->a;
    }
    if (rk_->b) {
      delete rk_->b;
    }
    rk_->a = a == NULL ? NULL : a->copy();
    rk_->b = b == NULL ? NULL : b->copy();
    if(updateRank)
        rank_ = rk_->rank();
}

template<typename T> std::string HMatrix<T>::toString() const {
    std::vector<const HMatrix<T> *> leaves;
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
            ", diagNorm=" << diagNorm << ")";
    return sstm.str();
}

template<typename T>
void EpsilonTruncate<T>::visit(HMatrix<T>* node, const Visit order) const {
  if (order != tree_leaf || !node->isRkMatrix()) return;
  RkMatrix<T> * rk = node->rk();
  rk->truncate(epsilon_);
  // Update rank
  node->rk(rk);
}

template<typename T>
void LeafEpsilonTruncate<T>::apply(HMatrix<T>* node) const {
  if (!node->isRkMatrix()) return;
  RkMatrix<T> * rk = node->rk();
  rk->truncate(epsilon_);
  // Update rank
  node->rk(rk);
}

template<typename T>
HMatrix<T> * HMatrix<T>::unmarshall(const MatrixSettings * settings, int rank, int appoxRank, char bitfield) {
    HMatrix<T> * m = new HMatrix<T>(settings);
    m->rank_ = rank;
    m->isUpper = (bitfield & 1 << 0 ? true : false);
    m->isLower = (bitfield & 1 << 1 ? true : false);
    m->isTriUpper = (bitfield & 1 << 2 ? true : false);
    m->isTriLower = (bitfield & 1 << 3 ? true : false);
    m->keepSameRows = (bitfield & 1 << 4 ? true : false);
    m->keepSameCols = (bitfield & 1 << 5 ? true : false);
    m->approximateRank_ = appoxRank;
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
    ownColsClusterTree_(false), localSettings(_children[0]->localSettings.global) {
    this->children = _children;
}

template<typename T> void HMatrix<T>::rank(int rank) {
    HMAT_ASSERT_MSG(rank_ >= 0, "HMatrix::rank can only be used on Rk blocks");
    HMAT_ASSERT_MSG(!rk() || rk()->a == NULL || rk()->rank() == rank,
        "HMatrix::rank can only be used on evicted blocks");
    rank_ = rank;
}

// Templates declaration
template class HMatrix<S_t>;
template class HMatrix<D_t>;
template class HMatrix<C_t>;
template class HMatrix<Z_t>;

template void reorderVector(ScalarArray<S_t>* v, int* indices);
template void reorderVector(ScalarArray<D_t>* v, int* indices);
template void reorderVector(ScalarArray<C_t>* v, int* indices);
template void reorderVector(ScalarArray<Z_t>* v, int* indices);

template void restoreVectorOrder(ScalarArray<S_t>* v, int* indices);
template void restoreVectorOrder(ScalarArray<D_t>* v, int* indices);
template void restoreVectorOrder(ScalarArray<C_t>* v, int* indices);
template void restoreVectorOrder(ScalarArray<Z_t>* v, int* indices);

template class EpsilonTruncate<S_t>;
template class EpsilonTruncate<D_t>;
template class EpsilonTruncate<C_t>;
template class EpsilonTruncate<Z_t>;

template class LeafEpsilonTruncate<S_t>;
template class LeafEpsilonTruncate<D_t>;
template class LeafEpsilonTruncate<C_t>;
template class LeafEpsilonTruncate<Z_t>;

}  // end namespace hmat

#include "recursion.cpp"

namespace hmat {

  // Explicit template instantiation
  template class RecursionMatrix<S_t, HMatrix<S_t> >;
  template class RecursionMatrix<C_t, HMatrix<C_t> >;
  template class RecursionMatrix<D_t, HMatrix<D_t> >;
  template class RecursionMatrix<Z_t, HMatrix<Z_t> >;

}  // end namespace hmat
