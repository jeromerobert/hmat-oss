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
#include "common/context.hpp"
#include "common/my_assert.h"

using namespace std;

namespace hmat {

// The default values below will be overwritten in default_engine.cpp by HMatSettings values
template<typename T> bool HMatrix<T>::coarsening = false;
template<typename T> bool HMatrix<T>::recompress = false;
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
}

template<typename T>
void reorderVector(FullMatrix<T>* v, int* indices) {
  DECLARE_CONTEXT;
  const int n = v->rows;
  Vector<T> tmp(n);
  for (int col = 0; col < v->cols; col++) {
    T* column = v->m + ((size_t) n) * col;
    for (int i = 0; i < n; i++) {
      tmp.v[i] = column[indices[i]];
    }
    memcpy(column, tmp.v, sizeof(T) * n);
  }
}


template<typename T>
void restoreVectorOrder(FullMatrix<T>* v, int* indices) {
  DECLARE_CONTEXT;
  const int n = v->rows;
  Vector<T> tmp(n);

  for (int col = 0; col < v->cols; col++) {
    T* column = v->m + ((size_t) n) * col;
    for (int i = 0; i < n; i++) {
      tmp.v[indices[i]] = column[i];
    }
    memcpy(column, tmp.v, sizeof(T) * n);
  }
}


template<typename T>
HMatrix<T>::HMatrix(ClusterTree* _rows, ClusterTree* _cols, const hmat::MatrixSettings * settings,
                    SymmetryFlag symFlag, AdmissibilityCondition * admissibilityCondition)
  : Tree<4>(NULL), rows_(_rows), cols_(_cols), rk_(NULL), rank_(-3),
    isUpper(false), isLower(false),
    isTriUpper(false), isTriLower(false), admissible(false), temporary(false),
    localSettings(settings)
{
  admissible = admissibilityCondition->isAdmissible(*(rows_), *(cols_));
  if (_rows->isLeaf() || _cols->isLeaf() || admissible) {
    if (admissible) {
      rk(new RkMatrix<T>(NULL, rows(), NULL, cols(), NoCompression));
    }
  } else {
    isUpper = false;
    isLower = (symFlag == kLowerSymmetric ? true : false);
    isTriUpper = false;
    isTriLower = false;
    for (int i = 0; i < 2; ++i) {
      ClusterTree* rowChild = static_cast<ClusterTree*>(_rows->getChild(i));
      for (int j = 0; j < 2; ++j) {
        if ((symFlag == kNotSymmetric) || (isUpper && (i <= j)) || (isLower && (i >= j))) {
          ClusterTree* colChild = static_cast<ClusterTree*>(_cols->getChild(j));
          this->insertChild(i, j, new HMatrix<T>(rowChild, colChild, settings, (i == j ? symFlag : kNotSymmetric), admissibilityCondition));
         }
       }
     }
  }
  admissibilityCondition->clean(*(rows_));
  admissibilityCondition->clean(*(cols_));
}

template<typename T>
HMatrix<T>::HMatrix(const hmat::MatrixSettings * settings) :
    Tree<4>(NULL), rows_(NULL), cols_(NULL), rk_(NULL), rank_(-3), isUpper(false),
    isLower(false), admissible(false), temporary(false), localSettings(settings)
    {}

template<typename T> HMatrix<T> * HMatrix<T>::internalCopy(bool temporary, bool withChildren) const {
    HMatrix<T> * r = new HMatrix<T>(localSettings.global);
    r->rows_ = rows_;
    r->cols_ = cols_;
    r->temporary = temporary;
    if(withChildren) {
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                HMatrix<T>* child = new HMatrix<T>(localSettings.global);
                child->temporary = temporary;
                assert(rows_->getChild(i) != NULL);
                assert(cols_->getChild(j) != NULL);
                child->rows_ = dynamic_cast<ClusterTree*>(rows_->getChild(i));
                child->cols_ = dynamic_cast<ClusterTree*>(cols_->getChild(j));
                child->rk(new RkMatrix<T>(NULL, &child->rows_->data,
                                                 NULL, &child->cols_->data,
                                                 NoCompression));
                r->insertChild(i, j, child);
            }
        }
    }
    return r;
}

template<typename T>
HMatrix<T>* HMatrix<T>::copyStructure() const {
  HMatrix<T>* h = internalCopy();
  h->isUpper = isUpper;
  h->isLower = isLower;
  h->isTriUpper = isTriUpper;
  h->isTriLower = isTriLower;
  h->admissible = admissible;
  h->rank_ = rank_ >= 0 ? 0 : rank_;
  if(!isLeaf()){
    for (int i = 0; i < 4; ++i) {
      if (getChild(i)) {
        const HMatrix<T>* child = static_cast<const HMatrix<T>*>(getChild(i));
        h->insertChild(i, child->copyStructure());
      }
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
  h->admissible = o->admissible;
  h->rank_ = o->rank_;
  if(!o->isLeaf()){
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        if (o->get(i, j)) {
          h->insertChild(i, j, HMatrix<T>::Zero(o->get(i, j)));
        }
      }
    }
  }
  return h;
}

template<typename T>
HMatrix<T>* HMatrix<T>::Zero(const ClusterTree* rows, const ClusterTree* cols,
                             const hmat::MatrixSettings * settings,
                             AdmissibilityCondition * admissibilityCondition) {
  // Leaves are filled by 0
  HMatrix<T> *h = new HMatrix<T>(settings);
  h->rows_ = rows;
  h->cols_ = cols;
  h->admissible = admissibilityCondition->isAdmissible(*(h->rows_), *(h->cols_));
  if (rows->isLeaf() || cols->isLeaf() || h->admissible) {
    if (h->admissible) {
      h->rank_ = 0;
    } else {
      h->rank_ = -1;
    }
  } else {
    for (int i = 0; i < 2; ++i) {
      const ClusterTree* rowChild = static_cast<const ClusterTree*>(h->rows_->getChild(i));
      for (int j = 0; j < 2; ++j) {
        const ClusterTree* colChild = static_cast<const ClusterTree*>(h->cols_->getChild(j));
        h->insertChild(i, j, HMatrix<T>::Zero(rowChild, colChild, settings, admissibilityCondition));
      }
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
    } else if(!isLeaf()) {
        for (int i = 0; i < 2; ++i) {
            const ClusterTree* rowChild = static_cast<const ClusterTree*>(rows->getChild(i));
            for (int j = 0; j < 2; ++j) {
                const ClusterTree* colChild = static_cast<const ClusterTree*>(cols->getChild(j));
                if(get(i, j))
                    get(i, j)->setClusterTrees(rowChild, colChild);
            }
        }
    }
}

template<typename T>
void HMatrix<T>::assemble(Assembly<T>& f, const AllocationObserver & ao) {
  if (isLeaf()) {
    // If the leaf is admissible, matrix assembly and compression.
    // if not we keep the matrix.
    FullMatrix<T> * m = NULL;
    RkMatrix<T>* assembledRk = NULL;
    f.assemble(localSettings, *rows_, *cols_, admissible, m, assembledRk, ao);
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
    assembled();
    for (int i = 0; i < 4; i++) {
      HMatrix<T> *child = static_cast<HMatrix*>(getChild(i));
      child->assemble(f, ao);
    }
    if (coarsening) {
      // If all children are Rk leaves, then we try to merge them into a single Rk-leaf.
      // This is done if the memory of the resulting leaf is less than the sum of the initial
      // leaves. Note that this operation could be used hierarchically.

      bool allRkLeaves = true;
      const RkMatrix<T>* childrenArray[4];
      size_t childrenElements = 0;
      for (int i = 0; i < 4; i++) {
        HMatrix<T> *child = static_cast<HMatrix*>(getChild(i));
        if (!child->isRkMatrix()) {
          allRkLeaves = false;
          break;
        } else {
          childrenArray[i] = child->rk();
          childrenElements += ((size_t)childrenArray[i]->rows->size()
                               + childrenArray[i]->cols->size()) * childrenArray[i]->rank();
        }
      }
      if (allRkLeaves) {
        RkMatrix<T> dummy(NULL, rows(), NULL, cols(), NoCompression);
        T alpha[4] = {Constants<T>::pone, Constants<T>::pone, Constants<T>::pone, Constants<T>::pone};
        RkMatrix<T>* candidate = dummy.formattedAddParts(alpha, childrenArray, 4);
        size_t elements = (((size_t) candidate->rows->size()) + candidate->cols->size()) * candidate->rank();
        if (elements < childrenElements) {
          cout << "Coarsening ! " << elements << " < " << childrenElements << endl;
          for (int i = 0; i < 4; i++) {
            removeChild(i);
          }
          delete[] children;
          children = NULL;
          rk(candidate);
          assert(isLeaf());
          assert(isRkMatrix());
        } else {
          delete candidate;
        }
      }
    }
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

  if (isLeaf()) {
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
        if(upper->rk() != NULL)
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
    assembled();
    if (onlyLower) {
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          if ((*rows() == *cols()) && (j > i)) {
            continue;
          }
          get(i,j)->assembleSymmetric(f, NULL, true, ao);
        }
      }
    } else {
      if (this == upper) {
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j <= i; j++) {
            HMatrix<T> *child = get(i, j);
            HMatrix<T> *upperChild = get(j, i);
            assert(child != NULL);
            child->assembleSymmetric(f, upperChild, false, ao);
          }
        }
      } else {
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++) {
            HMatrix<T> *child = get(i, j);
            HMatrix<T> *upperChild = upper->get(j, i);
            child->assembleSymmetric(f, upperChild, false, ao);
          }
        }
        upper->assembled();
        if (coarsening) {
            // If all children are Rk leaves, then we try to merge them into a single Rk-leaf.
            // This is done if the memory of the resulting leaf is less than the sum of the initial
          bool allRkLeaves = true;
          const RkMatrix<T>* childrenArray[4];
          size_t childrenElements = 0;
          for (int i = 0; i < 4; i++) {
            HMatrix<T> *child = static_cast<HMatrix*>(getChild(i));
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
            T alpha[4] = {Constants<T>::pone, Constants<T>::pone, Constants<T>::pone, Constants<T>::pone};
            RkMatrix<T> dummy(NULL, rows(), NULL, cols(), NoCompression);
            RkMatrix<T>* candidate = dummy.formattedAddParts(alpha, childrenArray, 4);
            size_t elements = (((size_t) candidate->rows->size()) + candidate->cols->size()) * candidate->rank();
            if (elements < childrenElements) {
              cout << "Coarsening ! " << elements << " < " << childrenElements << endl;
              for (int i = 0; i < 4; i++) {
                removeChild(i);
                upper->removeChild(i);
              }
              delete[] children;
              children = NULL;
              delete[] upper->children;
              upper->children = NULL;
              rk(candidate);
              upper->rk(new RkMatrix<T>(candidate->b->copy(), upper->rows(),
                                        candidate->a->copy(), upper->cols(), candidate->method));
              assert(isLeaf() && upper->isLeaf());
              assert(isRkMatrix() && upper->isRkMatrix());
            } else {
              delete candidate;
            }
          }
        }
      }
    }
  }
}

template<typename T> void HMatrix<T>::info(hmat_info_t & result) {
    result.nr_block_clusters++;
    if(isLeaf()) {
        size_t s = ((size_t)rows()->size()) * cols()->size();
        result.uncompressed_size += s;
        if(isRkMatrix()) {
            if(!isNull()) {
                size_t mem = rank() * (((size_t)rows()->size()) + cols()->size());
                result.compressed_size += mem;
                int dim = result.largest_rk_dim_cols + result.largest_rk_dim_rows;
                if(rows()->size() + cols()->size() > dim) {
                    result.largest_rk_dim_cols = cols()->size();
                    result.largest_rk_dim_rows = rows()->size();
                }

                size_t old_s = ((size_t)result.largest_rk_mem_cols + result.largest_rk_mem_rows) * result.largest_rk_mem_rank;
                if(mem > old_s) {
                    result.largest_rk_mem_cols = cols()->size();
                    result.largest_rk_mem_rows = rows()->size();
                    result.largest_rk_mem_rank = rank();
                }
            }
            result.rk_count++;
            result.rk_size += s;
        } else {
            result.compressed_size += s;
            result.full_count ++;
            result.full_size += s;
        }
    } else {
        for (int i = 0; i < 4; i++) {
            HMatrix<T> *child = getChild(i);
            if (child)
                child->info(result);
        }
    }
}

template<typename T>
void HMatrix<T>::eval(FullMatrix<T>* result, bool renumber) const {
  if (isLeaf()) {
    FullMatrix<T> *mat = isRkMatrix() ? rk()->eval() : full();
    int *rowIndices = rows()->indices() + rows()->offset();
    int rowCount = rows()->size();
    int *colIndices = cols()->indices() + cols()->offset();
    int colCount = cols()->size();
    for (int i = 0; i < rowCount; i++) {
      for (int j = 0; j < colCount; j++) {
        if(renumber)
          result->get(rowIndices[i], colIndices[j]) = mat->get(i, j);
        else
          result->get(rows()->offset() + i, cols()->offset() + j) = mat->get(i, j);
      }
    }
    if (isRkMatrix()) {
      delete mat;
    }
  } else {
    for (int i = 0; i < 4; i++) {
      if (getChild(i)) {
        static_cast<HMatrix<T>*>(getChild(i))->eval(result, renumber);
      }
    }
  }
}

template<typename T>
void HMatrix<T>::evalPart(FullMatrix<T>* result, const IndexSet* _rows,
                          const IndexSet* _cols) const {
  if (isLeaf()) {
    FullMatrix<T> *mat = isRkMatrix() ? rk()->eval() : full();
    int rowOffset = rows()->offset() - _rows->offset();
    int rowCount = rows()->size();
    int colOffset = cols()->offset() - _cols->offset();
    int colCount = cols()->size();
    for (int i = 0; i < rowCount; i++) {
      for (int j = 0; j < colCount; j++) {
        result->get(i + rowOffset, j + colOffset) = mat->get(i, j);
      }
    }
    if (isRkMatrix()) {
      delete mat;
    }
  } else {
    for (int i = 0; i < 4; i++) {
      if (getChild(i)) {
        static_cast<HMatrix<T>*>(getChild(i))->evalPart(result, _rows, _cols);
      }
    }
  }
}

template<typename T> double HMatrix<T>::normSqr() const {
  double result = 0.;
  if (rows()->size() == 0 || cols()->size() == 0) {
    return result;
  }
  if (isLeaf() && !isNull()) {
    if (isRkMatrix()) {
      // Approximate ||a * bt|| by ||a||*||b|| so we return a
      // upper bound of the actual norm
      result = rk()->a->normSqr() * rk()->b->normSqr();
    } else {
      result = full()->normSqr();
    }
  } else if(!isLeaf()){
    for (int i = 0; i < 4; i++) {
      if (getChild(i)) {
        result += getChild(i)->normSqr();
      }
    }
  }
  return result;
}

template<typename T>
void HMatrix<T>::scale(T alpha) {
  if (isLeaf()) {
    if (isNull()) {
      // nothing to do
    } else if (isRkMatrix()) {
      rk()->scale(alpha);
    } else {
      assert(isFullMatrix());
      full()->scale(alpha);
    }
  } else {
    for (int i = 0; i < 4; i++) {
      if (getChild(i)) {
        HMatrix* child = static_cast<HMatrix<T>*>(getChild(i));
        child->scale(alpha);
      }
    }
  }
}

template<typename T>
void HMatrix<T>::gemv(char trans, T alpha, const Vector<T>* x, T beta, Vector<T>* y) const {
  if (rows()->size() == 0 || cols()->size() == 0) return;
  FullMatrix<T> mx(x->v, x->rows, 1);
  FullMatrix<T> my(y->v, y->rows, 1);
  gemv(trans, alpha, &mx, beta, &my);
}

template<typename T>
void HMatrix<T>::gemv(char matTrans, T alpha, const FullMatrix<T>* x, T beta, FullMatrix<T>* y) const {
  if (rows()->size() == 0 || cols()->size() == 0) return;
  if (beta != Constants<T>::pone) {
    y->scale(beta);
  }
  beta = Constants<T>::pone;

  if (!isLeaf()) {
    const ClusterData* myRows = rows();
    const ClusterData* myCols = cols();
    for (int i = 0; i < 4; i++) {
      HMatrix<T>* child = static_cast<HMatrix<T>*>(getChild(i));
      char trans = matTrans;
      if(!child)
      {
        if (isTriLower || isTriUpper)
          continue;
        else if (isLower)
        {
          assert(i == 2);
          child = static_cast<HMatrix<T>*>(get(1, 0));
          trans = (trans == 'N' ? 'T' : 'N');
        }
        else
        {
          assert(isUpper);
          assert(i == 1);
          child = static_cast<HMatrix<T>*>(get(0, 1));
          trans = (trans == 'N' ? 'T' : 'N');
        }
      }
      const ClusterData* childRows = child->rows();
      const ClusterData* childCols = child->cols();
      int rowsOffset = childRows->offset() - myRows->offset();
      int colsOffset = childCols->offset() - myCols->offset();
      if (trans == 'N') {
        assert(colsOffset + childCols->size() <= x->rows);
        assert(rowsOffset + childRows->size() <= y->rows);
        FullMatrix<T> subX(x->m + colsOffset, childCols->size(), x->cols, x->lda);
        FullMatrix<T> subY(y->m + rowsOffset, childRows->size(), y->cols, y->lda);
        child->gemv(trans, alpha, &subX, beta, &subY);
      } else {
        assert(trans == 'T');
        assert(rowsOffset + childRows->size() <= x->rows);
        assert(colsOffset + childCols->size() <= y->rows);
        FullMatrix<T> subX(x->m + rowsOffset, childRows->size(), x->cols, x->lda);
        FullMatrix<T> subY(y->m + colsOffset, childCols->size(), y->cols, y->lda);
        child->gemv(trans, alpha, &subX, beta, &subY);
      }
    }
  } else {
    if (isFullMatrix()) {
      y->gemm(matTrans, 'N', alpha, full(), x, beta);
    } else if(!isNull()){
      rk()->gemv(matTrans, alpha, x, beta, y);
    } else if(beta != Constants<T>::pone){
      y->scale(beta);
    }
  }
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
        for(int i = 0; i < 4; i++) {
            if(!listAllRk(m->getChild(i), result))
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
    if(*rows() == *x->rows() && *cols() == *x->cols()) {
        if (isLeaf()) {
            if (isRkMatrix()) {
                if(x->isRkMatrix()) {
                    if (x->isNull()) {
                        return;
                    }
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
                    full(new FullMatrix<T>(rows()->size(), cols()->size()));
                if(x->isFullMatrix()) {
                    full()->axpy(alpha, x->full());
                } else if(x->isRkMatrix()) {
                    FullMatrix<T> * f = x->rk()->eval();
                    full()->axpy(alpha, f);
                    delete f;
                } else {
                    // nothing TODO
                }
            }
        } else {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    HMatrix<T>* child = get(i, j);
                    const HMatrix<T>* bChild = x->isLeaf() ? x : x->get(i, j);
                    child->axpy(alpha, bChild);
                }
            }
        }
    } else {
        if(x->isFullMatrix()) {
            axpy(alpha, x->full(), x->rows(), x->cols());
            return;
        }
        else if(x->isRkMatrix()) {
            axpy(alpha, x->rk());
            return;
        }
        else if(x->isLeaf()){
            // X is an empty leaf, so nothing to do
        } else {
            HMAT_ASSERT(false);
        }
    }
}

/** @brief AXPY between this an a subset of B with B a RkMatrix */
template<typename T>
void HMatrix<T>::axpy(T alpha, const RkMatrix<T>* b) {
  DECLARE_CONTEXT;
  // this += alpha * b
  assert(b->rows->isSuperSet(*rows()));
  assert(b->cols->isSuperSet(*cols()));

  if (b->rank() == 0 || rows()->size() == 0 || cols()->size() == 0) {
    return;
  }

  if (!isLeaf()) {
    for (int i = 0; i < 4; i++) {
      if (getChild(i)) {
        static_cast<HMatrix<T>*>(getChild(i))->axpy(alpha, b);
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
      if(isFullMatrix()) {
        full()->axpy(alpha, rkMat);
        delete rkMat;
      } else {
        full(rkMat);
        full()->scale(alpha);
      }
    }
    if (needResizing) {
      delete newRk;
    }
  }
}

/** @brief AXPY between this an a subset of B with B a FullMatrix */
template<typename T>
void HMatrix<T>::axpy(T alpha, const FullMatrix<T>* b, const IndexSet* rows,
                      const IndexSet* cols) {
  DECLARE_CONTEXT;
  // this += alpha * b
  assert(rows->isSuperSet(*this->rows()) && cols->isSuperSet(*this->cols()));
  if (!isLeaf()) {
    for (int i = 0; i < 4; i++) {
      HMatrix<T>* child = static_cast<HMatrix<T>*>(getChild(i));
      if (!child) {
        continue;
      }
      IndexSet childRows, childCols;
      childRows.intersection(*child->rows(), *rows);
      childCols.intersection(*child->cols(), *cols);
      if (childRows.size() > 0 && childCols.size() > 0) {
        int rowOffset = childRows.offset() - rows->offset();
        int colOffset = childCols.offset() - cols->offset();
        FullMatrix<T> subB(b->m + rowOffset + colOffset * b->lda,
                           childRows.size(), childCols.size(), b->lda);
        child->axpy(alpha, &subB, &childRows, &childCols);
      }
    }
  } else {
    int rowOffset = this->rows()->offset() - rows->offset();
    int colOffset = this->cols()->offset() - cols->offset();
    FullMatrix<T> subMat(b->m + rowOffset + ((size_t) colOffset) * b->lda,
                         this->rows()->size(), this->cols()->size(), b->lda);
    if (isFullMatrix()) {
      full()->axpy(alpha, &subMat);
    } else {
      assert(isRkMatrix());
      rk()->axpy(alpha, &subMat);
      rank_ = rk()->rank();
    }
  }
}

template<typename T> HMatrix<T> * HMatrix<T>::subset(
    const IndexSet * rows, const IndexSet * cols) const
{
    if((this->rows() == rows && this->cols() == cols) ||
       (*(this->rows()) == *rows && *(this->cols()) == *cols) ||
       (!rows->isSubset(*(this->rows())) || !cols->isSubset(*(this->cols()))))
        return const_cast<HMatrix<T>*>(this);

    if(this->isLeaf()) {
        HMatrix<T> * tmpMatrix = new HMatrix<T>(this->localSettings.global);
        tmpMatrix->temporary=true;
        if(this->isRkMatrix()) {
            tmpMatrix->rk(const_cast<RkMatrix<T>*>(rk()->subset(rows, cols)));
            tmpMatrix->rows_ = rows_->slice(rows->offset(), rows->size());
            tmpMatrix->cols_ = cols_->slice(cols->offset(), cols->size());
        } else {
            //TODO not yet implemented but will happen
            HMAT_ASSERT(false);
        }
        return tmpMatrix;
    } else {
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

    // suppose that A is bigger than B
    const IndexSet * cdb = row_b ? in_b->rows() : in_b->cols();
    if(row_a)
        out_a = in_a->subset(cdb, in_a->cols());
    else
        out_a = in_a->subset(in_a->rows(), cdb);

    if(out_a == in_a) { // if a has not changed, b won't change either so we bypass this second step
        // suppose than B is bigger than A
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
 *  We compute the product alpha.f(a).f(b)+beta.c -> c (with c=this)
 *  f(a)=transpose(a) if transA='T', f(a)=a if transA='N' (idem for b)
 */
template<typename T> void HMatrix<T>::uncompatibleGemm(char transA, char transB, T alpha,
                                                  const HMatrix<T>* a, const HMatrix<T>* b, T beta) {
    if (a->rows()->size() == 0 || a->cols()->size() == 0) return;
    HMatrix<T> * va = NULL;
    HMatrix<T> * vb = NULL;
    HMatrix<T> * vc = NULL;;
    HMatrix<T> * vva = NULL;
    HMatrix<T> * vvb = NULL;
    HMatrix<T> * vvc = NULL;

    // Create va & vb = the subsets of a & b that match each other for doing the product f(a).f(b)
    // We modify the columns of f(a) and the rows of f(b)
    makeCompatible<T>(transA != 'N', transB == 'N', a, b, va, vb);

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
    vvc->leafGemm(transA, transB, alpha, vva, vvb, beta);

    // Delete the temporary matrices
    if(vva != a)
        delete vva;
    if(vvb != b)
        delete vvb;
    if(vvc != this)
        delete vvc;
}

template<typename T> void
HMatrix<T>::recursiveGemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>*b, T beta) {
    // Computing a(m,0) * b(0,n) here may give wrong results because of format conversions, exit early
    if (a->rows()->size() == 0 || a->cols()->size() == 0) return;

    // None of the matrices is a leaf
    if (!isLeaf() && !a->isLeaf() && !b->isLeaf()) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                HMatrix<T>* child = get(i, j);
                if (!child) { // symmetric or triangular case
                    continue;
                }
                // Void child
                if (child->rows()->size() == 0 || child->cols()->size() == 0) continue;
                for (int k = 0; k < 2; k++) {
                    char tA = transA, tB = transB;
                    // childA states :
                    // if A is symmetric and childA_ik is NULL
                    // then childA_ki^T is used and transA is changed accordingly.
                    // However A may be triangular( upper/lower ) so childA_ik is NULL
                    // and must be taken as 0.
                    const HMatrix<T>* childA = (tA == 'N' ? a->get(i, k) : a->get(k, i));
                    const HMatrix<T>* childB = (tB == 'N' ? b->get(k, j) : b->get(j, k));
                    if (!childA && (a->isTriUpper || a->isTriLower)) {
                        assert(*a->rows() == *a->cols());
                        continue;
                    }
                    if (!childB && (b->isTriUpper || b->isTriLower)) {
                        assert(*b->rows() == *b->cols());
                        continue;
                    }

                    if (!childA) {
                        tA = (tA == 'N' ? 'T' : 'N');
                        childA = (tA == 'N' ? a->get(i, k) : a->get(k, i));
                    }
                    if (!childB) {
                        tB = (tB == 'N' ? 'T' : 'N');
                        childB = (tB == 'N' ? b->get(k, j) : b->get(j, k));
                    }
                    child->gemm(tA, tB, alpha, childA, childB, beta);
                }
            }
        }
        return;
    }
    else
        uncompatibleGemm(transA, transB, alpha, a, b, beta);
}

template<typename T> void
HMatrix<T>::leafGemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>*b, T beta) {
    assert((transA == 'N' ? *a->cols() : *a->rows()) == ( transB == 'N' ? *b->rows() : *b->cols())); // pour le produit A*B
    assert((transA == 'N' ? *a->rows() : *a->cols()) == *this->rows()); // compatibility of A*B + this : Rows
    assert((transB == 'N' ? *b->cols() : *b->rows()) == *this->cols()); // compatibility of A*B + this : columns

    // One of the matrices is a leaf
    assert(isLeaf() || a->isLeaf() || b->isLeaf());

    // the resulting matrix is not a leaf.
    if (!isLeaf()) {
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
            axpy(alpha, fullMat, rows(), cols());
            delete fullMat;
        }
        return;
    }

    // This matrix is not yet initialized but we know it will be a RkMatrix if
    // a or b is a RkMatrix
    if((rank_ == -3 && (a->isRkMatrix() || b->isRkMatrix() ||
        // this choice might be bad if a or b contains lot's of full matrices
        // but this case should almost never happen
        (!a->isLeaf() && !b->isLeaf())))
        ||(isRkMatrix() && rk() == NULL))
    {
        rk(new RkMatrix<T>(NULL, rows(), NULL, cols(), NoCompression));
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
        rk()->gemmRk(transA, transB, alpha, a, b, beta);
        rank_ = rk()->rank();
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
        fullMat = HMatrix<T>::multiplyFullMatrix(transA, transB, a, b);
    } else {
        // TODO not yet implemented
        HMAT_ASSERT(false);
    }
    if(isFullMatrix()) {
        full()->axpy(alpha, fullMat);
        delete fullMat;
    } else {
        // It's not optimal to concider that the result is a FullMatrix but
        // this is a H*F case and it almost never happen
        fullMat->scale(alpha);
        full(fullMat);
    }
}

/**
 * Get a subset of the "a" or "b" part of this RkMatrix and return
 * it as a full HMatrix.
 * The columns (or row if col is true) subset of the return matrix has no meaning
 * @param subset the row subset to extract or the columns subset if col is true
 * @param col true to get the "b" part of the matrix
 */
template<typename T> HMatrix<T> * HMatrix<T>::fullRkSubset(const IndexSet* subset, bool col) const {
    assert(isRkMatrix() && !isNull());
    HMatrix<T> * r = this->subset(col ? this->rows() : subset, col ? subset : this->cols());
    FullMatrix<T> * a = col ? r->rk()->b : r->rk()->a;
    FullMatrix<T> * newFull = new FullMatrix<T>(a->m, a->rows, a->cols, a->lda);
    if(col) {
        // the "b" part of a rk matrice is stored transposed
        std::swap(r->rows_, r->cols_);
    }
    r->cols_ = r->cols_->slice(r->cols_->data.offset(), a->cols);
    delete r->rk();
    r->full(newFull);
    return r;
}

template<typename T>
void HMatrix<T>::gemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>* b, T beta) {
  // Computing a(m,0) * b(0,n) here may give wrong results because of format conversions, exit early
  if (rows()->size() == 0 || cols()->size() == 0) return;

  if ((transA == 'T') && (transB == 'T')) {
    // This code has *not* been tested because it's currently not used.
    HMAT_ASSERT(false);
    this->transpose();
    this->gemm('N', 'N', alpha, b, a, beta);
    this->transpose();
    return;
  }
  if(isRkMatrix() && !isNull() && b->isRkMatrix() && !b->isNull() && rk()->b == b->rk()->b) {
    HMatrix<T> * cSubset = this->fullRkSubset(a->rows(), false);
    HMatrix<T> * bSubset = b->fullRkSubset(a->cols(), false);
    cSubset->gemm(transA, transB, alpha, a, bSubset, beta);
    delete cSubset;
    delete bSubset;
    return;
  }

  if(isRkMatrix() && !isNull() && a->isRkMatrix() && !a->isNull() && rk()->a == a->rk()->a) {
    HMatrix<T> * cSubset = this->fullRkSubset(transB == 'N' ? b->cols() : b->rows(), true);
    HMatrix<T> * aSubset = a->fullRkSubset(transB == 'N' ? b->rows() : b->cols(), true);
    // transpose because cSubset and aSubset are transposed
    cSubset->gemm(transB == 'N' ? 'T' : 'N', transA, alpha, b, aSubset, beta);
    delete cSubset;
    delete aSubset;
    return;
  }

 // Scaling this
  if (beta != Constants<T>::pone) {
    if (beta == Constants<T>::zero) {
      this->clear();
    } else {
      this->scale(beta);
    }
  }
  if((a->isLeaf() && a->isNull()) || (b->isLeaf() && b->isNull())) {
      if(!isAssembled() && isLeaf())
          rk(new RkMatrix<T>(NULL, rows(), NULL, cols(), NoCompression));
      return;
  }

  // Once the scaling is done, beta is reset to 1
  // to avoid an other scaling.
  recursiveGemm(transA, transB, alpha, a, b, Constants<T>::pone);
}

template<typename T>
FullMatrix<T>* HMatrix<T>::multiplyFullH(char transM, char transH,
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
FullMatrix<T>* HMatrix<T>::multiplyHFull(char transH, char transM,
                                         const HMatrix<T>* h,
                                         const FullMatrix<T>* mat) {
  assert((transH == 'N' ? h->cols()->size() : h->rows()->size())
           == (transM == 'N' ? mat->rows : mat->cols));
  FullMatrix<T>* result =
    new FullMatrix<T>((transH == 'N' ? h->rows()->size() : h->cols()->size()),
                      (transM == 'N' ? mat->cols : mat->rows));
  if (transM == 'N') {
    h->gemv(transH, Constants<T>::pone, mat, Constants<T>::zero, result);
  } else {
    FullMatrix<T>* matT = mat->copyAndTranspose();
    h->gemv(transH, Constants<T>::pone, matT, Constants<T>::zero, result);
    delete matT;
  }
  return result;
}

template<typename T>
RkMatrix<T>* HMatrix<T>::multiplyRkMatrix(char transA, char transB, const HMatrix<T>* a, const HMatrix<T>* b){
  // We know that one of the matrices is a RkMatrix
  assert((transA == 'N') || (transB == 'N')); // Exclusion of the case At * Bt
  assert(a->isRkMatrix() || b->isRkMatrix());
  RkMatrix<T> *rk = NULL;
  // Matrices range compatibility
  if((transA == 'N') && (transB == 'N'))
    assert(a->cols()->size() == b->rows()->size());
  if((transA == 'T') && (transB == 'N'))
    assert(a->rows()->size() == b->rows()->size());
  if((transA == 'N') && (transB == 'T'))
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
  assert((transA == 'N') || (transB == 'N'));// Not for the products At*Bt
  assert(a->isFullMatrix() || b->isFullMatrix());
  assert(!(a->isRkMatrix() || b->isRkMatrix()));
  FullMatrix<T> *result = NULL;
  // The cases are:
  //  - A H, B F
  //  - A F, B H
  //  - A F, B F
  if (!a->isLeaf() && b->isFullMatrix()) {
    result = HMatrix<T>::multiplyHFull(transA, transB, a, b->full());
    HMAT_ASSERT(result);
  } else if (a->isFullMatrix() && !b->isLeaf()) {
    result = HMatrix<T>::multiplyFullH(transA, transB, a->full(), b);
    HMAT_ASSERT(result);
  } else if (a->isFullMatrix() && b->isFullMatrix()) {
    int aRows = ((transA == 'N')? a->full()->rows : a->full()->cols);
    int bCols = ((transB == 'N')? b->full()->cols : b->full()->rows);
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

  if (rows()->size() == 0 || cols()->size() == 0) return;

  // The symmetric matrix must be taken into account: lower or upper
  if (!isLeaf()) {
    get(0,0)->multiplyWithDiag(d->get(0,0), left, inverse);
    get(1,1)->multiplyWithDiag(d->get(1,1), left, inverse);
    if (get(0, 1)) {
      get(0, 1)->multiplyWithDiag(left ? d->get(0, 0) : d->get(1, 1), left, inverse);
    }
    if (get(1, 0)) {
      get(1, 0)->multiplyWithDiag(left ? d->get(1, 1) : d->get(0, 0), left, inverse);
    }
  } else if (isRkMatrix() && !isNull()) {
    assert(!rk()->a->isTriUpper() && !rk()->b->isTriUpper());
    assert(!rk()->a->isTriLower() && !rk()->b->isTriLower());
    rk()->multiplyWithDiagOrDiagInv(d, inverse, left);
  } else if(isFullMatrix()){
    if (d->isFullMatrix()) {
      full()->multiplyWithDiagOrDiagInv(d->full()->diagonal, inverse, left);
    } else {
      Vector<T> diag(d->rows()->size());
      d->extractDiagonal(diag.v, d->rows()->size());
      full()->multiplyWithDiagOrDiagInv(&diag, inverse, left);
    }
  } else {
    // this is a null matrix (either full of Rk) so nothing to do
  }
}

template<typename T> void HMatrix<T>::transposeNoRecurse() {
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
    swap(children[1 + 0 * 2], children[0 + 1 * 2]);
    swap(rows_, cols_);
}

template<typename T>
void HMatrix<T>::transpose() {
  if (!isLeaf()) {
    this->transposeNoRecurse();
    get(1, 1)->transpose();
    get(0, 0)->transpose();
    if (get(1, 0)) get(1, 0)->transpose();
    if (get(0, 1)) get(0, 1)->transpose();
  } else {
    swap(rows_, cols_);
    if (isRkMatrix() && rk()) {
      // To transpose an Rk-matrix, simple exchange A and B : (AB^T)^T = (BA^T)
      swap(rk()->a, rk()->b);
      swap(rk()->rows, rk()->cols);
    } else if (isFullMatrix()) {
      assert(full()->lda == full()->rows);
      full()->transpose();
    }
  }
}

template<typename T>
void HMatrix<T>::copyAndTranspose(const HMatrix<T>* o) {
  assert(o);
  assert(*this->rows() == *o->cols());
  assert(*this->cols() == *o->rows());
  assert(isLeaf() == o->isLeaf());

  if (isLeaf()) {
    if (o->isRkMatrix()) {
      assert(!isFullMatrix());
      if (rk()) {
        delete rk();
      }
      const RkMatrix<T>* oRk = o->rk();
      FullMatrix<T>* newA = oRk->b ? oRk->b->copy() : NULL;
      FullMatrix<T>* newB = oRk->a ? oRk->a->copy() : NULL;
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
            full()->diagonal = new Vector<T>(oF->rows);
            HMAT_ASSERT(full()->diagonal);
          }
          memcpy(full()->diagonal->v, oF->diagonal->v, oF->rows * sizeof(T));
        }
      }
    }
  } else {
    get(0, 0)->copyAndTranspose(o->get(0, 0));
    get(1, 1)->copyAndTranspose(o->get(1, 1));
    if (get(0, 1)) {
      get(0, 1)->copyAndTranspose(o->get(1, 0));
    }
    if (get(1, 0)) {
      get(1, 0)->copyAndTranspose(o->get(0, 1));
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

template<typename T>
void HMatrix<T>::createPostcriptFile(const std::string& filename) const {
    PostscriptDumper<T> dumper;
    dumper.write(this, filename);
}

template<typename T>
void HMatrix<T>::dumpTreeToFile(const std::string& filename, const HMatrixNodeDumper<T>& nodeDumper) const {
  ofstream file;
  const DofCoordinates* points = rows()->coordinates();
  const double* coord = &points->get(0, 0);
  const int* indices = rows()->indices();
  const int dimension = points->dimension();
  string delimiter;

  file.open(filename.c_str());
  // Points
  file << "{" << endl
       << "  \"points\": [" << endl;
  delimiter = "";
  for (int i = 0; i < points->size(); i++) {
    file << "    " << delimiter << "[";
    if (dimension > 0) {
      file << coord[dimension*i];
      for (int dim = 1; dim < dimension; ++dim) {
        file << ", " << coord[dimension*i+dim];
      }
    }
    file << "]" << endl;
    delimiter = " ,";
  }
  // Mapping
  file << "  ]," << endl
       << "  \"mapping\": [" << endl << "    ";
  delimiter = "";
  for (int i = 0; i < points->size(); i++) {
    file << delimiter << indices[i];
    delimiter = " ,";
  }
  file << "]," << endl;
  file << "  \"tree\":" << endl;
  dumpSubTree(file, 0, nodeDumper);
  file << "}" << endl;
}

template<typename T>
void HMatrix<T>::dumpSubTree(ofstream& f, int depth, const HMatrixNodeDumper<T>& nodeDumper) const {
  string prefix("    ");
  for (int i = 0; i < depth; i++) {
    prefix += "  ";
  }
  AxisAlignedBoundingBox rows_bbox(rows_->data);
  AxisAlignedBoundingBox cols_bbox(cols_->data);
  const int rows_dimension(rows_->data.coordinates()->dimension());
  const int cols_dimension(cols_->data.coordinates()->dimension());

  f << prefix << "{\"isLeaf\": " << (isLeaf() ? "true" : "false") << "," << endl
    << prefix << " \"depth\": " << depth << "," << endl
    << prefix << " \"rows\": " << "{\"offset\": " << rows()->offset() << ", \"n\": " << rows()->size() << ", "
    << "\"boundingBox\": [[" << rows_bbox.bbMin[0];
  for (int dim = 1; dim < rows_dimension; ++dim) {
    f << ", " << rows_bbox.bbMin[dim];
  }
  f << "], [" << rows_bbox.bbMax[0];
  for (int dim = 1; dim < rows_dimension; ++dim) {
    f << ", " << rows_bbox.bbMax[dim];
  }
  f << "]]}," << endl
    << prefix << " \"cols\": " << "{\"offset\": " << cols()->offset() << ", \"n\": " << cols()->size() << ", "
    << "\"boundingBox\": [[" << cols_bbox.bbMin[0];
  for (int dim = 1; dim < cols_dimension; ++dim) {
    f << ", " << cols_bbox.bbMin[dim];
  }
  f << "], [" << cols_bbox.bbMax[0];
  for (int dim = 1; dim < cols_dimension; ++dim) {
    f << ", " << cols_bbox.bbMax[dim];
  }
  f << "]]}," << endl;
  const std::string extra_info(nodeDumper.dumpExtraInfo(*this, " "+prefix));
  if (!extra_info.empty()) {
    f << prefix << " " << extra_info << "," << endl;
  }
  if (!isLeaf()) {
    f << prefix << " \"children\": [" << endl;
    string delimiter("");
    for (int i = 0; i < 4; i++) {
      const HMatrix<T>* child = static_cast<const HMatrix<T>*>(getChild(i));
      if (!child) continue;
      f << delimiter;
      child->dumpSubTree(f, depth + 1, nodeDumper);
      f << endl;
      delimiter = ",";
    }
    f << prefix << " ]";
  } else {
    // It's a leaf
    if (isFullMatrix()) {
      f << prefix << " \"leaf_type\": \"Full\"";
    } else if (isRkMatrix()) {
      f << prefix << " \"leaf_type\": \"Rk\", \"k\": " << rank() << ",";
      // f << endl << prefix << " \"eta\": " << this->data.rows->getEta(this->data.cols) << ",";
      f << prefix << " \"method\": " << rk()->method;
    } else {
      f << prefix << " \"leaf_type\": \"N/A\"";
    }
  }
  f << "}";
}

template<typename T>
void HMatrix<T>::copy(const HMatrix<T>* o) {
  DECLARE_CONTEXT;

  assert(*rows() == *o->rows());
  assert(*cols() == *o->cols());

  isLower = o->isLower;
  isUpper = o->isUpper;
  isTriUpper = o->isTriUpper;
  isTriLower = o->isTriLower;
  if (isLeaf()) {
    assert(o->isLeaf());
    if (isAssembled() && isNull() && o->isNull()) {
      return;
    }
    // When the matrix has not allocated but only the structure
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
    rank_ = o->rank_;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        if (o->get(i, j)) {
          assert(get(i, j));
          get(i, j)->copy(o->get(i, j));
        } else {
          assert(!get(i, j));
        }
      }
    }
  }
}

template<typename T>
void HMatrix<T>::clear() {
  if (rows()->size() == 0 || cols()->size() == 0) return;
  if (isLeaf()) {
    if (isFullMatrix()) {
      full()->clear();
    } else if(isRkMatrix()){
      delete rk();
      rk(new RkMatrix<T>(NULL, rows(), NULL, cols(), NoCompression));
    }
  } else {
    for (int i = 0; i < 4; i++) {
      HMatrix<T>* child = static_cast<HMatrix<T>*>(getChild(i));
      if (child)
        child->clear();
    }
  }
}

template<typename T>
void HMatrix<T>::inverse(HMatrix<T>* tmp, int depth) {
  DECLARE_CONTEXT;

  bool hasToFreeTmp = false;
  if (!tmp) {
    hasToFreeTmp = true;
    tmp = HMatrix<T>::Zero(this);
  }

  assert(*(rows()) == *(tmp->rows()));
  assert(*(cols()) == *(tmp->cols()));

  if (this->isLeaf()) {
    assert(isFullMatrix());
    full()->inverse();
  } else {
    assert(!tmp->isLeaf());
    HMatrix<T>* m11 = static_cast<HMatrix<T>*>(get(0, 0));
    HMatrix<T>* m21 = static_cast<HMatrix<T>*>(get(1, 0));
    HMatrix<T>* m12 = static_cast<HMatrix<T>*>(get(0, 1));
    HMatrix<T>* m22 = static_cast<HMatrix<T>*>(get(1, 1));
    HMatrix<T>* x11 = static_cast<HMatrix<T>*>(tmp->get(0, 0));
    HMatrix<T>* x21 = static_cast<HMatrix<T>*>(tmp->get(1, 0));
    HMatrix<T>* x12 = static_cast<HMatrix<T>*>(tmp->get(0, 1));
    HMatrix<T>* x22 = static_cast<HMatrix<T>*>(tmp->get(1, 1));

    // cout << prefix << "X11 <- X11^-1" << endl;
    // Destroy x11
    // X11 <- X11^-1
    m11->inverse(x11, depth + 1);

    // cout << prefix << "X12 <- -M11^-1 M12" << endl;
    // X12 <- - M11^-1 M12
    x12->gemm('N', 'N', Constants<T>::mone, m11, m12, Constants<T>::zero);

    // cout << prefix << "X21 <- M21 M11^-1" << endl;
    // X21 <- M21 * M11^-1
    x21->gemm('N', 'N', Constants<T>::pone, m21, m11, Constants<T>::zero);

    // cout << prefix << "M22 <- M22 - M21 M11^-1 M12" << endl;
    // M22 <- M22 - M21 M11^-1 M12 = S
    m22->gemm('N', 'N', Constants<T>::pone, m21, x12, Constants<T>::pone);

    // cout << prefix << "M22 < S^-1" << endl;
    // M22 <- S^-1
    // The contents of X22 is deleted
    m22->inverse(x22, depth + 1);

    // cout << prefix << "M12 <- -M11^-1 M12 S^-1" << endl;
    // M12 <- -M11^-1 M12 S^-1
    m12->gemm('N', 'N', Constants<T>::pone, x12, m22, Constants<T>::zero);

    // cout << prefix << "M11 <- M11 + M11^-1 M12 S^-1 M21 M11^-1" << endl;
    // M11 <- M11 + M11^-1 M12 S^-1 M21 M11^-1
    m11->gemm('N', 'N', Constants<T>::mone, m12, x21, Constants<T>::pone);

    // cout << prefix << "M21 <- S^-1 M21 M11^-1" << endl;
    // M21 <- -S^-1 M21 M11^-1
    m21->gemm('N', 'N', Constants<T>::mone, m22, x21, Constants<T>::zero);
  }

  if (hasToFreeTmp) {
    delete tmp;
  }
}

template<typename T>
void HMatrix<T>::solveLowerTriangularLeft(HMatrix<T>* b, bool unitriangular) const {
  DECLARE_CONTEXT;
  if (rows()->size() == 0 || cols()->size() == 0) return;
  // At first, the recursion one (simple case)
  if (!isLeaf() && !b->isLeaf()) {
    //  Forward substitution:
    //  [ L11 |  0  ]    [ X11 | X12 ]   [ b11 | b12 ]
    //  [ ----+---- ] *  [-----+-----] = [ ----+---- ]
    //  [ L21 | L22 ]    [ X21 | X22 ]   [ b21 | b22 ]
    //
    //  L11 * X11 = b11 (by recursive forward substitution)
    //  L11 * X12 = b12 (by recursive forward substitution)
    //  L21 * X11 + L22 * X21 = b21 (forward substitution of L22*X21=b21-L21*X11)
    //  L21 * X12 + L22 * X22 = b22 (forward substitution of L22*X22=b22-L21*X12)
    //
    const HMatrix<T>* l11 = get(0, 0);
    const HMatrix<T>* l21 = get(1, 0);
    const HMatrix<T>* l22 = get(1, 1);
    HMatrix<T>* b11 = b->get(0, 0);
    HMatrix<T>* b21 = b->get(1, 0);
    HMatrix<T>* b12 = b->get(0, 1);
    HMatrix<T>* b22 = b->get(1, 1);

    l11->solveLowerTriangularLeft(b11, unitriangular);
    l11->solveLowerTriangularLeft(b12, unitriangular);
    b21->gemm('N', 'N', Constants<T>::mone, l21, b11, Constants<T>::pone);
    l22->solveLowerTriangularLeft(b21, unitriangular);
    b22->gemm('N', 'N', Constants<T>::mone, l21, b12, Constants<T>::pone);
    l22->solveLowerTriangularLeft(b22, unitriangular);
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
        HMatrix<T> * tmp;
        if(*cols() == *b->rows())
            tmp = b;
        else
            tmp = b->subset(this->cols(), b->cols());
        this->solveLowerTriangularLeft(tmp->rk()->a, unitriangular);
        if(tmp != b)
            delete tmp;
      }
    } else {
      // B isn't a leaf, then 'this' is one
      assert(this->isLeaf());
      // Evaluate B, solve by column, and restore in the matrix
      // TODO: check if it's not too bad
      FullMatrix<T>* bFull = new FullMatrix<T>(b->rows()->size(), b->cols()->size());

      b->evalPart(bFull, b->rows(), b->cols());
      this->solveLowerTriangularLeft(bFull, unitriangular);
      b->clear();
      b->axpy(Constants<T>::pone, bFull, b->rows(), b->cols());
      delete bFull;
    }
  }
}

template<typename T>
void HMatrix<T>::solveLowerTriangularLeft(FullMatrix<T>* b, bool unitriangular) const {
  DECLARE_CONTEXT;
  assert(*rows() == *cols());
  assert(cols()->size() == b->rows); // Here : the change : OK or not ??????? : cols <-> rows
  if (rows()->size() == 0 || cols()->size() == 0) return;
  if (this->isLeaf()) {
    assert(this->isFullMatrix());
    // LAPACK resolution
    full()->solveLowerTriangularLeft(b, unitriangular);
  } else {
    const HMatrix<T>* l11 = get(0, 0);
    const HMatrix<T>* l21 = get(1, 0);
    const HMatrix<T>* l22 = get(1, 1);
    FullMatrix<T> b1(b->m, l11->cols()->size(), b->cols, b->lda);
    FullMatrix<T> b2(b->m + l11->cols()->size(), l22->cols()->size(), b->cols, b->lda);
    l11->solveLowerTriangularLeft(&b1, unitriangular);
    l21->gemv('N', Constants<T>::mone, &b1, Constants<T>::pone, &b2);
    l22->solveLowerTriangularLeft(&b2, unitriangular);
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangularRight(HMatrix<T>* b, bool unitriangular, bool lowerStored) const {
  DECLARE_CONTEXT;
  if (rows()->size() == 0 || cols()->size() == 0) return;
  // The recursion one (simple case)
  if (!isLeaf() && !b->isLeaf()) {
    const HMatrix<T>* u11 = get(0, 0);
    const HMatrix<T>* u12 = lowerStored ? get(1, 0) : get(0, 1);
    const HMatrix<T>* u22 = get(1, 1);
    HMatrix<T>* b11 = b->get(0, 0);
    HMatrix<T>* b21 = b->get(1, 0);
    HMatrix<T>* b12 = b->get(0, 1);
    HMatrix<T>* b22 = b->get(1, 1);

    u11->solveUpperTriangularRight(b11, unitriangular, lowerStored);
    u11->solveUpperTriangularRight(b21, unitriangular, lowerStored);
    // B12 <- -B11*U12 + B12
    b12->gemm('N', lowerStored ? 'T' : 'N', Constants<T>::mone, b11, u12, Constants<T>::pone);
    // B12 <- the solution of X * U22 = B12
    u22->solveUpperTriangularRight(b12, unitriangular, lowerStored);
    // B22 <- - B21*U12 + B22
    b22->gemm('N', lowerStored ? 'T' : 'N', Constants<T>::mone, b21, u12, Constants<T>::pone);
    // B22 <- the solution of X*U22 = B22
    u22->solveUpperTriangularRight(b22, unitriangular, lowerStored);

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
      assert(isLeaf());
      assert(isFullMatrix());
      // Evaluate B, solve by column and restore all in the matrix
      // TODO: check if it's not too bad
      FullMatrix<T>* bFull = new FullMatrix<T>(b->rows()->size(), b->cols()->size());
      b->evalPart(bFull, b->rows(), b->cols());
      bFull->transpose();
      this->solveUpperTriangularRight(bFull, unitriangular, lowerStored);
      bFull->transpose();
      // int bRows = b->rows()->size();
      // int bCols = b->cols()->size();
      // Vector<T> bRow(bCols);
      // for (int row = 0; row < bRows; row++) {
      //   blasCopy<T>(bCols, bFull->m + row, bRows, bRow.v, 1);
      //   FullMatrix<T> tmp(bRow.v, bRow.rows, 1);
      //   this->solveUpperTriangularRight(&tmp);
      //   blasCopy<T>(bCols, bRow.v, 1, bFull->m + row, bRows);      // }
      // }
      b->clear();
      b->axpy(Constants<T>::pone, bFull, b->rows(), b->cols());
      delete bFull;
    }
  }
}

/* Resolve U.X=B, solution saved in B, with B Hmat
   Only called by luDecomposition
 */
template<typename T>
void HMatrix<T>::solveUpperTriangularLeft(HMatrix<T>* b, bool unitriangular, bool lowerStored) const {
  DECLARE_CONTEXT;
  if (rows()->size() == 0 || cols()->size() == 0) return;
  // At first, the recursion one (simple case)
  if (!isLeaf() && !b->isLeaf()) {
    //  Backward substitution:
    //  [ U11 | U12 ]    [ X11 | X12 ]   [ b11 | b12 ]
    //  [ ----+---- ] *  [-----+-----] = [ ----+---- ]
    //  [  0  | U22 ]    [ X21 | X22 ]   [ b21 | b22 ]
    //
    //  U22 * X21 = b21 (by recursive backward substitution)
    //  U22 * X22 = b22 (by recursive backward substitution)
    //  U11 * X12 + U12 * X22 = b12 (backward substitution of U11*X12=b12-U12*X22)
    //  U11 * X11 + U12 * X21 = b11 (backward substitution of U11*X11=b11-U12*X21)
    const HMatrix<T>* u11 = get(0, 0);
    const HMatrix<T>* u12 = lowerStored ? get(1, 0) : get(0, 1);
    const HMatrix<T>* u22 = get(1, 1);
    HMatrix<T>* b11 = b->get(0, 0);
    HMatrix<T>* b21 = b->get(1, 0);
    HMatrix<T>* b12 = b->get(0, 1);
    HMatrix<T>* b22 = b->get(1, 1);

    u22->solveUpperTriangularLeft(b21, unitriangular, lowerStored);
    u22->solveUpperTriangularLeft(b22, unitriangular, lowerStored);
    b12->gemm(lowerStored ? 'T' : 'N', 'N', Constants<T>::mone, u12, b22, Constants<T>::pone);
    u11->solveUpperTriangularLeft(b12, unitriangular, lowerStored);
    b11->gemm(lowerStored ? 'T' : 'N', 'N', Constants<T>::mone, u12, b21, Constants<T>::pone);
    u11->solveUpperTriangularLeft(b11, unitriangular, lowerStored);
  } else {
    // if B is a leaf, the resolve is done by column
    if (b->isLeaf()) {
      if (b->isFullMatrix()) {
        this->solveUpperTriangularLeft(b->full(), unitriangular, lowerStored);
      } else if(!b->isNull()){
        assert(b->isRkMatrix());
        this->solveUpperTriangularLeft(b->rk()->a, unitriangular, lowerStored);
      }
    } else {
      // B isn't a leaf, then so is L
      assert(this->isLeaf());
      // Evaluate B, solve by column, and restore in the matrix
      // TODO: check if it's not too bad
      FullMatrix<T>* bFull = new FullMatrix<T>(b->rows()->size(), b->cols()->size());
      b->evalPart(bFull, b->rows(), b->cols());
      this->solveUpperTriangularLeft(bFull, unitriangular, lowerStored);
      b->clear();
      b->axpy(Constants<T>::pone, bFull, b->rows(), b->cols());
      delete bFull;
    }
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangularRight(FullMatrix<T>* b, bool unitriangular, bool lowerStored) const {
  DECLARE_CONTEXT;
  assert(*rows() == *cols());
  if (rows()->size() == 0 || cols()->size() == 0) return;
  // B is supposed given in form of a row vector, but transposed
  // so we can deal it with a subset as usual.
  if (this->isLeaf()) {
    assert(this->isFullMatrix());
    FullMatrix<T>* bCopy = b->copyAndTranspose();
    full()->solveUpperTriangularRight(bCopy, unitriangular, lowerStored);
    bCopy->transpose();
    b->copyMatrixAtOffset(bCopy, 0, 0);
    delete bCopy;
  } else {
    const HMatrix<T>* u11 = get(0, 0);
    const HMatrix<T>* u12 = lowerStored ? get(1, 0) : get(0, 1);
    const HMatrix<T>* u22 = get(1, 1);

    FullMatrix<T> b1(b->m, u11->rows()->size(), b->cols, b->lda);
    FullMatrix<T> b2(b->m + u11->rows()->size(), u22->rows()->size(), b->cols, b->lda);
    u11->solveUpperTriangularRight(&b1, unitriangular, lowerStored);
    // b2 <- -x1 U12 + b2 = -U12^t x1^t + b2
    u12->gemv(lowerStored ? 'N' : 'T', Constants<T>::mone, &b1, Constants<T>::pone, &b2);
    u22->solveUpperTriangularRight(&b2, unitriangular, lowerStored);
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangularLeft(FullMatrix<T>* b, bool unitriangular, bool lowerStored) const {
  DECLARE_CONTEXT;
  assert(*rows() == *cols());
  if (rows()->size() == 0 || cols()->size() == 0) return;
  if (this->isLeaf()) {
    full()->solveUpperTriangularLeft(b, unitriangular, lowerStored);
  } else {
    const HMatrix<T>* u11 = get(0, 0);
    const HMatrix<T>* u12 = (lowerStored ? get(1, 0) : get(0, 1));
    const HMatrix<T>* u22 = get(1, 1);

    FullMatrix<T> b1(b->m, u11->cols()->size(), b->cols, b->lda);
    FullMatrix<T> b2(b->m + u11->cols()->size(), u22->cols()->size(), b->cols, b->lda);

    u22->solveUpperTriangularLeft(&b2, unitriangular, lowerStored);
    // b1 <- -U12 b2 + b1
    u12->gemv(lowerStored ? 'T' : 'N', Constants<T>::mone, &b2, Constants<T>::pone, &b1);
    u11->solveUpperTriangularLeft(&b1, unitriangular, lowerStored);
  }
}
template<typename T> void HMatrix<T>::lltDecomposition() {

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
    assertLower(this);
    if (rows()->size() == 0 || cols()->size() == 0) return;
    if(isLeaf()) {
        full()->lltDecomposition();
    } else {
        HMAT_ASSERT(isLower);
        HMAT_ASSERT(!get(0,1));
        HMatrix<T>* h11 = get(0,0);
        HMatrix<T>* h21 = get(1,0);
        HMatrix<T>* h22 = get(1,1);
        h11->lltDecomposition();
        h11->solveUpperTriangularRight(h21, false, true);
        h22->gemm('N', 'T', Constants<T>::mone, h21, h21, Constants<T>::pone);
        h22->lltDecomposition();
    }
    isTriLower = true;
}

template<typename T>
void HMatrix<T>::luDecomposition() {
// |     |     |    |     |     |   |     |     |
// | h11 | h12 |    | L11 |     |   | U11 | U12 |
// |-----|-----| =  |-----|-----| * |-----|-----|
// | h21 | h22 |    | L21 | L22 |   |     | U22 |
// |     |     |    |     |     |   |     |     |
//
// h11 = L11 * U11 => (L11,U11) = h11.luDecomposition
// h12 = L11 * U12 => trsm L
// h21 = L21 * U11 => trsm R
// h22 = L21 * U12 + L22 * U22 => (L22,U22) = (h22 - L21*U12).lltDecomposition()
//
  DECLARE_CONTEXT;

  if (rows()->size() == 0 || cols()->size() == 0) return;
  if (isLeaf()) {
    assert(isFullMatrix());
    full()->luDecomposition();
    full()->checkNan();
  } else {
    HMatrix<T>* h11 = get(0, 0);
    HMatrix<T>* h21 = get(1, 0);
    HMatrix<T>* h12 = get(0, 1);
    HMatrix<T>* h22 = get(1, 1);

    // H11 <- L11 * U11
    h11->luDecomposition();
    const HMatrix<T>* l11 = h11;
    const HMatrix<T>* u11 = h11;
    // Solve L11 U12 = H12 (get U12)
    l11->solveLowerTriangularLeft(h12, true);
    // Solve L21 U11 = H21 (get L21)
    u11->solveUpperTriangularRight(h21, false, false);
    // H22 <- H22 - L21 U12
    h22->gemm('N', 'N', Constants<T>::mone, h21, h12, Constants<T>::pone);
    h22->luDecomposition();
  }
}

template<typename T>
void HMatrix<T>::mdmtProduct(const HMatrix<T>* m, const HMatrix<T>* d) {
  DECLARE_CONTEXT;
  if (rows()->size() == 0 || cols()->size() == 0) return;
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

  if (m->rows()->size() == 0 || m->cols()->size() == 0) return;
  if(!isLeaf()) {
    HMatrix<T>* h11 = get(0,0);
    HMatrix<T>* h21 = get(1,0);
    HMatrix<T>* h22 = get(1,1);
    if (!m->isLeaf()) {
      HMatrix<T>* m11 = m->get(0,0);
      HMatrix<T>* m21 = m->get(1,0);
      HMatrix<T>* m12 = m->get(0,1);
      HMatrix<T>* m22 = m->get(1,1);

      HMatrix<T>* d11 = d->get(0,0);
      HMatrix<T>* d22 = d->get(1,1);

      h11->mdmtProduct(m11, d11);
      h11->mdmtProduct(m12, d22);

      HMatrix<T>* x = Zero(m21);
      x->copy(m21);
      x->multiplyWithDiag(d11);
      h21->gemm('N', 'T', Constants<T>::mone, x, m11, Constants<T>::pone);
      delete x;

      HMatrix<T>* y = Zero(m22);
      y->copy(m22);
      assert(*y->cols() == *d22->rows());
      y->multiplyWithDiag(d22);
      h21->gemm('N', 'T', Constants<T>::mone, y, m12, Constants<T>::pone);
      delete y;

      h22->mdmtProduct(m21, d11);
      h22->mdmtProduct(m22, d22);
    } else if (m->isRkMatrix() && !m->isNull()) {
      HMatrix<T>* m_copy = Zero(m);
      m_copy->copy(m);

      assert(*m->cols() == *d->rows());
      assert(*m_copy->rk()->cols == *d->rows());
      m_copy->multiplyWithDiag(d); // right multiplication by D
      RkMatrix<T>* rkMat = RkMatrix<T>::multiplyRkRk('N', 'T', m_copy->rk(), m->rk());
      delete m_copy;

      this->axpy(Constants<T>::mone, rkMat);
      delete rkMat;
    } else if(m->isFullMatrix()){
      HMatrix<T>* copy_m = Zero(m);
      HMAT_ASSERT(copy_m);
      copy_m->copy(m);
      copy_m->multiplyWithDiag(d); // right multiplication by D

      FullMatrix<T>* fullMat = HMatrix<T>::multiplyFullMatrix('N', 'T', copy_m, m);
      HMAT_ASSERT(fullMat);
      delete copy_m;

      this->axpy(Constants<T>::mone, fullMat, rows(), cols());
      delete fullMat;
    } else {
      // m is a null matrix (either Rk or Full) so nothing to do.
    }
  } else {
    assert(isFullMatrix());

    if (m->isRkMatrix() && !m->isNull()) {
      // this : full
      // m    : rk
      // Strategy: compute mdm^T as FullMatrix and then do this<-this - mdm^T

      // 1) copy  m = AB^T : m_copy
      // 2) m_copy <- m_copy * D    (multiplyWithDiag)
      // 3) rkMat <- multiplyRkRk ( m_copy , m^T)
      // 4) fullMat <- evaluation as a FullMatrix of the product rkMat = (A*(D*B)^T) * (A*B^T)^T
      // 5) this <- this - fullMat
      HMatrix<T>* m_copy = Zero(m);
      m_copy->copy(m);
      m_copy->multiplyWithDiag(d);

      RkMatrix<T>* rkMat = RkMatrix<T>::multiplyRkRk('N', 'T', m_copy->rk(), m->rk());
      FullMatrix<T>* fullMat = rkMat->eval();
      delete m_copy;
      delete rkMat;
      full()->axpy(Constants<T>::mone, fullMat);
      delete fullMat;
    } else if (m->isFullMatrix()) {
      // S <- S - M*D*M^T
      assert(!full()->isTriUpper());
      assert(!full()->isTriLower());
      assert(!m->full()->isTriUpper());
      assert(!m->full()->isTriLower());
      FullMatrix<T> mTmp(m->full()->rows, m->full()->cols);
      mTmp.copyMatrixAtOffset(m->full(), 0, 0);
      if (d->isFullMatrix()) {
        mTmp.multiplyWithDiagOrDiagInv(d->full()->diagonal, false, false);
      } else {
        Vector<T> diag(d->cols()->size());
        d->extractDiagonal(diag.v, d->cols()->size());
        mTmp.multiplyWithDiagOrDiagInv(&diag, false, false);
      }
      full()->gemm('N', 'T', Constants<T>::mone, &mTmp, m->full(), Constants<T>::pone);
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
        assertLdlt(me->get(0,0));
        assertLdlt(me->get(1,1));
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
        assert(!me->get(0,1));
        assertLower(me->get(0,0));
        assertLower(me->get(1,1));
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
        assert(!me->get(1,0));
        assertUpper(me->get(0,0));
        assertUpper(me->get(1,1));
    }
#else
    ignore_unused_arg(me);
#endif
}

template<typename T>
void HMatrix<T>::ldltDecomposition() {
  DECLARE_CONTEXT;
  assertLower(this);

  if (rows()->size() == 0 || cols()->size() == 0) return;
  if (isLeaf()) {
    //The basic case of the recursion is necessarily a full matrix leaf
    //since the recursion is done with *rows() == *cols().

    assert(isFullMatrix());
    full()->ldltDecomposition();
    assert(full()->diagonal);
  } else {
    HMatrix<T>* h11 = get(0,0);
    HMatrix<T>* h21 = get(1,0);
    HMatrix<T>* h22 = get(1,1);

    // H11 <- L11 and D11 is stored additionally to each diagonal leaf
    h11->ldltDecomposition();
    assertLdlt(h11);

    HMatrix<T>* y = Zero(h11); // H11 is lower triangular, therefore either upper or lower part is NULL
    y->copy(h11); // X <- copy(L11)
    // Y <- Y*D11 , D11 is stocked in h11 (in diagonal leaves)

    y->multiplyWithDiag(h11); // MultiplyWithDiag takes into account the fact that "y" is Upper or Lower
    // Y <- Y^T
    assertLower(y);

    // The transpose function keeps the Upper or Lower matrix but
    //     reverse storage.
    y->transpose();
    // H21 <- solution of X*Y = H21, with Y = (L11 * D11)^T give L21
    // stored in H21
    assert(y->isUpper || y->isLeaf());
    assert(!y->isLower);
    y->solveUpperTriangularRight(h21, false, false);
    assertUpper(y);
    delete y;

    // H22 <- H22 - L21 * D11 * L21^T
    // D11 is contained on the diagonal leaves (which are full)
    h22->mdmtProduct(h21, h11);
    assertLower(h22);
    h22->ldltDecomposition();
  }
  isTriLower = true;
}

template<typename T>
void HMatrix<T>::solve(FullMatrix<T>* b) const {
  DECLARE_CONTEXT;
  // Solve (LU) X = b
  // First compute L Y = b
  this->solveLowerTriangularLeft(b, true);
  // Then compute U X = Y
  this->solveUpperTriangularLeft(b, false, false);
}

template<typename T>
void HMatrix<T>::extractDiagonal(T* diag, int size) const {
  DECLARE_CONTEXT;
  if (rows()->size() == 0 || cols()->size() == 0) return;
  if(isLeaf()) {
    assert(isFullMatrix());
    if(full()->diagonal) {
      // LDLt
      memcpy(diag, full()->diagonal->v, full()->rows * sizeof(T));
    } else {
      // LLt
      for (int i = 0; i < full()->rows; ++i)
        diag[i] = full()->m[i*full()->rows + i];
    }
  } else {
    assert(size == get(0,0)->rows()->size() + get(1,1)->rows()->size());
    get(0,0)->extractDiagonal(diag, get(0,0)->rows()->size());
    get(1,1)->extractDiagonal(diag + get(0,0)->rows()->size(), get(1,1)->rows()->size());
  }
}

/* Solve M.X=B with M hmat LU factorized*/
template<typename T> void HMatrix<T>::solve(
        HMatrix<T>* b,
        hmat_factorization_t factorizationType) const {
    DECLARE_CONTEXT;
    /* Solve LX=B, result in B */
    this->solveLowerTriangularLeft(b, true);
    /* Solve UX=B, result in B */
    switch(factorizationType) {
    case hmat_factorization_lu:
        this->solveUpperTriangularLeft(b, false, false);
        break;
    case hmat_factorization_ldlt:
        this->multiplyWithDiag(b, true, true);
        this->solveUpperTriangularLeft(b, true, true);
        break;
    case hmat_factorization_llt:
        this->solveUpperTriangularLeft(b, false, true);
        break;
    default:
        HMAT_ASSERT(false);
    }
}

template<typename T> void HMatrix<T>::solveDiagonal(FullMatrix<T>* b) const {
    // Solve D*X = B and store result into B
    // Diagonal extraction
    T* diag;
    bool extracted = false;
    if (rows()->size() == 0 || cols()->size() == 0) return;
    if(isFullMatrix() && full()->diagonal) {
        // LDLt
        diag = full()->diagonal->v;
    } else {
        // LLt
        diag = new T[cols()->size()];
        extractDiagonal(diag, cols()->size());
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

template<typename T>
void HMatrix<T>::solveLdlt(FullMatrix<T>* b) const {
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
void HMatrix<T>::solveLlt(FullMatrix<T>* b) const {
  DECLARE_CONTEXT;
  // L*L^T * X = B
  // B <- solution of L * Y = B : Y = L^T * X
  this->solveLowerTriangularLeft(b, false);

  // B <- solution of L^T X = B :  the solution X we are looking for is stored in B
  this->solveUpperTriangularLeft(b, false, true);
}

template<typename T>
void HMatrix<T>::checkNan() const {
  return;
  if (isLeaf()) {
    if (isFullMatrix()) {
      full()->checkNan();
    }
    if (isRkMatrix()) {
      rk()->checkNan();
    }
  } else {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        if (get(i, j)) {
          get(i, j)->checkNan();
        }
      }
    }
  }
}

template<typename T> void HMatrix<T>::setTriLower(bool value)
{
    isTriLower = value;
    if(!isLeaf())
    {
        get(0, 0)->setTriLower(value);
        get(1, 1)->setTriLower(value);
    }
}

template<typename T>  void HMatrix<T>::rk(const FullMatrix<T> * a, const FullMatrix<T> * b, bool updateRank) {
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
    std::vector<Tree<4>*> leaves;
    listAllLeaves(leaves);
    int nbAssembled = 0;
    int nbNullFull = 0;
    int nbNullRk = 0;
    double diagNorm = 0;
    for(unsigned int i = 0; i < leaves.size(); i++) {
        HMatrix<T> * l = static_cast<HMatrix<T>*>(leaves[i]);
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

// Templates declaration
template class HMatrix<S_t>;
template class HMatrix<D_t>;
template class HMatrix<C_t>;
template class HMatrix<Z_t>;

template class HMatrixNodeDumper<S_t>;
template class HMatrixNodeDumper<D_t>;
template class HMatrixNodeDumper<C_t>;
template class HMatrixNodeDumper<Z_t>;

template void reorderVector(FullMatrix<S_t>* v, int* indices);
template void reorderVector(FullMatrix<D_t>* v, int* indices);
template void reorderVector(FullMatrix<C_t>* v, int* indices);
template void reorderVector(FullMatrix<Z_t>* v, int* indices);

template void restoreVectorOrder(FullMatrix<S_t>* v, int* indices);
template void restoreVectorOrder(FullMatrix<D_t>* v, int* indices);
template void restoreVectorOrder(FullMatrix<C_t>* v, int* indices);
template void restoreVectorOrder(FullMatrix<Z_t>* v, int* indices);

}  // end namespace hmat
