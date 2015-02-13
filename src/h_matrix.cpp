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
#include "fromdouble.hpp"

using namespace std;

// The default values below will be overwritten in default_engine.cpp by HMatSettings values
template<typename T> bool HMatrix<T>::coarsening = false;
template<typename T> bool HMatrix<T>::recompress = false;
template<typename T> bool HMatrix<T>::validateCompression = false;
template<typename T> bool HMatrix<T>::validationReRun = false;
template<typename T> bool HMatrix<T>::validationDump = false;
template<typename T> double HMatrix<T>::validationErrorThreshold = 0;

template<typename T> HMatrixData<T>::~HMatrixData() {
  if (rk) {
    delete rk;
  }
  if (m) {
    delete m;
  }
}

template<typename T> bool HMatrixData<T>::isAdmissibleLeaf(const hmat::MatrixSettings * settings) const {
   size_t max_size;
   if(RkMatrix<T>::approx.method == AcaPartial || RkMatrix<T>::approx.method == AcaPlus)
       max_size = 2E9;
   else
       max_size = settings->getMaxElementsPerBlock();
   return rows->isAdmissibleWith(cols, settings->getAdmissibilityFactor(), max_size);
}

template<typename T>
void reorderVector(FullMatrix<T>* v, int* indices) {
  DECLARE_CONTEXT;
  const size_t n = v->rows;
  Vector<T> tmp(n);
  for (int col = 0; col < v->cols; col++) {
    T* column = v->m + (n * col);
    for (size_t i = 0; i < n; i++) {
      tmp.v[i] = column[indices[i]];
    }
    memcpy(column, tmp.v, sizeof(T) * n);
  }
}


template<typename T>
void restoreVectorOrder(FullMatrix<T>* v, int* indices) {
  DECLARE_CONTEXT;
  const size_t n = v->rows;
  Vector<T> tmp(n);

  for (int col = 0; col < v->cols; col++) {
    T* column = v->m + (n * col);
    for (size_t i = 0; i < n; i++) {
      tmp.v[indices[i]] = column[i];
    }
    memcpy(column, tmp.v, sizeof(T) * n);
  }
}


template<typename T>
HMatrix<T>::HMatrix(ClusterTree* _rows, ClusterTree* _cols, const hmat::MatrixSettings * settings, SymmetryFlag symFlag)
  : Tree<4>(NULL),
    data(HMatrixData<T>(_rows, _cols)),
    isUpper(false), isLower(false),
    isTriUpper(false), isTriLower(false), localSettings(settings) {
  bool adm = _rows->isAdmissibleWith(_cols, settings->getAdmissibilityFactor(), settings->getMaxElementsPerBlock());
  if (_rows->isLeaf() || _cols->isLeaf() || adm) {
    if (adm) {
      data.rk = new RkMatrix<T>(NULL, rows(), NULL, cols(), NoCompression);
    }
    return;
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
          this->insertChild(i + j * 2, new HMatrix<T>(rowChild, colChild, settings, (i == j ? symFlag : kNotSymmetric)));
         }
       }
     }
  }
}

template<typename T>
HMatrix<T>::HMatrix(const hmat::MatrixSettings * settings) :
    Tree<4>(NULL), data(HMatrixData<T>()), isUpper(false),
    isLower(false), localSettings(settings) {}

template<typename T>
HMatrix<T>* HMatrix<T>::copyStructure() const {
  HMatrix<T>* h = new HMatrix<T>(localSettings.global);
  h->data.rows = data.rows;
  h->data.cols = data.cols;

  h->isUpper = isUpper;
  h->isLower = isLower;
  h->isTriUpper = isTriUpper;
  h->isTriLower = isTriLower;

  if (isLeaf()) {
    h->data.rk = NULL;
    h->data.m = NULL;
    if (isRkMatrix()) {
      // We have to create a RkMatrix <T> because
      // h->isRkMatrix () returns false otherwise,
      // which may cause trouble for some operations.
      h->data.rk = new RkMatrix<T>(NULL, data.rk->rows, NULL, data.rk->cols, data.rk->method);
    }
  } else {
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
  HMatrix<T> *h = new HMatrix<T>(o->localSettings.global);
  h->data.rows = o->data.rows;
  h->data.cols = o->data.cols;

  h->isLower = o->isLower;
  h->isUpper = o->isUpper;
  h->isTriUpper = o->isTriUpper;
  h->isTriLower = o->isTriLower;

  if (o->isLeaf()) {
    if (o->isRkMatrix()) {
      h->data.rk = new RkMatrix<T>(NULL, h->rows(), NULL, h->cols(), o->data.rk->method);
      h->data.m = NULL;
    } else {
      h->data.rk = NULL;
      h->data.m = FullMatrix<T>::Zero(h->rows()->n, h->cols()->n);
    }
  } else {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        if (o->get(i, j)) {
          h->insertChild(i + j * 2, HMatrix<T>::Zero(o->get(i, j)));
        }
      }
    }
  }
  return h;
}

template<typename T>
HMatrix<T>* HMatrix<T>::Zero(const ClusterTree* rows, const ClusterTree* cols, const hmat::MatrixSettings * settings) {
  // Leaves are filled by 0
  HMatrix<T> *h = new HMatrix<T>(settings);
  h->data.rows = (ClusterTree *) rows;
  h->data.cols = (ClusterTree *) cols;

  if (rows->isLeaf() || cols->isLeaf() || h->data.isAdmissibleLeaf(settings)) {
    if (h->data.isAdmissibleLeaf(settings)) {
      h->data.rk = new RkMatrix<T>(NULL, h->rows(), NULL, h->cols(), NoCompression);
      h->data.m = NULL;
    } else {
      h->data.rk = NULL;
      h->data.m = FullMatrix<T>::Zero(h->rows()->n, h->cols()->n);
    }
  } else {
    for (int i = 0; i < 2; ++i) {
      const ClusterTree* rowChild = static_cast<const ClusterTree*>(h->data.rows->getChild(i));
      for (int j = 0; j < 2; ++j) {
        const ClusterTree* colChild = static_cast<const ClusterTree*>(h->data.cols->getChild(j));
        h->insertChild(i + j * 2, HMatrix<T>::Zero(rowChild, colChild, settings));
      }
    }
  }
  return h;
}

template<typename T>
void HMatrix<T>::assemble(const AssemblyFunction<T>& f) {
  if (isLeaf()) {
    // If the leaf is admissible, matrix assembly and compression.
    // if not we keep the matrix.
    if (data.isAdmissibleLeaf(localSettings.global)) {
      RkMatrix<typename Types<T>::dp>* rkDp =
        compress(RkMatrix<T>::approx.method, f, rows(), cols());
      if (recompress) {
        rkDp->truncate();
      }
      RkMatrix<T>* rk = fromDoubleRk<T>(rkDp);

      data.m = NULL;
      data.rk->swap(*rk);
      delete rk;
    } else {
      FullMatrix<T>* mat = fromDoubleFull<T>(f.assemble(rows(), cols()));
      data.rk = NULL;
      data.m = mat;
    }
  } else {
    data.m = NULL;
    data.rk = NULL;
    for (int i = 0; i < 4; i++) {
      HMatrix<T> *child = static_cast<HMatrix*>(getChild(i));
      child->assemble(f);
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
          childrenArray[i] = child->data.rk;
          childrenElements += (childrenArray[i]->rows->n
                               + childrenArray[i]->cols->n) * childrenArray[i]->k;
        }
      }
      if (allRkLeaves) {
        RkMatrix<T> dummy(NULL, rows(), NULL, cols(), NoCompression);
        T alpha[4] = {Constants<T>::pone, Constants<T>::pone, Constants<T>::pone, Constants<T>::pone};
        RkMatrix<T>* candidate = dummy.formattedAddParts(alpha, childrenArray, 4);
        size_t elements = (candidate->rows->n + candidate->cols->n) * candidate->k;
        if (elements < childrenElements) {
          cout << "Coarsening ! " << elements << " < " << childrenElements << endl;
          for (int i = 0; i < 4; i++) {
            removeChild(i);
          }
          delete[] children;
          children = NULL;
          data.m = NULL;
          data.rk = candidate;
          myAssert(isLeaf());
          myAssert(isRkMatrix());
        } else {
          delete candidate;
        }
      }
    }
  }
}

template<typename T>
void HMatrix<T>::assembleSymmetric(const AssemblyFunction<T>& f,
   HMatrix<T>* upper, bool onlyLower) {
  if (!onlyLower) {
    if (!upper){
      upper = this;
    }
    myAssert(*this->rows() == *upper->cols());
    myAssert(*this->cols() == *upper->rows());
  }

  if (isLeaf()) {
    // If the leaf is admissible, matrix assembly and compression.
    // if not we keep the matrix.
    if (data.isAdmissibleLeaf(localSettings.global)) {
      this->assemble(f);
      if ((!onlyLower) && (upper != this)) {
        // Admissible leaf: a matrix represented by AB^t is transposed by exchanging A and B.
        RkMatrix<T>* rk = new RkMatrix<T>(NULL, upper->rows(),
                                          NULL, upper->cols(), this->data.rk->method);
        rk->k = this->data.rk->k;
        rk->a = (data.rk->b ? data.rk->b->copy() : NULL);
        rk->b = (data.rk->a ? data.rk->a->copy() : NULL);
        if(upper->data.rk != NULL)
            delete upper->data.rk;
        upper->data.rk = rk;
        upper->data.m = NULL;
      }
    } else {
      this->assemble(f);
      if ((!onlyLower) && ( upper != this)) {
        upper->data.rk = NULL;
        upper->data.m = data.m->copyAndTranspose();
      }
    }
  } else {
    if (onlyLower) {
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          if ((*rows() == *cols()) && (j > i)) {
            continue;
          }
          get(i,j)->assembleSymmetric(f, NULL, true);
        }
      }
    } else {
      if (this == upper) {
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j <= i; j++) {
            HMatrix<T> *child = get(i, j);
            HMatrix<T> *upperChild = get(j, i);
            myAssert(child != NULL);
            child->assembleSymmetric(f, upperChild);
          }
        }
      } else {
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++) {
            HMatrix<T> *child = get(i, j);
            HMatrix<T> *upperChild = upper->get(j, i);
            child->assembleSymmetric(f, upperChild);
          }
        }
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
              childrenArray[i] = child->data.rk;
              childrenElements += (childrenArray[i]->rows->n
                                   + childrenArray[i]->cols->n) * childrenArray[i]->k;
            }
          }
          if (allRkLeaves) {
            T alpha[4] = {Constants<T>::pone, Constants<T>::pone, Constants<T>::pone, Constants<T>::pone};
            RkMatrix<T> dummy(NULL, rows(), NULL, cols(), NoCompression);
            RkMatrix<T>* candidate = dummy.formattedAddParts(alpha, childrenArray, 4);
            size_t elements = (candidate->rows->n + candidate->cols->n) * candidate->k;
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

              data.m = NULL;
              data.rk = candidate;
              upper->data.m = NULL;
              upper->data.rk = new RkMatrix<T>(NULL, upper->rows(),
                                               NULL, upper->cols(), data.rk->method);
              upper->data.rk->k = data.rk->k;
              upper->data.rk->a = data.rk->b->copy();
              upper->data.rk->b = data.rk->a->copy();

              myAssert(isLeaf() && upper->isLeaf());
              myAssert(isRkMatrix() && upper->isRkMatrix());
            } else {
              delete candidate;
            }
          }
        }
      }
    }
  }
}

template<typename T>
pair<size_t, size_t> HMatrix<T>::compressionRatio() const {
  pair<size_t, size_t> result = pair<size_t, size_t>(0, 0);
  if (isLeaf()) {
    if (data.m) {
      result.first = data.m->rows * data.m->cols;
      result.second = result.first;
    } else {
      if (data.rk) {
        result = data.rk->compressionRatio();
      }
    }
    return result;
  }
  for (int i = 0; i < 4; i++) {
    HMatrix<T> *child = static_cast<HMatrix<T>*>(getChild(i));
    if (child) {
      pair<size_t, size_t> p = child->compressionRatio();
      result.first += p.first;
      result.second += p.second;
    }
  }
  return result;
}

template<typename T>
void HMatrix<T>::eval(FullMatrix<T>* result, bool renumber) const {
  if (isLeaf()) {
    FullMatrix<T> *mat = data.m;
    if (data.rk) {
      mat = data.rk->eval();
    }
    int *rowIndices = rows()->indices + rows()->offset;
    size_t rowCount = rows()->n;
    int *colIndices = cols()->indices + cols()->offset;
    size_t colCount = cols()->n;
    for (size_t i = 0; i < rowCount; i++) {
      for (size_t j = 0; j < colCount; j++) {
        if(renumber)
          result->get(rowIndices[i], colIndices[j]) = mat->get(i, j);
        else
          result->get(rows()->offset + i, cols()->offset + j) = mat->get(i, j);
      }
    }
    if (data.rk) {
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
void HMatrix<T>::evalPart(FullMatrix<T>* result, const ClusterData* _rows,
                          const ClusterData* _cols) const {
  if (isLeaf()) {
    FullMatrix<T> *mat = data.m;
    if (data.rk) {
      mat = data.rk->eval();
    }
    int rowOffset = rows()->offset - _rows->offset;
    int rowCount = rows()->n;
    int colOffset = cols()->offset - _cols->offset;
    int colCount = cols()->n;
    for (int i = 0; i < rowCount; i++) {
      for (int j = 0; j < colCount; j++) {
        result->get(i + rowOffset, j + colOffset) = mat->get(i, j);
      }
    }
    if (data.rk) {
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

template<typename T>
double HMatrix<T>::norm() const {
  double result = 0.;
  if (isLeaf()) {
    FullMatrix<T>* mat = data.m;
    if (data.rk) {
      // TODO: This is not optimized and problematic for
      // RKMatrix of too big size.
      mat = data.rk->eval();
      result = mat->norm();
      delete mat;
    } else {
      result = mat->norm();
    }
  } else {
    for (int i = 0; i < 4; i++) {
      if (getChild(i)) {
        result += pow(static_cast<HMatrix<T>*>(getChild(i))->norm(), 2.) ;
      }
    }
    result = sqrt(result) ;
  }
  return result;
}

template<typename T>
void HMatrix<T>::scale(T alpha) {
  if (isLeaf()) {
    if (isRkMatrix()) {
      data.rk->scale(alpha);
    } else {
      myAssert(isFullMatrix());
      data.m->scale(alpha);
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
  FullMatrix<T> mx(x->v, x->rows, 1);
  FullMatrix<T> my(y->v, y->rows, 1);
  gemv(trans, alpha, &mx, beta, &my);
}

template<typename T>
void HMatrix<T>::gemv(char trans, T alpha, const FullMatrix<T>* x, T beta, FullMatrix<T>* y) const {
  if (beta != Constants<T>::pone) {
    y->scale(beta);
  }
  beta = Constants<T>::pone;

  if (!isLeaf()) {
    const ClusterData* myRows = rows();
    const ClusterData* myCols = cols();
    // TODO: make this work with symmetric lower-stored matrices.
    for (int i = 0; i < 4; i++) {
      HMatrix<T>* child = static_cast<HMatrix<T>*>(getChild(i));
      if((isTriLower || isLower) && !child)
          continue;
      const ClusterData* childRows = child->rows();
      const ClusterData* childCols = child->cols();
      size_t rowsOffset = childRows->offset - myRows->offset;
      size_t colsOffset = childCols->offset - myCols->offset;
      if (trans == 'N') {
        myAssert(colsOffset + childCols->n <= (size_t) x->rows);
        myAssert(rowsOffset + childRows->n <= (size_t) y->rows);
        FullMatrix<T> subX(x->m + colsOffset, childCols->n, x->cols, x->lda);
        FullMatrix<T> subY(y->m + rowsOffset, childRows->n, y->cols, y->lda);
        child->gemv(trans, alpha, &subX, beta, &subY);
      } else {
        myAssert(trans == 'T');
        myAssert(rowsOffset + childRows->n <= (size_t) x->rows);
        myAssert(colsOffset + childCols->n <= (size_t) y->rows);
        FullMatrix<T> subX(x->m + rowsOffset, childRows->n, x->cols, x->lda);
        FullMatrix<T> subY(y->m + colsOffset, childCols->n, y->cols, y->lda);
        child->gemv(trans, alpha, &subX, beta, &subY);
      }
    }
  } else {
    if (data.m) {
      y->gemm(trans, 'N', alpha, data.m, x, beta);
    } else {
      data.rk->gemv(trans, alpha, x, beta, y);
    }
  }
}

template<typename T>
void HMatrix<T>::axpy(T alpha, const HMatrix<T>* b) {
  myAssert(*rows() == *b->rows());
  myAssert(*cols() == *b->cols());

  if (isLeaf()) {
    if (isRkMatrix()) {
      myAssert(b->isRkMatrix());
      if (b->data.rk->k == 0) {
        return;
      }
      data.rk->axpy(alpha, b->data.rk);
    } else {
      myAssert(b->isFullMatrix());
      data.m->axpy(alpha, b->data.m);
    }
  } else {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        HMatrix<T>* child = get(i, j);
        HMatrix<T>* bChild = b->get(i, j);
        child->axpy(alpha, bChild);
      }
    }
  }
}

template<typename T>
void HMatrix<T>::axpy(T alpha, const RkMatrix<T>* b) {
  DECLARE_CONTEXT;
  // this += alpha * b
  myAssert(b->rows->isSuperSet(*rows()));
  myAssert(b->cols->isSuperSet(*cols()));

  if (b->k == 0) {
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
    const RkMatrix<T>* rk = b;
    if (needResizing) {
      rk = b->subset(rows(), cols());
    }
    if (isFullMatrix()) {
      // In this case, the matrix has small size
      // then evaluating the Rk-matrix is cheaper
      FullMatrix<T>* rkMat = rk->eval();
      data.m->axpy(alpha, rkMat);
      delete rkMat;
    } else {
      myAssert(isRkMatrix());
      data.rk->axpy(alpha, rk);
    }
    if (needResizing) {
      delete rk;
    }
  }
}

template<typename T>
void HMatrix<T>::axpy(T alpha, const FullMatrix<T>* b, const ClusterData* rows,
                      const ClusterData* cols) {
  DECLARE_CONTEXT;
  // this += alpha * b
  myAssert(rows->isSuperSet(*this->rows()) && cols->isSuperSet(*this->cols()));
  if (!isLeaf()) {
    for (int i = 0; i < 4; i++) {
      HMatrix<T>* child = static_cast<HMatrix<T>*>(getChild(i));
      if (!child) {
        continue;
      }
      if (child->rows()->intersects(*rows) && child->cols()->intersects(*cols)) {
        const ClusterData *childRows = child->rows()->intersection(*rows);
        const ClusterData *childCols = child->cols()->intersection(*cols);
        int rowOffset = childRows->offset - rows->offset;
        int colOffset = childCols->offset - cols->offset;
        FullMatrix<T> subB(b->m + rowOffset + colOffset * b->lda,
                           childRows->n, childCols->n, b->lda);
        child->axpy(alpha, &subB, childRows, childCols);
        delete childRows;
        delete childCols;
      }
    }
  } else {
    size_t rowOffset = this->rows()->offset - rows->offset;
    size_t colOffset = this->cols()->offset - cols->offset;
    FullMatrix<T> subMat(b->m + rowOffset + (colOffset * b->lda),
                         this->rows()->n, this->cols()->n, b->lda);
    if (isFullMatrix()) {
      data.m->axpy(alpha, &subMat);
    } else {
      myAssert(isRkMatrix());
      data.rk->axpy(alpha, &subMat);
    }
  }
}

template<typename T>
void HMatrix<T>::gemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>* b, T beta, int depth) {
  myAssert((transA == 'N' ? *a->cols() : *a->rows()) == ( transB == 'N' ? *b->rows() : *b->cols())); // pour le produit A*B
  if ((transA == 'T') && (transB == 'T')) {
    strongAssert(false); // This assertion is only a warning, this code has *not* been tested.
    const ClusterData& tmp_rows = *this->rows();
    const ClusterData& tmp_cols = *this->cols();
    this->transpose();
    myAssert(tmp_rows == *cols());
    myAssert(tmp_cols == *rows());
    this->gemm('N', 'N', alpha, b, a, beta);
    this->transpose();
    myAssert(tmp_rows == *rows());
    myAssert(tmp_cols == *cols());
    return;
  }

  myAssert((transA == 'N' ? *a->rows() : *a->cols()) == *this->rows()); // compatibility of A*B + this : Rows
  myAssert((transB == 'N' ? *b->cols() : *b->rows()) == *this->cols()); // compatibility of A*B + this : columns

 // Scaling this
  if (beta != Constants<T>::pone) {
    if (beta == Constants<T>::zero) {
      this->clear();
    } else {
      this->scale(beta);
    }
  }

  // Once the scaling is done, beta is reset to 1
  // to avoid an other scaling.
  beta = Constants<T>::pone;

  // None of the matrices is a leaf
  if (!isLeaf() && !a->isLeaf() && !b->isLeaf()) {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        HMatrix<T>* child = get(i, j);
        if (!child) { // symmetric or triangular case
          continue;
        }
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
            myAssert(*a->rows() == *a->cols());
            continue;
          }
          if (!childB && (b->isTriUpper || b->isTriLower)) {
            myAssert(*b->rows() == *b->cols());
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

  // One of the matrices is a leaf
  myAssert(isLeaf() || a->isLeaf() || b->isLeaf());

  // the resulting matrix is not a leaf.
  if (!isLeaf()) {
    // If the resulting matrix is subdivided then at least one of the matrices of the product is a leaf.
    RkMatrix<T>* rkMat = NULL;
    FullMatrix<T>* fullMat = NULL;

    // One matrix is a RkMatrix
    if (a->isRkMatrix() || b->isRkMatrix()) {
      if ((a->isRkMatrix() && (a->data.rk->k == 0))
          || (b->isRkMatrix() && (b->data.rk->k == 0))) {
        return;
      }
      rkMat = HMatrix<T>::multiplyRkMatrix(transA, transB, a, b);
    } else {
      // None of the matrices of the product is a Rk-matrix so one of them is
      // a full matrix so as the result.
      myAssert(a->isFullMatrix() || b->isFullMatrix());
      fullMat = HMatrix<T>::multiplyFullMatrix(transA, transB, a, b);
    }
    // The resulting matrix is added to a H-matrix with the H-matrix class operations.
    if (rkMat) {
      axpy(alpha, rkMat);
      delete rkMat;
    } else {
      myAssert(fullMat);
      axpy(alpha, fullMat, rows(), cols());
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
    myAssert(isRkMatrix());
    myAssert((transA == 'N' ? *a->cols() : *a->rows()) == (transB == 'N' ? *b->rows() : *b->cols()));
    myAssert(*rows() == (transA == 'N' ? *a->rows() : *a->cols()));
    myAssert(*cols() == (transB == 'N' ? *b->cols() : *b->rows()));
    data.rk->gemmRk(transA, transB, alpha, a, b, beta);
    return;
  }

  // The resulting matrix is a full matrix
  myAssert(isFullMatrix());
  if (a->isRkMatrix() || b->isRkMatrix()) {
    myAssert(a->isRkMatrix() || b->isRkMatrix());
    if ((a->isRkMatrix() && (a->data.rk->k == 0))
        || (b->isRkMatrix() && (b->data.rk->k == 0))) {
      return;
    }
    RkMatrix<T>* rkMat = HMatrix<T>::multiplyRkMatrix(transA, transB, a, b);
    FullMatrix<T>* fullMat = rkMat->eval();
    delete rkMat;
    data.m->axpy(alpha, fullMat);
    delete fullMat;
  } else {
    myAssert(a->isFullMatrix() || b->isFullMatrix());
    FullMatrix<T>* fullMat = HMatrix<T>::multiplyFullMatrix(transA, transB, a, b);
    this->data.m->axpy(alpha, fullMat);
    delete fullMat;
  }
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
  myAssert((transH == 'N' ? h->cols()->n : h->rows()->n)
           == (transM == 'N' ? mat->rows : mat->cols));
  FullMatrix<T>* result =
    new FullMatrix<T>((transH == 'N' ? h->rows()->n : h->cols()->n),
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
  myAssert(a->isRkMatrix() || b->isRkMatrix());
  RkMatrix<T> *rk = NULL;
  // Matrices range compatibility
  if((transA == 'N') && (transB == 'N'))
    assert(a->cols()->n == b->rows()->n);
  if((transA == 'T') && (transB == 'N'))
    assert(a->rows()->n == b->rows()->n);
  if((transA == 'N') && (transB == 'T'))
    assert(a->cols()->n == b->cols()->n);

  // The cases are:
  //  - A Rk, B H
  //  - A H,  B Rk
  //  - A Rk, B Rk
  //  - A Rk, B F
  //  - A F,  B Rk
  if (a->isRkMatrix() && b->isHMatrix()) {
    rk = RkMatrix<T>::multiplyRkH(transA, transB, a->data.rk, b);
    strongAssert(rk);
  }
  else if (a->isHMatrix() && b->isRkMatrix()) {
    rk = RkMatrix<T>::multiplyHRk(transA, transB, a, b->data.rk);
    strongAssert(rk);
  }
  else if (a->isRkMatrix() && b->isRkMatrix()) {
    rk = RkMatrix<T>::multiplyRkRk(transA, transB, a->data.rk, b->data.rk);
    strongAssert(rk);
  }
  else if (a->isRkMatrix() && b->isFullMatrix()) {
    rk = RkMatrix<T>::multiplyRkFull(transA, transB, a->data.rk, b->data.m, (transB == 'N' ? b->cols() : b->rows()));
    strongAssert(rk);
  }
  else if (a->isFullMatrix() && b->isRkMatrix()) {
    rk = RkMatrix<T>::multiplyFullRk(transA, transB, a->data.m, b->data.rk, (transA == 'N' ? a->rows() : a->cols()));
    strongAssert(rk);
  } else {
    // None of the above cases, impossible.
    strongAssert(false);
  }
  return rk;
}

template<typename T>
FullMatrix<T>* HMatrix<T>::multiplyFullMatrix(char transA, char transB,
                                              const HMatrix<T>* a,
                                              const HMatrix<T>* b) {
  // At least one full matrix, and not RkMatrix.
  assert((transA == 'N') || (transB == 'N'));// Not for the products At*Bt
  myAssert(a->isFullMatrix() || b->isFullMatrix());
  myAssert(!(a->isRkMatrix() || b->isRkMatrix()));
  FullMatrix<T> *result = NULL;
  // The cases are:
  //  - A H, B F
  //  - A F, B H
  //  - A F, B F
  if (a->isHMatrix() && b->isFullMatrix()) {
    result = HMatrix<T>::multiplyHFull(transA, transB, a, b->data.m);
    strongAssert(result);
  } else if (a->isFullMatrix() && b->isHMatrix()) {
    result = HMatrix<T>::multiplyFullH(transA, transB, a->data.m, b);
    strongAssert(result);
  } else if (a->isFullMatrix() && b->isFullMatrix()) {
    int aRows = ((transA == 'N')? a->data.m->rows : a->data.m->cols);
    int bCols = ((transB == 'N')? b->data.m->cols : b->data.m->rows);
    result = new FullMatrix<T>(aRows, bCols);
    result->gemm(transA, transB, Constants<T>::pone, a->data.m, b->data.m,
                 Constants<T>::zero);
    strongAssert(result);
  } else {
    // None of above, impossible
    strongAssert(false);
  }
  return result;
}

template<typename T>
void HMatrix<T>::multiplyWithDiag(const HMatrix<T>* d, bool left) {
  DECLARE_CONTEXT;
  multiplyWithDiagOrDiagInv(d, false, left);
}

template<typename T>
void HMatrix<T>::multiplyWithDiagOrDiagInv(const HMatrix<T>* d, bool inverse, bool left) {
  myAssert(*d->rows() == *d->cols());
  myAssert(left || (*cols() == *d->rows()));
  myAssert(!left || (*rows() == *d->cols()));

  // The symmetric matrix must be taken into account: lower or upper
  if (isHMatrix()) {
    get(0,0)->multiplyWithDiagOrDiagInv(d->get(0,0), inverse, left);
    get(1,1)->multiplyWithDiagOrDiagInv(d->get(1,1), inverse, left);
    if (get(0, 1)) {
      get(0, 1)->multiplyWithDiagOrDiagInv(left ? d->get(0, 0) : d->get(1, 1), inverse, left);
    }
    if (get(1, 0)) {
      get(1, 0)->multiplyWithDiagOrDiagInv(left ? d->get(1, 1) : d->get(0, 0), inverse, left);
    }
  } else if (isRkMatrix()) {
    myAssert(!data.rk->a->isTriUpper && !data.rk->b->isTriUpper);
    myAssert(!data.rk->a->isTriLower && !data.rk->b->isTriLower);
    data.rk->multiplyWithDiagOrDiagInv(d, inverse, left);
  } else {
    myAssert(isFullMatrix());
    if (d->isFullMatrix()) {
      data.m->multiplyWithDiagOrDiagInv(d->data.m->diagonal, inverse, left);
    } else {
      Vector<T> diag(d->rows()->n);
      d->getDiag(&diag);
      data.m->multiplyWithDiagOrDiagInv(&diag, inverse, left);
    }
  }
}

template<typename T>
void HMatrix<T>::transpose() {
  if (!isLeaf()) {
    if (isLower || isUpper) { // if the matrix is symmetric, inverting it(Upper/Lower)
      isLower = !isLower;
      isUpper = !isUpper;
    }
    if (isTriLower || isTriUpper) { // if the matrix is triangular, on inverting it (isTriUpper/isTriLower)
      isTriLower = !isTriLower;
      isTriUpper = !isTriUpper;
    }
    get(1, 1)->transpose();
    get(0, 0)->transpose();
    swap(children[1 + 0 * 2], children[0 + 1 * 2]);
    if (get(1, 0)) get(1, 0)->transpose();
    if (get(0, 1)) get(0, 1)->transpose();
    swap(data.rows, data.cols);
  } else {
    swap(data.rows, data.cols);
    if (isRkMatrix()) {
      // To transpose an Rk-matrix, simple exchange A and B : (AB^T)^T = (BA^T)
      swap(data.rk->a, data.rk->b);
      swap(data.rk->rows, data.rk->cols);
    } else if (isFullMatrix()) {
      myAssert(data.m->lda == data.m->rows);
      data.m->transpose();
    }
  }
}

template<typename T>
void HMatrix<T>::copyAndTranspose(const HMatrix<T>* o) {
  myAssert(o);
  myAssert(*this->rows() == *o->cols());
  myAssert(*this->cols() == *o->rows());
  myAssert(isLeaf() == o->isLeaf());

  if (isLeaf()) {
    if (o->isRkMatrix()) {
      myAssert(!isFullMatrix());
      if (data.rk) {
        delete data.rk;
      }
      const RkMatrix<T>* oRk = o->data.rk;
      FullMatrix<T>* newA = oRk->b->copy();
      FullMatrix<T>* newB = oRk->a->copy();
      data.rk = new RkMatrix<T>(newA, oRk->cols, newB, oRk->rows, oRk->method);
    } else {
      myAssert(o->isFullMatrix());
      if (data.m) {
        delete data.m;
      }
      const FullMatrix<T>* oF = o->data.m;
      data.m = oF->copyAndTranspose();
      if (oF->diagonal) {
        if (!data.m->diagonal) {
          data.m->diagonal = new Vector<T>(oF->rows);
          strongAssert(data.m->diagonal);
        }
        memcpy(data.m->diagonal->v, oF->diagonal->v, oF->rows * sizeof(T));
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
  return &(data.rows->data);
}

template<typename T>
const ClusterData* HMatrix<T>::cols() const {
  return &(data.cols->data);
}

template<typename T>
void HMatrix<T>::createPostcriptFile(const char* filename) const {
    PostscriptDumper<T> dumper;
    dumper.write(this, filename);
}

template<typename T>
void HMatrix<T>::dumpTreeToFile(const char* filename) const {
  ofstream file;
  const vector<Point>* points = rows()->points;
  int* indices = rows()->indices;
  string delimiter;

  file.open(filename);
  // Points
  file << "{" << endl
       << "  \"points\": [" << endl;
  delimiter = "";
  for (size_t i = 0; i < points->size(); i++) {
    file << "    " << delimiter << "[" << (*points)[i].x << ", " << (*points)[i].y << ", " << (*points)[i].z
         << "]" << endl;
    delimiter = " ,";
  }
  // Mapping
  file << "  ]," << endl
       << "  \"mapping\": [" << endl << "    ";
  delimiter = "";
  for (size_t i = 0; i < points->size(); i++) {
    file << delimiter << indices[i];
    delimiter = " ,";
  }
  file << "]," << endl;
  file << "  \"tree\":" << endl;
  dumpSubTree(file, 0);
  file << "}" << endl;
}

template<typename T>
void HMatrix<T>::dumpSubTree(ofstream& f, int depth) const {
  string prefix("    ");
  for (int i = 0; i < depth; i++) {
    prefix += "  ";
  }
  f << prefix << "{\"isLeaf\": " << (isLeaf() ? "true" : "false") << "," << endl
    << prefix << " \"depth\": " << depth << "," << endl
    << prefix << " \"rows\": " << "{\"offset\": " << rows()->offset << ", \"n\": " << rows()->n << ", "
    << "\"boundingBox\": [[" << data.rows->boundingBox[0].x << ", " << data.rows->boundingBox[0].y
    << ", " << data.rows->boundingBox[0].z << "], ["
    << data.rows->boundingBox[1].x << ", " << data.rows->boundingBox[1].y
    << ", " << data.rows->boundingBox[1].z << "]]},"
    << endl
    << prefix << " \"cols\": " << "{\"offset\": " << cols()->offset << ", \"n\": " << cols()->n << ", "
    << "\"boundingBox\": [[" << data.cols->boundingBox[0].x << ", " << data.cols->boundingBox[0].y
    << ", " << data.cols->boundingBox[0].z << "], ["
    << data.cols->boundingBox[1].x << ", " << data.cols->boundingBox[1].y
    << ", " << data.cols->boundingBox[1].z << "]]},";
  if (!isLeaf()) {
    f << endl << prefix << " \"children\": [" << endl;
    string delimiter("");
    for (int i = 0; i < 4; i++) {
      const HMatrix<T>* child = static_cast<const HMatrix<T>*>(getChild(i));
      if (!child) continue;
      f << delimiter;
      child->dumpSubTree(f, depth + 1);
      f << endl;
      delimiter = ",";
    }
    f << prefix << " ]";
  } else {
    // It's a leaf
    if (isFullMatrix()) {
      f << endl << prefix << " \"leaf_type\": \"Full\"";
    } else {
      myAssert(isRkMatrix());
      f << endl << prefix << " \"leaf_type\": \"Rk\", \"k\": " << data.rk->k << ",";
      f << endl << prefix << " \"eta\": " << this->data.rows->getEta(this->data.cols) << ",";
      f << endl << prefix << " \"method\": " << this->data.rk->method;
    }
  }
  f << "}";
}

template<typename T>
void HMatrix<T>::copy(const HMatrix<T>* o) {
  DECLARE_CONTEXT;

  myAssert(*rows() == *o->rows());
  myAssert(*cols() == *o->cols());

  isLower = o->isLower;
  isUpper = o->isUpper;
  isTriUpper = o->isTriUpper;
  isTriLower = o->isTriLower;

  if (isLeaf()) {
    myAssert(o->isLeaf());
    if ((!data.m && !o->data.m) && (!data.rk && !o->data.rk)) {
      return;
    }
    // When the matrix has not allocated but only the structure
    if (o->isFullMatrix() && (!data.m)) {
      data.m = FullMatrix<T>::Zero(o->data.m->rows, o->data.m->cols);
    } else if (o->isRkMatrix() && (!data.rk)) {
      data.rk = new RkMatrix<T>(NULL, o->data.rk->rows, NULL, o->data.rk->cols, o->data.rk->method);
    }
    myAssert((isRkMatrix() == o->isRkMatrix())
           && (isFullMatrix() == o->isFullMatrix()));
    if (o->isRkMatrix()) {
      data.rk->copy(o->data.rk);
    } else {
      myAssert(isFullMatrix());
      if (o->data.m->diagonal) {
        if (!data.m->diagonal) {
          data.m->diagonal = new Vector<T>(o->data.m->rows);
          strongAssert(data.m->diagonal);
        }
        memcpy(data.m->diagonal->v, o->data.m->diagonal->v, o->data.m->rows * sizeof(T));
      }
      data.m->isTriLower = o->data.m->isTriLower;
      data.m->isTriUpper = o->data.m->isTriUpper;
      data.m->copyMatrixAtOffset(o->data.m, 0, 0);
      myAssert(data.m->m);
    }
  } else {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        if (o->get(i, j)) {
          myAssert(get(i, j));
          get(i, j)->copy(o->get(i, j));
        } else {
          myAssert(!get(i, j));
        }
      }
    }
  }
}

template<typename T>
void HMatrix<T>::clear() {
  if (isLeaf()) {
    if (isFullMatrix()) {
      data.m->clear();
    } else {
      myAssert(isRkMatrix());
      delete data.rk;
      data.rk = new RkMatrix<T>(NULL, rows(), NULL, cols(), NoCompression);
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

  myAssert(*(rows()) == *(tmp->rows()));
  myAssert(*(cols()) == *(tmp->cols()));

  if (this->isLeaf()) {
    myAssert(isFullMatrix());
    data.m->inverse();
  } else {
    myAssert(!tmp->isLeaf());
    HMatrix<T>* m11 = static_cast<HMatrix<T>*>(getChild(0));
    HMatrix<T>* m21 = static_cast<HMatrix<T>*>(getChild(1));
    HMatrix<T>* m12 = static_cast<HMatrix<T>*>(getChild(2));
    HMatrix<T>* m22 = static_cast<HMatrix<T>*>(getChild(3));
    HMatrix<T>* x11 = static_cast<HMatrix<T>*>(tmp->getChild(0));
    HMatrix<T>* x21 = static_cast<HMatrix<T>*>(tmp->getChild(1));
    HMatrix<T>* x12 = static_cast<HMatrix<T>*>(tmp->getChild(2));
    HMatrix<T>* x22 = static_cast<HMatrix<T>*>(tmp->getChild(3));

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
void HMatrix<T>::solveLowerTriangular(HMatrix<T>* b) const {
  DECLARE_CONTEXT;
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

    l11->solveLowerTriangular(b11);
    l11->solveLowerTriangular(b12);
    b21->gemm('N', 'N', Constants<T>::mone, l21, b11, Constants<T>::pone);
    l22->solveLowerTriangular(b21);
    b22->gemm('N', 'N', Constants<T>::mone, l21, b12, Constants<T>::pone);
    l22->solveLowerTriangular(b22);
  } else {
    // if B is a leaf, the resolve is done by column
    if (b->isLeaf()) {
      if (b->isFullMatrix()) {
        this->solveLowerTriangular(b->data.m);
      } else {
        myAssert(b->isRkMatrix());
        if (b->data.rk->k == 0) {
          return;
        }
        this->solveLowerTriangular(b->data.rk->a);
      }
    } else {
      // B isn't a leaf, then 'this' is one
      myAssert(this->isLeaf());
      // Evaluate B, solve by column, and restore in the matrix
      // TODO: check if it's not too bad
      FullMatrix<T>* bFull = new FullMatrix<T>(b->rows()->n, b->cols()->n);

      b->evalPart(bFull, b->rows(), b->cols());
      this->solveLowerTriangular(bFull);
      b->clear();
      b->axpy(Constants<T>::pone, bFull, b->rows(), b->cols());
      delete bFull;
    }
  }
}

template<typename T>
void HMatrix<T>::solveLowerTriangular(FullMatrix<T>* b) const {
  DECLARE_CONTEXT;
  myAssert(*rows() == *cols());
  myAssert(cols()->n == b->rows); // Here : the change : OK or not ??????? : cols <-> rows
  if (this->isLeaf()) {
    myAssert(this->isFullMatrix());
    // LAPACK resolution
    this->data.m->solveLowerTriangular(b);
  } else {
    const HMatrix<T>* l11 = get(0, 0);
    const HMatrix<T>* l21 = get(1, 0);
    const HMatrix<T>* l22 = get(1, 1);
    FullMatrix<T> b1(b->m, l11->cols()->n, b->cols, b->lda);
    FullMatrix<T> b2(b->m + l11->cols()->n, l22->cols()->n, b->cols, b->lda);
    l11->solveLowerTriangular(&b1);
    l21->gemv('N', Constants<T>::mone, &b1, Constants<T>::pone, &b2);
    l22->solveLowerTriangular(&b2);
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangular(HMatrix<T>* b, bool lowerStored) const {
  DECLARE_CONTEXT;
  myAssert(*b->cols() == *this->rows());
  myAssert(*this->rows() == *this->cols());
  myAssert(*b->cols() == *this->cols());

  // The recursion one (simple case)
  if (!isLeaf() && !b->isLeaf()) {
    const HMatrix<T>* u11 = get(0, 0);
    const HMatrix<T>* u12 = lowerStored ? get(1, 0) : get(0, 1);
    const HMatrix<T>* u22 = get(1, 1);
    HMatrix<T>* b11 = b->get(0, 0);
    HMatrix<T>* b21 = b->get(1, 0);
    HMatrix<T>* b12 = b->get(0, 1);
    HMatrix<T>* b22 = b->get(1, 1);

    u11->solveUpperTriangular(b11, lowerStored);
    u11->solveUpperTriangular(b21, lowerStored);
    // B12 <- -B11*U12 + B12
    b12->gemm('N', lowerStored ? 'T' : 'N', Constants<T>::mone, b11, u12, Constants<T>::pone);
    // B12 <- the solution of X * U22 = B12
    u22->solveUpperTriangular(b12, lowerStored);
    // B22 <- - B21*U12 + B22
    b22->gemm('N', lowerStored ? 'T' : 'N', Constants<T>::mone, b21, u12, Constants<T>::pone);
    // B22 <- the solution of X*U22 = B22
    u22->solveUpperTriangular(b22, lowerStored);

  } else {
    // if B is a leaf, the resolve is done by row
    if (b->isLeaf()) {
      if (b->isFullMatrix()) {
        b->data.m->transpose();
        this->solveUpperTriangular(b->data.m, lowerStored);
        b->data.m->transpose();
      } else {
        // Xa Xb^t U = Ba Bb^t
        //   - Xa = Ba
        //   - Xb^t U = Bb^t
        // Xb is stored without been transposed
        // it become again a resolution by column of Bb
        myAssert(b->isRkMatrix());
        if (b->data.rk->k == 0) {
          return;
        }
        this->solveUpperTriangular(b->data.rk->b, lowerStored);
      }
    } else {
      // B is not a leaf, then so is L
      myAssert(isLeaf());
      myAssert(isFullMatrix());
      // Evaluate B, solve by column and restore all in the matrix
      // TODO: check if it's not too bad
      FullMatrix<T>* bFull = new FullMatrix<T>(b->rows()->n, b->cols()->n);
      b->evalPart(bFull, b->rows(), b->cols());
      bFull->transpose();
      this->solveUpperTriangular(bFull, lowerStored);
      bFull->transpose();
      // int bRows = b->rows()->n;
      // int bCols = b->cols()->n;
      // Vector<T> bRow(bCols);
      // for (int row = 0; row < bRows; row++) {
      //   blasCopy<T>(bCols, bFull->m + row, bRows, bRow.v, 1);
      //   FullMatrix<T> tmp(bRow.v, bRow.rows, 1);
      //   this->solveUpperTriangular(&tmp);
      //   blasCopy<T>(bCols, bRow.v, 1, bFull->m + row, bRows);      // }
      // }
      b->clear();
      b->axpy(Constants<T>::pone, bFull, b->rows(), b->cols());
      delete bFull;
    }
  }
}

/* Resolve U.X=B, solution saved in B, with B Hmat */
template<typename T>
void HMatrix<T>::solveUpperTriangularLeft(HMatrix<T>* b) const {
  DECLARE_CONTEXT;
  // At first, the recursion one (simple case)
  if (!isLeaf() && !b->isLeaf()) {
    const HMatrix<T>* u11 = get(0, 0);
    const HMatrix<T>* u12 = get(0, 1);
    const HMatrix<T>* u22 = get(1, 1);
    HMatrix<T>* b11 = b->get(0, 0);
    HMatrix<T>* b21 = b->get(1, 0);
    HMatrix<T>* b12 = b->get(0, 1);
    HMatrix<T>* b22 = b->get(1, 1);

    u22->solveUpperTriangularLeft(b21);
    u22->solveUpperTriangularLeft(b22);
    b12->gemm('N', 'N', Constants<T>::mone, u12, b22, Constants<T>::pone);
    u11->solveUpperTriangularLeft(b12);
    b11->gemm('N', 'N', Constants<T>::mone, u12, b21, Constants<T>::pone);
    u11->solveUpperTriangularLeft(b11);
  } else {
    // if B is a leaf, the resolve is done by column
    if (b->isLeaf()) {
      if (b->isFullMatrix()) {
        this->solveUpperTriangularLeft(b->data.m);
      } else {
        myAssert(b->isRkMatrix());
        if (b->data.rk->k != 0) {
          this->solveUpperTriangularLeft(b->data.rk->a);
        }
      }
    } else {
      // B isn't a leaf, then so is L
      myAssert(this->isLeaf());
      // Evaluate B, solve by column, and restore in the matrix
      // TODO: check if it's not too bad
      FullMatrix<T>* bFull = new FullMatrix<T>(b->rows()->n, b->cols()->n);
      b->evalPart(bFull, b->rows(), b->cols());
      this->solveUpperTriangularLeft(bFull);
      b->clear();
      b->axpy(Constants<T>::pone, bFull, b->rows(), b->cols());
      delete bFull;
    }
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangular(FullMatrix<T>* b, bool loweredStored) const {
  DECLARE_CONTEXT;
  myAssert(*rows() == *cols());
  // B is supposed given in form of a row vector, but transposed
  // so we can deal it with a subset as usual.
  if (this->isLeaf()) {
    myAssert(this->isFullMatrix());
    FullMatrix<T>* bCopy = b->copyAndTranspose();
    this->data.m->solveUpperTriangular(bCopy, loweredStored);
    bCopy->transpose();
    b->copyMatrixAtOffset(bCopy, 0, 0);
    delete bCopy;
  } else {
    const HMatrix<T>* u11 = get(0, 0);
    const HMatrix<T>* u12 = loweredStored ? get(1, 0) : get(0, 1);
    const HMatrix<T>* u22 = get(1, 1);

    FullMatrix<T> b1(b->m, u11->rows()->n, b->cols, b->lda);
    FullMatrix<T> b2(b->m + u11->rows()->n, u22->rows()->n, b->cols, b->lda);
    u11->solveUpperTriangular(&b1, loweredStored);
    // b2 <- -x1 U12 + b2 = -U12^t x1^t + b2
    u12->gemv(loweredStored ? 'N' : 'T', Constants<T>::mone, &b1, Constants<T>::pone, &b2);
    u22->solveUpperTriangular(&b2, loweredStored);
  }
}

template<typename T>
void HMatrix<T>::solveUpperTriangularLeft(FullMatrix<T>* b, bool lowerStored) const {
  DECLARE_CONTEXT;
  myAssert(*rows() == *cols());
  if (this->isLeaf()) {
    this->data.m->solveUpperTriangularLeft(b, lowerStored);
  } else {
    const HMatrix<T>* u11 = get(0, 0);
    const HMatrix<T>* u12 = (lowerStored ? get(1, 0) : get(0, 1));
    const HMatrix<T>* u22 = get(1, 1);

    FullMatrix<T> b1(b->m, u11->cols()->n, b->cols, b->lda);
    FullMatrix<T> b2(b->m + u11->cols()->n, u22->cols()->n, b->cols, b->lda);

    u22->solveUpperTriangularLeft(&b2, lowerStored);
    // b1 <- -U12 b2 + b1
    u12->gemv(lowerStored ? 'T' : 'N', Constants<T>::mone, &b2, Constants<T>::pone, &b1);
    u11->solveUpperTriangularLeft(&b1, lowerStored);
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

#ifdef DEBUG_LDLT
    assertLower();
#endif
    if(isLeaf()) {
        data.m->lltDecomposition();
    } else {
        strongAssert(isLower);
        strongAssert(!get(0,1));
        HMatrix<T>* h11 = get(0,0);
        HMatrix<T>* h21 = get(1,0);
        HMatrix<T>* h22 = get(1,1);
        h11->lltDecomposition();
        h11->solveUpperTriangular(h21, true);
        h22->gemm('N', 'T', Constants<T>::mone, h21, h21, Constants<T>::pone);
        h22->lltDecomposition();
    }
    isTriLower = true;
}

template<typename T>
void HMatrix<T>::luDecomposition() {
  DECLARE_CONTEXT;

  if (isLeaf()) {
    myAssert(isFullMatrix());
    data.m->luDecomposition();
    data.m->checkNan();
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
    l11->solveLowerTriangular(h12);
    // Solve L21 U11 = H21 (get L21)
    u11->solveUpperTriangular(h21);
    // H22 <- H22 - L21 U12
    h22->gemm('N', 'N', Constants<T>::mone, h21, h12, Constants<T>::pone);
    h22->luDecomposition();
  }
}

template<typename T>
void HMatrix<T>::mdmtProduct(const HMatrix<T>* m, const HMatrix<T>* d) {
  DECLARE_CONTEXT;
  // this <- this - M * D * M^T
  //
  // D is stored separately in full matrix of diagonal leaves (see full_matrix.hpp).
  // this is symmetric and stored as lower triangular.
  // Warning: d must be the result of an ldlt factorization
#ifdef DEBUG_LDLT
  assertLower();
#endif
  myAssert(*d->rows() == *d->cols());       // D is square
  myAssert(*this->rows() == *this->cols()); // this is square
  myAssert(*m->cols() == *d->rows());       // Check if we can have the produit M*D and D*M^T
  myAssert(*this->rows() == *m->rows());

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
      myAssert(*y->cols() == *d22->rows());
      y->multiplyWithDiag(d22, false);
      h21->gemm('N', 'T', Constants<T>::mone, y, m12, Constants<T>::pone);
      delete y;

      h22->mdmtProduct(m21, d11);
      h22->mdmtProduct(m22, d22);
    } else if (m->isRkMatrix()) {
      HMatrix<T>* m_copy = Zero(m);
      m_copy->copy(m);

      myAssert(*m->cols() == *d->rows());
      myAssert(*m_copy->data.rk->cols == *d->rows());
      m_copy->multiplyWithDiagOrDiagInv(d, false, false); // right multiplication by D
      RkMatrix<T>* rkMat = RkMatrix<T>::multiplyRkRk('N', 'T', m_copy->data.rk, m->data.rk);
      delete m_copy;

      this->axpy(Constants<T>::mone, rkMat);
      delete rkMat;
    } else {
      myAssert(m->isFullMatrix());
      HMatrix<T>* copy_m = Zero(m);
      strongAssert(copy_m);
      copy_m->copy(m);
      copy_m->multiplyWithDiagOrDiagInv(d, false, false); // right multiplication by D

      FullMatrix<T>* fullMat = HMatrix<T>::multiplyFullMatrix('N', 'T', copy_m, m);
      strongAssert(fullMat);
      delete copy_m;

      this->axpy(Constants<T>::mone, fullMat, rows(), cols());
      delete fullMat;
    }
  } else {
    myAssert(isFullMatrix());

    if (m->isRkMatrix()) {
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
      m_copy->multiplyWithDiagOrDiagInv(d, false, false);

      RkMatrix<T>* rkMat = RkMatrix<T>::multiplyRkRk('N', 'T', m_copy->data.rk, m->data.rk);
      FullMatrix<T>* fullMat = rkMat->eval();
      delete m_copy;
      delete rkMat;
      data.m->axpy(Constants<T>::mone, fullMat);
      delete fullMat;
    } else if (m->isFullMatrix()) {
      // S <- S - M*D*M^T
      myAssert(!data.m->isTriUpper);
      myAssert(!data.m->isTriLower);
      myAssert(!m->data.m->isTriUpper);
      myAssert(!m->data.m->isTriLower);
      FullMatrix<T> mTmp(m->data.m->rows, m->data.m->cols);
      mTmp.copyMatrixAtOffset(m->data.m, 0, 0);
      if (d->isFullMatrix()) {
        mTmp.multiplyWithDiagOrDiagInv(d->data.m->diagonal, false, false);
      } else {
        Vector<T> diag(d->cols()->n);
        d->getDiag(&diag);
        mTmp.multiplyWithDiagOrDiagInv(&diag, false, false);
      }
      data.m->gemm('N', 'T', Constants<T>::mone, &mTmp, m->data.m, Constants<T>::pone);
    }
  }
}

#ifdef DEBUG_LDLT
template<typename T>
bool HMatrix<T>::assertLdlt() const {
  if (isLeaf()) {
    myAssert(this->isFullMatrix());
    myAssert(data.m->diagonal);
    myAssert(data.m->isTriLower);
    return ((data.m->diagonal != NULL) && data.m->isTriLower);
  } else
    assert(isTriLower);
    return (get(0,0)->assertLdlt()) && (get(1,1)->assertLdlt());
}

template<typename T>
void HMatrix<T>::assertLower() {
  if (this->isLeaf()) {
    return;
  } else {
    myAssert(isLower);
    myAssert(!get(0,1));
    get(0,0)->assertLower(); //isLower || get(0,0)->isLeaf());
    get(1,1)->assertLower(); // isLower || get(1,1)->isLeaf());
  }
}

template<typename T>
void HMatrix<T>::assertUpper() {
  if (this->isLeaf()) {
    return;
  } else {
    myAssert(isUpper);
    myAssert(!get(1,0));
    get(0,0)->assertUpper(); //isLower || get(0,0)->isLeaf());
    get(1,1)->assertUpper(); // isLower || get(1,1)->isLeaf());
    get(0,1)->assertAllocFull();
  }
}

template<typename T>
void HMatrix<T>::assertAllocFull() {
  if (this->isLeaf()) {
    if (isFullMatrix())
      myAssert(data.m->m);
  } else {
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        if (get(i,j))
          get(i,j)->assertAllocFull();
  }
}
#endif

template<typename T>
void HMatrix<T>::ldltDecomposition() {
  DECLARE_CONTEXT;
#ifdef DEBUG_LDLT
  this->assertLower();
#endif

  if (isLeaf()) {
    //The basic case of the recursion is necessarily a full matrix leaf
    //since the recursion is done with *rows() == *cols().

    myAssert(isFullMatrix());
    this->data.m->ldltDecomposition();
    myAssert(this->data.m->diagonal);
  } else {
    HMatrix<T>* h11 = get(0,0);
    HMatrix<T>* h21 = get(1,0);
    HMatrix<T>* h22 = get(1,1);

    // H11 <- L11 and D11 is stored additionally to each diagonal leaf


    h11->ldltDecomposition();
#ifdef DEBUG_LDLT
    myAssert(h11->assertLdlt());
#endif

    HMatrix<T>* y = Zero(h11); // H11 is lower triangular, therefore either upper or lower part is NULL
    y->copy(h11); // X <- copy(L11)
    // Y <- Y*D11 , D11 is stocked in h11 (in diagonal leaves)

    y->multiplyWithDiag(h11); // MultiplyWithDiag takes into account the fact that "y" is Upper or Lower
    // Y <- Y^T
#ifdef DEBUG_LDLT
    y->assertLower();
#endif
    // The transpose function keeps the Upper or Lower matrix but
    //     reverse storage.
    y->transpose();
    // H21 <- solution of X*Y = H21, with Y = (L11 * D11)^T give L21
    // stored in H21
    myAssert(y->isUpper || y->isLeaf());
    myAssert(!y->isLower);
    y->solveUpperTriangular(h21);
#ifdef DEBUG_LDLT
    y->assertUpper();
    y->assertAllocFull();
#endif
    delete y;

    // H22 <- H22 - L21 * D11 * L21^T
    // D11 is contained on the diagonal leaves (which are full)
    h22->mdmtProduct(h21, h11);
#ifdef DEBUG_LDLT
    h22->assertAllocFull();
    h22->assertLower();
#endif
    h22->ldltDecomposition();
  }
  isTriLower = true;
}

template<typename T>
void HMatrix<T>::solve(FullMatrix<T>* b) const {
  DECLARE_CONTEXT;
  this->solveLowerTriangular(b);
  this->solveUpperTriangularLeft(b, false);
}

template<typename T>
void HMatrix<T>::getDiag(Vector<T>* diag, int start) const {
  DECLARE_CONTEXT;
  if(isLeaf()) {
    myAssert(isFullMatrix());
    if(data.m->diagonal) {
      // LDLt
      memcpy(diag->v + start, data.m->diagonal->v, data.m->rows * sizeof(T));
    } else {
      // LLt
      for (int i = 0; i < data.m->rows; ++i)
        diag->v[start + i] = data.m->m[i*data.m->rows + i];
    }
  } else {
    get(0,0)->getDiag(diag, start);
    get(1,1)->getDiag(diag, start + get(0,0)->rows()->n);
  }
}

/* Solve M.X=B with M hmat LU factorized*/
template<typename T>
void HMatrix<T>::solve(HMatrix<T>* b) const {
  DECLARE_CONTEXT;

  /* Solve LX=B, result in B */
  this->solveLowerTriangular(b);
  /* Solve UX=B, result in B */
  this->solveUpperTriangularLeft(b);
}

template<typename T>
void HMatrix<T>::solveDiagonal(FullMatrix<T>* b) const {
  // Diagonal extraction
  Vector<T> diag(cols()->n);
  getDiag(&diag);
  // TODO: use BLAS for that
  for (int j = 0; j < b->cols; j++) {
    for (int i = 0; i < b->rows; i++) {
      b->get(i, j) = b->get(i, j) / diag.v[i];
    }
  }
}

template<typename T>
void HMatrix<T>::solveLdlt(FullMatrix<T>* b) const {
  DECLARE_CONTEXT;
#ifdef DEBUG_LDLT
  assertLdlt();
#endif
  // L*D*L^T * X = B
  // B <- solution of L * Y = B : Y = D*L^T * X
  this->solveLowerTriangular(b);

  // B <- D^{-1} Y : solution of D*Y = B : Y = L^T * X
  this->solveDiagonal(b);

  // B <- solution of L^T X = B :  the solution X we are looking for is stored in B
  this->solveUpperTriangularLeft(b, true);
}

template<typename T>
void HMatrix<T>::checkNan() const {
  return;
  if (isLeaf()) {
    if (isFullMatrix()) {
      data.m->checkNan();
    }
    if (isRkMatrix()) {
      data.rk->checkNan();
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

// Templates declaration
template class HMatrixData<S_t>;
template class HMatrixData<D_t>;
template class HMatrixData<C_t>;
template class HMatrixData<Z_t>;

template class HMatrix<S_t>;
template class HMatrix<D_t>;
template class HMatrix<C_t>;
template class HMatrix<Z_t>;

template void reorderVector(FullMatrix<S_t>* v, int* indices);
template void reorderVector(FullMatrix<D_t>* v, int* indices);
template void reorderVector(FullMatrix<C_t>* v, int* indices);
template void reorderVector(FullMatrix<Z_t>* v, int* indices);

template void restoreVectorOrder(FullMatrix<S_t>* v, int* indices);
template void restoreVectorOrder(FullMatrix<D_t>* v, int* indices);
template void restoreVectorOrder(FullMatrix<C_t>* v, int* indices);
template void restoreVectorOrder(FullMatrix<Z_t>* v, int* indices);
