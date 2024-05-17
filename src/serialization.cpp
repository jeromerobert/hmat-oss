/*
  HMat-OSS (HMatrix library, open source software)

  Copyright (C) 2014-2016 Airbus Group SAS

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

#include "serialization.hpp"
#include <vector>
#include "compression.hpp"
#include "rk_matrix.hpp"
#include "common/my_assert.h"

namespace hmat {

// TODO decide what to todo with read/write return value
template<typename T>
void MatrixStructMarshaller<T>::write(const ClusterTree * clusterTree) {
    const ClusterData & d = clusterTree->data;
    int s = d.coordinates()->numberOfDof();
    int dim = d.coordinates()->dimension();
    writeValue(s);
    writeValue(dim);
    for(int j = 0; j < s; j++) {
        for(int i = 0; i < dim; i++) {
            writeValue(d.coordinates()->spanCenter(j, i));
        }
    }
    if(d.group_index() == NULL)
        writeValue(0);
    else {
        writeValue(1);
        writeFunc_(d.group_index(), s * sizeof(int), userData_);
    }
    writeFunc_(d.indices(), s * sizeof(int), userData_);
    writeTree<ClusterTree>(clusterTree);
}

/** Write a scalar value */
template<typename T>
template<typename VT> void MatrixStructMarshaller<T>::writeValue(VT v) {
    writeFunc_(&v, sizeof(v), userData_);
}

/** Write a node of a clusterTree */
template<typename T>
void MatrixStructMarshaller<T>::writeTreeNode(const ClusterTree * clusterTree) {
    if(clusterTree == NULL) {
        writeValue(-1);
    } else {
        writeValue(clusterTree->data.offset());
        writeValue(clusterTree->data.size());
    }
}

/** Write a node of a HMatrix */
template<typename T>
void MatrixStructMarshaller<T>::writeTreeNode(const HMatrix<T> * m) {
    if(m == NULL) {
        writeValue<unsigned char>(1 << 7);
        return;
    }
    unsigned char bitfield = 0;
    bitfield |= m->isUpper;
    bitfield |= m->isLower        ? 1 << 1 : 0;
    bitfield |= m->isTriUpper     ? 1 << 2 : 0;
    bitfield |= m->isTriLower     ? 1 << 3 : 0;
    bitfield |= m->keepSameRows   ? 1 << 4 : 0;
    bitfield |= m->keepSameCols   ? 1 << 5 : 0;
    writeValue(bitfield);
    writeValue(m->approximateRank());
    if(m->isAssembled()) {
        if(m->isLeaf()) {
            if(m->isRkMatrix())
                writeValue(m->rank());
            else
                writeValue(FULL_BLOCK);
        } else {
            writeValue(NONLEAF_BLOCK);
        }
    } else {
        writeValue(UNINITIALIZED_BLOCK);
    }
    writeValue(m->lowRankEpsilon());
}

/**
 * Write any sub class of Tree.
 * - TR is the Tree class
 * - matching writeTreeNode methods must be available
 * - writeTreeNode must support NULL node
 */
template<typename T>
template<typename TR> int MatrixStructMarshaller<T>::writeTree(const TR * tree) {
    writeTreeNode(tree);
    if(tree != NULL) {
        writeValue((int)tree->nrChild());
        for(int i = 0; i < tree->nrChild(); i++) {
            writeTree(tree->getChild(i));
        }
    }
    return 0;
}

template<typename T>
void MatrixStructMarshaller<T>::write(const HMatrix<T> * matrix, Factorization factorization){
    writeValue(Types<T>::TYPE);
    writeValue(convert_factorization_to_int(factorization));
    write(matrix->rowsTree());
    write(matrix->colsTree());
    writeTree(matrix);
    return;
}

template<typename T>
ClusterTree * MatrixStructUnmarshaller<T>::readClusterTree() {
    int size = readValue<int>();
    int dim = readValue<int>();
    double * coordinates = new double[size * dim];
    readFunc_(coordinates, sizeof(double) * size * dim, userData_);
    DofCoordinates * dofCoordinates = new DofCoordinates(coordinates, dim, size, true);
    delete[] coordinates;
    int * group_index = NULL;
    if (readValue<int>()) {
        group_index = new int[size];
        readFunc_(group_index, sizeof(int) * size, userData_);
    }
    dofData_ = new DofData(*dofCoordinates, group_index);
    delete dofCoordinates;
    delete[] group_index;
    // dummy cluster tree to access the indices array
    ClusterTree dummyClusterTree(dofData_);
    // avoid DofData deletion
    dummyClusterTree.father = &dummyClusterTree;
    readFunc_(dummyClusterTree.data.indices(), sizeof(int) * size, userData_);
    for(int i = 0; i < size; ++i) {
        dummyClusterTree.data.indices_rev()[dummyClusterTree.data.indices()[i]] = i;
    }
    return readTree<ClusterTree>(NULL);
}

/**
 * Read a clusterTree node
 * @param clusterTree parent node
 */
template<typename T>
ClusterTree * MatrixStructUnmarshaller<T>::readTreeNode(const ClusterTree * parent) {
    ClusterTree * toReturn;
    int offset = readValue<int>();
    if(offset == -1)
        return NULL;
    int size = readValue<int>();
    if(parent == NULL) {
        // this is the root
        toReturn = new ClusterTree(dofData_, offset, size );
    }
    else
        toReturn = parent->slice(offset, size);
    return toReturn;
}

template<typename T>
HMatrix<T> * MatrixStructUnmarshaller<T>::readTreeNode(HMatrix<T> *) {
    unsigned char bitfield = readValue<char>();
    if(bitfield & (1 << 7))
        return NULL;
    int rankApprox = readValue<int>();
    int rank = readValue<int>();
    double epsilon = readValue<double>();
    return HMatrix<T>::unmarshall(settings_, rank, rankApprox, bitfield, epsilon);
}

/** Read a scalar value */
template<typename T>
template<typename VT> VT MatrixStructUnmarshaller<T>::readValue() {
    VT v;
    readFunc_(&v, sizeof(v), userData_);
    return v;
}

/**
 * Read a sub class of Tree
 * - TR is the Tree class
 * - readTreeNode methods must be available
 * - readTreeNode methods may return NULL
 */
template<typename T>
template<typename TR> TR * MatrixStructUnmarshaller<T>::readTree(TR * tree) {
    const int depth = (tree == NULL ? 0 : tree->depth + 1);
    tree = readTreeNode(tree);
    if(tree != NULL) {
        tree->depth = depth;
        int nrChild = readValue<int>();
        for(int i = 0; i < nrChild; i++) {
            tree->insertChild(i, readTree(tree));
        }
    }
    return tree;
}

template<typename T>
HMatrix<T> * MatrixStructUnmarshaller<T>::read(){
    int type = readValue<int>();
    HMAT_ASSERT_MSG(type == Types<T>::TYPE,
                    "Type mismatch. Unmarshaller type is %d while data type is %d",
                    Types<T>::TYPE, type);
    factorization_ = convert_int_to_factorization(readValue<int>());
    ClusterTree * rows = readClusterTree();
    // There is no easy way for reader to know that a Tree parsing is over.
    // We call readFunc_ with 0 size after reading all Tree instances.
    // This is harmless for casual readers, and may be needed for advanced usages.
    // First argument could be NULL, but this may trigger an assertion in debug
    // mode, so use a non-NULL pointer instead.
    readFunc_(&factorization_, 0, userData_);
    ClusterTree * cols = readClusterTree();
    readFunc_(&factorization_, 0, userData_);
    HMatrix<T> * r = readTree<HMatrix<T> >(NULL);
    readFunc_(&factorization_, 0, userData_);
    r->setClusterTrees(rows, cols);
    r->ownClusterTrees(true, true);
    return r;
}

template<typename T>
void MatrixDataMarshaller<T>::writeLeaf(const HMatrix<T> * matrix) {
    if(!matrix->isAssembled()) {
        writeInt(UNINITIALIZED_BLOCK);
    } else if(matrix->isRkMatrix()){
        writeInt(matrix->rank());
        if(!matrix->isNull()) {
          matrix->rk()->writeArray(writeFunc_, userData_);
          writeInt(matrix->rk()->a->getOrtho());
          writeInt(matrix->rk()->b->getOrtho());
        }
    } else if(matrix->isNull()){
        // null full block
        writeInt(1);
    } else {
        int r = matrix->rows()->size();
        // bit 1: null or not, bit 2: have pivot, bit 3: have diagonal
        int bitfield = 0;
        bool pivot = matrix->full()->pivots != NULL;
        bool diag = matrix->full()->diagonal != NULL;
        if(pivot)
            bitfield |= 2;
        if(diag)
            bitfield |= 4;
        writeInt(bitfield);
        writeScalarArray(&matrix->full()->data);
        if(pivot)
            writeFunc_(matrix->full()->pivots, sizeof(int) * r, userData_);
        if(diag)
          matrix->full()->diagonal->writeArray(writeFunc_, userData_);
    }
}

template<typename T>
void MatrixDataMarshaller<T>::writeScalarArray(ScalarArray<T> * a) {
  a->writeArray(writeFunc_, userData_);
}

template<typename T>
void MatrixDataMarshaller<T>::writeInt(int v) {
    writeFunc_(&v, sizeof(v), userData_);
}

template<typename T>
void MatrixDataMarshaller<T>::write(const HMatrix<T> * matrix){
    std::vector<const HMatrix<T> *> stack;
    stack.push_back(matrix);
    while(!stack.empty()) {
        const HMatrix<T> * m = stack.back();
        stack.pop_back();
        if(m->isLeaf()) {
            writeLeaf(m);
        } else {
            for(int i = m->nrChild() - 1; i >= 0; --i) {
                if(m->getChild(i) != NULL && !m->getChild(i)->isVoid())
                    stack.push_back(m->getChild(i));
            }
        }
    }
}

template<typename T>
void MatrixDataUnmarshaller<T>::readLeaf(HMatrix<T> * matrix) {
    const IndexSet * r = matrix->rows();
    const IndexSet * c = matrix->cols();
    int header;
    readFunc_(&header, sizeof(header), userData_);
    if(matrix->isRkMatrix()) {
        if(matrix->rk() != NULL)
            delete matrix->rk();
        int rank = header;
        if(rank > 0) {
            ScalarArray<T> * a = readScalarArray(r->size(), rank);
            ScalarArray<T> * b = readScalarArray(c->size(), rank);
            matrix->rk(new RkMatrix<T>(a, r, b, c));
            int orth;
            readFunc_(&orth, sizeof(int), userData_);
            matrix->rk()->a->setOrtho(orth);
            readFunc_(&orth, sizeof(int), userData_);
            matrix->rk()->b->setOrtho(orth);
        } else {
            matrix->rk(NULL);
        }
    } else if(!(header & 1)) {
        bool pivot = header & 2;
        bool diagonal = header & 4;
        // leak check
        assert(!matrix->isAssembled() || matrix->full() == NULL);
        FullMatrix<T> *fmat = new FullMatrix<T>(r, c, true);
	fmat->data.readArray(readFunc_, userData_);
	matrix->full( fmat );
        if(pivot) {
            matrix->full()->pivots = (int*) calloc(r->size(), sizeof(int));
            readFunc_(matrix->full()->pivots, r->size() * sizeof(int), userData_);
        }
        if(diagonal) {
            matrix->full()->diagonal = new Vector<T>(r->size());
            matrix->full()->diagonal->readArray(readFunc_, userData_);
        }
    }
}

template<typename T>
ScalarArray<T> * MatrixDataUnmarshaller<T>::readScalarArray(int rows, int cols) {
    ScalarArray<T> * r = new ScalarArray<T>(rows, cols);
    r->readArray(readFunc_, userData_);
    return r;
}

template<typename T>
void MatrixDataUnmarshaller<T>::read(HMatrix<T> * matrix){
    std::vector<HMatrix<T> *> stack;
    stack.push_back(matrix);
    while(!stack.empty()) {
        HMatrix<T> * m = stack.back();
        stack.pop_back();
        if(m->isLeaf()) {
            readLeaf(m);
        } else {
            for(int i = m->nrChild() - 1; i >= 0; --i) {
                if(m->getChild(i) != NULL && !m->getChild(i)->isVoid())
                    stack.push_back(m->getChild(i));
            }
        }
    }
    // Comment in MatrixStructUnmarshaller<T>::read explains why readFunc_ is called there
    readFunc_(&stack, 0, userData_);
}

// Templates declaration
template class MatrixStructMarshaller<S_t>;
template class MatrixStructMarshaller<D_t>;
template class MatrixStructMarshaller<C_t>;
template class MatrixStructMarshaller<Z_t>;
template class MatrixStructUnmarshaller<S_t>;
template class MatrixStructUnmarshaller<D_t>;
template class MatrixStructUnmarshaller<C_t>;
template class MatrixStructUnmarshaller<Z_t>;
template class MatrixDataMarshaller<S_t>;
template class MatrixDataMarshaller<D_t>;
template class MatrixDataMarshaller<C_t>;
template class MatrixDataMarshaller<Z_t>;
template class MatrixDataUnmarshaller<S_t>;
template class MatrixDataUnmarshaller<D_t>;
template class MatrixDataUnmarshaller<C_t>;
template class MatrixDataUnmarshaller<Z_t>;
}
