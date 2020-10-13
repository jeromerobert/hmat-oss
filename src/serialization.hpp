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
#pragma once

#include <h_matrix.hpp>

namespace hmat {

template<typename T> class MatrixStructMarshaller {
    void write(const ClusterTree * clusterTree);
    template<typename VT> void writeValue(VT v);
    void writeTreeNode(const ClusterTree * clusterTree);
    void writeTreeNode(const HMatrix<T> * m);
    template<typename TR> int writeTree(const TR * tree);
    hmat_iostream writeFunc_;
    void * userData_;
public:
    /**
     * Create matrix structure marshaller
     * @param writefunc the stream to write to
     * @param user_data a pointer to be passed to writefunc
     */
    MatrixStructMarshaller(hmat_iostream writefunc, void * user_data):
        writeFunc_(writefunc), userData_(user_data) {}

    void write(const HMatrix<T> * matrix, Factorization factorization = Factorization::NONE);
};

template<typename T> class MatrixStructUnmarshaller {
    ClusterTree * readClusterTree();
    ClusterTree * readTreeNode(const ClusterTree * parent);
    HMatrix<T> * readTreeNode(HMatrix<T> *);
    template<typename VT> VT readValue();
    template<typename TR> TR * readTree(TR * tree);

    hmat_iostream readFunc_;
    void * userData_;
    DofData * dofData_;
    MatrixSettings * settings_;
    Factorization factorization_;
public:
    MatrixStructUnmarshaller(MatrixSettings * settings, hmat_iostream readfunc, void * user_data):
        readFunc_(readfunc), userData_(user_data), settings_(settings),
        factorization_(Factorization::NONE){}
    HMatrix<T> * read();
    Factorization factorization() {
        return factorization_;
    }
};

/** Save matrix blocks to a stream */
template<typename T> class MatrixDataMarshaller {
    void writeLeaf(const HMatrix<T> * matrix);
    void writeScalarArray(ScalarArray<T> * a);
    void writeInt(int v);
    hmat_iostream writeFunc_;
    void * userData_;

public:
    MatrixDataMarshaller(hmat_iostream writefunc, void * user_data):
        writeFunc_(writefunc), userData_(user_data){}

    void write(const HMatrix<T> * matrix);
};

/**
 * Read matrix blocks from a stream to an existing matrix
 * structure.
 */
template<typename T> class MatrixDataUnmarshaller {
    void readLeaf(HMatrix<T> * matrix);
    ScalarArray<T> * readScalarArray(int rows, int cols);
    hmat_iostream readFunc_;
    void * userData_;
public:
    MatrixDataUnmarshaller(hmat_iostream readfunc, void * user_data):
        readFunc_(readfunc), userData_(user_data){}

    void read(HMatrix<T> * matrix);
};
}
