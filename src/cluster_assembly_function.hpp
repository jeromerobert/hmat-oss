#pragma once

/*
  HMat-OSS (HMatrix library, open source software)

  Copyright (C) 2019 Airbus S.A.S.

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

/* Convenience class to lighten the getRow() / getCol() / assemble calls
   and use information from block_info_t to speed up things for sparse and null blocks.
*/

#include "cluster_tree.hpp"
#include "assembly.hpp"

namespace hmat {


  template<typename T>
  class ClusterAssemblyFunction {
    const Function <T> &f;

  public:
    const ClusterData *rows;
    const ClusterData *cols;
    hmat_block_info_t info;
    int stratum;
    const AllocationObserver &allocationObserver_;

    ClusterAssemblyFunction(const Function <T> &_f,
                            const ClusterData *_rows, const ClusterData *_cols,
                            const AllocationObserver &allocationObserver);

    ~ClusterAssemblyFunction();

    void getRow(int index, Vector<typename Types<T>::dp> &result) const;

    void getCol(int index, Vector<typename Types<T>::dp> &result) const;

    typename Types<T>::dp getElement(int rowIndex, int colIndex) const;


    FullMatrix<typename Types<T>::dp> *assemble() const;

  private:
    ClusterAssemblyFunction(ClusterAssemblyFunction &o) : f(o.f), rows(o.rows), cols(o.cols), allocationObserver_(o.allocationObserver_) {} // No copy
  };

}