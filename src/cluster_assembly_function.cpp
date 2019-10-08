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

#include "cluster_assembly_function.hpp"
#include "scalar_array.hpp"
#include "h_matrix.hpp"

namespace hmat {


  template<typename T>
  void hmat::ClusterAssemblyFunction<T>::getRow(int index, Vector<typename Types<T>::dp> &result) const {
    if (!HMatrix<T>::validateNullRowCol) {
      // Normal mode: we compute except if a function is_guaranteed_null_row() is provided and tells it's null
      if (!info.is_guaranteed_null_row || !info.is_guaranteed_null_row(&info, index, stratum))
        f.getRow(rows, cols, index, info.user_data, &result, stratum);
    } else {
      // Validation mode: we always compute, and if a function is_guaranteed_null_row() tells it's null then we check that
      f.getRow(rows, cols, index, info.user_data, &result, stratum);
      if (info.is_guaranteed_null_row && info.is_guaranteed_null_row(&info, index, stratum))
        assert(result.isZero());
      // TODO: in validation mode, we could also warn about undetected null rows or columns
    }
  }

  template<typename T>
  void hmat::ClusterAssemblyFunction<T>::getCol(int index, Vector<typename Types<T>::dp> &result) const {
    if (!HMatrix<T>::validateNullRowCol) {
      // Normal mode: we compute except if a function is_guaranteed_null_col() is provided and tells it's null
      if (!info.is_guaranteed_null_col || !info.is_guaranteed_null_col(&info, index, stratum))
        f.getCol(rows, cols, index, info.user_data, &result, stratum);
    } else {
      // Validation mode: we always compute, and if a function is_guaranteed_null_col() tells it's null then we check that
      f.getCol(rows, cols, index, info.user_data, &result, stratum);
      if (info.is_guaranteed_null_col && info.is_guaranteed_null_col(&info, index, stratum))
        assert(result.isZero());
    }
  }

  template<typename T>
  typename Types<T>::dp hmat::ClusterAssemblyFunction<T>::getElement(int rowIndex, int colIndex) const {
    if (!HMatrix<T>::validateNullRowCol) {
      // Normal mode: we compute except if a function is_guaranteed_null_col/row() is provided and tells it's null
      bool colNotGuaranteedNull =
          !info.is_guaranteed_null_col || !info.is_guaranteed_null_col(&info, colIndex, stratum);
      bool rowNotGuaranteedNull =
          !info.is_guaranteed_null_row || !info.is_guaranteed_null_row(&info, rowIndex, stratum);
      if (colNotGuaranteedNull && rowNotGuaranteedNull)
        return f.getElement(rows, cols, rowIndex, colIndex, info.user_data, stratum);
      return (typename Types<T>::dp)0;
    } else {
      // Validation mode: we always compute, and if a function is_guaranteed_null_col() tells it's null then we check that
      typename Types<T>::dp result = f.getElement(rows, cols, rowIndex, colIndex, info.user_data, stratum);
      bool colGuaranteedNull = info.is_guaranteed_null_col && info.is_guaranteed_null_col(&info, colIndex, stratum);
      bool rowGuaranteedNull = info.is_guaranteed_null_row && info.is_guaranteed_null_row(&info, rowIndex, stratum);
      if (colGuaranteedNull || rowGuaranteedNull)
        assert(result == (typename Types<T>::dp) 0);
      return result;
    }
  }

  template<typename T>
  FullMatrix<typename Types<T>::dp> *hmat::ClusterAssemblyFunction<T>::assemble() const {
    if(stratum != -1) {
      ScalarArray<typename Types<T>::dp> *mat = new ScalarArray<typename Types<T>::dp>(rows->size(), cols->size());
      for(int j = 0 ; j < cols->size(); j++) {
        Vector<typename Types<T>::dp> vec(*mat, j);
        getCol(j, vec);
      }
      return new FullMatrix<typename Types<T>::dp>(mat, rows, cols);
    }
    if (info.block_type != hmat_block_null)
      return f.assemble(rows, cols, &info, allocationObserver_) ;
    else
      // TODO return NULL
      return new FullMatrix<typename Types<T>::dp>(rows, cols);

  }

  template<typename T>
  ClusterAssemblyFunction<T>::~ClusterAssemblyFunction() {
    f.releaseBlock(&info, allocationObserver_);
  }

  template<typename T>
  ClusterAssemblyFunction<T>::ClusterAssemblyFunction(const Function<T> &_f, const ClusterData *_rows,
                                                      const ClusterData *_cols,
                                                      const AllocationObserver &allocationObserver)
      : f(_f), rows(_rows), cols(_cols), stratum(-1), allocationObserver_(allocationObserver) {
    f.prepareBlock(rows, cols, &info, allocationObserver_);
    assert((info.user_data == NULL) == (info.release_user_data == NULL));
  }

  // Declaration of the used templates
  template class ClusterAssemblyFunction<S_t>;
  template class ClusterAssemblyFunction<D_t>;
  template class ClusterAssemblyFunction<C_t>;
  template class ClusterAssemblyFunction<Z_t>;

}