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

/* Interactions between the elements of the matrix */
#include "interaction.hpp"
#include "full_matrix.hpp"
#include "common/context.hpp"
#include "common/my_assert.h"
#include <iostream>

namespace hmat {

template<typename T>
FullMatrix<typename Types<T>::dp>*
SimpleAssemblyFunction<T>::assemble(const ClusterData* rows,
                                    const ClusterData* cols,
                                    const hmat_block_info_t * block_info) const {
  FullMatrix<typename Types<T>::dp>* result =
    new FullMatrix<typename Types<T>::dp>(rows->size(), cols->size());
  const int* rows_indices = rows->indices() + rows->offset();
  const int* cols_indices = cols->indices() + cols->offset();
  for (int j = 0; j < cols->size(); ++j) {
    int col = cols_indices[j];
    for (int i = 0; i < rows->size(); ++i) {
      int row = rows_indices[i];
      result->get(i, j) = interaction(row, col);
    }
  }
  return result;
}

template<typename T>
void SimpleAssemblyFunction<T>::getRow(const ClusterData* rows, const ClusterData* cols,
                                       int rowIndex, void* handle,
                                       Vector<typename Types<T>::dp>* result) const {
  const int row = *(rows->indices() + rows->offset() + rowIndex);
  const int* cols_indices = cols->indices() + cols->offset();
  for (int j = 0; j < cols->size(); j++) {
    result->v[j] = interaction(row, cols_indices[j]);
  }
}

template<typename T>
void SimpleAssemblyFunction<T>::getCol(const ClusterData* rows, const ClusterData* cols,
                                       int colIndex, void* handle,
                                       Vector<typename Types<T>::dp>* result) const {
  const int col = *(cols->indices() + cols->offset() + colIndex);
  const int* rows_indices = rows->indices() + rows->offset();
  for (int i = 0; i < rows->size(); i++) {
    result->v[i] = interaction(rows_indices[i], col);
  }
}


template<typename T>
BlockAssemblyFunction<T>::BlockAssemblyFunction(const ClusterData* rowData,
                                                  const ClusterData* colData,
                                                  void* matrixUserData,
                                                  hmat_prepare_func_t _prepare,
                                                  compute_func _compute)
  : prepare(_prepare), compute(_compute), matrixUserData(matrixUserData) {
  rowMapping = rowData->indices();
  colMapping = colData->indices();
  rowReverseMapping = rowData->indices_rev();
  colReverseMapping = colData->indices_rev();
}

template<typename T>
BlockAssemblyFunction<T>::~BlockAssemblyFunction() {}

template<typename T>
FullMatrix<typename Types<T>::dp>*
BlockAssemblyFunction<T>::assemble(const ClusterData* rows,
                                   const ClusterData* cols,
                                   const hmat_block_info_t * block_info) const {
  DECLARE_CONTEXT;
  FullMatrix<typename Types<T>::dp>* result =
    FullMatrix<typename Types<T>::dp>::Zero(rows->size(), cols->size());

  hmat_block_info_t local_block_info ;

  if (!block_info)
    prepareBlock(rows, cols, &local_block_info);
  else
    local_block_info = *block_info ;

  if (local_block_info.block_type != hmat_block_null)
    compute(local_block_info.user_data, 0, rows->size(), 0, cols->size(), (void*) result->m);

  if (!block_info)
    releaseBlock(&local_block_info);

  return result;
}

template<typename T>
void BlockAssemblyFunction<T>::prepareBlock(const ClusterData* rows, const ClusterData* cols,
    hmat_block_info_t * block_info) const {
  // TODO factorize block_info init with ClusterAssemblyFunction
  block_info->block_type = hmat_block_full;
  block_info->release_user_data = NULL;
  block_info->is_null_col = NULL;
  block_info->is_null_row = NULL;
  block_info->user_data = NULL;
  prepare(rows->offset(), rows->size(), cols->offset(), cols->size(), rowMapping, rowReverseMapping,
          colMapping, colReverseMapping, matrixUserData, block_info);
  // check memory leak
  myAssert((block_info->user_data == NULL) == (block_info->release_user_data == NULL));
}

template<typename T>
void BlockAssemblyFunction<T>::releaseBlock(hmat_block_info_t * block_info) const {
  if(block_info->release_user_data)
    block_info->release_user_data(block_info->user_data);
}

template<typename T>
void BlockAssemblyFunction<T>::getRow(const ClusterData* rows,
                                       const ClusterData* cols,
                                       int rowIndex, void* handle,
                                       Vector<typename Types<T>::dp>* result) const {
  DECLARE_CONTEXT;
  myAssert(handle);
  compute(handle, rowIndex, 1, 0, cols->size(), (void*) result->v);
}

template<typename T>
void BlockAssemblyFunction<T>::getCol(const ClusterData* rows,
                                       const ClusterData* cols,
                                       int colIndex, void* handle,
                                       Vector<typename Types<T>::dp>* result) const {
  DECLARE_CONTEXT;
  myAssert(handle);
  compute(handle, 0, rows->size(), colIndex, 1, (void*) result->v);

  // for (int i = 0; i < rows->size(); i++) {
  //   if (result->v[i] == Constants<T>::zero) {
  //     myAssert(false);
  //   }
  // }
}


// Template declaration
template class SimpleAssemblyFunction<S_t>;
template class SimpleAssemblyFunction<D_t>;
template class SimpleAssemblyFunction<C_t>;
template class SimpleAssemblyFunction<Z_t>;

template class BlockAssemblyFunction<S_t>;
template class BlockAssemblyFunction<D_t>;
template class BlockAssemblyFunction<C_t>;
template class BlockAssemblyFunction<Z_t>;

}  // end namespace hmat

