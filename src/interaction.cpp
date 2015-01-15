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

template<typename T>
FullMatrix<typename Types<T>::dp>*
SimpleAssemblyFunction<T>::assemble(const ClusterData* rows,
                                    const ClusterData* cols,
                                    void *handle,
                                    hmat_block_info_t * block_info) const {
  FullMatrix<typename Types<T>::dp>* result =
    new FullMatrix<typename Types<T>::dp>(rows->n, cols->n);
  for (int i = 0; i < rows->n; ++i) {
    int row = rows->indices[i + rows->offset];
    for (int j = 0; j < cols->n; ++j) {
      int col = cols->indices[j + cols->offset];
      result->get(i, j) = interaction(row, col);
    }
  }
  return result;
}

template<typename T>
Vector<typename Types<T>::dp>*
SimpleAssemblyFunction<T>::getRow(const ClusterData* rows, const ClusterData* cols,
                               int rowIndex, void* handle) const {
  Vector<typename Types<T>::dp>* result =
    new Vector<typename Types<T>::dp>(cols->n);
  getRow(rows, cols, rowIndex, handle, result);
  return result;
}

template<typename T>
void SimpleAssemblyFunction<T>::getRow(const ClusterData* rows, const ClusterData* cols,
                                       int rowIndex, void* handle,
                                       Vector<typename Types<T>::dp>* result) const {
  int row = rows->indices[rowIndex + rows->offset];
  for (int j = 0; j < cols->n; j++) {
    int col = cols->indices[j + cols->offset];
    result->v[j] = interaction(row, col);
  }
}

template<typename T>
Vector<typename Types<T>::dp>*
SimpleAssemblyFunction<T>::getCol(const ClusterData* rows, const ClusterData* cols,
                                  int colIndex, void* handle) const {
  Vector<typename Types<T>::dp>* result =
    new Vector<typename Types<T>::dp>(rows->n);
  getCol(rows, cols, colIndex, handle, result);
  return result;
}

template<typename T>
void SimpleAssemblyFunction<T>::getCol(const ClusterData* rows, const ClusterData* cols,
                                       int colIndex, void* handle,
                                       Vector<typename Types<T>::dp>* result) const {
  int col = cols->indices[colIndex + cols->offset];
  for (int i = 0; i < rows->n; i++) {
    int row = rows->indices[i + rows->offset];
    result->v[i] = interaction(row, col);
  }
}


template<typename T>
BlockAssemblyFunction<T>::BlockAssemblyFunction(const ClusterData* rowData,
                                                  const ClusterData* colData,
                                                  void* _user_context,
                                                  prepare_func _prepare,
                                                  compute_func _compute,
                                                  release_func _free_data)
  : prepare(_prepare), compute(_compute), free_data(_free_data), user_context(_user_context) {
  rowMapping = rowData->indices;
  colMapping = colData->indices;
  rowReverseMapping = new int[rowData->n];
  colReverseMapping = new int[colData->n];
  for (int i = 0; i < (int) rowData->n; i++) {
    rowReverseMapping[rowMapping[i]] = i;
  }
  for (int i = 0; i < (int) colData->n; i++) {
    colReverseMapping[colMapping[i]] = i;
  }
}

template<typename T>
BlockAssemblyFunction<T>::~BlockAssemblyFunction() {
  delete[] rowReverseMapping;
  delete[] colReverseMapping;
}

template<typename T>
typename Types<T>::dp BlockAssemblyFunction<T>::interaction(int i, int j) const {
  //TODO: FIXME
  strongAssert(false);
  return Constants<typename Types<T>::dp>::zero;
}

template<typename T>
FullMatrix<typename Types<T>::dp>*
BlockAssemblyFunction<T>::assemble(const ClusterData* rows,
                                   const ClusterData* cols,
                                   void *handle,
                                   hmat_block_info_t * block_info) const {
  DECLARE_CONTEXT;
  FullMatrix<typename Types<T>::dp>* result =
    FullMatrix<typename Types<T>::dp>::Zero(rows->n, cols->n);

  hmat_block_info_t local_block_info ;

  if (!block_info)
    prepareBlock(rows, cols, &handle, &local_block_info);
  else
    local_block_info = *block_info ;

  if (local_block_info.block_type != hmat_block_null)
    compute(handle, 0, rows->n, 0, cols->n, (void*) result->m);

  if (!block_info)
    releaseBlock(handle, &local_block_info);

  return result;
}

template<typename T>
void BlockAssemblyFunction<T>::prepareBlock(const ClusterData* rows, const ClusterData* cols,
    void** handle, hmat_block_info_t * block_info) const {
  block_info->block_type = hmat_block_full;
  block_info->rowMask = (char*)calloc(rows->n, sizeof(char)) ;
  block_info->colMask = (char*)calloc(cols->n, sizeof(char)) ;
  prepare(rows->offset, rows->n, cols->offset, cols->n, rowMapping, rowReverseMapping,
          colMapping, colReverseMapping, user_context, handle, block_info);
}

template<typename T>
void BlockAssemblyFunction<T>::releaseBlock(void* handle, hmat_block_info_t * block_info) const {
  if (block_info->colMask) free(block_info->colMask) ;
  if (block_info->rowMask) free(block_info->rowMask) ;
  free_data(handle);
}

template<typename T>
Vector<typename Types<T>::dp>*
BlockAssemblyFunction<T>::getRow(const ClusterData* rows,
                                  const ClusterData* cols,
                                  int rowIndex, void* handle) const {
  Vector<typename Types<T>::dp>* result =
    Vector<typename Types<T>::dp>::Zero(cols->n);
  strongAssert(result);
  getRow(rows, cols, rowIndex, handle, result);
  return result;
}

template<typename T>
void BlockAssemblyFunction<T>::getRow(const ClusterData* rows,
                                       const ClusterData* cols,
                                       int rowIndex, void* handle,
                                       Vector<typename Types<T>::dp>* result) const {
  DECLARE_CONTEXT;
  myAssert(handle);
  compute(handle, rowIndex, 1, 0, cols->n, (void*) result->v);
}


template<typename T>
Vector<typename Types<T>::dp>*
BlockAssemblyFunction<T>::getCol(const ClusterData* rows,
                                  const ClusterData* cols,
                                  int colIndex, void* handle) const {
  DECLARE_CONTEXT;
  Vector<typename Types<T>::dp>* result =
    Vector<typename Types<T>::dp>::Zero(rows->n);
  strongAssert(result);
  getCol(rows, cols, colIndex, handle, result);
  return result;
}

template<typename T>
void BlockAssemblyFunction<T>::getCol(const ClusterData* rows,
                                       const ClusterData* cols,
                                       int colIndex, void* handle,
                                       Vector<typename Types<T>::dp>* result) const {
  DECLARE_CONTEXT;
  myAssert(handle);
  compute(handle, 0, rows->n, colIndex, 1, (void*) result->v);

  // for (int i = 0; i < rows->n; i++) {
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

