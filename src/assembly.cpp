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
#include "assembly.hpp"
#include "full_matrix.hpp"
#include "common/context.hpp"
#include <assert.h>
#include <iostream>
#include "h_matrix.hpp"
#include "rk_matrix.hpp"
#include "fromdouble.hpp"

namespace hmat {

template<typename T>
void AssemblyFunction<T>::assemble(const LocalSettings & settings,
                                     const ClusterTree &rows,
                                     const ClusterTree &cols,
                                     bool admissible,
                                     FullMatrix<T> *&fullMatrix,
                                     RkMatrix<T> *&rkMatrix,
                                     const AllocationObserver & allocationObserver) {
    if (admissible) {

      RkMatrix<typename Types<T>::dp>* rkDp = compress<T>(settings.approx, function_, &(rows.data), &(cols.data),
                                                          allocationObserver);
      if (HMatrix<T>::recompress) {
        rkDp->truncate();
      }
      rkMatrix = fromDoubleRk<T>(rkDp);
    } else {
      fullMatrix = fromDoubleFull<T>(function_.assemble(&(rows.data), &(cols.data), NULL, allocationObserver));
    }
}

template<typename T>
FullMatrix<typename Types<T>::dp>*
SimpleFunction<T>::assemble(const ClusterData* rows,
                            const ClusterData* cols,
                            const hmat_block_info_t *,
                            const AllocationObserver &) const {
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
void SimpleFunction<T>::getRow(const ClusterData* rows, const ClusterData* cols,
                                       int rowIndex, void* handle,
                                       Vector<typename Types<T>::dp>* result) const {
  const int row = *(rows->indices() + rows->offset() + rowIndex);
  const int* cols_indices = cols->indices() + cols->offset();
  for (int j = 0; j < cols->size(); j++) {
    result->v[j] = interaction(row, cols_indices[j]);
  }
}

template<typename T>
void SimpleFunction<T>::getCol(const ClusterData* rows, const ClusterData* cols,
                                       int colIndex, void* handle,
                                       Vector<typename Types<T>::dp>* result) const {
  const int col = *(cols->indices() + cols->offset() + colIndex);
  const int* rows_indices = rows->indices() + rows->offset();
  for (int i = 0; i < rows->size(); i++) {
    result->v[i] = interaction(rows_indices[i], col);
  }
}


template<typename T>
BlockFunction<T>::BlockFunction(const ClusterData* rowData,
                                                  const ClusterData* colData,
                                                  void* matrixUserData,
                                                  hmat_prepare_func_t _prepare,
                                                  hmat_compute_func_t _compute)
  : prepare(_prepare), compute(_compute), matrixUserData(matrixUserData) {
  rowMapping = rowData->indices();
  colMapping = colData->indices();
  rowReverseMapping = rowData->indices_rev();
  colReverseMapping = colData->indices_rev();
}

template<typename T>
BlockFunction<T>::~BlockFunction() {}

template<typename T>
FullMatrix<typename Types<T>::dp>*
BlockFunction<T>::assemble(const ClusterData* rows,
                                   const ClusterData* cols,
                                   const hmat_block_info_t * block_info,
                                   const AllocationObserver & allocator) const {
  DECLARE_CONTEXT;
  FullMatrix<typename Types<T>::dp>* result = NULL;
  hmat_block_info_t local_block_info ;

  if (!block_info)
    prepareBlock(rows, cols, &local_block_info, allocator);
  else
    local_block_info = *block_info ;

  if (local_block_info.block_type != hmat_block_null) {
    result = FullMatrix<typename Types<T>::dp>::Zero(rows->size(), cols->size());
    compute(local_block_info.user_data, 0, rows->size(), 0, cols->size(), (void*) result->m);
  }

  if (!block_info)
    releaseBlock(&local_block_info, allocator);

  return result;
}

void initBlockInfo(hmat_block_info_t * info) {
    info->block_type = hmat_block_full;
    info->release_user_data = NULL;
    info->is_null_col = NULL;
    info->is_null_row = NULL;
    info->user_data = NULL;
    info->needed_memory = HMAT_NEEDED_MEMORY_UNSET;
}

template<typename T>
void BlockFunction<T>::prepareBlock(const ClusterData* rows, const ClusterData* cols,
                                    hmat_block_info_t * block_info, const AllocationObserver & ao) const {
    initBlockInfo(block_info);
    prepareImpl(rows, cols, block_info);
    if(block_info->needed_memory != HMAT_NEEDED_MEMORY_UNSET) {
        ao.allocate(block_info->needed_memory);
        prepareImpl(rows, cols, block_info);
    }
    // check memory leak
    assert((block_info->user_data == NULL) == (block_info->release_user_data == NULL));
}

template<typename T>
void BlockFunction<T>::prepareImpl(const ClusterData* rows, const ClusterData* cols,
    hmat_block_info_t * block_info) const {
  prepare(rows->offset(), rows->size(), cols->offset(), cols->size(), rowMapping, rowReverseMapping,
          colMapping, colReverseMapping, matrixUserData, block_info);
}

template<typename T>
void BlockFunction<T>::releaseBlock(hmat_block_info_t * block_info, const AllocationObserver & ao) const {
  if(block_info->release_user_data)
    block_info->release_user_data(block_info->user_data);
  if(block_info->needed_memory != HMAT_NEEDED_MEMORY_UNSET)
    ao.free(block_info->needed_memory);
}

template<typename T>
void BlockFunction<T>::getRow(const ClusterData* rows,
                                       const ClusterData* cols,
                                       int rowIndex, void* handle,
                                       Vector<typename Types<T>::dp>* result) const {
  DECLARE_CONTEXT;
  assert(handle);
  compute(handle, rowIndex, 1, 0, cols->size(), (void*) result->v);
}

template<typename T>
void BlockFunction<T>::getCol(const ClusterData* rows,
                                       const ClusterData* cols,
                                       int colIndex, void* handle,
                                       Vector<typename Types<T>::dp>* result) const {
  DECLARE_CONTEXT;
  assert(handle);
  compute(handle, 0, rows->size(), colIndex, 1, (void*) result->v);

  // for (int i = 0; i < rows->size(); i++) {
  //   if (result->v[i] == Constants<T>::zero) {
  //     assert(false);
  //   }
  // }
}
template<typename T>
void Function<T>::prepareBlock(const ClusterData* rows, const ClusterData* cols,
             hmat_block_info_t * block_info, const AllocationObserver &) const {
   initBlockInfo(block_info);
}


// Template declaration
template class BlockFunction<S_t>;
template class BlockFunction<D_t>;
template class BlockFunction<C_t>;
template class BlockFunction<Z_t>;

template class SimpleAssemblyFunction<S_t>;
template class SimpleAssemblyFunction<D_t>;
template class SimpleAssemblyFunction<C_t>;
template class SimpleAssemblyFunction<Z_t>;

template class BlockAssemblyFunction<S_t>;
template class BlockAssemblyFunction<D_t>;
template class BlockAssemblyFunction<C_t>;
template class BlockAssemblyFunction<Z_t>;

}  // end namespace hmat

