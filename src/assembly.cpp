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

#include <cstdlib>
#include <cstring>

namespace hmat {

template<typename T, template <typename> class F>
void AssemblyFunction<T, F>::assemble(const LocalSettings &,
                                     const ClusterTree &rows,
                                     const ClusterTree &cols,
                                     bool admissible,
                                     FullMatrix<T> *&fullMatrix,
                                     RkMatrix<T> *&rkMatrix,
                                     const AllocationObserver & allocationObserver) {
    if (admissible) {
      // Always compress the smallest blocks using an SVD. Small blocks tend to have
      // a bad compression ratio anyways, and the SVD is not very costly in this
      // case.
      CompressionMethod method = RkMatrix<T>::approx.method;
      if (std::max(rows.data.size(), cols.data.size()) < RkMatrix<T>::approx.compressionMinLeafSize) {
        method = Svd;
      }
      rkMatrix = fromDoubleRk<T>(compress<T>(method, function_, &rows.data, &cols.data, allocationObserver));
    } else if (rows.data.size() && cols.data.size()) {
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
    new FullMatrix<typename Types<T>::dp>(rows, cols);
  const int* rows_indices = rows->indices() + rows->offset();
  const int* cols_indices = cols->indices() + cols->offset();
  for (int j = 0; j < cols->size(); ++j) {
    int col = cols_indices[j];
    for (int i = 0; i < rows->size(); ++i) {
      int row = rows_indices[i];
      compute_(userContext_, row, col, &result->get(i, j));
    }
  }
  return result;
}

template<typename T>
void SimpleFunction<T>::getRow(const ClusterData* rows, const ClusterData* cols,
                               int rowIndex, void*,
                               Vector<typename Types<T>::dp>* result, int stratum) const {
  (void)stratum; //unused with NDEBUG
  assert(stratum == -1); // stratum not supported here
  const int row = *(rows->indices() + rows->offset() + rowIndex);
  const int* cols_indices = cols->indices() + cols->offset();
  for (int j = 0; j < cols->size(); j++) {
    compute_(userContext_, row, cols_indices[j], result->m + j);
  }
}

template<typename T>
void SimpleFunction<T>::getCol(const ClusterData* rows, const ClusterData* cols,
                               int colIndex, void*,
                               Vector<typename Types<T>::dp>* result, int stratum) const {
  (void)stratum; //unused with NDEBUG
  assert(stratum == -1); // stratum not supported here
  const int col = *(cols->indices() + cols->offset() + colIndex);
  const int* rows_indices = rows->indices() + rows->offset();
  for (int i = 0; i < rows->size(); i++) {
    compute_(userContext_, rows_indices[i], col, result->m + i);
  }
}


template<typename T>
BlockFunction<T>::BlockFunction(const ClusterData* rowData,
                                const ClusterData* colData,
                                void* matrixUserData,
                                hmat_prepare_func_t _prepare,
                                hmat_compute_func_t legacyCompute,
                                void (*compute)(struct hmat_block_compute_context_t*))
  : prepare(_prepare), compute_(compute), legacyCompute_(legacyCompute), matrixUserData_(matrixUserData) {
  rowMapping = rowData->indices();
  colMapping = colData->indices();
  rowReverseMapping = rowData->indices_rev();
  colReverseMapping = colData->indices_rev();
  assert(legacyCompute_ || compute_);
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

  if (local_block_info.block_type == hmat_block_null) {
    // Nothing to do
  } else if(compute_ == NULL) {
    result = new FullMatrix<typename Types<T>::dp>(rows, cols);
    legacyCompute_(local_block_info.user_data, 0, rows->size(), 0, cols->size(), result->data.m);
  } else {
    result = new FullMatrix<typename Types<T>::dp>(rows, cols);
    struct hmat_block_compute_context_t ac;
    ac.block = result->data.m;
    ac.col_count = cols->size();
    ac.col_start = 0;
    ac.row_count = rows->size();
    ac.row_start = 0;
    ac.stratum=-1;
    ac.user_data=local_block_info.user_data;
    compute_(&ac);
  }

  if (!block_info)
    releaseBlock(&local_block_info, allocator);

  return result;
}

void initBlockInfo(hmat_block_info_t * info) {
    info->block_type = hmat_block_full;
    info->release_user_data = NULL;
    info->is_guaranteed_null_col = NULL;
    info->is_guaranteed_null_row = NULL;
    info->user_data = NULL;
    info->needed_memory = 0;
    info->number_of_strata = 1;
}

template<typename T>
void BlockFunction<T>::prepareBlock(const ClusterData* rows, const ClusterData* cols,
                                    hmat_block_info_t * block_info, const AllocationObserver & ao) const {
    initBlockInfo(block_info);
    prepareImpl(rows, cols, block_info);
    if(block_info->needed_memory != 0) {
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
          colMapping, colReverseMapping, matrixUserData_, block_info);
}

template<typename T>
void BlockFunction<T>::releaseBlock(hmat_block_info_t * block_info, const AllocationObserver & ao) const {
  if(block_info->release_user_data)
    block_info->release_user_data(block_info->user_data);
  if(block_info->needed_memory != 0)
    ao.free(block_info->needed_memory);
}

template<typename T>
void BlockFunction<T>::getRow(const ClusterData*, const ClusterData* cols,
                              int rowIndex, void* handle,
                              Vector<typename Types<T>::dp>* result, int stratum) const {
    DECLARE_CONTEXT;
    assert(handle);
    if(compute_ == NULL) {
      assert(stratum == -1); // statum not supported here
      legacyCompute_(handle, rowIndex, 1, 0, cols->size(), result->m);
    } else {
        struct hmat_block_compute_context_t ac;
        ac.block = result->m;
        ac.col_count = cols->size();
        ac.col_start = 0;
        ac.row_count = 1;
        ac.row_start = rowIndex;
        ac.stratum=stratum;
        ac.user_data=handle;
        compute_(&ac);
    }
}

template<typename T>
void BlockFunction<T>::getCol(const ClusterData* rows,
                              const ClusterData*, int colIndex, void* handle,
                              Vector<typename Types<T>::dp>* result, int stratum) const {
    DECLARE_CONTEXT;
    assert(handle);
    if(compute_ == NULL) {
      assert(stratum == -1); // statum not supported here
      legacyCompute_(handle, 0, rows->size(), colIndex, 1, result->m);
    } else {
        struct hmat_block_compute_context_t ac;
        ac.block = result->m;
        ac.col_count = 1;
        ac.col_start = colIndex;
        ac.row_count = rows->size();
        ac.row_start = 0;
        ac.stratum=stratum;
        ac.user_data=handle;
        compute_(&ac);
    }
}
template<typename T>
void Function<T>::prepareBlock(const ClusterData*, const ClusterData*,
             hmat_block_info_t * block_info, const AllocationObserver &) const {
   initBlockInfo(block_info);
}


// Template declaration
template class Function<S_t>;
template class Function<D_t>;
template class Function<C_t>;
template class Function<Z_t>;

template class SimpleFunction<S_t>;
template class SimpleFunction<D_t>;
template class SimpleFunction<C_t>;
template class SimpleFunction<Z_t>;

template class BlockFunction<S_t>;
template class BlockFunction<D_t>;
template class BlockFunction<C_t>;
template class BlockFunction<Z_t>;

template class AssemblyFunction<S_t, SimpleFunction>;
template class AssemblyFunction<D_t, SimpleFunction>;
template class AssemblyFunction<C_t, SimpleFunction>;
template class AssemblyFunction<Z_t, SimpleFunction>;

template class AssemblyFunction<S_t, BlockFunction>;
template class AssemblyFunction<D_t, BlockFunction>;
template class AssemblyFunction<C_t, BlockFunction>;
template class AssemblyFunction<Z_t, BlockFunction>;
}  // end namespace hmat

