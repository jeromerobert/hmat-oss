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

#ifndef _C_WRAPPING_HPP
#define _C_WRAPPING_HPP

#include <string>
#include <cstring>

#include "common/context.hpp"
#include "common/my_assert.h"
#include "full_matrix.hpp"
#include "h_matrix.hpp"
#include "uncompressed_values.hpp"
#include "serialization.hpp"
#include "hmat_cpp_interface.hpp"
#include "disable_threading.hpp"

namespace
{
template<typename T, template <typename> class E>
hmat_matrix_t * create_empty_hmatrix_admissibility(
  const hmat_cluster_tree_t* rows_tree,
  const hmat_cluster_tree_t* cols_tree, int lower_sym,
  hmat_admissibility_t* condition)
{
  DECLARE_CONTEXT;
    hmat::SymmetryFlag sym = lower_sym ? hmat::kLowerSymmetric : hmat::kNotSymmetric;
    hmat::IEngine<T>* engine = new E<T>();
    return (hmat_matrix_t*) new hmat::HMatInterface<T>(
            engine,
            reinterpret_cast<const hmat::ClusterTree*>(rows_tree),
            reinterpret_cast<const hmat::ClusterTree*>(cols_tree),
            sym, (hmat::AdmissibilityCondition*)condition);
}

template<typename T, template <typename> class E>
int assemble_generic(hmat_matrix_t* matrix, hmat_assemble_context_t * ctx) {
    DECLARE_CONTEXT;
    hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*)matrix;
    bool assembleOnly = ctx->factorization == hmat_factorization_none;
    hmat::SymmetryFlag sf = ctx->lower_symmetric ? hmat::kLowerSymmetric : hmat::kNotSymmetric;
    try {
        if (ctx->lower_symmetric) {
          HMAT_ASSERT(hmat->engine().hmat->rowsTree() == hmat->engine().hmat->colsTree());
        }
        HMAT_ASSERT_MSG(ctx->compression, "No compression algorithm defined in hmat_assemble_context_t");
        hmat::CompressionAlgorithm* compression = (hmat::CompressionAlgorithm*)ctx->compression;
        if(ctx->assembly != NULL) {
            HMAT_ASSERT(ctx->block_compute == NULL && ctx->advanced_compute == NULL && ctx->simple_compute == NULL);
            hmat::Assembly<T> * cppAssembly = (hmat::Assembly<T> *)ctx->assembly;
            hmat->assemble(*cppAssembly, sf, ctx->progress);
        } else if(ctx->block_compute != NULL || ctx->advanced_compute != NULL) {
            HMAT_ASSERT(ctx->simple_compute == NULL && ctx->assembly == NULL);
            HMAT_ASSERT(ctx->prepare != NULL);
            hmat::BlockFunction<T> blockFunction(hmat->rows(), hmat->cols(),
                ctx->user_context, ctx->prepare, ctx->block_compute, ctx->advanced_compute);
            hmat::AssemblyFunction<T, hmat::BlockFunction> * f =
                new hmat::AssemblyFunction<T, hmat::BlockFunction>(blockFunction, compression);
            hmat->assemble(*f, sf, true, ctx->progress, true);
        } else if(ctx->simple_compute != NULL) {
            HMAT_ASSERT(ctx->block_compute == NULL && ctx->advanced_compute == NULL && ctx->assembly == NULL);
            hmat::AssemblyFunction<T, hmat::SimpleFunction> * f =
                new hmat::AssemblyFunction<T, hmat::SimpleFunction>(
                hmat::SimpleFunction<T>(ctx->simple_compute, ctx->user_context), compression);
            hmat->assemble(*f, sf, true, ctx->progress, true);
        } else
          HMAT_ASSERT_MSG(0, "No valid assembly method in assemble_generic()");

        if(!assembleOnly)
            hmat->factorize(hmat::convert_int_to_factorization(ctx->factorization), ctx->progress);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}

template<typename T, template <typename> class E>
hmat_matrix_t* copy(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  return (hmat_matrix_t*) ((hmat::HMatInterface<T>*) holder)->copy();
}

template<typename T, template <typename> class E>
hmat_matrix_t* copy_struct(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  return (hmat_matrix_t*) ((hmat::HMatInterface<T>*) holder)->copy(true);
}

template<typename T, template <typename> class E>
int destroy(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  delete (hmat::HMatInterface<T>*)(holder);
  return 0;
}

template<typename T, template <typename> class E>
int destroy_child(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*) holder;
  hmat->setHMatrix();
  delete hmat;
  return 0;
}

template<typename T, template <typename> class E>
int inverse(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  try {
      ((hmat::HMatInterface<T>*) holder)->inverse();
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int factorize_generic(hmat_matrix_t* holder, hmat_factorization_context_t * ctx) {
    DECLARE_CONTEXT;
    hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*) holder;
    try {
        hmat->factorize(hmat::convert_int_to_factorization(ctx->factorization), ctx->progress);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}

template<typename T, template <typename> class E>
int factor(hmat_matrix_t* holder, hmat_factorization_t t) {
  DECLARE_CONTEXT;
    hmat_factorization_context_t ctx;
    hmat_factorization_context_init(&ctx);
    ctx.factorization = t;
    return factorize_generic<T, E>(holder, &ctx);
}

template<typename T, template <typename> class E>
int finalize() {
  DECLARE_CONTEXT;
  E<T>::finalize();
  return 0;
}

template<typename T, template <typename> class E>
int full_gemm(char transA, char transB, int mc, int nc, void* c,
                             void* alpha, void* a, hmat_matrix_t * holder, void* beta) {
  DECLARE_CONTEXT;

  try {
      const hmat::HMatInterface<T>* b = (hmat::HMatInterface<T>*)holder;
      hmat::ScalarArray<T> matC((T*)c, mc, nc);
      hmat::ScalarArray<T>* matA = NULL;
      const hmat::ClusterData* bDataRows = (transB == 'N' ? b->rows(): b->cols());
      const hmat::ClusterData* bDataCols = (transB == 'N' ? b->cols(): b->rows());
      hmat::reorderVector(&matC, bDataCols->indices(), 1);
      if (transA == 'N') {
        matA = new hmat::ScalarArray<T>((T*)a, mc, bDataRows->size());
        hmat::reorderVector(matA, bDataRows->indices(), 1);
      } else {
        matA = new hmat::ScalarArray<T>((T*)a, bDataRows->size(), mc);
        hmat::reorderVector(matA, bDataRows->indices(), 0);
      }
      hmat::HMatInterface<T>::gemm(matC, transA, transB, *((T*)alpha), *matA, *b, *((T*)beta));
      hmat::restoreVectorOrder(&matC, bDataCols->indices(), 1);
      if (transA == 'N') {
          hmat::restoreVectorOrder(matA, bDataRows->indices(), 1);
      } else {
          hmat::restoreVectorOrder(matA, bDataRows->indices(), 0);
      }
      delete matA;
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int gemm(char trans_a, char trans_b, const void *alpha, hmat_matrix_t * holder,
         hmat_matrix_t * holder_b, const void *beta, hmat_matrix_t * holder_c) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat_a = (hmat::HMatInterface<T>*)holder;
  hmat::HMatInterface<T>* hmat_b = (hmat::HMatInterface<T>*)holder_b;
  hmat::HMatInterface<T>* hmat_c = (hmat::HMatInterface<T>*)holder_c;
  try {
      hmat_c->gemm(trans_a, trans_b, *((T*)alpha), hmat_a, hmat_b, *((T*)beta));
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int axpy(void *a, hmat_matrix_t * x, hmat_matrix_t * y) {
  DECLARE_CONTEXT;
  DISABLE_THREADING_IN_BLOCK;
  hmat::HMatInterface<T>* hmat_x = reinterpret_cast<hmat::HMatInterface<T>*>(x);
  hmat::HMatInterface<T>* hmat_y = reinterpret_cast<hmat::HMatInterface<T>*>(y);
  try {
      hmat_y->engine().hmat->axpy(*((T*)a), hmat_x->engine().hmat);
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int gemv(char trans_a, void* alpha, hmat_matrix_t * holder, void* vec_b,
                   void* beta, void* vec_c, int nrhs) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*)holder;
  const hmat::ClusterData* bData = (trans_a == 'N' ? hmat->cols(): hmat->rows());
  const hmat::ClusterData* cData = (trans_a == 'N' ? hmat->rows(): hmat->cols());
  try {
      hmat::ScalarArray<T> mb((T*) vec_b, bData->size(), nrhs);
      hmat::ScalarArray<T> mc((T*) vec_c, cData->size(), nrhs);
      hmat::reorderVector(&mb, bData->indices(), 0);
      hmat::reorderVector(&mc, cData->indices(), 0);
      hmat->gemv(trans_a, *((T*)alpha), mb, *((T*)beta), mc);
      hmat::restoreVectorOrder(&mb, bData->indices(), 0);
      hmat::restoreVectorOrder(&mc, cData->indices(), 0);
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int gemm_scalar( char trans_a, void* alpha, hmat_matrix_t * holder, void* vec_b,
		 void* beta, void* vec_c, int nrhs ) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*)holder;
  const hmat::ClusterData* bData = (trans_a == 'N' ? hmat->cols(): hmat->rows());
  const hmat::ClusterData* cData = (trans_a == 'N' ? hmat->rows(): hmat->cols());
  try {
      hmat::ScalarArray<T> mb((T*) vec_b, bData->size(), nrhs);
      hmat::ScalarArray<T> mc((T*) vec_c, cData->size(), nrhs);

      hmat->gemm_scalar(trans_a, *((T*)alpha), mb, *((T*)beta), mc);
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

inline bool is_trans(char trans) {
    return trans == 'T' || trans == 'C';
}

inline bool is_conj(char trans) {
    return trans == 'J' || trans == 'C';
}

inline char switch_flag_trans(char trans) {
    switch (trans) {
    case 'N': return 'T';
    case 'T': return 'N';
    case 'C': return 'J';
    case 'J': return 'C';
    default: HMAT_ASSERT(false);
    }
}

inline char switch_flag_conj(char trans) {
    switch (trans) {
    case 'N': return 'J';
    case 'T': return 'C';
    case 'C': return 'T';
    case 'J': return 'N';
    default: HMAT_ASSERT(false);
    }
}

template<typename T, template <typename> class E>
int gemm_dense(char trans_b, char trans_x, char side, const void* alpha, hmat_matrix_t* holder,
               void* vec_x, const void* beta, void* vec_y, int nrhs) {
  char trans_y = 'N';
  T alphaT = *((T*)alpha);
  T betaT = *((T*)beta);
  alpha = &alphaT;
  beta = &betaT;
  if (side == 'R') {
      // Y <- alpha X * B + beta Y <=>  Y^t <- alpha B^t X^t + beta Y^t or Y^H <- bar(alpha) B^H * X^H + bar(beta) Y^H
      if (trans_b == 'C') {
          trans_b = 'N';
          trans_x = switch_flag_conj(switch_flag_trans(trans_x));
          trans_y = 'C';
          alphaT = hmat::conj(alphaT);
          betaT = hmat::conj(betaT);
      } else {
          trans_b = switch_flag_trans(trans_b);
          trans_x = switch_flag_trans(trans_x);
          trans_y = 'T';
      }
  }

  DECLARE_CONTEXT;
  DISABLE_THREADING_IN_BLOCK;

  // Now side='L': op(Y) <- alpha op(B) op(X) + beta op(Y)
  const hmat::HMatInterface<T>* b = (hmat::HMatInterface<T>*)holder;
  const hmat::IndexSet* bDataRows = !is_trans(trans_b) ? b->rows(): b->cols();
  const hmat::IndexSet* bDataCols = !is_trans(trans_b) ? b->cols(): b->rows();
  try {
      hmat::ScalarArray<T>* mx = !is_trans(trans_x) ?
          new hmat::ScalarArray<T>((T*) vec_x, bDataCols->size(), nrhs) :
          new hmat::ScalarArray<T>((T*) vec_x, nrhs, bDataCols->size());
      hmat::ScalarArray<T>* my =  !is_trans(trans_y) ?
          new hmat::ScalarArray<T>((T*) vec_y, bDataRows->size(), nrhs) :
          new hmat::ScalarArray<T>((T*) vec_y, nrhs, bDataRows->size());

      // Apply transformations on x and y
      if (is_trans(trans_x))
          mx->transpose();
      if (is_conj(trans_x))
          mx->conjugate();
      if (is_trans(trans_y))
          my->transpose();
      if (is_conj(trans_y))
          my->conjugate();

      b->gemv(trans_b, *((T*)alpha), *mx, *((T*)beta), *my);

      // Apply inverse transformations on x and y
      if (is_trans(trans_x))
          mx->transpose();
      if (is_trans(trans_y))
          my->transpose();
      if (is_conj(trans_y))
          my->conjugate();

      delete mx;
      delete my;
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int trsm( char side, char uplo, char transa, char diag, int m, int n,
	  void *alpha, hmat_matrix_t *A, int is_b_hmat, void *B )
{
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmatA = (hmat::HMatInterface<T>*)A;

  try {
      if ( is_b_hmat ) {
          hmat::HMatInterface<T>* hmatB = (hmat::HMatInterface<T>*)B;
          hmatA->trsm( side, uplo, transa, diag, *((T*)alpha), hmatB );
      }
      else {
          bool isleft = (side == 'l') || (side == 'L');
          hmat::ScalarArray<T> mB( (T*)B, (isleft ? m : n), (isleft ? n : m ) );
          hmatA->trsm( side, uplo, transa, diag, *((T*)alpha), mB );
      }
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int add_identity(hmat_matrix_t* holder, void *alpha) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*)holder;
  try {
      hmat->addIdentity(*((T*)alpha));
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int init() {
  DECLARE_CONTEXT;
  return E<T>::init();
}

template<typename T, template <typename> class E>
double norm(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  return reinterpret_cast<hmat::HMatInterface<T>*>(holder)->engine().norm();
}

template<typename T, template <typename> class E>
int logdet(hmat_matrix_t* holder, double * result) {
  DECLARE_CONTEXT;
  try {
    auto r = reinterpret_cast<hmat::HMatInterface<T>*>(holder)->engine().logdet();
    *reinterpret_cast<typename hmat::Types<T>::dp*>(result)=r;
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int scale(void *alpha, hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  try {
      ((hmat::HMatInterface<T>*)holder)->scale(*((T*)alpha));
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int truncate(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  try {
      ((hmat::HMatInterface<T>*)holder)->truncate();
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int vector_reorder(void* vec_b, const hmat_cluster_tree_t *rows_ct, int rows, const hmat_cluster_tree_t *cols_ct, int cols) {
  DECLARE_CONTEXT;
  try {
      HMAT_ASSERT_MSG(rows_ct != NULL || rows != 0, "either row cluster tree or rows must be non null");
      HMAT_ASSERT_MSG(cols_ct != NULL || cols != 0, "either col cluster tree or cols must be non null");
      const hmat::ClusterTree *clusterTreeRows = reinterpret_cast<const hmat::ClusterTree*>(rows_ct);
      const hmat::ClusterTree *clusterTreeCols = reinterpret_cast<const hmat::ClusterTree*>(cols_ct);
      int nrows = clusterTreeRows == NULL ? rows : clusterTreeRows->data.size();
      int ncols = clusterTreeCols == NULL ? cols : clusterTreeCols->data.size();
      hmat::ScalarArray<T> mb((T*) vec_b, nrows, ncols);
      if (clusterTreeRows) {
        hmat::reorderVector(&mb, clusterTreeRows->data.indices(), 0);
      }
      if (clusterTreeCols) {
        hmat::reorderVector(&mb, clusterTreeCols->data.indices(), 1);
      }
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int vector_restore(void* vec_b, const hmat_cluster_tree_t *rows_ct, int rows, const hmat_cluster_tree_t *cols_ct, int cols) {
  DECLARE_CONTEXT;
  try {
      HMAT_ASSERT_MSG(rows_ct != NULL || rows != 0, "either row cluster tree or rows must be non null");
      HMAT_ASSERT_MSG(cols_ct != NULL || cols != 0, "either col cluster tree or cols must be non null");
      const hmat::ClusterTree *clusterTreeRows = reinterpret_cast<const hmat::ClusterTree*>(rows_ct);
      const hmat::ClusterTree *clusterTreeCols = reinterpret_cast<const hmat::ClusterTree*>(cols_ct);
      int nrows = clusterTreeRows == NULL ? rows : clusterTreeRows->data.size();
      int ncols = clusterTreeCols == NULL ? cols : clusterTreeCols->data.size();
      hmat::ScalarArray<T> mb((T*) vec_b, nrows, ncols);
      if (clusterTreeRows) {
        hmat::restoreVectorOrder(&mb, clusterTreeRows->data.indices(), 0);
      }
      if (clusterTreeCols) {
        hmat::restoreVectorOrder(&mb, clusterTreeCols->data.indices(), 1);
      }
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int solve_generic(hmat_matrix_t* holder, const struct hmat_solve_context_t* context) {
  HMAT_ASSERT_MSG(!(context->lower && context->upper),
                  "lower and upper cannot both be true in hmat_solve_context_t");
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*)holder;
  // Solve functions do not receive a progress bar as argument, they use the one
  // stored in HMatInterface.  Temporarily modify progress bar.
  hmat_progress_t * saved_progress = hmat->engine().progress();
  hmat->progress(context->progress);
  try {
      if (context->nr_rhs == 0) {
          // B is an HMatrix
          hmat::HMatInterface<T>* hmatB = (hmat::HMatInterface<T>*)context->values;
          if (context->lower)
              hmat->solveLower(*hmatB, false);
          else if (context->upper)
              hmat->solveLower(*hmatB, true);
          else
              hmat->solve(*hmatB);
      } else {
          hmat::ScalarArray<T> mb((T*)context->values, hmat->cols()->size(), context->nr_rhs);
          if (!context->no_permutation)
              hmat::reorderVector<T>(&mb, context->upper ? hmat->rows()->indices() : hmat->cols()->indices(), 0);
          if (context->lower)
              hmat->solveLower(mb, false);
          else if (context->upper)
              hmat->solveLower(mb, true);
          else
              hmat->solve(mb);
          if (!context->no_permutation)
              hmat::restoreVectorOrder<T>(&mb, context->upper ? hmat->rows()->indices() : hmat->cols()->indices(), 0);
      }
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      // Restore progress bar
      hmat->progress(saved_progress);
      return 1;
  }
  // Restore progress bar
  hmat->progress(saved_progress);
  return 0;
}

template<typename T, template <typename> class E>
int solve_mat(hmat_matrix_t* holder, hmat_matrix_t* hmatB) {
  struct hmat_solve_context_t ctx;
  hmat_solve_context_init(&ctx);
  ctx.values = hmatB;
  return solve_generic<T, E>(holder, &ctx);
}

template<typename T, template <typename> class E>
int solve_systems(hmat_matrix_t* holder, void* b, int nrhs) {
  struct hmat_solve_context_t ctx;
  hmat_solve_context_init(&ctx);
  ctx.values = b;
  ctx.nr_rhs = nrhs;
  return solve_generic<T, E>(holder, &ctx);
}

template<typename T, template <typename> class E>
int solve_dense(hmat_matrix_t* holder, void* b, int nrhs) {
  struct hmat_solve_context_t ctx;
  hmat_solve_context_init(&ctx);
  ctx.values = b;
  ctx.nr_rhs = nrhs;
  ctx.no_permutation = 1;
  return solve_generic<T, E>(holder, &ctx);
}

template<typename T, template <typename> class E>
int transpose(hmat_matrix_t* hmat) {
  DECLARE_CONTEXT;
  try {
      ((hmat::HMatInterface<T>*)hmat)->transpose();
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int hmat_get_info(hmat_matrix_t* holder, hmat_info_t* info) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*) holder;
  try {
      hmat->info(*info);
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int hmat_dump_info(hmat_matrix_t* holder, char* prefix) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*) holder;
  try {
      std::string filejson(prefix);
      filejson += ".json";
      hmat->dumpTreeToFile( filejson );
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int get_cluster_trees(hmat_matrix_t* holder, const hmat_cluster_tree_t ** rows, const hmat_cluster_tree_t ** cols) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*) holder;
  try {
      if (rows)
        *rows = static_cast<const hmat_cluster_tree_t*>(static_cast<const void*>(hmat->engine().hmat->rowsTree()));
      if (cols)
        *cols = static_cast<const hmat_cluster_tree_t*>(static_cast<const void*>(hmat->engine().hmat->colsTree()));
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int set_cluster_trees(hmat_matrix_t* holder, const hmat_cluster_tree_t * rows, const hmat_cluster_tree_t * cols) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*) holder;
  try {
      hmat->engine().hmat->setClusterTrees(
        reinterpret_cast<const hmat::ClusterTree*>(rows),
        reinterpret_cast<const hmat::ClusterTree*>(cols));
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
void own_cluster_trees(hmat_matrix_t* holder, int owns_row, int owns_col)
{
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*) holder;
  hmat->engine().hmat->ownClusterTrees(owns_row != 0, owns_col != 0);
}

template<typename T, template <typename> class E>
void set_low_rank_epsilon(hmat_matrix_t* holder, double epsilon)
{
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*) holder;
  hmat->engine().hmat->lowRankEpsilon(epsilon);
}

template<typename T, template <typename> class E>
int extract_diagonal(hmat_matrix_t* holder, void* diag, int size)
{
  DECLARE_CONTEXT;
  (void)size; //for API compatibility
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*) holder;
  try {
      hmat->engine().hmat->extractDiagonal(static_cast<T*>(diag));
      hmat::ScalarArray<T> permutedDiagonal(static_cast<T*>(diag), hmat->cols()->size(), 1);
      hmat::restoreVectorOrder(&permutedDiagonal, hmat->cols()->indices(), 0);
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int extract_diagonal_block(hmat_matrix_t* holder, int components, void* diag)
{
  DECLARE_CONTEXT;
  hmat::HMatInterface<T>* hmat = (hmat::HMatInterface<T>*) holder;
  try {
      hmat->engine().hmat->extractDiagonal(static_cast<T*>(diag), components);
      hmat::ScalarArray<T> permutedDiagonal(static_cast<T*>(diag), components, hmat->cols()->size());
      hmat::restoreVectorOrder(&permutedDiagonal, hmat->cols()->indices(), 1);
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  return 0;
}

template<typename T, template <typename> class E>
int solve_lower_triangular(hmat_matrix_t* holder, int transpose, void* b, int nrhs)
{
  struct hmat_solve_context_t ctx;
  hmat_solve_context_init(&ctx);
  ctx.values = b;
  ctx.nr_rhs = nrhs;
  if (transpose)
     ctx.upper = 1;
  else
     ctx.lower = 1;
  return solve_generic<T, E>(holder, &ctx);
}

template<typename T, template <typename> class E>
int solve_lower_triangular_dense(hmat_matrix_t* holder, int transpose, void* b, int nrhs)
{
  struct hmat_solve_context_t ctx;
  hmat_solve_context_init(&ctx);
  ctx.values = b;
  ctx.nr_rhs = nrhs;
  ctx.no_permutation = 1;
  if (transpose)
     ctx.upper = 1;
  else
     ctx.lower = 1;
  return solve_generic<T, E>(holder, &ctx);
}

template <typename T, template <typename> class E>
hmat_matrix_t *get_child( hmat_matrix_t *hmatrix, int i, int j ) {
    DECLARE_CONTEXT;
    hmat::HMatInterface<T> *hmat = (hmat::HMatInterface<T> *)hmatrix;

    hmat::HMatrix<T> *m = hmat->get( i, j );
    hmat::IEngine<T>* engine = new E<T>();
    hmat::HMatInterface<T> *r = new hmat::HMatInterface<T>( engine, m, hmat->factorization() );
    return (hmat_matrix_t*) r;
}

template <typename T, template <typename> class E>
int set_diagonal_children(hmat_matrix_t* holder, hmat_matrix_t** holder_children) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T> *hmat = reinterpret_cast<hmat::HMatInterface<T>*>(holder);
  hmat::HMatrix<T>* m = hmat->engine().hmat;
  try {
    HMAT_ASSERT_MSG(m->nrChildRow() == m->nrChildCol(), "Cannot call set_diagonal_children on non symmetric matrix");
    for (int i = 0; i < m->nrChildRow(); ++i, ++holder_children) {
      hmat::HMatrix<T>* currentChild = m->get(i, i);
      if (currentChild) {
        // Detach it; there may be memory leaks
        currentChild->father = NULL;
        currentChild->ownClusterTrees(false, false);
        delete currentChild;
      }
      hmat::HMatrix<T>* newChild = reinterpret_cast<hmat::HMatInterface<T>*>(*holder_children)->engine().hmat;
      // insertChild takes care of father and depth
      m->insertChild(i, i, newChild);
    }
  } catch (const std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
  }
  // Call setClusterTrees to overwrite children ClusterTree
  m->setClusterTrees(m->rowsTree(), m->colsTree());
  return 0;
}

template <typename T, template <typename> class E>
int get_block(struct hmat_get_values_context_t *ctx) {
  DECLARE_CONTEXT;
  DISABLE_THREADING_IN_BLOCK;
    hmat::HMatInterface<T> *hmat = (hmat::HMatInterface<T> *)ctx->matrix;
    try {
        HMAT_ASSERT_MSG(hmat->factorization() != hmat::Factorization::HODLRSYM &&
                        hmat->factorization() != hmat::Factorization::HODLR,
                        "Unsupported operation");
        hmat::IndexSet rows(ctx->row_offset, ctx->row_size);
        hmat::IndexSet cols(ctx->col_offset, ctx->col_size);
        typename E<T>::UncompressedBlock view;
        const E<T>& engine = dynamic_cast<const E<T>&>(hmat->engine());
        view.uncompress(engine.getHandle(), rows, cols, (T*)ctx->values);
        hmat::HMatrix<T>* compressed = hmat->engine().hmat;
        // Symmetrize values when requesting a full symmetric matrix
        if (compressed->isLower &&
            ctx->row_offset == 0 && ctx->col_offset == 0 &&
            ctx->row_size == compressed->rows()->size() && ctx->col_size == compressed->cols()->size())
        {
          T* ptr = static_cast<T*>(ctx->values);
          for (int i = 0; i < ctx->row_size; i++) {
            for (int j = i + 1; j < ctx->col_size; j++) {
              ptr[j*ctx->row_size + i] = ptr[i*ctx->row_size + j];
            }
          }
        }
        if (ctx->renumber_rows)
            view.renumberRows();
        ctx->col_indices = view.colsNumbering();
        ctx->row_indices= view.rowsNumbering();
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}

template <typename T, template <typename> class E>
int get_values(struct hmat_get_values_context_t *ctx) {
  DECLARE_CONTEXT;
    // No need to call DISABLE_THREADING_IN_BLOCK here, there is no BLAS call
    hmat::HMatInterface<T> *hmat = (hmat::HMatInterface<T> *)ctx->matrix;
    try {
        HMAT_ASSERT_MSG(hmat->factorization() != hmat::Factorization::HODLRSYM &&
                        hmat->factorization() != hmat::Factorization::HODLR,
                        "Unsupported operation");
        typename E<T>::UncompressedValues view;
        const E<T>& engine = reinterpret_cast<const E<T>&>(hmat->engine());
        view.uncompress(engine.getHandle(),
                        ctx->row_indices, ctx->row_size,
                        ctx->col_indices, ctx->col_size,
                        (T*)ctx->values);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}

template <typename T, template <typename> class E>
int walk(hmat_matrix_t* holder, hmat_procedure_t* proc) {
  DECLARE_CONTEXT;
    hmat::HMatInterface<T> *hmat = (hmat::HMatInterface<T> *) holder;
    try {
        hmat::TreeProcedure<hmat::HMatrix<T> > *functor = (hmat::TreeProcedure<hmat::HMatrix<T> > *) proc->internal;
        hmat->walk(functor);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}

template <typename T, template <typename> class E>
int apply_on_leaf(hmat_matrix_t* holder, const hmat_leaf_procedure_t* proc) {
  DECLARE_CONTEXT;
    hmat::HMatInterface<T> *hmat = (hmat::HMatInterface<T> *) holder;
    try {
        const hmat::LeafProcedure<hmat::HMatrix<T> > *functor = static_cast<const hmat::LeafProcedure<hmat::HMatrix<T> > *>(proc->internal);
        hmat->apply_on_leaf(*functor);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}

template <typename T, template <typename> class E>
hmat_matrix_t * read_struct(hmat_iostream readfunc, void * user_data) {
    hmat::MatrixStructUnmarshaller<T> unmarshaller(&hmat::HMatSettings::getInstance(), readfunc, user_data);
    hmat::HMatrix<T> * m = unmarshaller.read();
    E<T>* engine = new E<T>();
    hmat::HMatInterface<T> * r = new hmat::HMatInterface<T>(engine, m, unmarshaller.factorization());
    return (hmat_matrix_t*) r;
}

template <typename T, template <typename> class E>
void read_data(hmat_matrix_t * matrix, hmat_iostream readfunc, void * user_data) {
    hmat::HMatInterface<T> * hmi = (hmat::HMatInterface<T> *) matrix;
    hmat::MatrixDataUnmarshaller<T>(readfunc, user_data).read(hmi->engine().hmat);
}

template <typename T, template <typename> class E>
void write_struct(hmat_matrix_t* matrix, hmat_iostream writefunc, void * user_data) {
    hmat::HMatInterface<T> * hmi = (hmat::HMatInterface<T> *) matrix;
    hmat::MatrixStructMarshaller<T>(writefunc, user_data).write(
        hmi->engine().hmat, hmi->factorization());
}

template <typename T, template <typename> class E>
void write_data(hmat_matrix_t* matrix, hmat_iostream writefunc, void * user_data) {
    hmat::HMatInterface<T> * hmi = (hmat::HMatInterface<T> *) matrix;
    hmat::MatrixDataMarshaller<T>(writefunc, user_data).write(hmi->engine().hmat);
}

template <typename T>
void set_progressbar(hmat_matrix_t * matrix, hmat_progress_t * progress) {
    reinterpret_cast<hmat::HMatInterface<T> *>(matrix)->progress(progress);
}

}  // end anonymous namespace

namespace hmat {

template<typename T, template <typename> class E>
static void createCInterface(hmat_interface_t * i)
{
  DECLARE_CONTEXT;
    i->copy = copy<T, E>;
    i->copy_struct = copy_struct<T, E>;
    i->create_empty_hmatrix_admissibility = create_empty_hmatrix_admissibility<T, E>;
    i->destroy = destroy<T, E>;
    i->get_child = get_child<T, E>;
    i->destroy_child = destroy_child<T, E>;
    i->inverse = inverse<T, E>;
    i->finalize = finalize<T, E>;
    i->full_gemm = full_gemm<T, E>;
    i->gemm = gemm<T, E>;
    i->gemv = gemv<T, E>;
    i->gemm_scalar = gemm_scalar<T, E>;
    i->add_identity = add_identity<T, E>;
    i->init = init<T, E>;
    i->norm = norm<T, E>;
    i->logdet = logdet<T, E>;
    i->scale = scale<T, E>;
    i->solve_generic = solve_generic<T, E>;
    i->solve_mat = solve_mat<T, E>;
    i->solve_systems = solve_systems<T, E>;
    i->solve_dense = solve_dense<T, E>;
    i->transpose = transpose<T, E>;
    i->internal = NULL;
    i->get_info  = hmat_get_info<T, E>;
    i->dump_info = hmat_dump_info<T, E>;
    i->get_cluster_trees = get_cluster_trees<T, E>;
    i->set_cluster_trees = set_cluster_trees<T, E>;
    i->own_cluster_trees = own_cluster_trees<T, E>;
    i->set_low_rank_epsilon = set_low_rank_epsilon<T, E>;
    i->extract_diagonal = extract_diagonal<T, E>;
    i->extract_diagonal_block = extract_diagonal_block<T, E>;
    i->solve_lower_triangular = solve_lower_triangular<T, E>;
    i->solve_lower_triangular_dense = solve_lower_triangular_dense<T, E>;
    i->assemble_generic = assemble_generic<T, E>;
    i->factorize_generic = factorize_generic<T, E>;
    i->get_values = get_values<T, E>;
    i->get_block = get_block<T, E>;
    i->walk = walk<T, E>;
    i->read_struct = read_struct<T, E>;
    i->write_struct = write_struct<T, E>;
    i->write_data = write_data<T, E>;
    i->read_data = read_data<T, E>;
    i->apply_on_leaf = apply_on_leaf<T, E>;
    i->axpy = axpy<T, E>;
    i->trsm = trsm<T, E>;
    i->truncate = truncate<T, E>;
    i->set_progressbar = set_progressbar<T>;
    i->gemm_dense = gemm_dense<T, E>;
    i->vector_reorder = vector_reorder<T, E>;
    i->vector_restore = vector_restore<T, E>;
    i->set_diagonal_children = set_diagonal_children<T, E>;
}

}  // end namespace hmat

#endif  // _C_WRAPPING_HPP
