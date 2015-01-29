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

/*! \file
  \ingroup HMatrix
  \brief C interface to the HMatrix library.
*/
#ifndef _HMAT_H
#define _HMAT_H

#include <stdlib.h>
#include "hmat/config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief All the scalar types */
typedef enum {
  HMAT_SIMPLE_PRECISION=0,
  HMAT_DOUBLE_PRECISION=1,
  HMAT_SIMPLE_COMPLEX=2,
  HMAT_DOUBLE_COMPLEX=3,
} hmat_value_t;

/** Type of ClusterTree */
typedef enum {
  hmat_cluster_geometric,
  hmat_cluster_median,
  hmat_cluster_hybrid
} hmat_cluster_t;

/** Choice of the compression method */
typedef enum {
  hmat_compress_svd,
  hmat_compress_aca_full,
  hmat_compress_aca_partial,
  hmat_compress_aca_plus
} hmat_compress_t;

/*! \brief Coordinates of a degree of freedom

  \a globalIndex is the global index
*/
typedef struct {
  int globalIndex;
  double x;
  double y;
  double z;
} DofCoordinate;

/*! \brief Prepare block assembly.

 \param row_start starting row
 \param row_count number of rows
 \param col_start starting column
 \param col_count number of columns
 \param row_hmat2client renumbered rows -> global row indices mapping
 \param row_client2hmat global row indices -> renumbered rows mapping
 \param col_hmat2client renumbered cols -> global col indices mapping
 \param col_client2hmat global col indices -> renumbered cols mapping
 \param context user provided data
 \param **data opaque pointer to data related to this block. Must be
               freed using \a release_func().
 */
typedef void (*prepare_func)(int row_start,
    int row_count,
    int col_start,
    int col_count,
    int *row_hmat2client,
    int *row_client2hmat,
    int *col_hmat2client,
    int *col_client2hmat,
    void *context,
    void **data);

/*! \brief Compute a sub-block.

\warning The computation has to be prepared with \a prepare_func() first.

The supported cases are:
- Compute the full block
- Compute exactly one row
- Compute exactly one column

Regarding the indexing: For all indices in this function, the "C"
conventions are followed, ie indices start at 0, the lower bound is
included and the upper bound is excluded. The indexing is relative
to the block, that is the row i is the i-th row within the block.

\param v_data opaque pointer, as set by \a prepare_func()
\param row_start starting row
\param row_count number of rows
\param col_start starting column
\param col_count number of columns
\param block pointer to the output buffer. No padding is allowed,
that is the leading dimension of the buffer must be its real leading
dimension (1 for a row). Column-major order is assumed.
 */
typedef void (*compute_func)(void* v_data, int row_start, int row_count,
                             int col_start, int col_count, void* block);

/*! \brief Release opaque pointer allocated by \a prepare_func().

\param v_data opaque pointer
 */
typedef void (*release_func)(void* v_data);

/*! \brief Compute a single matrix term

\param user_context pointer to user data, see \a assemble_hmatrix_simple_interaction function
\param row row index
\param col column index
\param result address where result is stored; result is a pointer to a double for real matrices,
              and a pointer to a double complex for complex matrices.
 */
typedef void (*simple_interaction_compute_func)(void* user_context, int row, int col, void* result);

typedef struct hmat_cluster_tree_struct hmat_cluster_tree_t;

/*! \brief Create a ClusterTree from the DoFs coordinates.

  \param dls the DoFs coordinates
  \param n the number of DoFs
  \return an opaque pointer to a ClusterTree, or NULL in case of error.
*/
hmat_cluster_tree_t * hmat_create_cluster_tree(DofCoordinate* dls, int n);

void hmat_delete_cluster_tree(hmat_cluster_tree_t * tree);

/*!
 * Return the number of nodes in a cluster tree
 */
int hmat_tree_nodes_count(hmat_cluster_tree_t * tree);

/** Information on the HMatrix */
typedef struct
{
  /*! Number of elements */
  size_t compressed_size;

  /*! Number of elements if HMatrix was not compressed */
  size_t uncompressed_size;

  /*! Number of block cluster tree nodes in the HMatrix */
  int nr_block_clusters;
} hmat_info_t;

typedef struct hmat_matrix_struct hmat_matrix_t;

typedef struct
{
    /*! Create an empty (not assembled) HMatrix from 2 \a ClusterTree instances.

      The HMatrix is built on a row and a column tree, that are provided as
      arguments.

      \param stype the scalar type
      \param rows_tree a ClusterTree as returned by \a create_cluster_tree().
      \param cols_tree a ClusterTree as returned by \a create_cluster_tree().
      \return an opaque pointer to an HMatrix, or NULL in case of error.
    */
    hmat_matrix_t* (*create_empty_hmatrix)(void* rows_tree, void* cols_tree);

    /*! Assemble a HMatrix.

      \param hmatrix The matrix to be assembled.
      \param user_context The user context to pass to the prepare function.
      \param prepare The prepare function as given in \a HMatOperations.
      \param compute The compute function as given in \a HMatOperations.
      \param free_data The free function as given in \a HMatOperations.
      \param lower_symmetric 1 if the matrix is lower symmetric, 0 otherwise
      \return 0 for success.
    */
    int (*assemble)(hmat_matrix_t* hmatrix, void* user_context, prepare_func prepare,
                         compute_func compute, release_func free_data,
                         int lower_symmetric);

    /*! Assemble a HMatrix then factorize it.

      \param hmatrix The matrix to be assembled.
      \param user_context The user context to pass to the prepare function.
      \param prepare The prepare function as given in \a HMatOperations.
      \param compute The compute function as given in \a HMatOperations.
      \param free_data The free function as given in \a HMatOperations.
      \param lower_symmetric 1 if the matrix is lower symmetric, 0 otherwise
      \return 0 for success.
    */
    int (*assemble_factor)(hmat_matrix_t* hmatrix, void* user_context, prepare_func prepare,
                         compute_func compute, release_func free_data,
                         int lower_symmetric);

    /*! Assemble a HMatrix.  This is a simplified interface, a single function is provided to
      compute matrix terms.

      \param hmatrix The matrix to be assembled.
      \param user_context The user context to pass to the compute function.
      \param compute The compute function
      \param lower_symmetric 1 if the matrix is lower symmetric, 0 otherwise
      \return 0 for success.
    */
    int (*assemble_simple_interaction)(hmat_matrix_t* hmatrix,
                                            void* user_context,
                                            simple_interaction_compute_func compute,
                                            int lower_symmetric);
    /*! \brief Return a copy of a HMatrix.

      \param from_hmat the source matrix
      \return a copy of the matrix, or NULL
    */
    hmat_matrix_t* (*copy)(hmat_matrix_t* hmatrix);
    /*! \brief Compute the norm of a HMatrix.

      \param hmatrix the matrix of which to compute the norm
      \return the norm
    */
    double (*norm)(hmat_matrix_t* hmatrix);
    /*! \brief Factor a HMatrix in place.

      \param hmatrix the matrix to factor
      \return 0 for success
    */
    int (*factor)(hmat_matrix_t *hmatrix);
    /*! \brief Destroy a HMatrix.

      \param hmatrix the matrix to destroy
      \return 0
    */
    int (*destroy)(hmat_matrix_t* hmatrix);
    /*! \brief Solve A X = B, with X overwriting B.

      In this function, B is a H-matrix
hmat
      \param hmatrix
      \param hmatrixb
      \return 0 for success
    */
    int (*solve_mat)(hmat_matrix_t* hmatrix, hmat_matrix_t* hmatrixB);
    /*! \brief Solve A x = b, with x overwriting b.

      In this function, b is a multi-column vector, with nrhs RHS.

      \param hmatrix
      \param b
      \param nrhs
      \return 0 for success
    */
    int (*solve_systems)(hmat_matrix_t* hmatrix, void* b, int nrhs);
    /*! \brief Transpose an HMatrix in place.

       \return 0 for success.
     */
    int (*transpose)(hmat_matrix_t *hmatrix);
    /*! \brief A <- alpha * A

      \param alpha
      \param hmatrix
    */
    int (*scale)(void * alpha, hmat_matrix_t *hmatrix);
    /*! \brief C <- alpha * A * B + beta * C

      \param trans_a 'N' or 'T'
      \param trans_b 'N' or 'T'
      \param alpha
      \param hmatrix
      \param hmatrix_b
      \param beta
      \param hmatrix_c
      \return 0 for success
    */
    int (*gemm)(char trans_a, char trans_b, void* alpha, hmat_matrix_t* hmatrix,
                     hmat_matrix_t* hmatrix_b, void* beta, hmat_matrix_t* hmatrix_c);
    /*! \brief c <- alpha * A * b + beta * c

      \param trans_a 'N' or 'T'
      \param alpha
      \param hmatrix
      \param vec_b
      \param beta
      \param vec_c
      \param nrhs
      \return 0 for success
    */
    int (*gemv)(char trans_a, void* alpha, hmat_matrix_t* hmatrix, void* vec_b,
                     void* beta, void* vec_c, int nrhs);
    /*! \brief C <- alpha * A * B + beta * C


      In this version, a, c: FullMatrix, b: HMatrix.

      \param trans_a
      \param trans_b
      \param mc number of rows of C
      \param nc number of columns of C
      \param c
      \param alpha
      \param a
      \param hmat_b
      \param beta
      \return 0 for success
     */
    int (*full_gemm)(char trans_a, char trans_b, int mc, int nc, void* c,
                               void* alpha, void* a, hmat_matrix_t* hmat_b, void* beta);
    /*! \brief Initialize library
     */
    int (*init)();

    /*! \brief Do the cleanup
     */
    int (*finalize)();

    /*! \brief Get current informations
        \param hmatrix A hmatrix
        \param info A structure to fill with current informations
     */
    int (*hmat_get_info)(hmat_matrix_t *hmatrix, hmat_info_t* info);

    /** For internal use only */
    void * internal;

}  hmat_interface_t;

void hmat_init_default_interface(hmat_interface_t * i, hmat_value_t type);

/*! \brief Set the maximum amount of memory used for some operations.

  This setting is not universal since the HMatrix solver uses as
  much memory as needed to store the matrices. It is only used for
  handling the vectors in gemv() and solve().

  \param memory_in_bytes the amount of memory in bytes.
  \return 0 for success
 */
int hmatrix_set_max_memory(size_t memory_in_bytes);

/*! Get the value set by \a hmatrix_set_max_memory().
 */
size_t hmatrix_get_max_memory();

typedef struct
{
  /*! \brief Tolerance for the assembly. */
  double assemblyEpsilon;
  /*! \brief Tolerance for the recompression (using SVD) */
  double recompressionEpsilon;
  int compressionMethod;
  /** \f$\eta\f$ in the admissiblity condition for two clusters \f$\sigma\f$ and \f$\tau\f$:
      \f[
      \min(diam(\sigma), diam(\tau)) < \eta \cdot d(\sigma, \tau)
      \f]
   */
  double admissibilityFactor;
  /*! \brief Type of ClusterTree */
  hmat_cluster_t clustering;
  /*! \brief Maximum size of a leaf in a ClusterTree (and of a non-admissible block in an HMatrix) */
  int maxLeafSize;
  /*! \brief max(|L0|) */
  int maxParallelLeaves;
  /*! \brief Maximum size of an admissible block. Should be size_t ! */
  int elementsPerBlock;
  /*! \brief Use an LU decomposition */
  int useLu;
  /*! \brief Use an LDL^t decomposition if possible */
  int useLdlt;
  /*! \brief Coarsen the matrix structure after assembly. */
  int coarsening;
  /*! \brief Recompress the matrix after assembly. */
  int recompress;
  /*! \brief Validate the rk-matrices after compression */
  int validateCompression;
  /*! \brief For blocks above error threshold, re-run the compression algorithm */
  int validationReRun;
  /*! \brief For blocks above error threshold, dump the faulty block to disk */
  int validationDump;
  /*! \brief Error threshold for the compression validation */
  double validationErrorThreshold;
} hmat_settings_t;

/*! \brief Get current settings
    \param settings A structure to fill with current settings
 */
void hmat_get_parameters(hmat_settings_t * settings);
/*! \brief Set current settings

\param structure containing parameters
\return 1 on failure, 0 otherwise.
*/
int hmat_set_parameters(hmat_settings_t*);

/*!
 * \brief hmat_get_version
 * \return The version of this library
 */
const char * hmat_get_version();

#ifdef __cplusplus
}
#endif
#endif  /* _HMAT_H */
