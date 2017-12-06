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

#if defined _WIN32
# define HMAT_HELPER_DLL_IMPORT __declspec(dllimport)
# define HMAT_HELPER_DLL_EXPORT __declspec(dllexport)
#else
# define HMAT_HELPER_DLL_IMPORT
# define HMAT_HELPER_DLL_EXPORT
#endif

#ifdef HMAT_STATIC /* We are building hmat as a static library */
# define HMAT_API
#elif defined HMAT_DLL_EXPORTS /* We are building hmat as a shared library */
# define HMAT_API HMAT_HELPER_DLL_EXPORT
#else              /* We are using hmat library */
# define HMAT_API HMAT_HELPER_DLL_IMPORT
#endif /* !HMAT_STATIC */

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief All the scalar types */
typedef enum {
  HMAT_SIMPLE_PRECISION=0,
  HMAT_DOUBLE_PRECISION=1,
  HMAT_SIMPLE_COMPLEX=2,
  HMAT_DOUBLE_COMPLEX=3
} hmat_value_t;

/** Choice of the compression method */
typedef enum {
  hmat_compress_svd,
  hmat_compress_aca_full,
  hmat_compress_aca_partial,
  hmat_compress_aca_plus
} hmat_compress_t;

typedef enum {
    hmat_block_full,
    hmat_block_null,
    hmat_block_sparse
} hmat_block_t;

typedef enum {
    hmat_factorization_none = -1,
    hmat_factorization_lu,
    hmat_factorization_ldlt,
    hmat_factorization_llt
} hmat_factorization_t;

/* -1 seems to be a good portable alternative to SIZE_T_MAX (=18446744073709551615 on linux 64 bits) */
/** The value of hmat_block_info_t.needed_memory when unset */
#define HMAT_NEEDED_MEMORY_UNSET ((size_t)-1)

typedef struct hmat_block_info_struct {
    hmat_block_t block_type;
    /**
     * user data to pass from prepare function to compute function.
     * Will also contains user data required to execute is_guaranteed_null_row and
     * is_guaranteed_null_col
     */
    void * user_data;
    /*! \brief Function provided by the user in hmat_prepare_func_t and used to release user_data */
    void (*release_user_data)(void* user_data);
    /*! \brief Function provided by the user in hmat_prepare_func_t and used to quickly detect null rows

      It should be able to detect null rows very quickly, possibly without to compute them,
      at the cost of possibly missing some null rows. It should return '\1' for a null row,
      '\0' means that the row can be anything (null or non-null).
      The block_row_offset argument is the row index within this block.

      Note 1: We have to use hmat_block_info_struct in this definition, but
              user prototype should be
        char is_guaranteed_null_row(const hmat_block_info_t * block_info, int i, int stratum);

      Note 2: It is convenient to store informations about null rows and columns for each stratum
              inside the prepare function, so that this function only performs a lookup to quickly
              determine whether this row is null or not.
    */
    char (*is_guaranteed_null_row)(const struct hmat_block_info_struct * block_info, int block_row_offset, int stratum);
    /*! \brief Function provided by the user in hmat_prepare_func_t and used to quickly detect null columns (equivalent to is_guaranteed_null_row) */
    char (*is_guaranteed_null_col)(const struct hmat_block_info_struct * block_info, int block_col_offset, int stratum);
    /**
     * The memory needed to assemble the block.
     * When set to HMAT_NEEDED_MEMORY_UNSET the hmat_prepare_func_t should reset it
     * to the expected value and return. The hmat_prepare_func_t will then
     * be called a second time to run the actual preparation.
     */
    size_t needed_memory;
    /** the number of strata in the block */
    int number_of_strata;
} hmat_block_info_t;

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
 */
typedef void (*hmat_prepare_func_t)(int row_start,
    int row_count,
    int col_start,
    int col_count,
    int *row_hmat2client,
    int *row_client2hmat,
    int *col_hmat2client,
    int *col_client2hmat,
    void *context,
    hmat_block_info_t * block_info);

/*! \brief Compute a sub-block.

\warning The computation has to be prepared with \a prepare_func() first.

The supported cases are:
- Compute the full block
- Compute exactly one row
- Compute exactly one column

Regarding the indexing: For all indices in this function, the "C"
conventions are followed, ie indices start at 0, the lower bound is
included and the upper bound is excluded. In contrast with
hmat_prepare_func_t, the indexing is relative
to the block, that is the row i is the i-th row within the block.

\param v_data opaque pointer, as set by \a prepare_func() in field user_data of hmat_block_info_t
\param block_row_start starting row in the block
\param block_row_count number of rows to be computed
\param block_col_start starting column in the block
\param block_col_count number of columns to be computed
\param block pointer to the output buffer. No padding is allowed,
that is the leading dimension of the buffer must be its real leading
dimension (1 for a row). Column-major order is assumed.
 */
typedef void (*hmat_compute_func_t)(void* v_data, int block_row_start, int block_row_count,
                             int block_col_start, int block_col_count, void* block);

/*! \brief Compute a single matrix term

\param user_context pointer to user data, see \a assemble_hmatrix_simple_interaction function
\param row row index
\param col column index
\param result address where result is stored; result is a pointer to a double for real matrices,
              and a pointer to a double complex for complex matrices.
 */
typedef void (*hmat_interaction_func_t)(void* user_context, int row, int col, void* result);

typedef struct hmat_clustering_algorithm hmat_clustering_algorithm_t;

/* Opaque pointer */
typedef struct hmat_cluster_tree_struct hmat_cluster_tree_t;

/* Median clustering */
HMAT_API hmat_clustering_algorithm_t* hmat_create_clustering_median(void);
/* Geometric clustering */
HMAT_API hmat_clustering_algorithm_t* hmat_create_clustering_geometric(void);
/* Hybrid clustering */
HMAT_API hmat_clustering_algorithm_t* hmat_create_clustering_hybrid(void);
/* Create a new clustering algorithm by setting the maximum number of degrees of freedom in a leaf */
HMAT_API hmat_clustering_algorithm_t* hmat_create_clustering_max_dof(const hmat_clustering_algorithm_t* algo, int max_dof);

/**
 * Create a new clustering algorithm which put large span apart and
 * delegate to an other algorithm.
 * @param algo the delegate algorithm
 * @param ratio ratio between the number of DOF in a cluster and
 * a span size so a DOF is concidered as large
 */
HMAT_API hmat_clustering_algorithm_t* hmat_create_clustering_span(
        const hmat_clustering_algorithm_t* algo, double ratio);

/* Create a new clustering algorithm (for tests purpose only) */
HMAT_API hmat_clustering_algorithm_t* hmat_create_void_clustering(const hmat_clustering_algorithm_t* algo);
/* Create a new clustering algorithm (for tests purpose only) */
HMAT_API hmat_clustering_algorithm_t* hmat_create_shuffle_clustering(const hmat_clustering_algorithm_t* algo, int from_divider, int to_divider);
/* Delete clustering algorithm */
HMAT_API void hmat_delete_clustering(hmat_clustering_algorithm_t *algo);
/* Set the clustering divider parameter */
HMAT_API void hmat_set_clustering_divider(hmat_clustering_algorithm_t* algo, int divider);

/*! \brief Create a ClusterTree from the DoFs coordinates.

  \param coord DoFs coordinates
  \param dimension spatial dimension
  \param size number of DoFs
  \param algo pointer to clustering algorithm
  \return an opaque pointer to a ClusterTree, or NULL in case of error.
*/
HMAT_API hmat_cluster_tree_t * hmat_create_cluster_tree(double* coord, int dimension, int size, hmat_clustering_algorithm_t* algo);

/* Opaque pointer */
typedef struct hmat_cluster_tree_builder hmat_cluster_tree_builder_t;

HMAT_API hmat_cluster_tree_builder_t* hmat_create_cluster_tree_builder(const hmat_clustering_algorithm_t* algo);

/* Specify an algorithm for nodes at given depth and below */
HMAT_API void hmat_cluster_tree_builder_add_algorithm(hmat_cluster_tree_builder_t* ctb, int level, const hmat_clustering_algorithm_t* algo);

HMAT_API void hmat_delete_cluster_tree_builder(hmat_cluster_tree_builder_t* ctb);

/*! \brief Create a ClusterTree from the DoFs coordinates.

  \param coord DoFs coordinates
  \param dimension spatial dimension
  \param size number of DoFs
  \param ctb pointer to an opaque ClusterTreeBuilder
  \return an opaque pointer to a ClusterTree, or NULL in case of error.
*/
HMAT_API hmat_cluster_tree_t * hmat_create_cluster_tree_from_builder(double* coord, int dimension, int size, const hmat_cluster_tree_builder_t* ctb);

HMAT_API void hmat_delete_cluster_tree(hmat_cluster_tree_t * tree);

HMAT_API hmat_cluster_tree_t * hmat_copy_cluster_tree(hmat_cluster_tree_t * tree);

struct hmat_cluster_tree_create_context_t {
    /** Spatial dimension */
    unsigned dimension;
    unsigned number_of_points;
    double * coordinates;
    unsigned number_of_dof;
    /**
     * Offset of each span in the spans array.
     * if span_offset is NULL each span is considered as a single point.
     * if not NULL the size of this array must be number_of_dof.
     * span_offsets[0] is the offset of the dof 1 (not 0).
     * span_offset[number_of_dof-1] is the length of the spans array.
     */
    unsigned * span_offsets;
    /** The id of the points in each span */
    unsigned * spans;
    /** pointer to an opaque ClusterTreeBuilder */
    const hmat_cluster_tree_builder_t* builder;
};

HMAT_API hmat_cluster_tree_t * hmat_create_cluster_tree_generic(struct hmat_cluster_tree_create_context_t*);

/*!
 * Return the number of nodes in a cluster tree
 */
HMAT_API int hmat_tree_nodes_count(hmat_cluster_tree_t * tree);

/** Information on a cluster tree */
typedef struct
{
  /* ! Spatial dimension of degrees of freedom */
  int spatial_dimension;

  /* ! Number of degrees of freedom */
  int dimension;

  /* ! Number of tree nodes */
  size_t nr_tree_nodes;
} hmat_cluster_info_t;

HMAT_API int hmat_cluster_get_info(hmat_cluster_tree_t *tree, hmat_cluster_info_t* info);

typedef struct {
    /** eta for Hackbusch condition */
    double eta;
    /**
     * Maximum number of element (m*n) of an admissible block when
     * assembling the full block is needed
     */
    size_t max_svd_elements;
    /**
     * Maximum number of element (m*n) of an admissible block when
     * only assembling rows or cols is needed
     */
    size_t max_aca_elements;
} hmat_admissibility_param_t;

/** Init an hmat_admissibility_param structure with default values */
HMAT_API void hmat_init_admissibility_param(hmat_admissibility_param_t *);

/* Opaque pointer */
typedef struct hmat_admissibility_condition hmat_admissibility_t;

/** Create an admissibility condition from parameters */
HMAT_API hmat_admissibility_t* hmat_create_admissibility(hmat_admissibility_param_t *);

/* Create a standard (Hackbusch) admissibility condition, with a given eta */
HMAT_API hmat_admissibility_t* hmat_create_admissibility_standard(double eta);

/**
 * @brief Create an admissibility condiction which set all blocks as admissible
 * @param max_block_size The maximum acceptable block size in number of values (rows * cols)
 * @param min_nr_block The minimum acceptable number of blocks created with this condition
 * @param split_rows Tel whether or not to split rows
 * @param split_cols Tel whether or not to split cols
 */
HMAT_API hmat_admissibility_t* hmat_create_admissibility_always(
        size_t max_size, unsigned int min_block, int split_rows, int split_cols);

/**
 * @brief Create an admissibility condiction which set all blocks as full
 * @param max_block_size The maximum acceptable block size in number of values (rows * cols)
 * @param min_nr_block The minimum acceptable number of blocks created with this condition
 * @param split_rows Tel whether or not to split rows
 * @param split_cols Tel whether or not to split cols
 */
HMAT_API hmat_admissibility_t* hmat_create_admissibility_never(
        size_t max_size, unsigned int min_block, int split_rows, int split_cols);

/* Delete admissibility condition */
HMAT_API void hmat_delete_admissibility(hmat_admissibility_t * cond);

/** Information on the HMatrix */
typedef struct
{
  /*! Number of allocated zeros & nnz */
  size_t full_fit_nnz;
  size_t rk_fit_nnz;
  size_t rk_square_fit_nnz;
  size_t full_square_fit_nnz;
  size_t rk_fit_compressed_size;
  size_t full_fit_compressed_size;
  size_t rk_compressed_zeros;
  size_t rk_compressed_nnz;
  size_t rk_eval_zeros;
  size_t rk_eval_nnz;
  size_t full_zeros;
  size_t full_nnz;
  size_t full_as_rk_size;
  size_t rk_as_full_size;
  /* ! Total number of terms stored in the full leaves of the HMatrix */
  size_t full_size;

  /** @deprecated Use compressed_size, uncompressed_size or full_size */
  size_t rk_size;

  /** Total number of full leaves of the HMatrix */
  size_t full_count;

  /** Total number of rk leaves of the HMatrix */
  size_t rk_count;

  /** Total number of terms stored in the HMatrix */
  size_t compressed_size;

  /*! Total number of terms that would be stored if the matrix was not compressed */
  size_t uncompressed_size;

  /*! Memory used to store the matrix. This includes meta data */
  size_t memory;

  /*! Total number of block cluster tree nodes in the HMatrix */
  int nr_block_clusters;

  /*! Number of rows in the largest Rk matrice with rows + cols criteria */
  int largest_rk_dim_rows;
  /*! Number of cols in the largest Rk matrice with rows + cols criteria */
  int largest_rk_dim_cols;
  /*! Number of rows in the largest Rk matrice with memory criteria */
  int largest_rk_mem_rows;
  /*! Number of cols in the largest Rk matrice with memory criteria */
  int largest_rk_mem_cols;
  /*! Rank of the largest Rk matrice with memory criteria */
  int largest_rk_mem_rank;
} hmat_info_t;

typedef struct hmat_matrix_struct hmat_matrix_t;

/** Allow to implement a progress bar associated to assemble or factorize */
typedef struct hmat_progress_struct {
    int max;
    int current;
    /** Called each time assembling or factorization progress */
    void (*update)(struct hmat_progress_struct* context);
    void * user_data;
} hmat_progress_t;

/**
 * Return the default progress bar.
 * This is a singleton which must/can not be freed.
 */
hmat_progress_t * hmat_default_progress(void);

/**
 * Function representing a generic stream.
 * It could be implemented as FILE*, unix fd, C++ iostream while using a C API.
 * @param buffer the buffer to read or write
 * @param n the size of the buffer
 */
typedef void (*hmat_iostream)(void * buffer, size_t n, void *user_data);

/** */
struct hmat_block_compute_context_t {
	/**
     * opaque pointer as set by \a prepare_func() in
     * hmat_block_info_t.user_data
     */
    void* user_data;
    /** Location and size of the block */
    int row_start, row_count, col_start, col_count;
    /** stratum id */
    int stratum;
    /**
     * block pointer to the output buffer. No padding is allowed,
	 * that is the leading dimension of the buffer must be its real
	 * leading.
	 */
    void* block;
};
/**
 * Argument of the assemble_generic function.
 * Only one of block_compute/advanced_compute, simple_compute or assembly can be non NULL.
 * If both block_compute & advanced_compute are set, advanced is used.
 */
typedef struct {
    /**
     * The hmat_matrix_t matrix has previouly been initialized, it is a
     * hierarchical structure containing inner nodes and leaves, which
     * contain either a FullMatrix or an RkMatrix (compressed form).
     *
     * There are 4 different ways to perform a matrix assembly;
     * the current recommended way is via advanced_compute, others
     * are kept for legacy code.
     *
     *  1. assembly: this member is a pointer to an Assembly instance;
     *       it provides an assemble method:
     *
     *   void assemble(const LocalSettings & settings,
     *                 const ClusterTree & rows, const ClusterTree & cols,
     *                 bool admissible,
     *                 FullMatrix<T> * & fullMatrix, RkMatrix<T> * & rkMatrix,
     *                 const AllocationObserver & ao)
     *
     *     If admissible argument is true, this method must compute the rkMatrix
     *     argument, otherwise fullMatrix is computed.
     *
     *  2. simple_compute: this member is a pointer to a function
     *
     *   void interaction(void* user_context, int row, int col, void* result)
     *
     *     If block is full, interaction is called for all (row,col) tuple of
     *     this block, the first argument user_context contains user data to
     *     perform these computations.
     *
     *     If block is compressed, the compression algorithm will either retrieve
     *     the full block (with SVD or ACA full algorithm) and compress it, or
     *     retrieve only some rows and columns (ACA partial or ACA+ algorithms).
     *     In all cases, it uses the interaction function.
     *     Note that row and col are numbered here according to user numbering.
     *
     *  3. block_compute: this member is a pointer to a function
     *
     *   void compute(void* user_context, int block_row_start, int block_row_count,
     *                int block_col_start, int block_col_count, void* block)
     *
     *     Moreover, user must also provide a prepare function.
     *     It is called to fill up an hmat_block_info_t structure, and potentially
     *     allocate hmat_block_info_t.user_data member (in which case
     *     hmat_block_info_t.release_user_data function pointer must be provided).
     *     If block_type is set to hmat_block_null in prepare function, nothing is
     *     computed.
     *     If block is full, compute is called on the whole block.
     *
     *     If block is compressed, the compression algorithm will either retrieve
     *     the full block (with SVD or ACA full algorithm) and compress it, or
     *     retrieve only some rows and columns (ACA partial or ACA+ algorithms).
     *
     *  4. advanced_compute: this member is a pointer to a function
     *
     *   void compute(struct hmat_block_compute_context_t*)
     *
     *     This case is similar to the previous one when block is full or compressed
     *     with SVD or ACA full algorithms.  But with ACA partial or ACA+ algorithms,
     *     assembly is done by material (called stratum), and interactions are summed
     *     up.  The prepare function must set hmat_block_info_t.number_of_strata, and
     *     a loop on strata is performed.  By convention, if hmat_block_info_t.stratum
     *     is -1, this callback must sum up interaction for all strata.  Otherwise, it
     *     must compute only the interactions of the given stratum.
     *
     * Only one of advanced_compute, block_compute, simple_compute and
     * assembly function pointers must be non null.
     *
     */
    /** First scenario */
    void * assembly;
    /** Second scenario */
    hmat_interaction_func_t simple_compute;
    /** Third scenario */
    hmat_compute_func_t block_compute;
    /** Fourth scenario */
    void (*advanced_compute)(struct hmat_block_compute_context_t*);

    /**
     * The user context used in all scenarii but the first one.  The default is NULL.
     */
    void* user_context;

    /** Auxiliary method used by third and fourth scenarii
      * This member is a pointer to function
      *    void prepare(int row_start, int row_count, int col_start, int col_count,
      *                 int *row_hmat2client, int *row_client2hmat,
      *                 int *col_hmat2client, int *col_client2hmat,
      *                 void *context,
      *                 hmat_block_info_t * block_info)
      * Eight first arguments are input arguments
      * Block_info is an output argument, it is initialized by hmat, and this
      * callback must fill it up; see comments in hmat_block_info_t.
      * Context should be passed to hmat_block_info_t.user_data.
      */
    hmat_prepare_func_t prepare;

    /** Copy left lower values to the upper right of the matrix */
    int lower_symmetric;
    /** The type of factorization to do after this assembling. The default is hmat_factorization_none. */
    hmat_factorization_t factorization;
    /** NULL disable progress display. The default is to use the hmat progress internal implementation. */
    hmat_progress_t * progress;
} hmat_assemble_context_t;

/** Init a hmat_assemble_context_t with default values */
HMAT_API void hmat_assemble_context_init(hmat_assemble_context_t * context);

typedef struct {
    /** The type of factorization to do after this assembling. The default is hmat_factorization_lu. */
    hmat_factorization_t factorization;
    /** NULL disable progress display. The default is to use the hmat progress internal implementation. */
    hmat_progress_t * progress;
} hmat_factorization_context_t;

/** Init a hmat_factorization_context_t with default values */
HMAT_API void hmat_factorization_context_init(hmat_factorization_context_t * context);

/** Context for the get_values and get_block function */
struct hmat_get_values_context_t {
    /** The matrix from witch to get values */
    hmat_matrix_t* matrix;
    /** Where output values are stored. Must be allocated by the caller */
    void * values;
    /**
     * @brief min row and col indices to get
     */
    int row_offset, col_offset;
    /** number of rows and cols to get */
    int row_size, col_size;
    /**
     * @brief Indirection array for numbering.
     */
    int * row_indices, * col_indices;
    /**
     * @brief If true renumber rows and set row_numbering to NULL
     * Setting this and original_numbering to true is not valid.
     */
    int renumber_rows:1;
};

/* Opaque pointer */
typedef struct
{
    hmat_value_t value_type;
    /** For internal use only */
    void * internal;
} hmat_procedure_t;

/* Delete a procedure */
HMAT_API void hmat_delete_procedure(hmat_procedure_t* proc);

/* Create a procedure to truncate Rk matrices */
HMAT_API hmat_procedure_t* hmat_create_procedure_epsilon_truncate(hmat_value_t type, double epsilon);

/* Opaque pointer */
typedef struct
{
    hmat_value_t value_type;
    /** For internal use only */
    const void * internal;
} hmat_leaf_procedure_t;

/* Delete a leaf procedure */
HMAT_API void hmat_delete_leaf_procedure(hmat_leaf_procedure_t* proc);

/* Create a procedure to truncate Rk matrices */
HMAT_API hmat_leaf_procedure_t* hmat_create_leaf_procedure_epsilon_truncate(hmat_value_t type, double epsilon);

typedef struct
{
    /*! Create an empty (not assembled) HMatrix from 2 \a ClusterTree instances.

      The HMatrix is built on a row and a column tree, that are provided as
      arguments.

      \param stype the scalar type
      \param rows_tree a ClusterTree as returned by \a hmat_create_cluster_tree().
      \param cols_tree a ClusterTree as returned by \a hmat_create_cluster_tree().
      \param lower_symmetric 1 if the matrix is lower symmetric, 0 otherwise
      \return an opaque pointer to an HMatrix, or NULL in case of error.
    */
    hmat_matrix_t* (*create_empty_hmatrix)(hmat_cluster_tree_t* rows_tree, hmat_cluster_tree_t* cols_tree, int lower_symmetric);

    /*! Create an empty (not assembled) HMatrix from 2 \a ClusterTree instances,
      and specify admissibility condition.

      The HMatrix is built on a row and a column tree, that are provided as
      arguments.

      \param stype the scalar type
      \param rows_tree a ClusterTree as returned by \a hmat_create_cluster_tree().
      \param cols_tree a ClusterTree as returned by \a hmat_create_cluster_tree().
      \param cond an admissibility condition, as returned by \a hmat_create_admissibility_standard().
      \param lower_symmetric 1 if the matrix is lower symmetric, 0 otherwise
      \return an opaque pointer to an HMatrix, or NULL in case of error.
    */
    hmat_matrix_t* (*create_empty_hmatrix_admissibility)(hmat_cluster_tree_t* rows_tree, hmat_cluster_tree_t* cols_tree,
                                                         int lower_symmetric, hmat_admissibility_t* cond);

    /*! Declare that the given HMatrix owns its cluster trees, which means that they will be
      automatically deallocated when HMatrix is destroyed.

      \param hmatrix HMatrix
      \param owns_row if 1, declare that this HMatrix owns its row cluster tree
      \param owns_col if 1, declare that this HMatrix owns its column cluster tree
    */
    void (*own_cluster_trees)(hmat_matrix_t* hmatrix, int owns_row, int owns_col);

    /*! Assemble a HMatrix.

      \param hmatrix The matrix to be assembled.
      \param user_context The user context to pass to the prepare function.
      \param prepare The prepare function as given in \a HMatOperations.
      \param compute The compute function as given in \a HMatOperations.
      \param free_data The free function as given in \a HMatOperations.
      \param lower_symmetric 1 if the matrix is lower symmetric, 0 otherwise
      \return 0 for success.
    */
    int (*assemble)(hmat_matrix_t* hmatrix, void* user_context, hmat_prepare_func_t prepare,
                         hmat_compute_func_t compute, int lower_symmetric);

    /*! Assemble a HMatrix then factorize it.

      \param hmatrix The matrix to be assembled.
      \param user_context The user context to pass to the prepare function.
      \param prepare The prepare function as given in \a HMatOperations.
      \param compute The compute function as given in \a HMatOperations.
      \param free_data The free function as given in \a HMatOperations.
      \param lower_symmetric 1 if the matrix is lower symmetric, 0 otherwise
      \return 0 for success.
    */
    int (*assemble_factorize)(hmat_matrix_t* hmatrix, void* user_context, hmat_prepare_func_t prepare,
                         hmat_compute_func_t compute, int lower_symmetric, hmat_factorization_t);

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
                                            hmat_interaction_func_t compute,
                                            int lower_symmetric);

    /*! Assemble a HMatrix */
    void (*assemble_generic)(hmat_matrix_t* matrix, hmat_assemble_context_t * context);

    /*! \brief Return a copy of a HMatrix.

      \param from_hmat the source matrix
      \return a copy of the matrix, or NULL
    */
    hmat_matrix_t* (*copy)(hmat_matrix_t* hmatrix);

    /** Return a null matrix with the same structure */
    hmat_matrix_t* (*copy_struct)(hmat_matrix_t* hmatrix);

    /*! \brief Compute the norm of a HMatrix.

      \param hmatrix the matrix of which to compute the norm
      \return the norm
    */
    double (*norm)(hmat_matrix_t* hmatrix);
    /*! \brief Inverse a HMatrix in place.

      \param hmatrix the matrix to inverse
      \return 0
    */
    int (*inverse)(hmat_matrix_t* hmatrix);
    /*! \brief Factor a HMatrix in place.

      \param hmatrix the matrix to factor
      \return 0 for success
    */
    int (*factorize)(hmat_matrix_t *hmatrix, hmat_factorization_t);

    /*! Factorize a HMatrix */
    void (*factorize_generic)(hmat_matrix_t* matrix, hmat_factorization_context_t * context);

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
    /*! \brief A <- A + alpha Id

      \param hmatrix
      \param alpha
      \return 0 for success
    */
    int (*add_identity)(hmat_matrix_t* hmatrix, void *alpha);
    /*! \brief Initialize library
     */
    int (*init)(void);

    /*! \brief Do the cleanup
     */
    int (*finalize)(void);

    /*! \brief Get current informations
        \param hmatrix A hmatrix
        \param info A structure to fill with current informations
     */
    int (*get_info)(hmat_matrix_t *hmatrix, hmat_info_t* info);

    /*! \brief Dump json & postscript informations about matrix
        \param hmatrix A hmatrix
        \param prefix A string to prefix files output */
    int (*dump_info)(hmat_matrix_t *hmatrix, char* prefix);

    /**
     * @brief Replace the cluster tree in a hmatrix
     * The provided cluster trees must be compatible with the structure of
     * the matrix.
     */
    int (*set_cluster_trees)(hmat_matrix_t* hmatrix, hmat_cluster_tree_t * rows, hmat_cluster_tree_t * cols);

    /**
     * @brief Extract matrix diagonal
     * \param hmatrix A hmatrix
     * \param diag allocated memory area in which diagonal values are written
     * \param size memory size
     */
    int (*extract_diagonal)(hmat_matrix_t* holder, void* diag, int size);

    /**
     * @brief Solve system op(L)*X=B
       \warning There is no check to ensure that matrix has been factorized.
     * \param hmatrix A hmatrix
     * \param transpose if different from 0, transposed matrix is used
     * \param b  right-hand sides, overwritten by solution X at exit
     * \param nrhs number of right-hand sides
     */
    int (*solve_lower_triangular)(hmat_matrix_t* hmatrix, int transpose, void* b, int nrhs);

    /**
     * @brief Extract and uncompress a block of the matrix.
     * After calling this function row_indices and col_indices will contains the
     * indices of the extract block.
     * @see struct hmat_get_values_context_t
     */
    int (*get_block)(struct hmat_get_values_context_t * ctx);

    /**
     * @brief Extract a set of values from the matrix
     * row_indices and col_indices must contains the rows ans columns to get.
     * row_offset, col_offset and renumber_rows are ignored
     * @see struct hmat_get_values_context_t
     */
    int (*get_values)(struct hmat_get_values_context_t * ctx);

    /**
     * @brief Apply a procedure to all nodes of a matrix
     * \param hmatrix A hmatrix
     * \param proc
     */
    int (*walk)(hmat_matrix_t* hmatrix, hmat_procedure_t* proc);

    /**
     * @brief Apply a procedure to all leaves of a matrix
     * \param hmatrix A hmatrix
     * \param proc
     */
    int (*apply_on_leaf)(hmat_matrix_t* hmatrix, const hmat_leaf_procedure_t* proc);

    hmat_value_t value_type;

    /** For internal use only */
    void * internal;

    hmat_matrix_t * (*read_struct)(hmat_iostream readfunc, void * user_data);
    void (*write_struct)(hmat_matrix_t* matrix, hmat_iostream writefunc, void * user_data);
    void (*read_data)(hmat_matrix_t* matrix, hmat_iostream readfunc, void * user_data);
    void (*write_data)(hmat_matrix_t* matrix, hmat_iostream writefunc, void * user_data);
}  hmat_interface_t;

HMAT_API void hmat_init_default_interface(hmat_interface_t * i, hmat_value_t type);

typedef struct
{
  /*! \brief Tolerance for the assembly. */
  double assemblyEpsilon;
  /*! \brief Tolerance for the recompression (using SVD) */
  double recompressionEpsilon;
  int compressionMethod;
  /*! \brief svd compression if max(rows->n, cols->n) < compressionMinLeafSize.*/
   int compressionMinLeafSize;
  /*! \brief Maximum size of a leaf in a ClusterTree (and of a non-admissible block in an HMatrix) */
  int maxLeafSize;
  /*! \brief max(|L0|) */
  int maxParallelLeaves;
  /*! \brief Padding for ABI backward compatiblity */
  int _dummyABI;
  /*! \brief Coarsen the matrix structure after assembly. */
  int coarsening;
  /*! \brief Recompress the matrix after assembly. */
  int recompress; //TODO: remove
  /*! \brief Validate the detection of null rows and columns */
  int validateNullRowCol;
  /*! \brief Validate the rk-matrices after compression */
  int validateCompression;
  /*! \brief Dump trace at the end of the algorithms (depends on the runtime) */
  int dumpTrace;
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
HMAT_API void hmat_get_parameters(hmat_settings_t * settings);
/*! \brief Set current settings

\param structure containing parameters
\return 1 on failure, 0 otherwise.
*/
HMAT_API int hmat_set_parameters(hmat_settings_t*);

/*! \brief Print current settings

\param structure containing parameters
*/
HMAT_API void hmat_print_parameters(hmat_settings_t*);

/*!
 * \brief hmat_get_version
 * \return The version of this library
 */
HMAT_API const char * hmat_get_version(void);
/*!
 * \brief hmat_get_build_date
 * \return The build date and time of this library
 */
HMAT_API void hmat_get_build_date(const char**, const char**);

/*!
 \brief hmat_tracing_dump Dumps the trace info in the given filename

 The file is in json format. Hmat library must be compiled with -DHAVE_CONTEXT for this to work.
\param filename the name of the output json file
*/
HMAT_API void hmat_tracing_dump(char *filename) ;

#ifdef __cplusplus
}
#endif
#endif  /* _HMAT_H */
