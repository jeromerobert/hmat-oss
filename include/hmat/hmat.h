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
  hmat_compress_aca_plus,
  hmat_compress_aca_random,
  hmat_compress_rrqr
} hmat_compress_t;

/**The different Floating Point compression methods to use */
typedef enum { 
  DEFAULT_COMPRESSOR = 0, 
  ZFP_COMPRESSOR = 1,
  SZ2_COMPRESSOR = 2, 
  SZ3_COMPRESSOR = 3
}hmat_FPcompress_t;

typedef enum {
    hmat_block_full,
    hmat_block_null,
    hmat_block_sparse
} hmat_block_t;

typedef enum {
    hmat_factorization_none = -1,
    hmat_factorization_lu,
    hmat_factorization_ldlt,
    hmat_factorization_llt,
    hmat_factorization_hodlr,
    hmat_factorization_hodlrsym
} hmat_factorization_t;

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
     * When set to 0, the hmat_prepare_func_t should reset it
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
/* NTiles recursive clustering */
HMAT_API hmat_clustering_algorithm_t* hmat_create_clustering_ntilesrecursive(int nTiles);
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

HMAT_API void hmat_delete_cluster_tree(const hmat_cluster_tree_t * tree);

HMAT_API hmat_cluster_tree_t * hmat_copy_cluster_tree(const hmat_cluster_tree_t * tree);

HMAT_API void hmat_swap_cluster_tree(hmat_cluster_tree_t *first, hmat_cluster_tree_t *second);

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
    /** Group index of dofs (may be NULL) */
    int * group_index;
    /** pointer to an opaque ClusterTreeBuilder */
    const hmat_cluster_tree_builder_t* builder;
};

HMAT_API hmat_cluster_tree_t * hmat_create_cluster_tree_generic(struct hmat_cluster_tree_create_context_t*);

/*!
 * Return the number of nodes in a cluster tree
 */
HMAT_API int hmat_tree_nodes_count(const hmat_cluster_tree_t * tree);
/*!
 * Returns the index-th son of a cluster tree root
 */
HMAT_API hmat_cluster_tree_t *hmat_cluster_get_son( hmat_cluster_tree_t * tree, int index );

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

/**
 * @brief Return the renumbering of the cluster tree
 * @param tree a cluster tree
 * @return The original position (before renumbering) of the ith value in the cluster
 * tree (after renumbering). This array must not be freed nor modified by the caller.
 */
HMAT_API const int * hmat_cluster_get_indices(const hmat_cluster_tree_t *tree);

typedef struct {
    /** eta for Hackbusch condition */
    double eta;
    /** ratio (in [0, 0.5]) to prevent tall and skinny blocks */
    double ratio;
    /** maximum block width */
    size_t max_width;
    /** A scale factor (default to -1 for "unspecified" */
    double scale;
} hmat_admissibility_param_t;

/** Init an hmat_admissibility_param structure with default values */
HMAT_API void hmat_init_admissibility_param(hmat_admissibility_param_t *);

/* Opaque pointer */
typedef struct hmat_admissibility_condition hmat_admissibility_t;

/** Create an admissibility condition from parameters */
HMAT_API hmat_admissibility_t* hmat_create_admissibility(hmat_admissibility_param_t *);

/** Update an admissibility condition with parameters */
HMAT_API void hmat_update_admissibility(hmat_admissibility_t*, hmat_admissibility_param_t *);

/* Create a standard (Hackbusch) admissibility condition, with a given eta */
HMAT_API hmat_admissibility_t* hmat_create_admissibility_standard(double eta);

/**
 * @brief Create an admissibility which will generate a HODLR matrix.
 *
 * All blocks except diagonal ones are admissible.
 */
HMAT_API hmat_admissibility_t* hmat_create_admissibility_hodlr(void);

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
 * @param max_block_size The maximum acceptable block size in number of values (rows * cols).
 * Use 0 to switch to autodetect.
 * @param min_nr_block The minimum acceptable number of blocks created with this condition
 * Use 0 to switch to autodetect.
 * @param split_rows Tel whether or not to split rows
 * @param split_cols Tel whether or not to split cols
 */
HMAT_API hmat_admissibility_t* hmat_create_admissibility_never(
        size_t max_size, unsigned int min_block, int split_rows, int split_cols);

/* Delete admissibility condition */
HMAT_API void hmat_delete_admissibility(hmat_admissibility_t * cond);

/* Opaque pointer */
typedef struct hmat_compression_algorithm_struct hmat_compression_algorithm_t;

/* Create a procedure to compress matrix blocks */
HMAT_API hmat_compression_algorithm_t* hmat_create_compression_svd(double epsilon);
HMAT_API hmat_compression_algorithm_t* hmat_create_compression_aca_full(double epsilon);
HMAT_API hmat_compression_algorithm_t* hmat_create_compression_aca_partial(double epsilon);
HMAT_API hmat_compression_algorithm_t* hmat_create_compression_aca_plus(double epsilon);
HMAT_API hmat_compression_algorithm_t* hmat_create_compression_aca_random(double epsilon);
HMAT_API hmat_compression_algorithm_t* hmat_create_compression_rrqr(double epsilon);

/* Delete a compression algorithm */
HMAT_API void hmat_delete_compression(const hmat_compression_algorithm_t* algo);

/** Information on the HMatrix */
typedef struct
{
  /*! Number of allocated zeros */
  int full_zeros;
  /** Total number of terms stored in the full leaves of the HMatrix */
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

/** Profile of the HMatrix */
typedef struct
{
  //TO DO : uses C Hashtable instead of C++ map for profiling the Matrices;
  
} hmat_profile_t;

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

/** Structure containing the FPcompression Ratio */
typedef struct {
  //The compression ratio for the Rk-Matrices only
  double rkRatio;

  //Total number of elements in the Rk-Matrices only
  size_t size_Rk;
  size_t size_Rk_compressed;

  //The compression ratio for the full blocs only
  double fullRatio;

  //Total number of elements in the full blocs only
  size_t size_Full;
  size_t size_Full_compressed;

  //The compression ratio for the Whole HMatrix
  double ratio;
} hmat_FPCompressionRatio_t;

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

    /** Compression algorithm (created by calling an hmat_create_compression_* function, or NULL for no compression) **/
    const hmat_compression_algorithm_t * compression;

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
     * Those are pointer to internal data and must not be freed by the caller.
     */
    int * row_indices, * col_indices;
    /**
     * @brief If true renumber rows when getting blocks.
     * This is only supported if row_offset = 0 and row_size = matrix row size.
     */
    int renumber_rows:1;
};

/**
 * Argument of the solve_generic function.
 */
struct hmat_solve_context_t {
    /**
     * RHS on input and solution on output.
     * Can be either an hmat_matrix_t or a dense array.
     */
    void* values;

    /** If 0, values is an hmatrix.  Otherwise, number of right hand sides */
    int nr_rhs;

    /** When input is a dense array, rows must be permuted to have the same numbering as column dofs.
     * If this permutation had already been made, no_permutation must be set.
     */
    int no_permutation;

    /** If true, solve L X = B; default is to solve A X = B.
     * @warning lower and upper cannot be both set.
     */
    int lower;

    /** If true, solve U X = B; default is to solve A X = B.
     * @warning lower and upper cannot be both set.
     */
    int upper;

    /** Not used, may be useful later. */
    hmat_progress_t * progress;
};

HMAT_API void hmat_solve_context_init(struct hmat_solve_context_t * context);

/* Opaque pointer */
typedef struct
{
    hmat_value_t value_type;
    /** For internal use only */
    void * internal;
} hmat_procedure_t;

/* Delete a procedure */
HMAT_API void hmat_delete_procedure(hmat_procedure_t* proc);

/* Opaque pointer */
typedef struct
{
    hmat_value_t value_type;
    /** For internal use only */
    const void * internal;
} hmat_leaf_procedure_t;

/* Delete a leaf procedure */
HMAT_API void hmat_delete_leaf_procedure(hmat_leaf_procedure_t* proc);

typedef struct
{
    /*! Create an empty (not assembled) HMatrix from 2 \a ClusterTree instances
      and an admissibility condition.

      The HMatrix is built on a row and a column tree, that are provided as
      arguments.

      \param stype the scalar type
      \param rows_tree a ClusterTree as returned by \a hmat_create_cluster_tree().
      \param cols_tree a ClusterTree as returned by \a hmat_create_cluster_tree().
      \param cond an admissibility condition, as returned by \a hmat_create_admissibility_standard().
      \param lower_symmetric 1 if the matrix is lower symmetric, 0 otherwise
      \return an opaque pointer to an HMatrix, or NULL in case of error.
    */
    hmat_matrix_t* (*create_empty_hmatrix_admissibility)(const hmat_cluster_tree_t* rows_tree, const hmat_cluster_tree_t* cols_tree,
                                                         int lower_symmetric, hmat_admissibility_t* cond);

    /*! Declare that the given HMatrix owns its cluster trees, which means that they will be
      automatically deallocated when HMatrix is destroyed.

      \param hmatrix HMatrix
      \param owns_row if 1, declare that this HMatrix owns its row cluster tree
      \param owns_col if 1, declare that this HMatrix owns its column cluster tree
    */
    void (*own_cluster_trees)(hmat_matrix_t* hmatrix, int owns_row, int owns_col);

    /*! Set threshold for low-rank recompressions.

      \param hmatrix HMatrix
      \param epsilon Threshold for low-rank recompressions
    */
    void (*set_low_rank_epsilon)(hmat_matrix_t* hmatrix, double epsilon);

    /*! Assemble a HMatrix */
    int (*assemble_generic)(hmat_matrix_t* matrix, hmat_assemble_context_t * context);

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

    /*! \brief Compute the logarithm of the determinante of a HMatrix. */
    int (*logdet)(hmat_matrix_t* hmatrix, double * result);

    /*! \brief Inverse a HMatrix in place.

      \param hmatrix the matrix to inverse
      \return 0
    */
    int (*inverse)(hmat_matrix_t* hmatrix);

    /*! Factorize a HMatrix */
    int (*factorize_generic)(hmat_matrix_t* matrix, hmat_factorization_context_t * context);

    /*! \brief Destroy a HMatrix.

      \param hmatrix the matrix to destroy
      \return 0
    */
    int (*destroy)(hmat_matrix_t* hmatrix);

    /*! \brief Get one of the children of an HMatrix.

      \param hmatrix the main matrix
      \param i the row coordinate of the child
      \param j the column coordinate of the child
      \return 0
    */
    hmat_matrix_t * (*get_child)( hmat_matrix_t *hmatrix, int i, int j );

    /*! \brief Destroy a child HMatrix.

      \param hmatrix the child matrix to destroy
      \return 0
    */
    int (*destroy_child)(hmat_matrix_t* hmatrix);

    /*! \brief Solve A X = B, with X overwriting B.

      \param hmatrix
      \param context
      \return 0 for success
    */
    int (*solve_generic)(hmat_matrix_t* matrix, const struct hmat_solve_context_t * context);

    /*! \brief Solve A X = B, with X overwriting B.

      In this function, B is a H-matrix
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
    /*! \brief Solve A x = b, with x overwriting b.

      In this function, b is a multi-column vector, with nrhs RHS.

      Functions vector_reorder and vector_restore must be called before
      and after this function to transform original numbering of b
      into/from hmat internal numbering.

      \param hmatrix
      \param b
      \param nrhs
      \return 0 for success
    */
    int (*solve_dense)(hmat_matrix_t* hmatrix, void* b, int nrhs);
    /*! \brief Transpose an HMatrix in place.

       \return 0 for success.
     */
    int (*transpose)(hmat_matrix_t *hmatrix);
    /*! \brief A <- alpha * A

      \param alpha
      \param hmatrix
    */
    int (*scale)(void * alpha, hmat_matrix_t *hmatrix);
    /*! \brief Recompress all Rk matrices to their respective epsilon_ values.

      This function is useful only after epsilon_ values have been modified.

      \param hmatrix
    */
    int (*truncate)(hmat_matrix_t *hmatrix);
    /*! \brief Permute values in a dense array

      \param vec_b values
      \param rows_ct cluster tree for rows; may be NULL, in which case rows argument must be set to define
      \param rows when rows_ct is NULL, this argument contains the number of rows, it is required to know data shape.
         When rows_ct is not NULL, this argument is unused.
      \param cols_ct cluster tree for cols; may be NULL, in which case cols argument must be set to define
      \param cols when cols_ct is NULL, this argument contains the number of columns, it is required to know data shape.
         When cols_ct is not NULL, this argument is unused.
    */
    int (*vector_reorder)(void* vec_b, const hmat_cluster_tree_t *rols_ct, int rows, const hmat_cluster_tree_t *cols_ct, int cols);
    /*! \brief Renumber values back to original numbering

      \param vec_b values
      \param rows_ct cluster tree for rows; may be NULL, in which case rows argument must be set to define
      \param rows when rows_ct is NULL, this argument contains the number of rows, it is required to know data shape.
         When rows_ct is not NULL, this argument is unused.
      \param cols_ct cluster tree for cols; may be NULL, in which case cols argument must be set to define
      \param cols when cols_ct is NULL, this argument contains the number of columns, it is required to know data shape.
         When cols_ct is not NULL, this argument is unused.
    */
    int (*vector_restore)(void* vec_b, const hmat_cluster_tree_t *rols_ct, int rows, const hmat_cluster_tree_t *cols_ct, int cols);
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
    int (*gemm)(char trans_a, char trans_b, const void* alpha, hmat_matrix_t* hmatrix,
                hmat_matrix_t* hmatrix_b, const void* beta, hmat_matrix_t* hmatrix_c);

    /*! \brief y := y + a * x */
    int (*axpy)(void* a, hmat_matrix_t* x, hmat_matrix_t* hmatrix_y);

    /*! \brief solve \alpha A * x */
    int (*trsm)( char side, char uplo, char transa, char diag, int m, int n,
		 void *alpha, hmat_matrix_t *A, int is_b_hmat, void *B );

    /*! @deprecated \brief c <- alpha * A * b + beta * c

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
    /*! @deprecated \brief Same as gemv, but without renumbering on vec_b and vec_c
    */
    int (*gemm_scalar)(char trans_a, void* alpha, hmat_matrix_t* hmatrix, void* vec_b,
		       void* beta, void* vec_c, int nrhs);
    /*! @deprecated \brief C <- alpha * A * B + beta * C


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
    /*! \brief Y <- alpha * op(B) * op(X) + beta * Y if side is 'L'
            or Y <- alpha * op(X) * op(B) + beta * Y if side is 'R'

      Functions vector_reorder and vector_restore must be called before
      and after this function to transform original numbering of X and Y
      into/from hmat internal numbering.

      \param trans_b 'N', 'T' or 'C'
      \param trans_x 'N', 'T' or 'C'
      \param side 'L' or 'R'
      \param alpha
      \param hmatrix
      \param vec_x
      \param beta
      \param vec_y
      \param nrhs
      \return 0 for success
    */
    int (*gemm_dense)(char trans_b, char trans_x, char side, const void* alpha, hmat_matrix_t* holder,
                      void* vec_x, const void* beta, void* vec_y, int nrhs);
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

    int (*get_profile)(hmat_matrix_t *hmatrix, hmat_profile_t* profile);

    int (*get_ratio)(hmat_matrix_t *hmatrix, hmat_FPCompressionRatio_t* ratio);

    int (*FPcompress)(hmat_matrix_t *hmatrix, double epsilon, int nb_blocs, hmat_FPcompress_t method);

    int (*FPdecompress)(hmat_matrix_t *hmatrix, hmat_FPcompress_t method);

    /*! \brief Dump json & postscript informations about matrix
        \param hmatrix A hmatrix
        \param prefix A string to prefix files output */
    int (*dump_info)(hmat_matrix_t *hmatrix, char* prefix);

    /**
     * @brief Get cluster trees
     * \param hmatrix A hmatrix
     * \param rows if not NULL, will contain a pointer to rows cluster tree
     * \param cols if not NULL, will contain a pointer to cols cluster tree
     */
    int (*get_cluster_trees)(hmat_matrix_t* hmatrix, const hmat_cluster_tree_t ** rows, const hmat_cluster_tree_t ** cols);

    /**
     * @brief Replace the cluster tree in a hmatrix
     * The provided cluster trees must be compatible with the structure of
     * the matrix.
     */
    int (*set_cluster_trees)(hmat_matrix_t* hmatrix, const hmat_cluster_tree_t * rows, const hmat_cluster_tree_t * cols);

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
     * @brief Solve system op(L)*X=B
       \warning There is no check to ensure that matrix has been factorized.
     * \param hmatrix A hmatrix
     * \param transpose if different from 0, transposed matrix is used
     * \param b  right-hand sides, overwritten by solution X at exit
     * \param nrhs number of right-hand sides
     */
    int (*solve_lower_triangular_dense)(hmat_matrix_t* hmatrix, int transpose, void* b, int nrhs);

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

    /**
     * @brief Set the progress bar associated to a matrix.
     *
     * Note that progress bar is also set by assemble_generic and factorize_generic.
     * @param matrix the matrix whose one want to change the progressbar
     * @param progress the new progress bar implementation. NULL disable progress
     * reporting.
     */
    void (*set_progressbar)(hmat_matrix_t * matrix, hmat_progress_t * progress);
    /**
     * @brief Extract matrix diagonal_block
     * \param hmatrix A hmatrix
     * \param components  extract each (components x components) block
     * \param diag allocated memory area in which diagonal values are written
     */
    int (*extract_diagonal_block)(hmat_matrix_t* holder, int components, void* diag);

}  hmat_interface_t;

HMAT_API void hmat_init_default_interface(hmat_interface_t * i, hmat_value_t type);

typedef struct
{
  /*! \brief svd compression if max(rows->n, cols->n) < compressionMinLeafSize.*/
  int compressionMinLeafSize;
  /*! \brief Tolerance for coarsening */
  double coarseningEpsilon;
  /*! \brief Maximum size of a leaf in a ClusterTree (and of a non-admissible block in an HMatrix) */
  int maxLeafSize;
  /*! \brief Coarsen the matrix structure after assembly. */
  int coarsening;
  /*! \brief Validate the detection of null rows and columns */
  int validateNullRowCol;
  /*! \brief Validate the rk-matrices after compression */
  int validateCompression;
  /*! \brief Validate the rk-matrices recompression */
  int validateRecompression;
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

/** \brief Set the function used to get the worker index.

    The function f() must return the worker Id (between 0 and nbWorkers-1) or -1 in a sequential section.
    This function is used within hmat-oss for timers and traces. It must be set if hmat-oss is called by
    multiple threads simultaneously.
*/
HMAT_API void hmat_set_worker_index_function(int (*f)(void));

#ifdef __cplusplus
}
#endif
#endif  /* _HMAT_H */
