/*
  HMat-OSS (HMatrix library, open source software)

  Copyright (C) 2021 Airbus SAS

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

/**
 * Factorize a matrix using LLt and HOLDR, test the factorization is valid and
 * test logdet function
 */
#define _GNU_SOURCE // for M_PI
#include <hmat/hmat.h>
#include "examples.h"
#include <math.h>
#include <string.h>
#include "common/chrono.h"
#include "common/my_assert.h"

static const double epsilon = 1e-3;
static const int nDoF = 3000;
static const float zero = 0;
static const float one = 1;

struct context_t {
  int n;
  double* points;
  double l;
  hmat_interface_t hmat;
  hmat_cluster_tree_t * clustertree;
};

/**
  Define interaction between 2 degrees of freedoms  (real case)
 */
void interaction_real(void* data, int i, int j, void* result)
{
  struct context_t* pdata = (struct context_t*) data;
  double* points = pdata->points;
  double r = distanceTo(points + 3 * i, points + 3 * j);
  *((double*)result) = exp(-fabs(r) / pdata->l) + (i == j ? nDoF/100. : 0);
}

void create_geometry(struct context_t* d) {
  d->n = nDoF;
  d->points = createCylinder(1, 1.75 * 3.14159 / sqrt(d->n), d->n);
  d->l = correlationLength(d->points, d->n);
}

hmat_matrix_t * create_matrix(struct context_t* d) {
  hmat_clustering_algorithm_t * ctm = hmat_create_clustering_median();
  /* Change maxLeafSize to have a deep tree */
  hmat_clustering_algorithm_t * ctmd = hmat_create_clustering_max_dof(ctm, 10);
  d->clustertree = hmat_create_cluster_tree(d->points, 3, d->n, ctmd);
  hmat_delete_clustering(ctmd);
  hmat_delete_clustering(ctm);
  hmat_admissibility_t * ac = hmat_create_admissibility_hodlr();
  hmat_matrix_t * matrix = d->hmat.create_empty_hmatrix_admissibility(d->clustertree, d->clustertree, 1, ac);
  d->hmat.own_cluster_trees(matrix, 1, 0);
  hmat_delete_admissibility(ac);
  d->hmat.set_low_rank_epsilon(matrix, epsilon);
  return matrix;
}

void assemble_matrix(struct context_t* ctx, hmat_matrix_t * m) {
  hmat_assemble_context_t ctx_assemble;
  hmat_assemble_context_init(&ctx_assemble);
  ctx_assemble.compression = hmat_create_compression_aca_random(epsilon);
  ctx_assemble.user_context = ctx;
  ctx_assemble.simple_compute = interaction_real;
  ctx_assemble.lower_symmetric = 1;
  HMAT_ASSERT(ctx->hmat.assemble_generic(m, &ctx_assemble) == 0);
  hmat_delete_compression(ctx_assemble.compression);
}

void facto_gemv(struct context_t* ctx, hmat_factorization_t f,
  hmat_matrix_t * matrix, float * vector,
  hmat_matrix_t ** o_matrix, float ** o_vector) {
  *o_vector = malloc(ctx->n * sizeof(float));
  float * vector_tmp = malloc(ctx->n * sizeof(float));
  hmat_factorization_context_t ctx_facto;
  hmat_factorization_context_init(&ctx_facto);
  ctx_facto.factorization = f;
  *o_matrix = ctx->hmat.copy(matrix);
  Time start = now();
  HMAT_ASSERT(ctx->hmat.factorize_generic(*o_matrix, &ctx_facto) == 0);
  HMAT_ASSERT(ctx->hmat.gemm_dense('T', 'N', 'L', &one, *o_matrix, vector, &zero, vector_tmp, 1) == 0);
  HMAT_ASSERT(ctx->hmat.gemm_dense('N', 'N', 'L', &one, *o_matrix, vector_tmp, &zero, *o_vector, 1) == 0);
  free(vector_tmp);
  Time end = now();
  printf("Facto+GEMV timer: %gs\n", time_diff(start, end));
}

void check_unsupported(struct context_t* ctx, hmat_matrix_t * matrix) {
  // Check that gemm on a factorized HOLDR is reported as unsupported
  HMAT_ASSERT(ctx->hmat.gemm('N', 'N', &one, matrix, matrix, &zero, matrix));
  struct hmat_get_values_context_t gvctx = { 0 };
  gvctx.matrix = matrix;
  // get_values is also unsupported
  HMAT_ASSERT(ctx->hmat.get_values(&gvctx));
}

void assert_equals(float * v1, float * v2, int n, double tol) {
  double diffnorm = 0;
  double norm = 0;
  for(int i = 0; i < n; i++) {
    double v = v1[i] - v2[i];
    diffnorm += v * v;
    norm += v1[i] * v1[i];
  }
  diffnorm = sqrt(diffnorm) / norm;
  printf("Relative error: %g\n", diffnorm);
  HMAT_ASSERT_MSG(diffnorm < tol, "%g > %g\n", diffnorm, tol);
}

float * to_float(double * v, int n) {
  float * r = malloc(sizeof(float) * n);
  for(int i = 0; i < n; i++)
    r[i] = v[i];
  free(v);
  return r;
}

int main(int argc, char **argv) {
  struct context_t ctx;
  hmat_init_default_interface(&ctx.hmat, HMAT_SIMPLE_PRECISION);
  create_geometry(&ctx);
  hmat_matrix_t * matrix = create_matrix(&ctx);
  float * vector = to_float(createRhs(ctx.points, ctx.n, ctx.l), ctx.n);
  ctx.hmat.vector_reorder(vector, ctx.clustertree, 0, NULL, 1);
  float * vector_nf = malloc(ctx.n * sizeof(float));
  memcpy(vector_nf, vector, ctx.n * sizeof(float));
  assemble_matrix(&ctx, matrix);
  ctx.hmat.gemm_dense('N', 'N', 'L', &one, matrix, vector, &zero, vector_nf, 1);

  hmat_matrix_t * llt_matrix = NULL, * hodlr_matrix = NULL;
  float * vector_llt = NULL, * vector_hodlr = NULL;
  printf("LLT factorization\n");
  facto_gemv(&ctx, hmat_factorization_llt, matrix, vector, &llt_matrix, &vector_llt);
  assert_equals(vector_nf, vector_llt, ctx.n, epsilon*1e-3);
  double lltlogdet = 0;
  ctx.hmat.logdet(llt_matrix, &lltlogdet);
  float lltlogdetf = lltlogdet;

  printf("\nHODLR factorization\n");
  facto_gemv(&ctx, hmat_factorization_hodlrsym, matrix, vector, &hodlr_matrix, &vector_hodlr);
  check_unsupported(&ctx, hodlr_matrix);
  // There is not truncate in HOLDR factorization so the error does not depends on epsilon
  assert_equals(vector_nf, vector_hodlr, ctx.n, 1e-9);
  double hodlrlogdet = 0;
  ctx.hmat.logdet(hodlr_matrix, &hodlrlogdet);
  float hodlrlogdetf = hodlrlogdet;
  printf("logdet with LLT: %g, logdet with HODLR: %g\n", lltlogdet, hodlrlogdet);
  assert_equals(&lltlogdetf, &hodlrlogdetf, 1, epsilon*1e-5);
  free(ctx.points);
  free(vector);
  free(vector_nf);
  free(vector_hodlr);
  free(vector_llt);
  ctx.hmat.destroy(matrix);
  ctx.hmat.destroy(llt_matrix);
  ctx.hmat.destroy(hodlr_matrix);
  return 0;
}
