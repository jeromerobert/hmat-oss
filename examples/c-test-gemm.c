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

// Cylinder
#include <stdio.h>
#include <math.h>
#ifdef __cplusplus
#include <complex>
typedef std::complex<double> double_complex;
typedef std::complex<float> float_complex;
#define make_double_complex(realPart, imagPart) \
    std::complex<double>(realPart, imagPart)
#define make_float_complex(realPart, imagPart) \
    std::complex<float>(realPart, imagPart)
#else
#include <complex.h>
typedef double complex double_complex;
typedef float complex float_complex;
#define make_double_complex(realPart, imagPart) \
    realPart + imagPart * _Complex_I
#define make_float_complex(realPart, imagPart) \
    realPart + imagPart * _Complex_I
#endif

#include "config.h"
#ifdef HAVE_MKL_CBLAS_H
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif


#include "hmat/hmat.h"

/** This is a simple example showing how to use the HMatrix library.

    In this example, we assemble and do a decomposition of a Matrix such that:
    \f[A_{ij} = \frac{e^{i\kappa |x_i - x_j|}}{4 \pi |x_i - x_j|}\f]
    with the points \f$(x_i)\f$ on a cylinder.
    In the real case we use 1 / r instead.
 */


/** Create an open cylinder point cloud.

    \param radius Radius of the cylinder
    \param step distance between two neighboring points
    \param n number of points
    \return a vector of points.
 */
double* createCylinder(double radius, double step, int n) {
  double* result = (double*) malloc(3 * n * sizeof(double));
  double length = 2 * M_PI * radius;
  int pointsPerCircle = length / step;
  double angleStep = 2 * M_PI / pointsPerCircle;
  int i;
  for (i = 0; i < n; i++) {
    result[3*i+0] = radius * cos(angleStep * i);
    result[3*i+1] = radius * sin(angleStep * i),
    result[3*i+2] = (step * i) / pointsPerCircle;
  }
  return result;
}

typedef struct {
  int n;
  double* points;
  double k;
  double l;
} problem_data_t;

double correlationLength(double * points, size_t n) {
  size_t i;
  double pMin[3], pMax[3];
  pMin[0] = points[0]; pMin[1] = points[1]; pMin[2] = points[2];
  pMax[0] = points[0]; pMax[1] = points[1]; pMax[2] = points[2];
  for (i = 0; i < n; i++) {
      if (points[3*i] < pMin[0]) pMin[0] =  points[3*i];
      if (points[3*i] > pMax[0]) pMax[0] =  points[3*i];

      if (points[3*i+1] < pMin[1]) pMin[1] =  points[3*i+1];
      if (points[3*i+1] > pMax[1]) pMax[1] =  points[3*i+1];

      if (points[3*i+2] < pMin[2]) pMin[2] =  points[3*i+2];
      if (points[3*i+2] > pMax[2]) pMax[2] =  points[3*i+2];
  }
  double l = pMax[0] - pMin[0];
  if (pMax[1] - pMin[1] > l) l = pMax[1] - pMin[1];
  if (pMax[2] - pMin[2] > l) l = pMax[2] - pMin[2];
  return 0.1 * l;
}

/**
  Define interaction between 2 degrees of freedoms  (real case)
*/
void interaction_real(void* data, int i, int j, void* result)
{
  problem_data_t* pdata = (problem_data_t*) data;
  double* points = pdata->points;
  double dx = points[3*i+0] - points[3*j+0];
  double dy = points[3*i+1] - points[3*j+1];
  double dz = points[3*i+2] - points[3*j+2];
  *((double*)result) = exp(-sqrt(dx*dx + dy*dy + dz*dz) / pdata->l);
}

/**
  Define interaction between 2 degrees of freedoms  (complex case)
*/
void interaction_complex(void* data, int i, int j, void* result)
{
  problem_data_t* pdata = (problem_data_t*) data;
  double* points = pdata->points;
  double k = pdata->k;
  double dx = points[3*i+0] - points[3*j+0];
  double dy = points[3*i+1] - points[3*j+1];
  double dz = points[3*i+2] - points[3*j+2];
  double distance = sqrt(dx*dx + dy*dy + dz*dz);
  double realPart = cos(k * distance) / (4 * M_PI);
  double imagPart = sin(k * distance) / (4 * M_PI);
  *((double_complex*)result) = make_double_complex(realPart, imagPart);
}

static void io_write(void * buffer, size_t n, void *user_data)
{
#if 0
  fwrite(buffer, n, 1, (FILE*) user_data);
#endif
}

int main(int argc, char **argv) {
  double radius, step, k;
  double* points;
  hmat_settings_t settings;
  int n;
  char arithmetic;
  hmat_clustering_algorithm_t* clustering;
  hmat_cluster_tree_t* cluster_tree;
  hmat_matrix_t *hmatrix, *hmatrix_result;
  hmat_interface_t hmat;
  hmat_value_t type;
  hmat_info_t mat_info;
  int rc;
  problem_data_t problem_data;
  hmat_admissibility_t * admissibilityCondition = hmat_create_admissibility_standard(3.0);
  struct hmat_get_values_context_t ctx_input;
  struct hmat_get_values_context_t ctx_output;
  char transFlags[3];
  CBLAS_TRANSPOSE transFlagsCblas[3];
  void *alpha, *beta, *minus_one;
  // int * identity;
  int i, j;
  char filename[200];
  FILE *hFile;

  if (argc != 3) {
    fprintf(stderr, "Usage: %s n_points (S|D|C|Z)\n", argv[0]);
    return 1;
  }

  n = atoi(argv[1]);
  arithmetic = argv[2][0];
  switch (arithmetic) {
  case 'S':
    type = HMAT_SIMPLE_PRECISION;
    break;
  case 'D':
    type = HMAT_DOUBLE_PRECISION;
    break;
  case 'C':
    type = HMAT_SIMPLE_COMPLEX;
    break;
  case 'Z':
    type = HMAT_DOUBLE_COMPLEX;
    break;
  default:
    fprintf(stderr, "Unknown arithmetic code %c, exiting...\n", arithmetic);
    return 1;
  }

  hmat_get_parameters(&settings);
  hmat_init_default_interface(&hmat, type);
  settings.compressionMethod = hmat_compress_aca_plus;
  settings.assemblyEpsilon = 1e-8;
  settings.recompressionEpsilon = 1e-8;

  hmat_set_parameters(&settings);
  if (0 != hmat.init())
  {
    fprintf(stderr, "Unable to initialize HMat library\n");
    return 1;
  }

  printf("Generating the point cloud...\n");

  radius = 1.;
  step = 1.75 * M_PI * radius / sqrt((double)n);
  k = 2 * M_PI / (10. * step); // 10 points / lambda
  points = createCylinder(radius, step, n);
  printf("done.\n");

  problem_data.n = n;
  problem_data.points = points;
  problem_data.k = k;
  problem_data.l = correlationLength(points, n);

  clustering = hmat_create_clustering_median();
  cluster_tree = hmat_create_cluster_tree(points, 3, n, clustering);
  hmat_delete_clustering(clustering);
  printf("ClusterTree node count = %d\n", hmat_tree_nodes_count(cluster_tree));
  hmatrix = hmat.create_empty_hmatrix_admissibility(cluster_tree, cluster_tree,
                                                    0, admissibilityCondition);
  hmat_delete_admissibility(admissibilityCondition);
  hmat.get_info(hmatrix, &mat_info);
  printf("HMatrix node count = %d\n", mat_info.nr_block_clusters);
  printf(" Rk leaves count = %ld\n", mat_info.rk_count);
  printf(" Full leaves count = %ld\n", mat_info.full_count);
  if (type == HMAT_SIMPLE_PRECISION || type == HMAT_DOUBLE_PRECISION)
    rc = hmat.assemble_simple_interaction(hmatrix, &problem_data, interaction_real, 0);
  else
    rc = hmat.assemble_simple_interaction(hmatrix, &problem_data, interaction_complex, 0);
  if (rc) {
    fprintf(stderr, "Error in assembly, return code is %d, exiting...\n", rc);
    hmat.finalize();
    return rc;
  }
  hFile = fopen("hmat.bin", "w");
  hmat.write_data(hmatrix, io_write, hFile);
  fclose(hFile);

  hmatrix_result = hmat.copy(hmatrix);

  transFlags[0] = 'N'; transFlagsCblas[0] = CblasNoTrans;
  transFlags[1] = 'T'; transFlagsCblas[1] = CblasTrans;
  transFlags[2] = 'C'; transFlagsCblas[2] = CblasConjTrans;
#if 0
  identity = malloc(n * sizeof(int));
  for(i = 0; i < n; ++i)
    identity[i] = i;
#endif

  ctx_input.matrix = hmatrix;
  ctx_input.values = malloc(n * n * sizeof(double_complex));
  ctx_input.row_offset = 0;
  ctx_input.row_size = n;
  ctx_input.col_offset = 0;
  ctx_input.col_size = n;
  ctx_input.row_indices = NULL;
  ctx_input.col_indices = NULL;
  ctx_input.renumber_rows = 0;

  ctx_output.matrix = hmatrix_result;
  ctx_output.values = malloc(n * n * sizeof(double_complex));
  ctx_output.row_offset = 0;
  ctx_output.row_size = n;
  ctx_output.col_offset = 0;
  ctx_output.col_size = n;
  ctx_output.row_indices = NULL;
  ctx_output.col_indices = NULL;
  ctx_output.renumber_rows = 0;

  alpha = malloc(sizeof(double_complex));
  beta = malloc(sizeof(double_complex));
  minus_one = malloc(sizeof(double_complex));
  switch (arithmetic) {
  case 'S':
    *((float*) alpha) = 1.0f;
    *((float*) beta)  = 0.0f;
    break;
  case 'D':
    *((double*) alpha) = 1.0;
    *((double*) beta)  = 0.0;
    break;
  case 'C':
    *((float_complex*) alpha) = make_float_complex(1.0, 0.0);
    *((float_complex*) beta)  = make_float_complex(0.0, 0.0);
    break;
  case 'Z':
    *((double_complex*) alpha) = make_double_complex(1.0, 0.0);
    *((double_complex*) beta)  = make_double_complex(0.0, 0.0);
    break;
  default:
    fprintf(stderr, "Unknown arithmetic code %c, exiting...\n", arithmetic);
    return 1;
  }
  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      fprintf(stdout, "gemm %c %c\n", transFlags[i], transFlags[j]);
      rc = hmat.gemm(transFlags[i], transFlags[j], alpha, hmatrix, hmatrix, beta, hmatrix_result);
      if (rc) {
        fprintf(stderr, "Error in gemm, return code is %d, exiting...\n", rc);
        hmat.finalize();
        return rc;
      }
      rc = hmat.get_block(&ctx_input);
      /* rc = hmat.get_values(&ctx_input); */
      if (rc) {
        fprintf(stderr, "Error in get_block (output), return code is %d, exiting...\n", rc);
        hmat.finalize();
        return rc;
      }
      rc = hmat.get_block(&ctx_output);
      /* rc = hmat.get_values(&ctx_output); */
      if (rc) {
        fprintf(stderr, "Error in get_block (output), return code is %d, exiting...\n", rc);
        hmat.finalize();
        return rc;
      }
      sprintf(filename, "gemm-%c%c.bin", transFlags[i], transFlags[j]);
      hFile = fopen(filename, "w");
      hmat.write_data(hmatrix_result, io_write, hFile);
      fclose(hFile);

      switch (arithmetic) {
      case 'S':
        cblas_sgemm(CblasColMajor, transFlagsCblas[i], transFlagsCblas[j], n, n, n,
           -1.0, (float*) ctx_input.values, n, (float*) ctx_input.values, n, 1.0, (float*) ctx_output.values, n);
        fprintf(stdout, "diff=%g\n", cblas_snrm2(n, ctx_output.values, 1) / n / n);
        break;
      case 'D':
        cblas_dgemm(CblasColMajor, transFlagsCblas[i], transFlagsCblas[j], n, n, n,
           -1.0, (double*) ctx_input.values, n, (double*) ctx_input.values, n, 1.0, (double*) ctx_output.values, n);
        fprintf(stdout, "diff=%g\n", cblas_dnrm2(n, ctx_output.values, 1) / n / n);
        break;
      case 'C':
        {
        *((float_complex*) minus_one) = make_float_complex(-1.0, 0.0);
        cblas_cgemm(CblasColMajor, transFlagsCblas[i], transFlagsCblas[j], n, n, n,
           minus_one, ctx_input.values, n, ctx_input.values, n, alpha, ctx_output.values, n);
        fprintf(stdout, "diff=%g\n", cblas_scnrm2(n, ctx_output.values, 1) / n / n);
        }
        break;
      case 'Z':
        {
        *((double_complex*) minus_one) = make_double_complex(-1.0, 0.0);
        cblas_zgemm(CblasColMajor, transFlagsCblas[i], transFlagsCblas[j], n, n, n,
           minus_one, ctx_input.values, n, ctx_input.values, n, alpha, ctx_output.values, n);
        fprintf(stdout, "diff=%g\n", cblas_dznrm2(n, ctx_output.values, 1) / n / n);
        }
        break;
      default:
        fprintf(stderr, "Unknown arithmetic code %c, exiting...\n", arithmetic);
        return 1;
      }

      sprintf(filename, "diff-%c%c.bin", transFlags[i], transFlags[j]);
      hFile = fopen(filename, "w");
      hmat.write_data(hmatrix_result, io_write, hFile);
      fclose(hFile);

    }
  }
  hmat.destroy(hmatrix_result);
  hmat.destroy(hmatrix);
  hmat_delete_cluster_tree(cluster_tree);
  hmat.finalize();
  free(alpha);
  free(beta);
  free(minus_one);
  return rc;
}
