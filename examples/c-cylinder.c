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
#define make_complex(realPart, imagPart) \
    std::complex<double>(realPart, imagPart)
#else
#include <complex.h>
typedef double complex double_complex;
#define make_complex(realPart, imagPart) \
    realPart + imagPart * _Complex_I
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

/** Write points into file. */
void pointsToFile(double* points, int size, const char* filename) {
  int i;
  FILE * fp = fopen(filename, "w");
  for (i = 0; i < size; i++) {
    fprintf(fp, "%e %e %e\n", points[3*i], points[3*i+1], points[3*i+2]);
  }
  fclose(fp);
}


/**
  Define interaction between 2 degrees of freedoms  (real case)
*/
double interaction_real(double* points, int i, int j)
{
  double dx = points[3*i+0] - points[3*j+0];
  double dy = points[3*i+1] - points[3*j+1];
  double dz = points[3*i+2] - points[3*j+2];
  return 1. / (sqrt(dx*dx + dy*dy + dz*dz) + 1e-10);
}

/**
  Define interaction between 2 degrees of freedoms  (complex case)
*/
double_complex
interaction_complex(double* points, double k, int i, int j)
{
  double dx = points[3*i+0] - points[3*j+0];
  double dy = points[3*i+1] - points[3*j+1];
  double dz = points[3*i+2] - points[3*j+2];
  double distance = sqrt(dx*dx + dy*dy + dz*dz) + 1e-10;
  double realPart = cos(k * distance) / (4 * M_PI * distance);
  double imagPart = sin(k * distance) / (4 * M_PI * distance);
  return make_complex(realPart, imagPart);
}

/**
  This structure contains data related to our problem.
*/
typedef struct {
  int type;
  int n;
  double* points;
  double k;
} problem_data_t;

/**
  This structure is a glue between hmat library and application.
*/
typedef struct {
  int row_start;
  int col_start;
  int* row_hmat2client;
  int* col_hmat2client;
  problem_data_t* user_context;
} block_data_t;

/**
  Function to free our block_data_t structure.  As we only store pointers, there is no
*/
void
free_hmat(void *data)
{
  free((block_data_t*) data);
}

/**
  prepare_hmat is called by hmat library to prepare assembly of
  a cluster block.  We allocate a block_data_t, which is then
  passed to the compute_hmat function.  We have to store all
  datas needed by compute_hmat into this block_data_t structure.
*/
void
prepare_hmat(int row_start,
             int row_count,
             int col_start,
             int col_count,
             int *row_hmat2client,
             int *row_client2hmat,
             int *col_hmat2client,
             int *col_client2hmat,
             void *user_context,
             hmat_block_info_t * block_info)
{
  block_info->user_data = calloc(1, sizeof(block_data_t));
  block_info->release_user_data = free_hmat;
  block_data_t* bdata = (block_data_t*) block_info->user_data;

  bdata->row_start = row_start;
  bdata->col_start = col_start;
  bdata->row_hmat2client = row_hmat2client;
  bdata->col_hmat2client = col_hmat2client;
  bdata->user_context = (problem_data_t*) user_context;
}

/**
  Compute all values of a block and store them into an array,
  which had already been allocated by hmat.  There is no
  padding, all values computed in this block are contiguous,
  and stored in a column-major order.
  This block is not necessarily identical to the one previously
  processed by prepare_hmat, it may be a sub-block.  In fact,
  it is either the full block, or a full column, or a full row.
*/
void
compute_hmat(void *data,
             int rowBlockBegin,
             int rowBlockCount,
             int colBlockBegin,
             int colBlockCount,
             void *values)
{
  int i, j;
  double *dValues = (double *) values;
  double_complex *zValues = (double_complex *) values;
  block_data_t* bdata = (block_data_t*) data;

  int type = bdata->user_context->type;
  int pos = 0;
  for (j = 0; j < colBlockCount; ++j) {
      int col = bdata->col_hmat2client[j + colBlockBegin + bdata->col_start];
      switch (type) {
	case HMAT_SIMPLE_PRECISION:
	case HMAT_DOUBLE_PRECISION:
	  for (i = 0; i < rowBlockCount; ++i, ++pos)
	    dValues[pos] = interaction_real(bdata->user_context->points, bdata->row_hmat2client[i + rowBlockBegin + bdata->row_start], col);
	  break;
	case HMAT_SIMPLE_COMPLEX:
	case HMAT_DOUBLE_COMPLEX:
	  for (i = 0; i < rowBlockCount; ++i, ++pos)
	    zValues[pos] = interaction_complex(bdata->user_context->points, bdata->user_context->k, bdata->row_hmat2client[i + rowBlockBegin + bdata->row_start], col);
	  break;
      }
  }
}

int main(int argc, char **argv) {
  double radius, step, k;
  double* points;
  hmat_settings_t settings;
  hmat_interface_t hmat;
  hmat_value_t scalar_type;
  hmat_info_t mat_info;
  int n;
  char arithmetic;
  hmat_coordinates_t* coordinates;
  hmat_clustering_algorithm_t* clustering;
  hmat_cluster_tree_t* cluster_tree;
  hmat_matrix_t* hmatrix;
  int rc;
  problem_data_t problem_data;

  if (argc != 3) {
    fprintf(stderr, "Usage: %s n_points (S|D|C|Z)\n", argv[0]);
    return 1;
  }

  n = atoi(argv[1]);
  arithmetic = argv[2][0];

  hmat_get_parameters(&settings);

  switch (arithmetic) {
  case 'S':
    scalar_type = HMAT_SIMPLE_PRECISION;
    break;
  case 'D':
    scalar_type = HMAT_DOUBLE_PRECISION;
    break;
  case 'C':
    scalar_type = HMAT_SIMPLE_COMPLEX;
    break;
  case 'Z':
    scalar_type = HMAT_DOUBLE_COMPLEX;
    break;
  default:
    fprintf(stderr, "Unknown arithmetic code %c\n", arithmetic);
    return 1;
  }

  hmat_init_default_interface(&hmat, scalar_type);
  settings.compressionMethod = hmat_compress_aca_plus;
  settings.admissibilityCondition = hmat_create_admissibility_standard(3.0);

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

  pointsToFile(points, n, "points.txt");

  problem_data.n = n;
  problem_data.points = points;
  problem_data.k = k;
  problem_data.type = scalar_type;

  coordinates = hmat_create_coordinates(points, 3, n);
  clustering = hmat_create_clustering_median();
  cluster_tree = hmat_create_cluster_tree(coordinates, clustering);
  hmat_delete_clustering(clustering);
  hmat_delete_coordinates(coordinates);
  printf("ClusterTree node count = %d\n", hmat_tree_nodes_count(cluster_tree));
  hmatrix = hmat.create_empty_hmatrix(cluster_tree, cluster_tree);
  hmat.hmat_get_info(hmatrix, &mat_info);
  printf("HMatrix node count = %d\n", mat_info.nr_block_clusters);
  rc = hmat.assemble(hmatrix, &problem_data, prepare_hmat, compute_hmat, 0);
  if (rc) {
    fprintf(stderr, "Error in assembly, return code is %d, exiting...\n", rc);
    hmat.finalize();
    return rc;
  }

  rc = hmat.factor(hmatrix);
  if (rc) {
    fprintf(stderr, "Error in factor, return code is %d, exiting...\n", rc);
    hmat.finalize();
    return rc;
  }

  hmat.finalize();
  hmat_delete_cluster_tree(cluster_tree);
  return 0;
}
