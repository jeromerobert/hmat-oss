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
DofCoordinate* createCylinder(double radius, double step, int n) {
  DofCoordinate* result = (DofCoordinate*) malloc(n * sizeof(DofCoordinate));
  double length = 2 * M_PI * radius;
  int pointsPerCircle = length / step;
  double angleStep = 2 * M_PI / pointsPerCircle;
  int i;
  for (i = 0; i < n; i++) {
    result[i].x = radius * cos(angleStep * i);
    result[i].y = radius * sin(angleStep * i),
    result[i].z = (step * i) / pointsPerCircle;
    result[i].globalIndex = i;
  }
  return result;
}

/** Write points into file. */
void pointsToFile(DofCoordinate* points, int size, const char* filename) {
  int i;
  FILE * fp = fopen(filename, "w");
  for (i = 0; i < size; i++) {
    fprintf(fp, "%e %e %e\n", points[i].x, points[i].y, points[i].z);
  }
  fclose(fp);
}

typedef struct {
  int n;
  DofCoordinate* points;
  double k;
} problem_data_t;


/**
  Define interaction between 2 degrees of freedoms  (real case)
*/
void interaction_real(void* data, int i, int j, void* result)
{
  problem_data_t* pdata = (problem_data_t*) data;
  DofCoordinate* points = pdata->points;
  double dx = points[i].x - points[j].x;
  double dy = points[i].y - points[j].y;
  double dz = points[i].z - points[j].z;
  *((double*)result) = 1. / (sqrt(dx*dx + dy*dy + dz*dz) + 1e-10);
}

/**
  Define interaction between 2 degrees of freedoms  (complex case)
*/
void interaction_complex(void* data, int i, int j, void* result)
{
  problem_data_t* pdata = (problem_data_t*) data;
  DofCoordinate* points = pdata->points;
  double k = pdata->k;
  double dx = points[i].x - points[j].x;
  double dy = points[i].y - points[j].y;
  double dz = points[i].z - points[j].z;
  double distance = sqrt(dx*dx + dy*dy + dz*dz) + 1e-10;
  double realPart = cos(k * distance) / (4 * M_PI * distance);
  double imagPart = sin(k * distance) / (4 * M_PI * distance);
  *((double_complex*)result) = make_complex(realPart, imagPart);
}

int main(int argc, char **argv) {
  double radius, step, k;
  DofCoordinate* points;
  hmat_settings_t settings;
  int n;
  char arithmetic;
  hmat_cluster_tree_t* cluster_tree;
  hmat_matrix_t* hmatrix;
  hmat_interface_t hmat;
  hmat_value_t type;
  hmat_info_t mat_info;
  int rc;
  problem_data_t problem_data;
  int is_parallel_run = 0;

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
  settings.admissibilityFactor = 3.;

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

  cluster_tree = hmat_create_cluster_tree(points, n);
  printf("ClusterTree node count = %d\n", hmat_tree_nodes_count(cluster_tree));
  hmatrix = hmat.create_empty_hmatrix(cluster_tree, cluster_tree);
  hmat.hmat_get_info(hmatrix, &mat_info);
  printf("HMatrix node count = %d\n", mat_info.nr_block_clusters);
  if (type == HMAT_SIMPLE_PRECISION || type == HMAT_DOUBLE_PRECISION)
    rc = hmat.assemble_simple_interaction(hmatrix, &problem_data, interaction_real, 0);
  else
    rc = hmat.assemble_simple_interaction(hmatrix, &problem_data, interaction_complex, 0);
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
  return rc;
}
