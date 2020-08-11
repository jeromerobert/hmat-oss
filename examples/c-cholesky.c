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

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "hmat/hmat.h"

/** This is a simple example showing how to use the HMatrix library.  */

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

double distanceTo(double* center, double* points){
  double r = sqrt((center[0] - points[0])*(center[0] - points[0]) +
                  (center[1] - points[1])*(center[1] - points[1]) +
                  (center[2] - points[2])*(center[2] - points[2]));
  return r;
}

double* createRhs(double *points, int n, double l) {
  double* rhs = (double*) calloc(n,  sizeof(double));
  int i;
  double center[3];
  center[0] = 0.;
  center[1] = 0.;
  center[2] = 0.;

  for (i = 0; i < n; i++) {
      center[0] += points[3*i+0];
      center[1] += points[3*i+1];
      center[2] += points[3*i+2];
  }
  center[0] /= n;
  center[1] /= n;
  center[2] /= n;

  for (i = 0; i < n; i++) {
      double r = distanceTo(center, &points[3*i]);
      rhs[i] = exp(-fabs(r) / l);
  }
  return rhs;
}

typedef struct {
  int n;
  double* points;
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
  double r = distanceTo(&points[3*i], &points[3*j]);

  *((double*)result) = exp(-fabs(r) / pdata->l);
}

void double_precision_error(problem_data_t *problem_data, int n, double *rhs, double *rhsCopy, double *result)
{
  int i,j;
  double rhsCopyNorm = 0.;
  double rhsNorm = 0.;
  double diffNorm = 0;
  double diff;
  double a;

  for (i = 0; i < n; i++) {
      rhsCopyNorm += rhsCopy[i] * rhsCopy[i];
  }

  rhsCopyNorm = sqrt(rhsCopyNorm);
  fprintf(stdout, "\n||b|| = %e\n", rhsCopyNorm);

  for (i = 0; i < n; i++) {
      rhsNorm += rhs[i] * rhs[i];
  }

  rhsNorm = sqrt(rhsNorm);
  fprintf(stdout, "||x|| = %e\n", rhsNorm);

  for (i = 0; i < n; i++) {
      diff = rhsCopy[i];
      for (j = 0; j < n; j++) {
	  interaction_real(problem_data, i, j, &a);
	  diff      -= a * rhs[j];
      }
      diffNorm += diff*diff;
  }

  *result = diffNorm/rhsCopyNorm;

}

void simple_precision_error(problem_data_t *problem_data, int n, float *rhs, float *rhsCopy, float *result)
{
  int i,j;
  float rhsCopyNorm = 0.;
  float rhsNorm = 0.;
  float diffNorm = 0;
  float diff;
  double a;

  for (i = 0; i < n; i++) {
      rhsCopyNorm += rhsCopy[i] * rhsCopy[i];
  }

  rhsCopyNorm = sqrt(rhsCopyNorm);
  fprintf(stdout, "\n||b|| = %e\n", rhsCopyNorm);

  for (i = 0; i < n; i++) {
      rhsNorm += rhs[i] * rhs[i];
  }

  rhsNorm = sqrt(rhsNorm);
  fprintf(stdout, "||x|| = %e\n", rhsNorm);

  for (i = 0; i < n; i++) {
      diff = rhsCopy[i];
      for (j = 0; j < n; j++) {
	  interaction_real(problem_data, i, j, &a);
	  diff      -= a * rhs[j];
      }
      diffNorm += diff*diff;
  }

  *result = diffNorm/rhsCopyNorm;

}

int main(int argc, char **argv) {
  int i;
  double radius, step;
  double* points;
  int n;
  hmat_interface_t hmat;
  hmat_settings_t settings;
  hmat_value_t type;
  hmat_info_t mat_info;
  char arithmetic;
  hmat_clustering_algorithm_t* clustering;
  hmat_clustering_algorithm_t* clustering_shuffle;
  hmat_clustering_algorithm_t* clustering_max_dof;
  hmat_cluster_tree_t* cluster_tree;
  hmat_matrix_t * hmatrix;
  int kLowerSymmetric = 1; /* =0 if not Symmetric */

  problem_data_t problem_data;
  double l;

  int nrhs = 1;
  double *drhs, *drhsCopy1, *drhsCopy2, *drhsCopy3, derr;
  float  *frhs, *frhsCopy1, *frhsCopy2, *frhsCopy3, ferr;
  double diffNorm;

  if (argc != 3) {
      fprintf(stderr, "Usage: %s n_points (S|D)\n", argv[0]);
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
    default:
      fprintf(stderr, "Unknown arithmetic code %c, exiting...\n", arithmetic);
      return 1;
  }

  hmat_get_parameters(&settings);
  hmat_init_default_interface(&hmat, type);

  settings.compressionMethod = hmat_compress_aca_plus;
  /*settings->recompress = 0;*/
  /*settings->admissibilityFactor = 3.;*/

  hmat_set_parameters(&settings);
  if (0 != hmat.init())
  {
    fprintf(stderr, "Unable to initialize HMat library\n");
    return 1;
  }

  printf("Generating the point cloud...\n");
  radius = 1.;
  step = 1.75 * M_PI * radius / sqrt((double)n);
  points = createCylinder(radius, step, n);
  printf("done.\n");
  printf("n = %d\n", n);

  l = correlationLength(points, n);
  printf("correlationLength = %le\n", l);
  problem_data.n = n;
  problem_data.points = points;
  problem_data.l = l;

  drhs = createRhs(points, n, l);
  drhsCopy1 = createRhs(points, n, l);
  drhsCopy2 = createRhs(points, n, l);
  drhsCopy3 = createRhs(points, n, l);
  frhs     =  (float*) calloc(n, sizeof(float));
  frhsCopy1 =  (float*) calloc(n, sizeof(float));
  frhsCopy2 =  (float*) calloc(n, sizeof(float));
  frhsCopy3 =  (float*) calloc(n, sizeof(float));
  for(i=0;i<n;i++) {
    frhs[i] = drhs[i];
    frhsCopy1[i] = drhsCopy1[i];
  }

  clustering = hmat_create_clustering_median();
  /* To test with n-ary trees, change final argument to a value > 2 */
  clustering_shuffle = hmat_create_shuffle_clustering(clustering, 2, 2);
  /* Change maxLeafSize to have a deep tree */
  clustering_max_dof = hmat_create_clustering_max_dof(clustering_shuffle, 10);
  cluster_tree = hmat_create_cluster_tree(points, 3, n, clustering_max_dof);
  hmat_delete_clustering(clustering_max_dof);
  hmat_delete_clustering(clustering_shuffle);
  hmat_delete_clustering(clustering);
  printf("ClusterTree node count = %d\n", hmat_tree_nodes_count(cluster_tree));
  hmat_admissibility_t * admissibilityCondition = hmat_create_admissibility_standard(2.0);
  hmatrix = hmat.create_empty_hmatrix_admissibility(
            cluster_tree, cluster_tree, 1, admissibilityCondition);
  hmat_delete_admissibility(admissibilityCondition);
  hmat.get_info(hmatrix, &mat_info);
  printf("HMatrix node count = %d\n", mat_info.nr_block_clusters);

  fprintf(stdout,"Assembly...");
  hmat_assemble_context_t ctx_assemble;
  hmat_assemble_context_init(&ctx_assemble);
  ctx_assemble.user_context = &problem_data;
  ctx_assemble.simple_compute = interaction_real;
  ctx_assemble.lower_symmetric = kLowerSymmetric;
  hmat.assemble_generic(hmatrix, &ctx_assemble);
  fprintf(stdout, "done.\n");
  hmat.get_info(hmatrix, &mat_info);
  printf("HMatrix size = %ld, uncompressed size = %ld\n", mat_info.compressed_size, mat_info.uncompressed_size);

  fprintf(stdout,"Factorisation...");
  hmat_factorization_context_t ctx_facto;
  hmat_factorization_context_init(&ctx_facto);
  ctx_facto.factorization = hmat_factorization_llt;
  hmat.factorize_generic(hmatrix, &ctx_facto);
  fprintf(stdout, "done.\n");
  hmat.get_info(hmatrix, &mat_info);
  printf("HMatrix size = %ld, uncompressed size = %ld\n", mat_info.compressed_size, mat_info.uncompressed_size);

  fprintf(stdout,"Solve...");
  if(type == HMAT_SIMPLE_PRECISION){
    hmat.solve_systems(hmatrix, frhs, nrhs);
  }else{
    hmat.solve_systems(hmatrix, drhs, nrhs);
  }
  fprintf(stdout, "done.\n");

  /* Store solution of AX=B */
  memcpy(frhsCopy2, frhs, n*sizeof(float));
  memcpy(drhsCopy2, drhs, n*sizeof(double));

  fprintf(stdout, "Accuracy...");
  if(type == HMAT_SIMPLE_PRECISION){
    simple_precision_error(&problem_data, n, frhs, frhsCopy1, &ferr);
    fprintf(stdout, "||Ax - b|| / ||b|| = %e\n",  ferr);
  }else{
    double_precision_error(&problem_data, n, drhs, drhsCopy1, &derr);
    fprintf(stdout, "||Ax - b|| / ||b|| = %le\n",  derr);
  }

  /* Restore right-hand side */
  memcpy(frhs, frhsCopy1, n*sizeof(float));
  memcpy(drhs, drhsCopy1, n*sizeof(double));
  memcpy(frhsCopy3, frhsCopy1, n*sizeof(float));
  memcpy(drhsCopy3, drhsCopy1, n*sizeof(double));
  fprintf(stdout,"Solve Ly=b...");
  if(type == HMAT_SIMPLE_PRECISION){
    hmat.solve_lower_triangular(hmatrix, 0, frhs, nrhs);
  }else{
    hmat.solve_lower_triangular(hmatrix, 0, drhs, nrhs);
  }
  fprintf(stdout, "done.\n");

  fprintf(stdout, "Accuracy...");
  diffNorm = 0.;
  if(type == HMAT_SIMPLE_PRECISION){
    float pone=1., zero=0. ;
    hmat.gemv('N', &pone, hmatrix, frhs, &zero, frhsCopy1, 1);
    for (i = 0; i < n; i++)
      diffNorm += (frhsCopy1[i] - frhsCopy3[i])*(frhsCopy1[i] - frhsCopy3[i]);
  }else{
    double pone=1., zero=0. ;
    hmat.gemv('N', &pone, hmatrix, drhs, &zero, drhsCopy1, 1);
    for (i = 0; i < n; i++)
      diffNorm += (drhsCopy1[i] - drhsCopy3[i])*(drhsCopy1[i] - drhsCopy3[i]);
  }
  fprintf(stdout, "\n||Ly-b|| = %e\n", sqrt(diffNorm));

  fprintf(stdout,"Solve Ltx=y...");
  memcpy(frhsCopy3, frhs, n*sizeof(float));
  memcpy(drhsCopy3, drhs, n*sizeof(double));
  if(type == HMAT_SIMPLE_PRECISION){
    hmat.solve_lower_triangular(hmatrix, 1, frhs, nrhs);
  }else{
    hmat.solve_lower_triangular(hmatrix, 1, drhs, nrhs);
  }
  fprintf(stdout, "done.\n");

  fprintf(stdout, "Accuracy...");
  diffNorm = 0.;
  if(type == HMAT_SIMPLE_PRECISION){
    float pone=1., zero=0. ;
    hmat.gemv('T', &pone, hmatrix, frhs, &zero, frhsCopy1, 1);
    for (i = 0; i < n; i++)
      diffNorm += (frhsCopy1[i] - frhsCopy3[i])*(frhsCopy1[i] - frhsCopy3[i]);
  }else{
    double pone=1., zero=0. ;
    hmat.gemv('T', &pone, hmatrix, drhs, &zero, drhsCopy1, 1);
    for (i = 0; i < n; i++)
      diffNorm += (drhsCopy1[i] - drhsCopy3[i])*(drhsCopy1[i] - drhsCopy3[i]);
  }
  fprintf(stdout, "\n||Ltx-y|| = %e\n", sqrt(diffNorm));

  fprintf(stdout, "Compare LLtX=b and AX=b solutions...");
  diffNorm = 0.;
  if(type == HMAT_SIMPLE_PRECISION){
    for (i = 0; i < n; i++)
      diffNorm += (frhs[i] - frhsCopy2[i])*(frhs[i] - frhsCopy2[i]);
  }else{
    for (i = 0; i < n; i++)
      diffNorm += (drhs[i] - drhsCopy2[i])*(drhs[i] - drhsCopy2[i]);
  }
  fprintf(stdout, "\n||x1-x2|| = %e\n", sqrt(diffNorm));

  free(frhs);
  free(frhsCopy1);
  free(frhsCopy2);
  free(frhsCopy3);
  free(drhs);
  free(drhsCopy1);
  free(drhsCopy2);
  free(drhsCopy3);
  hmat.destroy(hmatrix);
  hmat_delete_cluster_tree(cluster_tree);
  hmat.finalize();
  free(points);
  return 0;

}


