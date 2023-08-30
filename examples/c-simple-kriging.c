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

#ifndef _GNU_SOURCE
// for getline with old glibc
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <math.h>
#ifdef __cplusplus
#include <complex>
typedef std::complex<double> double_complex;
#else
#include <complex.h>
#ifdef _MSC_VER
typedef _Dcomplex double_complex;
#define make_complex(realPart, imagPart) _Cbuild(realPart, imagPart)
#else
typedef double complex double_complex;
#endif
#endif
#include "hmat/hmat.h"
#include "examples.h"

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#ifdef _WIN32
// getline is not defined in mingw
#include <stdlib.h>
size_t getline(char **lineptr, size_t *n, FILE *stream) {
    char *bufptr = NULL;
    char *p = bufptr;
    size_t size;
    int c;

    if (lineptr == NULL) {
        return -1;
    }
    if (stream == NULL) {
        return -1;
    }
    if (n == NULL) {
        return -1;
    }
    bufptr = *lineptr;
    size = *n;

    c = fgetc(stream);
    if (c == EOF) {
        return -1;
    }
    if (bufptr == NULL) {
        bufptr = (char*) malloc(128);
        if (bufptr == NULL) {
                return -1;
        }
        size = 128;
    }
    p = bufptr;
    while(c != EOF) {
        if ((p - bufptr) > (size - 1)) {
                size = size + 128;
                bufptr = (char*) realloc(bufptr, size);
                if (bufptr == NULL) {
                        return -1;
                }
        }
        *p++ = c;
        if (c == '\n') {
                break;
        }
        c = fgetc(stream);
    }

    *p++ = '\0';
    *lineptr = bufptr;
    *n = size;

    return p - bufptr - 1;
}
#endif


/** This is a simple example showing how to use the HMatrix library.

    c version of kriging.cpp.
    The kernel function is the interaction one!

 */

/** Read points from file. */
void readPointsFromFile(const char* filename, double **points, int *size) {
  FILE * fp=NULL;
  int  k;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;

  double x = 0, y = 0, z = 0;
  int np = 1000000, rnp = 500000;
  fp = fopen(filename, "r");
  *points = (double*) malloc(3 * np * sizeof(double));
  k = 0;
  while ((read = getline(&line, &len, fp)) != -1) {
      if(k>=np) *points = (double*) realloc(*points, 3*(k+rnp) * sizeof(double));
      sscanf(line, "%lf %lf %lf\n", &x, &y, &z);
      (*points)[3*k+0] = x;
      (*points)[3*k+1] = y;
      (*points)[3*k+2] = z;
      k++;
  }
   *points = (double*) realloc(*points, 3 * k * sizeof(double));
  fclose(fp);
  free(line);
  *size = k;
  return;
}

typedef struct {
  int n;
  double* points;
  double l;
} problem_data_t;

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

  char *pointsFilename = NULL;
  double* points;
  hmat_interface_t hmat;
  hmat_settings_t settings;
  hmat_value_t type;
  hmat_info_t mat_info;
  int n;
  char arithmetic;
  hmat_clustering_algorithm_t* clustering;
  hmat_cluster_tree_t* cluster_tree;
  hmat_matrix_t * hmatrix;
  int kLowerSymmetric = 1; /* =0 if not Symmetric */

  problem_data_t problem_data;
  double l;

  int nrhs = 1;
  double *drhs=NULL, *drhsCopy=NULL, derr;
  float  *frhs=NULL, *frhsCopy=NULL, ferr;

  if (argc != 3) {
      fprintf(stderr, "Usage: %s pointsfilename (S|D)\n", argv[0]);
      return 1;
  }

  pointsFilename = argv[1];
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

  /*settings->recompress = 0;*/
  /*settings->admissibilityFactor = 3.;*/

  hmat_set_parameters(&settings);
  if (0 != hmat.init())
  {
    fprintf(stderr, "Unable to initialize HMat library\n");
    return 1;
  }

  printf("Load points...");
  readPointsFromFile(pointsFilename, &points, &n);
  printf("done\n");
  printf("n = %d\n", n);

  l = correlationLength(points, n);
  printf("correlationLength = %le\n", l);
  problem_data.n = n;
  problem_data.points = points;
  problem_data.l = l;

  if(type == HMAT_SIMPLE_PRECISION){
    drhs = createRhs(points, n, l);
    drhsCopy = createRhs(points, n, l);
    frhs     =  (float*) calloc(n, sizeof(float));
    frhsCopy =  (float*) calloc(n, sizeof(float));
    for(i=0;i<n;i++) {
      frhs[i] = drhs[i];
      frhsCopy[i] = drhsCopy[i];
    }
    free(drhs);
    free(drhsCopy);
  }else{
    drhs = createRhs(points, n, l);
    drhsCopy = createRhs(points, n, l);
  }

  clustering = hmat_create_clustering_median();
  cluster_tree = hmat_create_cluster_tree(points, 3, n, clustering);
  hmat_delete_clustering(clustering);
  printf("ClusterTree node count = %d\n", hmat_tree_nodes_count(cluster_tree));
  hmat_admissibility_t * admissibilityCondition = hmat_create_admissibility_standard(2.0);
  hmatrix = hmat.create_empty_hmatrix_admissibility(
            cluster_tree, cluster_tree, 0, admissibilityCondition);
  hmat_delete_admissibility(admissibilityCondition);
  hmat.get_info(hmatrix, &mat_info);
  printf("HMatrix node count = %d\n", mat_info.nr_block_clusters);

  fprintf(stdout,"Assembly...");
  hmat_assemble_context_t ctx_assemble;
  hmat_assemble_context_init(&ctx_assemble);
  ctx_assemble.user_context = &problem_data;
  ctx_assemble.compression = hmat_create_compression_aca_plus(1e-5);
  ctx_assemble.simple_compute = interaction_real;
  ctx_assemble.lower_symmetric = kLowerSymmetric;
  hmat.assemble_generic(hmatrix, &ctx_assemble);
  fprintf(stdout, "done.\n");
  hmat_delete_compression(ctx_assemble.compression);

  hmat.get_info(hmatrix, &mat_info);
  printf("Rk size = %gM\n", 1e-6 * mat_info.compressed_size);
  hmat.set_low_rank_epsilon(hmatrix, 1.e-2);
  fprintf(stdout,"Post-process Rk-matrices...");
  hmat.truncate(hmatrix);
  fprintf(stdout, "done.\n");
  hmat.get_info(hmatrix, &mat_info);
  printf("Rk size = %gM\n", 1e-6 * mat_info.compressed_size);

  fprintf(stdout,"Factorisation...");
  hmat_factorization_context_t ctx_facto;
  hmat_factorization_context_init(&ctx_facto);
  ctx_facto.factorization = hmat_factorization_lu;
  hmat.factorize_generic(hmatrix, &ctx_facto);
  fprintf(stdout, "done.\n");

  fprintf(stdout,"Solve...");
  if(type == HMAT_SIMPLE_PRECISION){
    hmat.vector_reorder(frhs, cluster_tree, 0, NULL, nrhs);
    hmat.solve_dense(hmatrix, frhs, nrhs);
    hmat.vector_restore(frhs, cluster_tree, 0, NULL, nrhs);
  }else{
    hmat.vector_reorder(drhs, cluster_tree, 0, NULL, nrhs);
    hmat.solve_dense(hmatrix, drhs, nrhs);
    hmat.vector_restore(drhs, cluster_tree, 0, NULL, nrhs);
  }
  fprintf(stdout, "done.\n");

  fprintf(stdout, "Accuracy...");
  if(type == HMAT_SIMPLE_PRECISION){
    simple_precision_error(&problem_data, n, frhs, frhsCopy, &ferr);
    fprintf(stdout, "||Ax - b|| / ||b|| = %e\n",  ferr);
    free(frhs);
    free(frhsCopy);
  }else{
    double_precision_error(&problem_data, n, drhs, drhsCopy, &derr);
    fprintf(stdout, "||Ax - b|| / ||b|| = %le\n",  derr);
    free(drhs);
    free(drhsCopy);
  }

  hmat.destroy(hmatrix);
  hmat_delete_cluster_tree(cluster_tree);
  hmat.finalize();
  return 0;

}


