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
#include <complex.h>
#include "hmat/hmat.h"

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#if _WIN32
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
        bufptr = malloc(128);
        if (bufptr == NULL) {
                return -1;
        }
        size = 128;
    }
    p = bufptr;
    while(c != EOF) {
        if ((p - bufptr) > (size - 1)) {
                size = size + 128;
                bufptr = realloc(bufptr, size);
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


/** Write points into file. */
void pointsToFile(DofCoordinate* points, int size, const char* filename) {
  int i;
  FILE * fp = fopen(filename, "w");
  for (i = 0; i < size; i++) {
      fprintf(fp, "%e %e %e\n", points[i].x, points[i].y, points[i].z);
  }
  fclose(fp);
}

/** Read points from file. */
void readPointsFromFile(const char* filename, DofCoordinate **points, int *size) {
  FILE * fp=NULL;
  int  k;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;

  double x = 0, y = 0, z = 0;
  size_t np = 1000000, rnp = 500000;
  fp = fopen(filename, "r");
  *points = malloc(np * sizeof(DofCoordinate));
  k = 0;
  while ((read = getline(&line, &len, fp)) != -1) {
      if(k>=np) *points = realloc(*points, (k+rnp) * sizeof(DofCoordinate));
      sscanf(line, "%lf %lf %lf\n", &x, &y, &z);
      (*points)[k].globalIndex = k;
      (*points)[k].x = x;
      (*points)[k].y = y;
      (*points)[k].z = z;
      k++;
  }
   *points = realloc(*points, k * sizeof(DofCoordinate));
  fclose(fp);
  free(line);
  *size = k;
  return;
}


double distanceTo(DofCoordinate center, DofCoordinate points){
  double r = sqrt((center.x - points.x)*(center.x - points.x) +
                  (center.y - points.y)*(center.y - points.y) +
                  (center.z - points.z)*(center.z - points.z));
  return r;
}

double* createRhs(DofCoordinate *points, int n, double l) {
  double* rhs = calloc(n,  sizeof(double));
  int i;
  DofCoordinate center;
  center.x = 0.;
  center.y = 0.;
  center.z = 0.;

  for (i = 0; i < n; i++) {
      center.x += points[i].x;
      center.y += points[i].y;
      center.z += points[i].z;
  }
  center.x /= n;
  center.y /= n;
  center.z /= n;

  for (i = 0; i < n; i++) {
      double r = distanceTo(center, points[i]);
      rhs[i] = exp(-fabs(r) / l);
  }
  return rhs;
}

typedef struct {
  int n;
  DofCoordinate* points;
  double l;
} problem_data_t;

double correlationLength(DofCoordinate * points, size_t n) {
  size_t i;
  DofCoordinate pMin, pMax;
  pMin.x = points[0].x; pMin.y = points[0].y; pMin.z = points[0].z;
  pMax.x = points[0].x; pMax.y = points[0].y; pMax.z = points[0].z;
  for (i = 0; i < n; i++) {
      if (points[i].x < pMin.x) pMin.x =  points[i].x;
      if (points[i].x > pMax.x) pMax.x =  points[i].x;

      if (points[i].y < pMin.y) pMin.y =  points[i].y;
      if (points[i].y > pMax.y) pMax.y =  points[i].y;

      if (points[i].z < pMin.z) pMin.z =  points[i].z;
      if (points[i].z > pMax.z) pMax.z =  points[i].z;
  }
  double l = pMax.x - pMin.x;
  if (pMax.y - pMin.y > l) l = pMax.y - pMin.y;
  if (pMax.z - pMin.z > l) l = pMax.z - pMin.z;
  return 0.1 * l;
}
/**
  Define interaction between 2 degrees of freedoms  (real case)
 */
void interaction_real(void* data, int i, int j, void* result)
{
  problem_data_t* pdata = (problem_data_t*) data;
  DofCoordinate* points = pdata->points;
  double r = distanceTo(points[i], points[j]);

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
  DofCoordinate* points;
  hmat_interface_t hmat;
  hmat_settings_t settings;
  hmat_value_t type;
  int n;
  char arithmetic;
  void* cluster_tree;
  hmat_matrix_t * hmatrix;
  int kLowerSymmetric = 1; /* =0 if not Symmetric */
  int rc;

  problem_data_t problem_data;
  double l;

  int nrhs = 1;
  double *drhs, *drhsCopy, derr;
  float  *frhs, *frhsCopy, ferr;

  int is_parallel_run = 0;

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

  settings.compressionMethod = hmat_compress_aca_plus;
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
    frhs     =  calloc(n, sizeof(float));
    frhsCopy =  calloc(n, sizeof(float));
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

  cluster_tree = create_cluster_tree(points, n);
  printf("ClusterTree node count = %d\n", hmat_tree_nodes_count(cluster_tree));
  hmatrix = hmat.create_empty_hmatrix(cluster_tree, cluster_tree);

  fprintf(stdout,"Assembly...");
  rc = hmat.assemble_simple_interaction(hmatrix, &problem_data, interaction_real, kLowerSymmetric);
  if (rc) {
    fprintf(stderr, "Error in assembly, return code is %d, exiting...\n", rc);
    hmat.finalize();
    return rc;
  }
  fprintf(stdout, "done.\n");

  fprintf(stdout,"Factorisation...");
  rc = hmat.factor(hmatrix);
  if (rc) {
    fprintf(stderr, "Error in factorisation, return code is %d, exiting...\n", rc);
    hmat.finalize();
    return rc;
  }
  fprintf(stdout, "done.\n");

  fprintf(stdout,"Solve...");
  if(type == HMAT_SIMPLE_PRECISION){
    hmat.solve_systems(hmatrix, frhs, nrhs);
  }else{
    hmat.solve_systems(hmatrix, drhs, nrhs);
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

  hmat.finalize();
  return 0;

}


