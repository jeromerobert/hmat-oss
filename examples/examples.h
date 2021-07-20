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

/** Common functions to multiple examples */

#include <math.h>
#include <stdio.h>

/** Create an open cylinder point cloud.

    \param radius Radius of the cylinder
    \param step distance between two neighboring points
    \param n number of points
    \return a vector of points.
 */
inline static double* createCylinder(double radius, double step, int n) {
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
inline static void pointsToFile(double* points, int size, const char* filename) {
  int i;
  FILE * fp = fopen(filename, "w");
  for (i = 0; i < size; i++) {
      fprintf(fp, "%e %e %e\n", points[3*i], points[3*i+1], points[3*i+2]);
  }
  fclose(fp);
}

inline static double distanceTo(double* center, double* points){
  double r = sqrt((center[0] - points[0])*(center[0] - points[0]) +
                  (center[1] - points[1])*(center[1] - points[1]) +
                  (center[2] - points[2])*(center[2] - points[2]));
  return r;
}

inline static double* createRhs(double *points, int n, double l) {
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

inline static double correlationLength(double * points, size_t n) {
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

