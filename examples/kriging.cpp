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

/** Sample application for the HMatrix library.
*/
#include "full_matrix.hpp"
#include "interaction.hpp"
#include "data_types.hpp"
#include "hmat_cpp_interface.hpp"
#include <cstring>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>

using namespace hmat;

class KrigingAssemblyFunction : public SimpleAssemblyFunction<D_t> {
private:
  std::vector<Point> points;
  double l;

public:
  /** Constructor.

      \param _mat The FullMatrix<T> the values are taken from.
   */
  KrigingAssemblyFunction(std::vector<Point>& _points, double _l)
    : SimpleAssemblyFunction<D_t>(), points(_points), l(_l) {}

  D_t interaction(int i, int j) const {
    D_t r = points[i].distanceTo(points[j]);
    // Exponential
    return exp(-fabs(r) / l);
  }
};


ClusterTree* createClusterTree(const std::vector<Point>& points) {
  int n = (int) points.size();
  DofCoordinate* dls = new DofCoordinate[n];

  for (int i = 0; i < n; i++) {
    dls[i].x = points[i].x;
    dls[i].y = points[i].y;
    dls[i].z = points[i].z;
  }
  // We leak dls...
  return createClusterTree(dls, n);
}

void readPointsFromFile(const char* filename, std::vector<Point>& points) {
  std::ifstream f(filename);
  std::string line;
  while (getline(f, line)) {
    double x = 0, y = 0, z = 0;
    sscanf(line.c_str(), "%le %le %le", &x, &y, &z);
    points.push_back(Point(x, y, z));
  }
}

FullMatrix<D_t>* createRhs(const std::vector<Point>& points, double l) {
  const int n = (int) points.size();
  FullMatrix<D_t>* rhs = FullMatrix<D_t>::Zero(n, 1);

  Point center(0., 0., 0.);
  for (int i = 0; i < n; i++) {
    center.x += points[i].x;
    center.y += points[i].y;
    center.z += points[i].z;
  }
  center.x /= n;
  center.y /= n;
  center.z /= n;

  for (int i = 0; i < n; i++) {
    double r = center.distanceTo(points[i]);
    rhs->get(i, 0) = exp(-fabs(r) / l);
  }
  return rhs;
}

double correlationLength(const std::vector<Point>& points) {
  using std::max;
  using std::min;
  Point pMin(points[0]), pMax(points[0]);
  const size_t n = points.size();
  for (size_t i = 0; i < n; i++) {
    for (int coord = 0; coord < 3; coord++) {
      pMin.xyz[coord] = min(pMin.xyz[coord], points[i].xyz[coord]);
      pMax.xyz[coord] = max(pMax.xyz[coord], points[i].xyz[coord]);
    }
  }
  double l = .1 * max(max(pMax.x - pMin.x, pMax.y - pMin.y), pMax.z - pMin.z);
  return l;
}


template<template<typename> class E>
int go(const char* pointsFilename) {
  if (0 != HMatInterface<D_t, E>::init()) return 1;

  HMatSettings& settings = HMatSettings::getInstance();
  settings.compressionMethod = AcaPlus;
  settings.setParameters();

  std::cout << "Load points...";
  std::vector<Point> points;
  readPointsFromFile(pointsFilename, points);
  const int n = (int) points.size();
  std::cout << n << std::endl;

  const double l = correlationLength(points);
  FullMatrix<D_t>* rhs = createRhs(points, l);
  FullMatrix<D_t> rhsCopy(n, 1);
  rhsCopy.copyMatrixAtOffset(rhs, 0, 0);

  KrigingAssemblyFunction f(points, l);

  ClusterTree* ct = createClusterTree(points);
  std::cout << "ClusterTree node count = " << ct->nodesCount() << std::endl;
  // Either store lower triangular matrix and use LDLt (or LLt) factorization or
  // store full matrix and use LU factorization.
#if 0
  HMatInterface<D_t, E> hmat(ct, ct, kLowerSymmetric);
  settings.useLdlt = true;
  settings.useLu = false;
  settings.cholesky = false;
#else
  HMatInterface<D_t, E> hmat(ct, ct, kNotSymmetric);
  settings.useLdlt = false;
  settings.useLu = true;
  settings.cholesky = false;
#endif
  settings.setParameters();

  hmat.assemble(f, kLowerSymmetric);

  std::pair<size_t, size_t> compressionRatio = hmat.compressionRatio();
  std::cout << "Compression Ratio = "
            << 100 * ((double) compressionRatio.first) / compressionRatio.second
            << "%" << std::endl;

  std::cout << "done.\nFactorisation...";

  hmat.factorize();

  std::cout << "Resolution...";
  hmat.solve(*rhs);
  std::cout << "done." << std::endl;

  std::cout << "Accuracy...";
  double rhsCopyNorm = rhsCopy.norm();
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      rhsCopy.get(i, 0) -= f.interaction(i, j) * rhs->get(j, 0);
    }
  }
  double diffNorm = rhsCopy.norm();
  std::cout << "Done" << std::endl;
  std::cout << "||Ax - b|| / ||b|| = " << diffNorm / rhsCopyNorm << std::endl;

  delete rhs;
  HMatInterface<D_t, E>::finalize();
  return 0;
}


int main(int argc, char **argv) {
  if (argc != 2) {
      fprintf(stderr, "Usage: %s filename\n", argv[0]);
      return 1;
  }
  return go<DefaultEngine>(argv[1]);
}
