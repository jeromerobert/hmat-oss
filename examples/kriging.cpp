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
#include "assembly.hpp"
#include "data_types.hpp"
#include "hmat_cpp_interface.hpp"
#include "default_engine.hpp"
#include <cstring>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <algorithm>

using namespace hmat;

D_t
distanceTo(const DofCoordinates& points, int i, int j)
{
  D_t r = sqrt((points.get(0, i) - points.get(0, j))*(points.get(0, i) - points.get(0, j))+
               (points.get(1, i) - points.get(1, j))*(points.get(1, i) - points.get(1, j))+
               (points.get(2, i) - points.get(2, j))*(points.get(2, i) - points.get(2, j)));
  return r;
}

D_t
distanceTo(const DofCoordinates& points, int i, const Point& to)
{
  D_t r = sqrt((points.get(0, i) - to.x)*(points.get(0, i) - to.x)+
               (points.get(1, i) - to.y)*(points.get(1, i) - to.y)+
               (points.get(2, i) - to.z)*(points.get(2, i) - to.z));
  return r;
}

class KrigingFunction {
    const DofCoordinates& points;
    double l;

public:
    KrigingFunction(const DofCoordinates& _points, double _l):
         points(_points) , l(_l) {}

    D_t interaction(int i, int j) const {
        return exp(-fabs(distanceTo(points, i, j)) / l);
    }

    // static for C API
    static void compute(void* me, int i, int j, void* result) {
        *static_cast<D_t*>(result) = static_cast<KrigingFunction *>(me)->interaction(i, j);
    }
};

void readPointsFromFile(const char* filename, std::vector<Point>& points) {
  std::ifstream f(filename);
  std::string line;
  while (getline(f, line)) {
    double x = 0, y = 0, z = 0;
    sscanf(line.c_str(), "%le %le %le", &x, &y, &z);
    points.push_back(Point(x, y, z));
  }
}

ScalarArray<D_t>* createRhs(const DofCoordinates& coord, double l) {
  const int n = (int) coord.size();
  ScalarArray<D_t>* rhs = new ScalarArray<D_t>(n, 1);

  Point center(0., 0., 0.);
  for (int i = 0; i < n; i++) {
    center.x += coord.get(0, i);
    center.y += coord.get(1, i);
    center.z += coord.get(2, i);
  }
  center.x /= n;
  center.y /= n;
  center.z /= n;

  for (int i = 0; i < n; i++) {
    double r = distanceTo(coord, i, center);
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
  if (0 != E<D_t>::init()) return 1;

  HMatSettings& settings = HMatSettings::getInstance();
  settings.setParameters();

  std::cout << "Load points...";
  std::vector<Point> points;
  readPointsFromFile(pointsFilename, points);
  const int n = (int) points.size();
  std::cout << n << std::endl;
  double * xyz = new double[3*points.size()];
  for(size_t i = 0; i < points.size(); ++i)
  {
    xyz[3*i+0] = points[i].x;
    xyz[3*i+1] = points[i].y;
    xyz[3*i+2] = points[i].z;
  }
  DofCoordinates coord(xyz, 3, points.size(), true);

  const double l = correlationLength(points);
  ScalarArray<D_t>* rhs = createRhs(coord, l);
  ScalarArray<D_t> rhsCopy(n, 1);
  rhsCopy.copyMatrixAtOffset(rhs, 0, 0);
  KrigingFunction kriginFunction(coord, l);
  AssemblyFunction<D_t, SimpleFunction> f(
      SimpleFunction<D_t>(&KrigingFunction::compute, &kriginFunction), new CompressionAcaPlus(1e-4));

  ClusterTree* ct = createClusterTree(coord);
  std::cout << "ClusterTree node count = " << ct->nodesCount() << std::endl;
  // Either store lower triangular matrix and use LDLt (or LLt) factorization or
  // store full matrix and use LU factorization.
  IEngine<D_t>* engine = new E<D_t>();
  HMatInterface<D_t> hmat(engine, ct, ct, kNotSymmetric);
  settings.setParameters();

  hmat.assemble(f, kLowerSymmetric);
  hmat_info_t info;
  hmat.info(info);
  std::cout << "Compression Ratio = "
            << 100 * ((double) info.compressed_size) / info.uncompressed_size
            << "%" << std::endl;

  std::cout << "done.\nFactorisation...";

  hmat.factorize(hmat::Factorization::LU);

  std::cout << "Resolution...";
  hmat.solve(*rhs);
  std::cout << "done." << std::endl;

  std::cout << "Accuracy...";
  double rhsCopyNorm = rhsCopy.norm();
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      rhsCopy.get(i, 0) -= kriginFunction.interaction(i, j) * rhs->get(j, 0);
    }
  }
  double diffNorm = rhsCopy.norm();
  std::cout << "Done" << std::endl;
  std::cout << "||Ax - b|| / ||b|| = " << diffNorm / rhsCopyNorm << std::endl;

  delete rhs;
  E<D_t>::finalize();
  return 0;
}


int main(int argc, char **argv) {
  if (argc != 2) {
      fprintf(stderr, "Usage: %s filename\n", argv[0]);
      return 1;
  }
  return go<DefaultEngine>(argv[1]);
}
