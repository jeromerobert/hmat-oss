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
#include <iostream>
#include <cmath>

#include "hmat_cpp_interface.hpp"
#include "default_engine.hpp"
#include "my_chrono.hpp"

/** This is a simple example showing how to use the HMatrix library.

    In this example, we assemble and do a decomposition of a Matrix such that:
    \f[A_{ij} = \frac{e^{i\kappa |x_i - x_j|}}{4 \pi |x_i - x_j|}\f]
    with the points \f$(x_i)\f$ on a cylinder.
    In the real case we use 1 / r instead.

    The required steps are:
    - Create the point cloud in the \a createCylinder() function
    - Create a function returning any element (i, j) in the matrix:
      TestAssemblyFunction. We are using the unoptimzed and simpler
      SimpleAssemblyFunction as a base class here.
    - The rest of the code is in go():
      - Initialize the library
      - Create a ClusterTree (createClusterTree())
      - Create a HMatInterface<> instance
      - Assemble it hmat->assemble()
      - Factorisation: hmat->factorize()
 */


/** Create an open cylinder point cloud.

    \param radius Radius of the cylinder
    \param step distance between two neighboring points
    \param n number of points
    \return a vector of points.
 */
std::vector<Point> createCylinder(double radius, double step, int n) {
  std::vector<Point> result;
  double length = 2 * M_PI * radius;
  int pointsPerCircle = length / step;
  double angleStep = 2 * M_PI / pointsPerCircle;
  for (int i = 0; i < n; i++) {
    Point p(radius * cos(angleStep * i), radius * sin(angleStep * i),
            (step * i) / pointsPerCircle);
    result.push_back(p);
  }
  return result;
}

void pointsToFile(const std::vector<Point>& points, const char* filename) {
  std::ofstream f(filename);
  f << std::scientific;
  for (size_t i = 0; i < points.size(); i++) {
    f << points[i].x << " " << points[i].y << " " << points[i].z << std::endl;
  }
}


template<typename T>
class TestAssemblyFunction : public SimpleAssemblyFunction<T> {
public:
  /// Point coordinates
  std::vector<Point> points;
  /// Wavenumber for the complex case.
  double k;

public:
  /** Constructor.

      \param _points Point cloud
      \param _k Wavenumber
   */
  TestAssemblyFunction(std::vector<Point>& _points, double _k = 1.)
    : SimpleAssemblyFunction<T>(), points(_points), k(_k) {}
  typename Types<T>::dp interaction(int i, int j) const;
};


template<>
typename Types<S_t>::dp TestAssemblyFunction<S_t>::interaction(int i, int j) const {
  double distance = points[i].distanceTo(points[j]) + 1e-10;
  return 1. / distance;
}
template<>
typename Types<D_t>::dp TestAssemblyFunction<D_t>::interaction(int i, int j) const {
  double distance = points[i].distanceTo(points[j]) + 1e-10;
  return 1. / distance;
}
template<>
typename Types<C_t>::dp TestAssemblyFunction<C_t>::interaction(int i, int j) const {
  double distance = points[i].distanceTo(points[j]) + 1e-10;
  Z_t result = Constants<Z_t>::zero;
  result.real(cos(k * distance) / (4 * M_PI * distance));
  result.imag(sin(k * distance) / (4 * M_PI * distance));
  return result;
}
template<>
typename Types<Z_t>::dp TestAssemblyFunction<Z_t>::interaction(int i, int j) const {
  double distance = points[i].distanceTo(points[j]) + 1e-10;
  Z_t result = Constants<Z_t>::zero;
  result.real(cos(k * distance) / (4 * M_PI * distance));
  result.imag(sin(k * distance) / (4 * M_PI * distance));
  return result;
}


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

template<typename T, template<typename> class E> struct Configuration
{
    void configure(HMatInterface<T,E> & hmat){}
};

template<typename T, template<typename> class E>
void go(std::vector<Point>& points, double k) {
  if (0 != HMatInterface<T, E>::init())
    return;
  {
    ClusterTree* ct = createClusterTree(points);
    std::cout << "ClusterTree node count = " << ct->nodesCount() << std::endl;
    TestAssemblyFunction<T> f(points, k);
    HMatInterface<T, E> hmat(ct, ct);
    Configuration<T, E>().configure(hmat);
    Time tick = now();
    hmat.assemble(f, kNotSymmetric, false);
    hmat.factorize();
    Time tock = now();
    double duration = time_diff_in_nanos(tick, tock);
    std::cout << std::endl << "Assembly/LU time = " << duration / 1e9 << std::endl;
    std::pair<size_t, size_t> compressionRatio = hmat.compressionRatio();
    std::cout << "Compression Ratio = "
              << 100 * ((double) compressionRatio.first) / compressionRatio.second
              << "%" << std::endl;
    hmat.createPostcriptFile("h_matrix.ps");
  }
  HMatInterface<T, E>::finalize();
}

template<template<typename> class E>
void goA(char arithmetic, std::vector<Point>& points, double k) {
    switch (arithmetic) {
    case 'S':
        go<S_t, E>(points, k);
        break;
    case 'D':
        go<D_t, E>(points, k);
        break;
    case 'C':
        go<C_t, E>(points, k);
        break;
    case 'Z':
        go<Z_t, E>(points, k);
        break;
    default:
      std::cerr << "Unknown arithmetic code " << arithmetic << std::endl;
    }
}

int main(int argc, char **argv) {
  HMatSettings& settings = HMatSettings::getInstance();
  settings.maxParallelLeaves = 10000;

  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " n_points (S|D|C|Z)"
              << std::endl;
    return 0;
  }
  int n = atoi(argv[1]);
  char arithmetic = argv[2][0];

  settings.compressionMethod = AcaPlus;
  settings.admissibilityFactor = 3.;

  settings.setParameters();
  settings.printSettings();
  std::cout << "Generating the point cloud...";
  double radius = 1.;
  double step = 1.75 * M_PI * radius / sqrt((double)n);
  double k = 2 * M_PI / (10. * step); // 10 points / lambda
  std::vector<Point> points = createCylinder(radius, step, n);
  std::cout << "done.\n";

  pointsToFile(points, "points.txt");

  goA<DefaultEngine>(arithmetic, points, k);
  return 0;
}
