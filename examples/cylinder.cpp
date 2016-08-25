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

using namespace hmat;

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
  const DofCoordinates& points;
  /// Wavenumber for the complex case.
  double k;

public:
  /** Constructor.

      \param _points Point cloud
      \param _k Wavenumber
   */
  TestAssemblyFunction(const DofCoordinates& _points, double _k = 1.)
    : SimpleAssemblyFunction<T>(), points(_points), k(_k) {}
  typename Types<T>::dp interaction(int i, int j) const;
  double distanceTo(int i, int j) const;
};

template<typename T>
double
TestAssemblyFunction<T>::distanceTo(int i, int j) const
{
  double r = sqrt((points.get(0, i) - points.get(0, j))*(points.get(0, i) - points.get(0, j))+
                  (points.get(1, i) - points.get(1, j))*(points.get(1, i) - points.get(1, j))+
                  (points.get(2, i) - points.get(2, j))*(points.get(2, i) - points.get(2, j)));
  return r;
}


template<>
Types<S_t>::dp TestAssemblyFunction<S_t>::interaction(int i, int j) const {
  double distance = this->distanceTo(i, j) + 1e-10;
  return 1. / distance;
}
template<>
Types<D_t>::dp TestAssemblyFunction<D_t>::interaction(int i, int j) const {
  double distance = this->distanceTo(i, j) + 1e-10;
  return 1. / distance;
}
template<>
Types<C_t>::dp TestAssemblyFunction<C_t>::interaction(int i, int j) const {
  double distance = this->distanceTo(i, j) + 1e-10;
  Z_t result(cos(k * distance) / (4 * M_PI * distance), sin(k * distance) / (4 * M_PI * distance));
  return result;
}
template<>
Types<Z_t>::dp TestAssemblyFunction<Z_t>::interaction(int i, int j) const {
  double distance = this->distanceTo(i, j) + 1e-10;
  Z_t result(cos(k * distance) / (4 * M_PI * distance), sin(k * distance) / (4 * M_PI * distance));
  return result;
}


template<typename T, template<typename> class E> struct Configuration
{
    void configure(HMatInterface<T,E> &){}
};

hmat::StandardAdmissibilityCondition admissibilityCondition(3.);

template<typename T, template<typename> class E>
void go(const DofCoordinates& coord, double k) {
  if (0 != HMatInterface<T, E>::init())
    return;
  {
    ClusterTree* ct = createClusterTree(coord);
    std::cout << "ClusterTree node count = " << ct->nodesCount() << std::endl;
    TestAssemblyFunction<T> f(coord, k);
    HMatInterface<T, E> hmat(ct, ct, kNotSymmetric, &admissibilityCondition);
    std::cout << "HMatrix node count = " << hmat.nodesCount() << std::endl;
    Configuration<T, E>().configure(hmat);
    hmat.assemble(f, kNotSymmetric);
    hmat.factorize(hmat_factorization_lu);
    hmat_info_t info;
    hmat.info(info);
    std::cout << "Compression Ratio = "
              << 100 * ((double) info.compressed_size) / info.uncompressed_size
              << "%" << std::endl;
    hmat.createPostcriptFile("h_matrix.ps");
  }
  HMatInterface<T, E>::finalize();
}

template<template<typename> class E>
void goA(char arithmetic, const DofCoordinates& coord, double k) {
    switch (arithmetic) {
    case 'S':
        go<S_t, E>(coord, k);
        break;
    case 'D':
        go<D_t, E>(coord, k);
        break;
    case 'C':
        go<C_t, E>(coord, k);
        break;
    case 'Z':
        go<Z_t, E>(coord, k);
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
  settings.setParameters();
  settings.printSettings();
  std::cout << "Generating the point cloud...";
  double radius = 1.;
  double step = 1.75 * M_PI * radius / sqrt((double)n);
  double k = 2 * M_PI / (10. * step); // 10 points / lambda
  std::vector<Point> points = createCylinder(radius, step, n);
  double * xyz = new double[3*points.size()];
  for(size_t i = 0; i < points.size(); ++i)
  {
    xyz[3*i+0] = points[i].x;
    xyz[3*i+1] = points[i].y;
    xyz[3*i+2] = points[i].z;
  }
  DofCoordinates coord(xyz, 3, points.size(), true);
  std::cout << "done.\n";

  pointsToFile(points, "points.txt");

  goA<DefaultEngine>(arithmetic, coord, k);
  return 0;
}
