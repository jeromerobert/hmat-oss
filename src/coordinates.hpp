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

/*! \file
  \ingroup HMatrix
  \brief Geometric coordinates.
*/
#ifndef _COORDINATES_HPP
#define _COORDINATES_HPP

#include <cmath>

namespace hmat {

/*! \brief Class representing a 3D point.
 */
class Point {
public:
  union {
    struct {
      double x, y, z; /// Coordinates
    };
    double xyz[3]; /// Idem
  };
  Point(double _x = 0., double _y = 0., double _z = 0.) : x(_x), y(_y), z(_z) {}
  /*! \brief Return d(this, other)
   */
  inline double distanceTo(const Point &other) const {
    double result = 0.;
    double difference = 0.;
    difference = x - other.x;
    result += difference * difference;
    difference = y - other.y;
    result += difference * difference;
    difference = z - other.z;
    result += difference * difference;
    return sqrt(result);
  }
};


class DofCoordinates {
public:
  /*! \brief Create dof coordinates.

      \param coord  geometrical coordinates
      \param dim    spatial dimension
      \param size   number of points
      \param ownsMemory if true, coordinates are copied
   */
  DofCoordinates(double* coord, int dim, int size, bool ownsMemory = false);

  /*! \brief Copy constructor.

      \param other  instance being copied
   */
  DofCoordinates(const DofCoordinates& other);

  /*! \brief Destructor.
   */
  ~DofCoordinates();

  /*! \brief Get number of points.
   */
  inline int size() const { return size_; }

  /*! \brief Get spatial dimension.
   */
  inline int dimension() const { return dimension_; }

  /*! \brief Accessor to array element.
   */
  inline double& get(int i, int j) { return v_[j * dimension_ + i]; }

  /*! \brief Accessor to array element.
   */
  inline const double& get(int i, int j) const { return v_[j * dimension_ + i]; }

private:
  /// Array
  double* v_;

  /// Spatial dimension
  const int dimension_;

  /// Number of points
  const int size_;

  /// Flag to tell whether array has been copied and must be freed by destructor
  const bool ownsMemory_;
};

} // end namespace hmat

#endif
