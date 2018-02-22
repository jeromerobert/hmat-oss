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
#include <cstddef>

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
  //TODO: why do we always copy (aka ownsMemory is always true)
  DofCoordinates(double* coord, unsigned dim, unsigned size, bool ownsMemory = false,
                 unsigned number_of_dof=0, unsigned * span_offsets = NULL, unsigned * spans = NULL);

  /*! \brief Copy constructor.

      \param other  instance being copied
   */
  DofCoordinates(const DofCoordinates& other);

  /*! \brief Destructor.
   */
  ~DofCoordinates();

  /*! \brief Get number of points.
   * \deprecated Legacy API which does not support spans
   */
  int size() const;

  /*! \brief Get spatial dimension.
   */
  inline int dimension() const { return dimension_; }

  /*! \brief Accessor to array element.
   * \deprecated Legacy API which does not support spans
   */
  double& get(int i, int j);

  /*! \brief Accessor to array element.
   * \deprecated Legacy API which does not support spans
   */
  const double& get(int i, int j) const;

  unsigned numberOfPoints() const { return size_; }
  unsigned numberOfDof() const {
      return spanOffsets_ == NULL ? size_ : numberOfDof_;
  }

  unsigned spanSize(unsigned dof) const {
      if(spanOffsets_ == NULL)
          return 1;
      else
          return dof == 0 ? spanOffsets_[0] :
              spanOffsets_[dof] - spanOffsets_[dof - 1];
  }

  double spanPoint(unsigned dof, unsigned pointId, unsigned dim) const {
      if(spanOffsets_ == NULL) {
          return v_[dof * dimension_ + dim];
      } else {
          unsigned offset = dof == 0 ? 0 : spanOffsets_[dof - 1];
          return v_[spans_[offset + pointId] * dimension_ + dim];
      }
  }

  double spanCenter(unsigned dof, unsigned dim) const {
      if(spanOffsets_ == NULL) {
          return v_[dof * dimension_ + dim];
      } else {
          unsigned offset = dof == 0 ? 0 : spanOffsets_[dof - 1];
          double r = 0;
          int n = spanSize(dof);
          for(int i = 0; i < n; i++)
              r += v_[spans_[offset + i] * dimension_ + dim];
          return r / n;
      }
  }

private:
  /// Array
  double* v_;

  /// Spatial dimension
  unsigned dimension_;

  /// Number of points
  unsigned size_;

  /// Flag to tell whether array has been copied and must be freed by destructor
  const bool ownsMemory_;

  unsigned numberOfDof_;
  unsigned * spanOffsets_;
  unsigned * spans_;
  void init(double* coord, unsigned * span_offsets, unsigned * spans);
};

} // end namespace hmat

#endif
