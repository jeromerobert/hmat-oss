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
#include "coordinates.hpp"

#include <cstring>

namespace hmat {

DofCoordinates::DofCoordinates(double* coord, int dim, int size, bool ownsMemory)
  : dimension_(dim)
  , size_(size)
  , ownsMemory_(ownsMemory)
{
  if (ownsMemory_)
  {
    v_ = new double[size_ * dimension_];
    std::memcpy(v_, coord, sizeof(double) * size_ * dimension_);
  }
  else
    v_ = coord;
}

DofCoordinates::DofCoordinates(const DofCoordinates& other)
  : dimension_(other.dimension_)
  , size_(other.size_)
  , ownsMemory_(true)
{
  v_ = new double[size_ * dimension_];
  std::memcpy(v_, other.v_, sizeof(double) * size_ * dimension_);
}

DofCoordinates::~DofCoordinates()
{
  if (ownsMemory_)
    delete[] v_;
}

}  // end namespace hmat

