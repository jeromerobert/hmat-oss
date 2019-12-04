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
#include "common/my_assert.h"
#include <cstring>

namespace hmat {
void DofCoordinates::init(double* coord, unsigned* span_offsets, unsigned* spans)
{
    if (ownsMemory_) {
        v_ = new double[size_ * dimension_];
        std::memcpy(v_, coord, sizeof(double) * size_ * dimension_);
        if(span_offsets) {
            spanOffsets_ = new unsigned[numberOfDof_];
            std::memcpy(spanOffsets_, span_offsets, sizeof(unsigned) * numberOfDof_);
            unsigned n = span_offsets[numberOfDof_-1];
            spans_ = new unsigned[n];
            std::memcpy(spans_, spans, sizeof(unsigned) * n);
        } else {
            spanOffsets_ = NULL;
            spans_ = NULL;
        }
    } else {
        v_ = coord;
        spanOffsets_ = span_offsets;
        spans_ = spans;
    }
    if(spanOffsets_) {
        spanAABBs_ = new double[numberOfDof_ * dimension_ * 2];
        double * aabb = spanAABBs_;
        for(int dof = 0; dof < numberOfDof_; dof++) {
            unsigned offset = dof == 0 ? 0 : spanOffsets_[dof - 1];
            int n = spanSize(dof);
            double * v = v_ + spans_[offset] * dimension_;
            memcpy(aabb, v, dimension_ * sizeof(double));
            memcpy(aabb + dimension_, v, dimension_ * sizeof(double));
            for(int i = 1; i < n; i++) {
                v = v_ + spans_[offset + i] * dimension_;
                for(int dim = 0; dim < dimension_; dim++) {
                    aabb[dim] = std::min(aabb[dim], v[dim]);
                    aabb[dim + dimension_] = std::max(aabb[dim + dimension_], v[dim]);
                }
            }
            aabb += 2 * dimension_;
        }
    } else {
        spanAABBs_ = NULL;
    }
}

DofCoordinates::DofCoordinates(double* coord, unsigned dim, unsigned size, bool ownsMemory,
                               unsigned number_of_dof, unsigned * span_offsets, unsigned * spans)
  : dimension_(dim)
  , size_(size)
  , ownsMemory_(ownsMemory), numberOfDof_(number_of_dof)
{
    init(coord, span_offsets, spans);
}

DofCoordinates::DofCoordinates(const DofCoordinates& other)
  : dimension_(other.dimension_)
  , size_(other.size_)
  , ownsMemory_(true), numberOfDof_(other.numberOfDof_)
{
    init(other.v_, other.spanOffsets_, other.spans_);
}

DofCoordinates::~DofCoordinates()
{
  if (ownsMemory_) {
    delete[] v_;
    if(spanOffsets_ != NULL) {
      delete[] spanOffsets_;
      delete[] spans_;
    }
  }
  if(spanOffsets_ != NULL)
    delete[] spanAABBs_;
}

int DofCoordinates::size() const {
    HMAT_ASSERT(spanOffsets_ == NULL);
    return size_;
}

double& DofCoordinates::get(int i, int j) {
    HMAT_ASSERT(spanOffsets_ == NULL);
    return v_[j * dimension_ + i];
}

const double& DofCoordinates::get(int i, int j) const {
    HMAT_ASSERT(spanOffsets_ == NULL);
    return v_[j * dimension_ + i];
}

}  // end namespace hmat

