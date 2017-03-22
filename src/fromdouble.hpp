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
#pragma once

#include "full_matrix.hpp"
#include "rk_matrix.hpp"

namespace hmat {

/* Returns a potentially lower precision matrix.

    This functions takes a FullMatrix either in double precision, and
    returns a single precision matrix if required. This is used to
    perform some computations with higher precision and then to go back
    to the required precision.

    If the target type is double precision, this function is a
    no-op. Otherwise (single precision target type) it performs the
    conversion and destroy the original matrix.
 */

/** \brief Returns a conversion of the ScalarArray 'd' in arithmetics 'T'

  d is either of type T or T::dp. If del is true, d is deleted on exit.
  */
template<typename T> ScalarArray<T>* fromDoubleScalarArray(ScalarArray<typename Types<T>::dp>* d, bool del = true);

  /** \brief Returns a conversion of the FullMatrix 'f' in arithmetics 'T'

    f is either of type T or T::dp. f is deleted on exit.
    */
template<typename T> FullMatrix<T>* fromDoubleFull(FullMatrix<typename Types<T>::dp>* f);

  /** \brief Returns a conversion of the RkMatrix 'rk' in arithmetics 'T'

    rk is either of type T or T::dp. rk is deleted on exit.
    */
template<typename T> RkMatrix<T>* fromDoubleRk(RkMatrix<typename Types<T>::dp>* rk);

}  // end namespace hmat

