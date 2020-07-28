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

#ifndef _COMPRESSION_HPP
#define _COMPRESSION_HPP
/* Implementation of the algorithms of blocks compression */
#include "data_types.hpp"

/** Choice of the compression method.
 */
#include "assembly.hpp"

namespace hmat {

enum CompressionMethod {
  Svd, AcaFull, AcaPartial, AcaPlus, NoCompression, AcaRandom
};
class IndexSet;

/** Compress a FullMatrix into an RkMatrix.

    The compression uses the reduced SVD, and the accurarcy is
    controlled by \a RkMatrix::approx.

    \param m The matrix to compress. It is modified but not detroyed by the function.
    \return A RkMatrix approximationg the argument \a m.
*/
template<typename T>
RkMatrix<T>* truncatedSvd(FullMatrix<T>* m, double eps);

/** Compress a block into an RkMatrix.

    \param method The compression method
    \param f The assembly functions used to compute block elements
    \param rows The block rows
    \param cols The block colums
    \return A RkMatrix representation of the rows x cols block.
*/
template<typename T>
RkMatrix<typename Types<T>::dp>*
compress(CompressionMethod method, double compressionEpsilon, const Function<T>& f,
         const ClusterData* rows, const ClusterData* cols, double epsilon,
         const AllocationObserver & = AllocationObserver());

}  // end namespace hmat
#endif
