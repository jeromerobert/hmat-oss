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

/* Contient des definitions permettant d'acceder a certaines parties de LAPACK
   dans la MKL. */
#ifndef _LAPACK_OPERATIONS_HPP
#define _LAPACK_OPERATIONS_HPP
#include "data_types.hpp"
#include "full_matrix.hpp"

namespace hmat {

/** Makes an SVD with LAPACK in MKL.

    \param m
    \param u
    \param sigma
    \param vt
    \return
 */
template<typename T> int truncatedSvd(ScalarArray<T>* m, ScalarArray<T>** u, Vector<double>** sigma, ScalarArray<T>** vt);
/** QR matrix decomposition.

    Warning: m is modified!

    \param m
    \param tau
    \return
 */
template<typename T> T* qrDecomposition(ScalarArray<T>* m);

/** Do the product by Q.

    qr has to be factored using \a qrDecomposition.
    The arguments side and trans have the same meaning as in the
    LAPACK xORMQR function. Beware, only the 'L', 'N' case has been
    tested !

    \param side either 'L' or 'R', as in xORMQR
    \param trans either 'N' or 'T' as in xORMQR
    \param qr the matrix factored using \a qrDecomposition
    \param tau as created by \a qrDecomposition
    \param c as in xORMQR
    \return 0 for success
 */
template<typename T>
int productQ(char side, char trans, ScalarArray<T>* qr, T* tau, ScalarArray<T>* c);


/** Multiplication used in RkMatrix::truncate()

     A B -> computing "AB^t" with A and B full upper triangular
     (non-unitary diagonal)

 */
template<typename T> void myTrmm(ScalarArray<T>* aFull,
                                 ScalarArray<T>* bTri);

}  // end namespace hmat

#endif
