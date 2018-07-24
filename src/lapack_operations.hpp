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

  template<typename T> int sddCall(char jobz, int m, int n, T* a, int lda,
                                   double* sigma, T* u, int ldu, T* vt, int ldvt);
  template<typename T> int svdCall(char jobu, char jobv, int m, int n, T* a,
                                   int lda, double* sigma, T* u, int ldu, T* vt,
                                   int ldvt);

/** modified Gram-Schmidt algorithm

    Computes a QR-decomposition of a matrix A=[a_1,...,a_n] thanks to the
    modified Gram-Schmidt procedure with column pivoting.

    The matrix A is overwritten with a matrix Q=[q_1,...,q_r] whose columns are
    orthonormal and are a basis of Im(A).
    A pivoting strategy is used to improve stability:
    Each new qj vector is computed from the vector with the maximal 2-norm
    amongst the remaining a_k vectors.

    To further improve stability for each newly computed q_j vector its
    component is removed from the remaining columns a_k of A.

    Stopping criterion:
    whenever the maximal norm of the remaining vectors is smaller than
    prec * max(||ai||) the algorithm stops and the numerical rank at precision
    prec is the number of q_j vectors computed.

    Eventually the computed decomposition is:
    [a_{perm[0]},...,a_{perm[rank-1]}] = [q_1,...,q_{rank-1}] * [r]
    where [r] is an upper triangular matrix.

    \param prec is a small parameter describing a relative precision thus
    0 < prec < 1.
    WARNING: the lowest precision allowed is 1e-6.
    \return rank

    NB: On exit the orthonormal matrix stored in A is 'full' and not represented
    as a product of Householder reflectors. OR/ZU-MQR from LAPACK is NOT
    the way to apply the matrix: one has to use matrix-vector product instead.
*/
template<typename T> int modifiedGramSchmidt(ScalarArray<T> *a, ScalarArray<T> *r, double prec );
}  // end namespace hmat

#endif
