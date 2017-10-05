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
    \param maxNorm is the reference norm used in the stopping criterion. Usually
    it is the largest 2-norm of the columns. It allows for an orthogonalisation
    of a columns subset according to the whole matrix largest norm (see blocked below).
    Unused if negative.

    WARNING: the lowest precision allowed is 1e-6.
    \return rank

    NB: On exit the orthonormal matrix stored in A is 'full' and not represented
    as a product of Householder reflectors. OR/ZU-MQR from LAPACK is NOT
    the way to apply the matrix: one has to use matrix-vector product instead.
*/
template<typename T> int modifiedGramSchmidt(ScalarArray<T> *a, ScalarArray<T> *r, double prec, double maxNorm );

/** blocked modified Gram-Schmidt algorithm

    A blocked version of the modified Gram-Schmidt (mGS) algorithm.
    The matrix is sliced into panels [A1,A2,...,An] with a constant number of columns nb.

    Like the usual mGS algorithm the Frobenius norm of each panel is computed and
    the method selects the largest panel. This panel A1 is then orthonormalised thanks
    to the mGS method to produce the QR decomposition A1 = Q1.R11.
    The remaining panels Aj are updated as

     a) R1j := Q11^H Aj (BLAS 3 operation);
     b) Aj := Aj - Q1.R1j;

    and the panels norms are updated accordingly. The pivoting strategy is also used
    at the panel level with a stopping condition similar to the mGS and the method iterates.

    \param prec is a small parameter describing a relative precision thus
    0 < prec < 1.
    WARNING: the lowest precision allowed is 1e-6.

    \param nb is the block size used to slice the matrix.

    NB: On exit the orthonormal matrix stored in A is 'full' and not represented
    as a product of Householder reflectors. OR/ZU-MQR from LAPACK is NOT
    the way to apply the matrix: one has to use matrix-vector product instead.

    \return rank
*/

template<typename T> int blockedMGS(ScalarArray<T> *a, ScalarArray<T> *r, double prec, const int nb );
}  // end namespace hmat

#endif
