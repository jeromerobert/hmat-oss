/*
  HMat-OSS (HMatrix library, open source software)

  Copyright (C) 2021 Airbus SAS

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
#include "hmat/hmat.h"

namespace hmat {
template<typename T> class HMatrix;
template<typename T> class ScalarArray;
template<typename T> struct HODLRNode;

/**
 * @brief HODLR matrix factorization and solve.
 *
 * From Fast Direct Methods for Gaussian Processes
 * Sivaram Ambikasaran, Daniel Foreman-Mackey, Leslie Greengard, David W. Hogg, Michael O'Neil
 * arXiv:1403.6015
 * And from Fast symmetric factorization of hierarchical matrices with applications
 * Sivaram Ambikasaran, Michael O'Neil, Karan Raj Singh
 * arXiv:1405.0223
 */
template<typename T> class HODLR {
  HODLRNode<T> * root = nullptr;
public:
  void factorize(HMatrix<T> *, hmat_progress_t*);
  void factorizeSym(HMatrix<T> *, hmat_progress_t*);
  /** @brief solve with a Rk RHS */
  void solve(HMatrix<T> * const a, HMatrix<T> *b) const;
  void solve(HMatrix<T> * const a, ScalarArray<T> & b) const;
  void solveSymLower(HMatrix<T> * const a, ScalarArray<T> & b) const;
  void solveSymUpper(HMatrix<T> * const a, ScalarArray<T> & b) const;
  bool isFactorized() const;
  void gemv(char trans, T alpha, HMatrix<T> * const a, ScalarArray<T> & x, T beta, ScalarArray<T> & y) const;
  T logdet(HMatrix<T> * const a) const;
  ~HODLR();
};
}
