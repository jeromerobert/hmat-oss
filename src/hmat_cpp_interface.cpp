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

#include "hmat_cpp_interface.hpp"
#include "h_matrix.hpp"
#include "rk_matrix.hpp"
#include "cluster_tree.hpp"
#include "common/context.hpp"
#include "disable_threading.hpp"

#include <cstring>

namespace hmat {

// HMatInterface
template<typename T, template <typename> class E>
bool HMatInterface<T, E>::initialized = false;

template<typename T, template <typename> class E>
int HMatInterface<T, E>::init() {
  if (initialized) return 0;
  if (0 != E<T>::init()) return 1;
  initialized = true;
  return 0;
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::finalize() {
  if (!initialized) return;
  initialized = false;
  E<T>::finalize();
}

template<typename T, template <typename> class E>
HMatInterface<T, E>::HMatInterface(ClusterTree* _rows, ClusterTree* _cols, SymmetryFlag sym,
                                   AdmissibilityCondition * admissibilityCondition)
{
  DECLARE_CONTEXT;
  engine_.hmat = new HMatrix<T>(_rows, _cols, &HMatSettings::getInstance(), sym, admissibilityCondition);
}

template<typename T, template <typename> class E>
HMatInterface<T, E>::~HMatInterface() {
  engine_.destroy();
  delete engine_.hmat;
}

template<typename T, template <typename> class E>
HMatInterface<T, E>::HMatInterface(HMatrix<T>* h) :
    engine_(h)
{}

//TODO remove the synchronize parameter which is parallel specific
template<typename T, template <typename> class E>
void HMatInterface<T, E>::assemble(Assembly<T>& f, SymmetryFlag sym, bool synchronize, hmat_progress_t * progress) {
  DISABLE_THREADING_IN_BLOCK;
  engine_.progress(progress);
  engine_.assembly(f, sym, synchronize);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::factorize(hmat_factorization_t t, hmat_progress_t * progress) {
  DISABLE_THREADING_IN_BLOCK;
  engine_.progress(progress);
  engine_.factorization(t);
  factorizationType = t;
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::inverse(hmat_progress_t * progress) {
  DISABLE_THREADING_IN_BLOCK;
  engine_.progress(progress);
  engine_.inverse();
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::gemv(char trans, T alpha, FullMatrix<T>& x, T beta,
                            FullMatrix<T>& y) const {
  DISABLE_THREADING_IN_BLOCK;
  reorderVector(&x, trans == 'N' ? engine_.hmat->cols()->indices() : engine_.hmat->rows()->indices());
  reorderVector(&y, trans == 'N' ? engine_.hmat->rows()->indices() : engine_.hmat->cols()->indices());
  engine_.gemv(trans, alpha, x, beta, y);
  restoreVectorOrder(&x, trans == 'N' ? engine_.hmat->cols()->indices() : engine_.hmat->rows()->indices());
  restoreVectorOrder(&y, trans == 'N' ? engine_.hmat->rows()->indices() : engine_.hmat->cols()->indices());
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::gemm(char transA, char transB, T alpha,
                            const HMatInterface<T, E>* a,
                            const HMatInterface<T, E>* b, T beta) {
    DISABLE_THREADING_IN_BLOCK;
    engine_.gemm(transA, transB, alpha, a->engine_, b->engine_, beta);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::gemm(FullMatrix<T>& c, char transA, char transB, T alpha,
                            FullMatrix<T>& a, const HMatInterface<T, E>& b,
                            T beta) {
  // C <- AB + C  <=>  C^t <- B^t A^t + C^t
  // On fait les operations dans ce sens pour etre dans le bon sens
  // pour la memoire, et pour reordonner correctement les "vecteurs" A
  // et C.
  if (transA == 'N') {
    a.transpose();
  }
  c.transpose();
  b.gemv(transB == 'N' ? 'T' : 'N', alpha, a, beta, c);
  c.transpose();
  if (transA == 'N') {
    a.transpose();
  }
}


template<typename T, template <typename> class E>
void HMatInterface<T, E>::solve(FullMatrix<T>& b) const {
  DISABLE_THREADING_IN_BLOCK;
  reorderVector<T>(&b, engine_.hmat->cols()->indices());
  engine_.solve(b, factorizationType);
  restoreVectorOrder<T>(&b, engine_.hmat->cols()->indices());
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::solve(HMatInterface<T, E>& b) const {
  DISABLE_THREADING_IN_BLOCK;
  engine_.solve(b.engine_, factorizationType);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::solveLower(FullMatrix<T>& b, bool transpose) const {
  DISABLE_THREADING_IN_BLOCK;
  if (transpose)
    reorderVector<T>(&b, engine_.hmat->rows()->indices());
  else
    reorderVector<T>(&b, engine_.hmat->cols()->indices());
  engine_.solveLower(b, factorizationType, transpose);
  if (transpose)
    restoreVectorOrder<T>(&b, engine_.hmat->rows()->indices());
  else
    restoreVectorOrder<T>(&b, engine_.hmat->cols()->indices());
}

template<typename T, template <typename> class E>
HMatInterface<T, E>* HMatInterface<T, E>::copy() const {
  HMatInterface<T, E>* result = new HMatInterface<T, E>(NULL);
  engine_.copy(result->engine_);
  assert(result->engine_.hmat);
  return result;
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::transpose() {
  engine_.transpose();
}


template<typename T, template <typename> class E>
double HMatInterface<T, E>::norm() const {
  DISABLE_THREADING_IN_BLOCK;
  return engine_.norm();
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::eval(FullMatrix<T>* result, bool renumber) const {
  engine_.hmat->eval(result, renumber);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::scale(T alpha) {
  DISABLE_THREADING_IN_BLOCK;
  engine_.hmat->scale(alpha);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::addIdentity(T alpha) {
  DISABLE_THREADING_IN_BLOCK;
  engine_.addIdentity(alpha);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::info(hmat_info_t & result) const {
    memset(&result, 0, sizeof(hmat_info_t));
    engine_.hmat->info(result);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::createPostcriptFile(const std::string& filename) const {
    engine_.createPostcriptFile(filename);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::dumpTreeToFile(const std::string& filename) const {
    HMatrixVoidNodeDumper<T> dumper_extra;
    dumpTreeToFile(filename, dumper_extra);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::dumpTreeToFile(const std::string& filename, const HMatrixNodeDumper<T>& dumper_extra) const {
    engine_.dumpTreeToFile(filename, dumper_extra);
}

template<typename T, template <typename> class E>
int HMatInterface<T, E>::nodesCount() const {
  DISABLE_THREADING_IN_BLOCK;
  return engine_.hmat->nodesCount();
}
template<typename T, template <typename> class E>
void HMatInterface<T, E>::walk(TreeProcedure<HMatrix<T> > *proc){
  DISABLE_THREADING_IN_BLOCK;
  return engine_.hmat->walk(proc);
}
} // end namespace hmat

