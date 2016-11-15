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
#include "json.hpp"

#include <cstring>
#include <fstream>

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
                                   AdmissibilityCondition * admissibilityCondition) :
    engine_(), factorizationType(hmat_factorization_none)
{
  DECLARE_CONTEXT;
  engine_.hmat = new HMatrix<T>(_rows, _cols, &HMatSettings::getInstance(), 0, sym, admissibilityCondition);
}

template<typename T, template <typename> class E>
HMatInterface<T, E>::~HMatInterface() {
  engine_.destroy();
  delete engine_.hmat;
}

template<typename T, template <typename> class E>
HMatInterface<T, E>::HMatInterface(HMatrix<T>* h, hmat_factorization_t factorization) :
    engine_(h), factorizationType(factorization)
{}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::assemble(Assembly<T>& f, SymmetryFlag sym, bool,
                                   hmat_progress_t * progress, bool ownAssembly) {
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
  engine_.progress(progress);
  engine_.assembly(f, sym, ownAssembly);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::factorize(hmat_factorization_t t, hmat_progress_t * progress) {
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
  engine_.progress(progress);
  if(progress != NULL)
    progress->max = engine_.hmat->rows()->size();
  engine_.factorization(t);
  factorizationType = t;
  engine_.hmat->checkStructure();
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::inverse(hmat_progress_t * progress) {
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
  engine_.progress(progress);
  engine_.inverse();
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::gemv(char trans, T alpha, ScalarArray<T>& x, T beta,
                            ScalarArray<T>& y) const {
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
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
    DECLARE_CONTEXT;
    engine_.gemm(transA, transB, alpha, a->engine_, b->engine_, beta);
    engine_.hmat->checkStructure();
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::gemm(ScalarArray<T>& c, char transA, char transB, T alpha,
                            ScalarArray<T>& a, const HMatInterface<T, E>& b,
                            T beta) {
  DECLARE_CONTEXT;
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
void HMatInterface<T, E>::solve(ScalarArray<T>& b) const {
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
  reorderVector<T>(&b, engine_.hmat->cols()->indices());
  engine_.solve(b, factorizationType);
  restoreVectorOrder<T>(&b, engine_.hmat->cols()->indices());
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::solve(HMatInterface<T, E>& b) const {
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
  engine_.solve(b.engine_, factorizationType);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::solveLower(ScalarArray<T>& b, bool transpose) const {
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
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
HMatInterface<T, E>* HMatInterface<T, E>::copy(bool structOnly) const {
  DECLARE_CONTEXT;
  HMatInterface<T, E>* result = new HMatInterface<T, E>(NULL);
  engine_.copy(result->engine_, structOnly);
  assert(result->engine_.hmat);
  result->engine_.hmat->checkStructure();
  return result;
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::transpose() {
  DECLARE_CONTEXT;
  engine_.transpose();
  engine_.hmat->checkStructure();
}


template<typename T, template <typename> class E>
double HMatInterface<T, E>::norm() const {
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
  return engine_.norm();
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::eval(FullMatrix<T>* result, bool renumber) const {
  engine_.hmat->eval(result, renumber);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::scale(T alpha) {
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
  engine_.hmat->scale(alpha);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::addIdentity(T alpha) {
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
  engine_.addIdentity(alpha);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::addRand(double epsilon) {
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
  engine_.addRand(epsilon);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::info(hmat_info_t & result) const {
  DECLARE_CONTEXT;
    memset(&result, 0, sizeof(hmat_info_t));
    engine_.info(result);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::createPostcriptFile(const std::string& filename) const {
  DECLARE_CONTEXT;
    engine_.createPostcriptFile(filename);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::dumpTreeToFile(const std::string& filename) const {
  DECLARE_CONTEXT;
  std::ofstream out(filename.c_str());
  HMatrixJSONDumper<T>(engine_.hmat, out).dump();
}

template<typename T, template <typename> class E>
int HMatInterface<T, E>::nodesCount() const {
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
  return engine_.hmat->nodesCount();
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::walk(TreeProcedure<HMatrix<T> > *proc){
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
  return engine_.hmat->walk(proc);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::apply_on_leaf(const LeafProcedure<HMatrix<T> >& proc){
  DISABLE_THREADING_IN_BLOCK;
  DECLARE_CONTEXT;
  engine_.applyOnLeaf(proc);
}

} // end namespace hmat
