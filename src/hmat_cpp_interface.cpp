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
HMatInterface<T, E>::HMatInterface(ClusterTree* _rows, ClusterTree* _cols)
  : rows(_rows), cols(_cols) {
  DECLARE_CONTEXT;
  const HMatSettings& settings = HMatSettings::getInstance();
  SymmetryFlag sym = (settings.useLdlt ? kLowerSymmetric : kNotSymmetric);
  engine.hmat = new HMatrix<T>(rows, cols, &HMatSettings::getInstance(), sym);
}

template<typename T, template <typename> class E>
HMatInterface<T, E>::~HMatInterface() {
  engine.destroy();
  delete engine.hmat;
}

template<typename T, template <typename> class E>
HMatInterface<T, E>::HMatInterface(HMatrix<T>* h)
  : rows(h->data.rows), cols(h->data.cols), engine(h)
{}

//TODO remove the synchronize parameter which is parallel specific
template<typename T, template <typename> class E>
void HMatInterface<T, E>::assemble(AssemblyFunction<T>& f, SymmetryFlag sym, bool synchronize) {
  DISABLE_THREADING_IN_BLOCK;
  engine.assembly(f, sym, synchronize);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::factorize() {
  DISABLE_THREADING_IN_BLOCK;
  engine.factorization();
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::gemv(char trans, T alpha, FullMatrix<T>& x, T beta,
                            FullMatrix<T>& y) const {
  DISABLE_THREADING_IN_BLOCK;
  reorderVector(&x, trans == 'N' ? engine.hmat->cols()->indices : engine.hmat->rows()->indices);
  reorderVector(&y, trans == 'N' ? engine.hmat->rows()->indices : engine.hmat->cols()->indices);
  engine.gemv(trans, alpha, x, beta, y);
  restoreVectorOrder(&x, trans == 'N' ? engine.hmat->cols()->indices : engine.hmat->rows()->indices);
  restoreVectorOrder(&y, trans == 'N' ? engine.hmat->rows()->indices : engine.hmat->cols()->indices);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::gemm(char transA, char transB, T alpha,
                            const HMatInterface<T, E>* a,
                            const HMatInterface<T, E>* b, T beta) {
    DISABLE_THREADING_IN_BLOCK;
    engine.gemm(transA, transB, alpha, a->engine, b->engine, beta);
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
  reorderVector<T>(&b, engine.hmat->cols()->indices);
  engine.solve(b);
  restoreVectorOrder<T>(&b, engine.hmat->cols()->indices);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::solve(HMatInterface<T, E>& b) const {
  DISABLE_THREADING_IN_BLOCK;
  engine.solve(b.engine);
}


template<typename T, template <typename> class E>
HMatInterface<T, E>* HMatInterface<T, E>::copy() const {
  HMatrix<T>* hCopy = HMatrix<T>::Zero(engine.hmat);
  hCopy->copy(engine.hmat);
  HMatInterface<T, E>* result = new HMatInterface<T, E>(hCopy);
  engine.copy(result->engine);
  return result;
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::transpose() {
  engine.hmat->transpose();
  engine.transpose();
}


template<typename T, template <typename> class E>
double HMatInterface<T, E>::norm() const {
  DISABLE_THREADING_IN_BLOCK;
  return engine.norm();
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::scale(T alpha) {
  DISABLE_THREADING_IN_BLOCK;
  engine.hmat->scale(alpha);
}

template<typename T, template <typename> class E>
std::pair<size_t, size_t> HMatInterface<T, E>::compressionRatio() const {
  return engine.hmat->compressionRatio();
}

template<typename T, template <typename> class E>
std::pair<size_t, size_t> HMatInterface<T, E>::fullrkRatio() const {
  return engine.hmat->fullrkRatio();
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::createPostcriptFile(const char* filename) const {
    engine.createPostcriptFile(filename);
}

template<typename T, template <typename> class E>
void HMatInterface<T, E>::dumpTreeToFile(const char* filename) const {
    engine.dumpTreeToFile(filename);
}

template<typename T, template <typename> class E>
int HMatInterface<T, E>::nodesCount() const {
  DISABLE_THREADING_IN_BLOCK;
  return engine.hmat->nodesCount();
}

