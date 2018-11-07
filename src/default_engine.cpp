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

#include "default_engine.hpp"
#include "hmat_cpp_interface.hpp"
#include "common/context.hpp"
#include "common/my_assert.h"
#include "hmat/hmat.h"
#include "common/timeline.hpp"

namespace hmat {

static void default_progress_update(hmat_progress_t * ctx) {
    double progress = (100. * ctx->current) / ctx->max;
    std::cout << '\r' << "Progress: " << progress << "% ("
              << ctx->current << " / " << ctx->max << ")      ";
    if(ctx->current == ctx->max) {
        std::cout << std::endl;
    }
    std::cout.flush();
}

DefaultProgress::DefaultProgress() {
    delegate.max = 0;
    delegate.current = 0;
    delegate.user_data = NULL;
    delegate.update = default_progress_update;
}

hmat_progress_t * DefaultProgress::getInstance()
{
    static DefaultProgress instance;
    return &instance.delegate;
}

template<typename T>
static void setTemplatedParameters(const HMatSettings& s) {
  RkMatrix<T>::approx.assemblyEpsilon = s.assemblyEpsilon;
  RkMatrix<T>::approx.recompressionEpsilon = s.recompressionEpsilon;
  RkMatrix<T>::approx.method = s.compressionMethod;
  RkMatrix<T>::approx.compressionMinLeafSize = s.compressionMinLeafSize;
  HMatrix<T>::validateNullRowCol = s.validateNullRowCol;
  HMatrix<T>::validateCompression = s.validateCompression;
  HMatrix<T>::validationErrorThreshold = s.validationErrorThreshold;
  HMatrix<T>::validationReRun = s.validationReRun;
  HMatrix<T>::validationDump = s.validationDump;
  HMatrix<T>::coarsening = s.coarsening;
}


void HMatSettings::setParameters() const {
  HMAT_ASSERT(assemblyEpsilon > 0.);
  HMAT_ASSERT(recompressionEpsilon > 0.);
  HMAT_ASSERT(validationErrorThreshold >= 0.);
  setTemplatedParameters<S_t>(*this);
  setTemplatedParameters<D_t>(*this);
  setTemplatedParameters<C_t>(*this);
  setTemplatedParameters<Z_t>(*this);
}


void HMatSettings::printSettings(std::ostream& out) const {
  std::ios_base::fmtflags savedIosFlags = out.flags();
  out << std::scientific;
  out << "Assembly Epsilon           = " << assemblyEpsilon << std::endl;
  out << "Resolution Epsilon         = " << recompressionEpsilon << std::endl;
  out << "Compression Min Leaf Size  = " << compressionMinLeafSize << std::endl;
  out << "Validation Error Threshold = " << validationErrorThreshold << std::endl;
  switch (compressionMethod) {
  case Svd:
    out << "SVD Compression" << std::endl;
    break;
  case AcaFull:
    out << "ACA compression (Full Pivoting)" << std::endl;
    break;
  case AcaPartial:
    out << "ACA compression (Partial Pivoting)" << std::endl;
    break;
  case AcaPlus:
    out << "ACA+ compression" << std::endl;
    break;
  case NoCompression:
    // Should not happen
    break;
  }
  out.flags(savedIosFlags);
}

ClusterTree* createClusterTree(const DofCoordinates& dls, const ClusteringAlgorithm& algo) {
  DECLARE_CONTEXT;

  ClusterTreeBuilder ctb(algo);
  return ctb.build(dls);
}

template<typename T>
int DefaultEngine<T>::init(){
  Timeline::instance().init();
  return 0;
}

template<typename T>
void DefaultEngine<T>::assembly(Assembly<T>& f, SymmetryFlag sym, bool ownAssembly) {
  if (sym == kLowerSymmetric || hmat->isLower || hmat->isUpper) {
    hmat->assembleSymmetric(f, NULL, hmat->isLower || hmat->isUpper);
  } else {
    hmat->assemble(f);
  }
  if(ownAssembly)
      delete &f;
}

template<typename T>
void DefaultEngine<T>::factorization(hmat_factorization_t t) {
  switch(t)
  {
  case hmat_factorization_lu:
      hmat->luDecomposition(progress_);
      break;
  case hmat_factorization_ldlt:
      hmat->ldltDecomposition(progress_);
      break;
  case hmat_factorization_llt:
      hmat->lltDecomposition(progress_);
      break;
  default:
      HMAT_ASSERT(false);
  }
}

template<typename T>
void DefaultEngine<T>::inverse() {
  hmat->inverse();
}

template<typename T>
void DefaultEngine<T>::gemv(char trans, T alpha, ScalarArray<T>& x,
                                      T beta, ScalarArray<T>& y) const {
  hmat->gemv(trans, alpha, &x, beta, &y);
}

template<typename T>
void DefaultEngine<T>::gemm(char transA, char transB, T alpha,
                                      const DefaultEngine<T>& a,
                                      const DefaultEngine<T>& b, T beta) {
  hmat->gemm(transA, transB, alpha, a.hmat, b.hmat, beta);
}

template<typename T>
void DefaultEngine<T>::addRand(double epsilon) {
  hmat->addRand(epsilon);
}

template<typename T>
void DefaultEngine<T>::solve(ScalarArray<T>& b, hmat_factorization_t t) const {
  switch(t) {
  case hmat_factorization_lu:
      hmat->solve(&b);
      break;
  case hmat_factorization_ldlt:
      hmat->solveLdlt(&b);
      break;
  case hmat_factorization_llt:
      hmat->solveLlt(&b);
      break;
  default:
     // not supported
     HMAT_ASSERT(false);
  }
}

template<typename T>
void DefaultEngine<T>::solve(DefaultEngine<T>& b, hmat_factorization_t f) const {
    hmat->solve(b.hmat, f);
}

template<typename T>
void DefaultEngine<T>::solveLower(ScalarArray<T>& b, hmat_factorization_t t, bool transpose) const {
  bool unitriangular = (t == hmat_factorization_lu || t == hmat_factorization_ldlt);
  if (transpose)
    hmat->solveUpperTriangularLeft(&b, unitriangular, true);
  else
    hmat->solveLowerTriangularLeft(&b, unitriangular);
}

template<typename T>
void DefaultEngine<T>::createPostcriptFile(const std::string& filename) const {
    hmat->createPostcriptFile(filename);
}

template<typename T> void DefaultEngine<T>::copy(DefaultEngine<T> & result, bool structOnly) const {
    result.hmat = hmat->copyStructure();
    if(!structOnly)
        result.hmat->copy(hmat);
}

template<typename T> void DefaultEngine<T>::transpose() {
    hmat->transpose();
}

template<typename T> void DefaultEngine<T>::applyOnLeaf(const hmat::LeafProcedure<hmat::HMatrix<T> >&f) {
    hmat->apply_on_leaf(f);
}

}  // end namespace hmat

#include "hmat_cpp_interface.cpp"

namespace hmat {

// Explicit template instantiation
template class HMatInterface<S_t, DefaultEngine>;
template class HMatInterface<D_t, DefaultEngine>;
template class HMatInterface<C_t, DefaultEngine>;
template class HMatInterface<Z_t, DefaultEngine>;

}  // end namespace hmat

