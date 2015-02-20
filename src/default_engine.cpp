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
#include "default_engine.hpp"
#include "common/context.hpp"
#include "common/my_assert.h"

template<typename T>
static void setTemplatedParameters(const HMatSettings& s) {
  RkMatrix<T>::approx.assemblyEpsilon = s.assemblyEpsilon;
  RkMatrix<T>::approx.recompressionEpsilon = s.recompressionEpsilon;
  RkMatrix<T>::approx.method = s.compressionMethod;
  RkMatrix<T>::approx.compressionMinLeafSize = s.compressionMinLeafSize;
  HMatrix<T>::validateCompression = s.validateCompression;
  HMatrix<T>::validationErrorThreshold = s.validationErrorThreshold;
  HMatrix<T>::validationReRun = s.validationReRun;
  HMatrix<T>::validationDump = s.validationDump;
  HMatrix<T>::coarsening = s.coarsening;
  HMatrix<T>::recompress = s.recompress;
}


void HMatSettings::setParameters() const {
  strongAssert(assemblyEpsilon > 0.);
  strongAssert(recompressionEpsilon > 0.);
  strongAssert(validationErrorThreshold >= 0.);
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
  out << "Admissibility Condition    = " << admissibilityCondition->str() << std::endl;
  out << "Validation Error Threshold = " << validationErrorThreshold << std::endl;
  switch (clustering) {
  case kGeometric:
    out << "Geometric Clustering" << std::endl;
    break;
  case kMedian:
    out << "Median Clustering" << std::endl;
    break;
  case kHybrid:
    out << "Hybrid Clustering" << std::endl;
    break;
  }
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
  case RkNull:
    out << "RkNull compression : set admissible blocks to 0" << std::endl;
    break;
  case NoCompression:
    // Should not happen
    break;
  }
  out.flags(savedIosFlags);
}

ClusterTree* createClusterTree(DofCoordinate* dls, int n) {
  DECLARE_CONTEXT;
  std::vector<Point>* points = new std::vector<Point>(n);

  for (int i = 0; i < n; i++) {
    (*points)[i] = Point(dls[i].x, dls[i].y, dls[i].z);
  }
  int *indices = new int[n];
  for (int i = 0; i < n; i++) {
    indices[i] = i;
  }
  ClusterData rootData(indices, 0, n, points);
  Point boundingBox[2];
  rootData.computeBoundingBox(boundingBox);

  ClusterTree* ct;
  const HMatSettings& settings = HMatSettings::getInstance();
  switch (settings.clustering) {
  case kMedian:
    ct = new MedianBisectionClusterTree(boundingBox, rootData,
                                        settings.maxLeafSize);
    break;
  case kGeometric:
    ct = new GeometricBisectionClusterTree(boundingBox, rootData,
                                           settings.maxLeafSize);
    break;
  case kHybrid:
    ct = new HybridBisectionClusterTree(boundingBox, rootData,
                                        settings.maxLeafSize);
    break;
  default:
    strongAssert(false);
  }
  ct->divide();
  return ct;
}


template<typename T>
void DefaultEngine<T>::assembly(AssemblyFunction<T>& f, SymmetryFlag sym,
                                bool synchronize) {
  if (sym == kLowerSymmetric || hmat->isLower || hmat->isUpper) {
    hmat->assembleSymmetric(f, NULL, hmat->isLower || hmat->isUpper);
  } else {
    hmat->assemble(f);
  }
}

template<typename T>
void DefaultEngine<T>::factorization() {
  const HMatSettings& settings = HMatSettings::getInstance();
  strongAssert(settings.useLdlt ^ settings.useLu);
  if (settings.useLdlt) {
    if(settings.cholesky)
      hmat->lltDecomposition();
    else
      hmat->ldltDecomposition();
  } else {
    hmat->luDecomposition();
  }
}

template<typename T>
void DefaultEngine<T>::gemv(char trans, T alpha, FullMatrix<T>& x,
                                      T beta, FullMatrix<T>& y) const {
  hmat->gemv(trans, alpha, &x, beta, &y);
}

template<typename T>
void DefaultEngine<T>::gemm(char transA, char transB, T alpha,
                                      const DefaultEngine<T>& a,
                                      const DefaultEngine<T>& b, T beta) {
  hmat->gemm(transA, transB, alpha, a.hmat, b.hmat, beta);
}

template<typename T>
void DefaultEngine<T>::solve(FullMatrix<T>& b) const {
  const HMatSettings& settings = HMatSettings::getInstance();
  strongAssert(settings.useLu ^ settings.useLdlt);
  if (settings.useLu) hmat->solve(&b);
  else if (settings.useLdlt) hmat->solveLdlt(&b);
}

template<typename T>
void DefaultEngine<T>::solve(DefaultEngine<T>& b) const {
    hmat->solve(b.hmat);
}

template<typename T>
void DefaultEngine<T>::createPostcriptFile(const char* filename) const {
    hmat->createPostcriptFile(filename);
}

template<typename T>
void DefaultEngine<T>::dumpTreeToFile(const char* filename) const {
    hmat->dumpTreeToFile(filename);
}

template<typename T> double DefaultEngine<T>::norm() const {
    return hmat->norm();
}

template<typename T> void DefaultEngine<T>::copy(DefaultEngine<T> & result) const {
    result.hmat = hmat->copyStructure();
    result.hmat->copy(hmat);
}
#include "hmat_cpp_interface.cpp"

// Explicit template instantiation
template class HMatInterface<S_t, DefaultEngine>;
template class HMatInterface<D_t, DefaultEngine>;
template class HMatInterface<C_t, DefaultEngine>;
template class HMatInterface<Z_t, DefaultEngine>;
