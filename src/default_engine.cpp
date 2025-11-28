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
#include <numeric>

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
  RkMatrix<T>::approx.compressionMinLeafSize = s.compressionMinLeafSize;
  RkMatrix<T>::approx.coarseningEpsilon = s.coarseningEpsilon;
  HMatrix<T>::validateNullRowCol = s.validateNullRowCol;
  HMatrix<T>::validateCompression = s.validateCompression;
  HMatrix<T>::validateRecompression = s.validateRecompression;
  HMatrix<T>::validationErrorThreshold = s.validationErrorThreshold;
  HMatrix<T>::validationReRun = s.validationReRun;
  HMatrix<T>::validationDump = s.validationDump;
  HMatrix<T>::coarsening = s.coarsening;
}


void HMatSettings::setParameters() const {
  HMAT_ASSERT(coarseningEpsilon > 0.);
  HMAT_ASSERT(validationErrorThreshold >= 0.);
  setTemplatedParameters<S_t>(*this);
  setTemplatedParameters<D_t>(*this);
  setTemplatedParameters<C_t>(*this);
  setTemplatedParameters<Z_t>(*this);
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
  if (sym == kLowerSymmetric || this->hmat->isLower || this->hmat->isUpper) {
    this->hmat->assembleSymmetric(f, NULL, this->hmat->isLower || this->hmat->isUpper);
  } else {
    this->hmat->assemble(f);
  }
  if(ownAssembly)
      delete &f;
}

template<typename T>
void DefaultEngine<T>::factorization(Factorization algo) {
  switch(algo)
  {
  case Factorization::LU:
      this->hmat->luDecomposition(this->progress_);
      break;
  case Factorization::LDLT:
      this->hmat->ldltDecomposition(this->progress_);
      break;
  case Factorization::LLT:
      this->hmat->lltDecomposition(this->progress_);
      break;
  case Factorization::HODLR:
      this->hodlr.factorize(this->hmat, this->progress_);
      break;
  case Factorization::HODLRSYM:
      this->hodlr.factorizeSym(this->hmat, this->progress_);
      break;
  default:
      HMAT_ASSERT(false);
  }
}

template<typename T>
void DefaultEngine<T>::inverse() {
  this->hmat->inverse();
}

template<typename T>
void DefaultEngine<T>::gemv(char trans, T alpha, ScalarArray<T>& x,
                                      T beta, ScalarArray<T>& y) const {
  if(hodlr.isFactorized()) {
    this->hodlr.gemv(trans, alpha, this->hmat, x, beta, y);
  } else {
    this->hmat->gemv(trans, alpha, &x, beta, &y);
  }
}

template<typename T> typename Types<T>::dp DefaultEngine<T>::logdet() const {
  if(hodlr.isFactorized()) {
    return this->hodlr.logdet(this->hmat);
  } else if(this->hmat->isTriLower) {
    return this->hmat->logdet();
  } else {
    HMAT_ASSERT_MSG(false, "logdet is only supported for LLt or HODLR factorized matrices.");
  }
}

template<typename T> double DefaultEngine<T>::norm() const {
  return this->hmat->norm();
}

template<typename T>
void DefaultEngine<T>::gemm(char transA, char transB, T alpha,
                                      const IEngine<T>& a,
                                      const IEngine<T>& b, T beta) {
  HMAT_ASSERT_MSG(!hodlr.isFactorized(), "Unsupported operation");
  HMAT_ASSERT_MSG(!static_cast<const DefaultEngine &>(a).hodlr.isFactorized(),
                  "Unsupported operation");
  HMAT_ASSERT_MSG(!static_cast<const DefaultEngine &>(b).hodlr.isFactorized(),
                  "Unsupported operation");
  this->hmat->gemm(transA, transB, alpha, a.hmat, b.hmat, beta);
}

template<typename T>
void DefaultEngine<T>::trsm( char side, char  uplo, char trans, char diag, T alpha,
			     IEngine<T>& B ) const {
    this->hmat->trsm( side, uplo, trans, diag, alpha, B.hmat );
}

template<typename T>
void DefaultEngine<T>::trsm( char side, char  uplo, char trans, char diag, T alpha,
			     ScalarArray<T>& B ) const {
    this->hmat->trsm( side, uplo, trans, diag, alpha, &B );
}

template<typename T>
void DefaultEngine<T>::addIdentity(T alpha) {
  this->hmat->addIdentity(alpha);
}

template<typename T>
void DefaultEngine<T>::addRand(double epsilon) {
  this->hmat->addRand(epsilon);
}

template<typename T>
void DefaultEngine<T>::solve(ScalarArray<T>& b, Factorization algo) const {
  switch(algo) {
  case Factorization::LU:
      this->hmat->solve(&b);
      break;
  case Factorization::LDLT:
      this->hmat->solveLdlt(&b);
      break;
  case Factorization::LLT:
      this->hmat->solveLlt(&b);
      break;
  case Factorization::HODLR:
      this->hodlr.solve(this->hmat, b);
      break;
  case Factorization::HODLRSYM:
      this->hodlr.solveSymLower(this->hmat, b);
      this->hodlr.solveSymUpper(this->hmat, b);
      break;
  default:
     // not supported
     printf("\n\nMauvaise Facto !\n\n");
     HMAT_ASSERT(false);
  }
}

template<typename T>
void DefaultEngine<T>::solve(IEngine<T>& b, Factorization f) const {
  if(f == Factorization::HODLR) {
    this->hodlr.solve(this->hmat, b.hmat);
  }
  else
    this->hmat->solve(b.hmat, f);
}

template<typename T>
void DefaultEngine<T>::solveLower(ScalarArray<T>& b, Factorization algo, bool transpose) const {
  HMAT_ASSERT_MSG(algo != Factorization::HODLR, "solver lower not supported for non-symetric HODLR.");
  if(algo == Factorization::HODLRSYM) {
    if(transpose)
      this->hodlr.solveSymUpper(this->hmat, b);
    else
      this->hodlr.solveSymLower(this->hmat, b);
  } else {
    Diag diag = (algo == Factorization::LU || algo == Factorization::LDLT) ? Diag::UNIT : Diag::NONUNIT;
    if (transpose)
      this->hmat->solveUpperTriangularLeft(&b, algo, diag, Uplo::LOWER);
    else
      this->hmat->solveLowerTriangularLeft(&b, algo, diag, Uplo::LOWER);
  }
}

template<typename T>
void DefaultEngine<T>::solveLower(IEngine<T>& b, Factorization algo, bool transpose) const {
  // solveLower is well defined only for LLT (and maybe HODLRSYM?).
  // With LDLT, should we divide by sqrt(diag)?
  // With LU if transpose is true, do we want to solve L^T X = B or U X = B?
  HMAT_ASSERT_MSG(algo == Factorization::LLT, "solveLower only supported by LLt factorization.");
  if (transpose)
    this->hmat->solveUpperTriangularLeft(b.hmat, algo, Diag::NONUNIT, Uplo::LOWER);
  else
    this->hmat->solveLowerTriangularLeft(b.hmat, algo, Diag::NONUNIT, Uplo::LOWER);
}

template<typename T> void DefaultEngine<T>::copy(IEngine<T> & result, bool structOnly) const {
    result.hmat = this->hmat->copyStructure();
    if(!structOnly)
        result.hmat->copy(this->hmat);
}

template<typename T> void DefaultEngine<T>::transpose() {
  this->hmat->transpose();
}

template<typename T> void DefaultEngine<T>::applyOnLeaf(const hmat::LeafProcedure<hmat::HMatrix<T> >&f) {
  this->hmat->apply_on_leaf(f);
}

template<typename T> void DefaultEngine<T>::scale(T alpha) {
  this->hmat->scale(alpha);
}

template<typename T> void DefaultEngine<T>::info(hmat_info_t &i) const{
  this->hmat->info(i);
}

template <typename T>
void DefaultEngine<T>::profile(hmat_profile_t &p, const std::string& filename) const
{
  HMatProfile profile;

  this->hmat->profile(profile);

  profile.dump();
}

template <typename T> void DefaultEngine<T>::ratio(hmat_FPCompressionRatio_t &r) const
{

  r.ratio = 0;
  r.fullRatio = 0;
  r.rkRatio = 0;
  r.size_Full = 0;
  r.size_Rk = 0;
  r.size_Full_compressed = 0;
  r.size_Rk_compressed = 0;

  this->hmat->FPratio(r);

  if(r.size_Full_compressed>0) 
    {r.fullRatio = r.size_Full /r.size_Full_compressed;}
    else{r.fullRatio = 0;}

  if(r.size_Rk_compressed >0 )
    {r.rkRatio = r.size_Rk/r.size_Rk_compressed;}
    else{r.rkRatio = 0;}

  if(r.size_Full_compressed + r.size_Rk_compressed>0)
    {r.ratio = (r.size_Full + r.size_Rk)/(r.size_Full_compressed + r.size_Rk_compressed);}
    else{r.ratio = 0;}

}

template <typename T>
void DefaultEngine<T>::FPcompress()
{
  this->hmat->FPcompress();
}

template <typename T>
void DefaultEngine<T>::FPdecompress()
{
  this->hmat->FPdecompress();
}

template <typename T>
FPCompressionSettings DefaultEngine<T>::GetFPCompressionSettings() {
  return this->hmat->GetFPCompressionSettings();
};

template <typename T>
void DefaultEngine<T>::SetFPCompressionSettings(hmat_FPcompress_t compressor, int nb_blocs, float epsilonFP, bool compressFull, bool compressRk) {
  this->hmat->SetFPCompressionSettings(compressor, nb_blocs, epsilonFP, compressFull, compressRk);
};
  


}  // end namespace hmat

namespace hmat {

// Explicit template instantiation
template class DefaultEngine<S_t>;
template class DefaultEngine<D_t>;
template class DefaultEngine<C_t>;
template class DefaultEngine<Z_t>;

}  // end namespace hmat

