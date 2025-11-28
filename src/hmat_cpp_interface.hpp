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

/** @file
   @ingroup HMatrix
   @brief C++ interface to the HMatrix library.
 */
#ifndef HMAT_CPP_INTERFACE_HPP
#define HMAT_CPP_INTERFACE_HPP

/** C++ interface to the HMatrix library.

    This is the sole entry point to the HMatrix library.

    This interface is templated over the scalar type, T. This type has to be one
    of {S_t, D_t, C_t, Z_t}, using the standard BLAS notation. For the complex
    types, the C++ complex<> type is used. It is guaranteed to have the same
    layout as the equivalent FORTRAN types.

    The user code *has* to call \a HMatInterface<T>::init() before using any
    other function, and \a HMatInterface<T>::finalize() at the end.
*/

#include "hmat/hmat.h"

#include "clustering.hpp"
#include "compression.hpp"
#include "h_matrix.hpp"
#include "iengine.hpp"
#include "common/my_assert.h"

namespace hmat {

class ClusterTree;
class AdmissibilityCondition;

class DofCoordinates;
class ClusteringAlgorithm;

/** Settings for the HMatrix library.

    A single static instance of this class exist, but settings the values is not
    sufficient for the settings to take effect. One must call \a
    HMatSettings::setParameters().
*/
//TODO remove all global settings
class HMatSettings: public hmat::MatrixSettings {
public:
  int compressionMinLeafSize; ///< Force SVD compression if max(rows->n, cols->n) < compressionMinLeafSize
  double acaEpsilon; ///< Tolerance for the compression
  double coarseningEpsilon; ///< Tolerance for the coarsening
  /** \f$\eta\f$ in the admissiblity condition for two clusters \f$\sigma\f$ and \f$\tau\f$:
      \f[
      \min(diam(\sigma), diam(\tau)) < \eta \cdot d(\sigma, \tau)
      \f]
   */
  int maxLeafSize; ///< Maximum size of a leaf in a ClusterTree (and of a non-admissible block in an HMatrix)
  bool coarsening; ///< Coarsen the matrix structure after assembly.
  bool validateNullRowCol; ///< Validate the detection of null rows and columns
  bool validateCompression; ///< Validate the rk-matrices after compression
  bool validateRecompression; ///< Validate the rk-matrices recompression
  bool validationReRun; ///< For blocks above error threshold, re-run the compression algorithm
  bool dumpTrace; ///< Dump trace at the end of the algorithms (depends on the runtime)
  bool validationDump; ///< For blocks above error threshold, dump the faulty block to disk
  double validationErrorThreshold; ///< Error threshold for the compression validation
private:
  /** This constructor sets the default values.
   */
  HMatSettings() : compressionMinLeafSize(100),
                   coarseningEpsilon(1e-4),
                   maxLeafSize(200),
                   coarsening(false),
                   validateNullRowCol(false), validateCompression(false), validateRecompression(false),
                   validationReRun(false), dumpTrace(false), validationDump(false), validationErrorThreshold(0.) {
    setParameters();
  }
  // Disable the copy.
  HMatSettings(const HMatSettings&);
  void operator=(const HMatSettings&);

  public:
  // Classic Singleton pattern.
  static HMatSettings& getInstance() {
      static HMatSettings instance;
      return instance;
    }

  /** Change the settings of the HMatrix library.

      This method has to be called for the settings to take effect.
   */
  void setParameters() const;
};

DofCoordinates* createCoordinates(double* coord, int dim, int size);

/**
 * @deprecated use one of hmat_create_cluster_tree, hmat_create_cluster_tree_from_builder,
 * hmat_create_cluster_tree_generic
 */
ClusterTree* createClusterTree(const DofCoordinates& dls, const ClusteringAlgorithm& algo = MedianBisectionAlgorithm());

class DefaultProgress
{
    public:
        static hmat_progress_t * getInstance();
        hmat_progress_t delegate;
    private:
        DefaultProgress();
        DefaultProgress(DefaultProgress const &);
        void operator=(DefaultProgress const&);
};

    template<typename T>
class HMatInterface {
private:
  IEngine<T>* engine_;
  Factorization factorizationType;

public:
  /** Build a new HMatrix from two cluster sets.

      @note The ownership of the two ClusterTree instances (which don't need to
      be different) is transfered to the returned HMatInterface instance, and
      will be disposed at destruction time.

      @param _rows The row ClusterTree instance, built with \a createClusterTree()
      @param _cols The column ClusterTree instance, built with \a createClusterTree()
      @param symmetric If kLowerSymmetric, only lower triangular structure is created
      @return a new HMatInterface instance.
   */

  HMatInterface(IEngine<T>* engine, const ClusterTree* _rows, const ClusterTree* _cols, SymmetryFlag sym,
                AdmissibilityCondition * admissibilityCondition);

  HMatInterface(IEngine<T>* engine, HMatrix<T>* h, Factorization factorization = Factorization::NONE);

  /** Destroy an HMatInterface instance.

      @note This destructor is *not* virtual, as this class is not meant to be subclassed.
   */
  ~HMatInterface();
  /** Assemble an HMatrix.

      This builds an HMatrix using a provided AssemblyFunction. The compression
      method is determined by \a HMatSettings::compressionMethod, and the
      tolerance by \a HMatSettings::acaEpsilon.

      @param f The assembly function used to compute various matrix sub-parts
      @param sym If kLowerSymmetric, compute only the lower triangular matrix, and transpose
                 block to store upper counterpart.
      @param s: deprecated parameter
      @param ownAssembly true if &f should be deleted by the assemble function
   */
  void assemble(Assembly<T>& f, SymmetryFlag sym, bool s = true,
                hmat_progress_t * progress = DefaultProgress::getInstance(),
                bool ownAssembly=false);

  /** Compute a \f$LU\f$ or \f$LDL^T\f$ decomposition of the HMatrix, in place.

      An LDL^T decomposition is done if the HMatrix is symmetric and has been
      assembled as such (with sym = kLowerSymmetric in
      HMatInterface<T>::assemble()), and if HMatSettings::useLdlt is
      true. Otherwise an LU decomposition is done.
   */
  void factorize(Factorization, hmat_progress_t * progress = DefaultProgress::getInstance());

  /** Compute the inverse of the HMatrix, in place.
   */
  void inverse(hmat_progress_t * progress = DefaultProgress::getInstance());

  /** Matrix-Vector product.

      This computes \f$ y \gets \alpha . op(A) x + \beta y\f$, with A = this, x
      and y ScalarArray<T>. If trans == 'N', then op(A) = A, if trans == 'T',
      then op(A) = A^T, as in BLAS.

      @param trans 'N' or 'T'
      @param alpha
      @param x
      @param beta
      @param y
   */
  void gemv(char trans, T alpha, ScalarArray<T>& x, T beta, ScalarArray<T>& y) const;
  void gemm_scalar(char trans, T alpha, ScalarArray<T>& x, T beta, ScalarArray<T>& y) const;
  /** Matrix-Matrix product.

      This computes \f$ C \gets \alpha . op(A) \times op(B) + \beta C\f$ with A,
      B, C three HMatInterface<T> instancem, and C = this.  If trans* == 'N'
      then op(*) = *, if trans* == 'T', then op(*) = *^T, as in BLAS.

      @note transA == transB == 'T' is not supported.

      @param transA 'N' or 'T'
      @param transB 'N' or 'T'
      @param alpha
      @param a
      @param b
      @param beta
   */
  void gemm(char transA, char transB, T alpha, const HMatInterface<T>* a, const HMatInterface<T>* b, T beta);
  /** Matrix triangular solve.

      This solves \f$ alpha op(A) X = B or X op(A) = B\f$ where B is
      replaced by X on output.  A is a HMatInterface<T> instance, and
      B is either an HMatInterface<T> instance, or a ScalarArray<T>.
      A = this.
      If transa == 'N' then op(A) = A, if transa == 'T', then op(A) =
      A^T, and if transa == 'C', then op(A) = A^H as in BLAS.
      If side == 'L', then \f$ op(A) X = B \f$ is solved, else if 
      side == 'R', then \f$ X op(A) = B \f$ is solved.
      If uplo == 'U', then A is upper triangular, else if uplo == 'L',
      A is lower triangular.
      If diag == 'U', then A is unit triangular, else A is non unit triangular.

      @note cases LUT, RUT, and RLN are not supported.

      @param side   'L' or 'R'
      @param uplo   'U' or 'L'
      @param transA 'N' or 'T' or 'C'
      @param diag   'U' or 'N'
      @param alpha
      @param B
   */
  void trsm( char side, char uplo, char transa, char diag,
	     T alpha, HMatInterface<T>* B );
  void trsm( char side, char uplo, char transa, char diag,
	     T alpha, ScalarArray<T>& B );

  /** Full <- Full x HMatrix product.

      This computes the product \f$ C_F \gets \alpha . op(A_F) \times op(B_H) +
      \beta C_F\f$, with \f$A_F\f$, \f$C_F\f$ two ScalarArray<T>, and \f$B_H\f$
      an HMatrixInterface<T> instance.

      The meaning of the arguments is as in \a HMatInterface<T>::gemm(), and in
      BLAS.
   */
  static void gemm(ScalarArray<T>& c, char transA, char transB, T alpha, ScalarArray<T>& a, const HMatInterface<T>& b, T beta);
  /** Return a new copy of this.
   */
  HMatInterface<T>* copy(bool structOnly = false) const;
  /** Transpose this in place.
   */
  void transpose();
  /** Solve the system \f$A x = b\f$ in place, with A = this, and b a ScalarArray.

      @warning A has to be factored first with \a HMatInterface<T>::factorize().
   */
  void solve(ScalarArray<T>& b) const;
  /** Solve the system \f$A x = B\f$ in place, with A = this, and B a HMatInterface<T>.

      @warning A has to be factored first with \a HMatInterface<T>::factorize().
   */
  void solve(HMatInterface<T>& b) const;
  /** Solve the system \f$op(L) x = b\f$ in place, with L being the lower triangular part of
      an already factorized matrix, and b a ScalarArray.

      @warning A has to be factored first with \a HMatInterface<T>::factorize().
   */
  void solveLower(ScalarArray<T>& b, bool transpose=false) const;
  /** Solve the system \f$op(L) x = b\f$ in place, with L being the lower triangular part of
      an already factorized matrix, and b a HMatInterface<T>.

      @warning A has to be factored first with \a HMatInterface<T>::factorize().
   */
  void solveLower(HMatInterface<T>& b, bool transpose=false) const;
  /** this <- alpha * this
   */
  void scale(T alpha);

  /** Recompress all Rk matrices to their respective epsilon_ values.
   */
  void truncate();

  /** this <- this + alpha * Id
   */
  void addIdentity(T alpha);

  /** this <- this + A(norm = epsilon * norm(this)
   */
  void addRand(double epsilon);
  /**
   * Fill a hmat_info_t structure with information of this matrix.
   * @note This is only meaningful once the HMatrix has been assembled.
   */
  void info(hmat_info_t &) const;

  /**
   * Fill a hmat_profile_t structure with profile of this matrix.
   * @note This is only meaningful once the HMatrix has been assembled.
   */
  void profile(hmat_profile_t &, const std::string& filename = "profile.json") const;

  void ratio(hmat_FPCompressionRatio_t &result) const;

  /**
   * Apply a Floating-Point compression to the Hmatrix blocs
   */
  void FPcompress();

  /**
   * Uncompress the Hmatrix blocs resulting of the Floating-point compression
   */
  void FPdecompress();

  /**
   * Return the FP compression settings of this HMatrix
   */
  FPCompressionSettings GetFPCompressionSettings();

  /**
   * Set the FP compression settings of this HMatrix
   */
  void SetFPCompressionSettings(hmat_FPcompress_t compressor, int nb_blocs, float epsilonFP, bool compressFull, bool compressRk);

  /*! \brief Dump some HMatrix metadata to a Python-readable file.

    This function create a file that is readable by Python's eval()
    function, which contains a dictionnary with the following data:

    {'points': [(x1, y1, z1), ...],
     'mapping': [indices[0], indices[1], ...],
     'tree': {
       'isLeaf': False,
       'depth': 0,
       'rows': {'offset': 0, 'n': 15243, 'boundingBox': [(-0.0249617, -0.0249652, -0.0249586), (0.0962927, 0.0249652, 0.0249688)]},
       'cols': {'offset': 0, 'n': 15243, 'boundingBox': [(-0.0249617, -0.0249652, -0.0249586), (0.0962927, 0.0249652, 0.0249688)]},
       'children': [child1, child2, child3, child4]
     }
    }

    \param filename path to the output file.
   */
  void dumpTreeToFile(const std::string& filename) const;

  /** Return the number of block cluster tree nodes.
   */
  int nodesCount() const;
  /** Recursively apply a procedure to all nodes of an HMatrix.
   */
  void walk(TreeProcedure<HMatrix<T> > *proc);

  /** Recursively apply a procedure to all leaves of an HMatrix.
   */
  void apply_on_leaf(const LeafProcedure<HMatrix<T> >& proc);

  const ClusterData * rows() const {
      return engine_->hmat->rows();
  }

  const ClusterData * cols() const {
      return engine_->hmat->cols();
  }

  const IEngine<T> & engine() const {
      return *engine_;
  }

  EngineSettings& engineSettings() {
    return engine_->GetSettings();
  }

  Factorization factorization() {
      return factorizationType;
  }
 
  HMatrix<T>* get( int i, int j) const;

  void setHMatrix( HMatrix<T> *hmat = NULL) const {
      engine_->setHMatrix( hmat );
  }

  void progress(hmat_progress_t * progress) {
      engine_->progress(progress);
  }
private:
  /// Disallow the copy
  HMatInterface(const HMatInterface<T>& o);
};

}  // end namespace hmat

#endif
