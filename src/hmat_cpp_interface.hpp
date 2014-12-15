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
#include "hmat/hmat.h"

#include "compression.hpp"
#include "h_matrix.hpp"
#include "default_engine.hpp"

class ClusterTree;

/** Type of ClusterTree */
enum ClusteringType {kGeometric, kMedian, kHybrid};


/** Settings for the HMatrix library.

    A single static instance of this class exist, but settings the values is not
    sufficient for the settings to take effect. One must call \a
    HMatSettings::setParameters().
*/
//TODO remove all global settings
class HMatSettings: public hmat::MatrixSettings {
public:
  double assemblyEpsilon; ///< Tolerance for the assembly.
  double recompressionEpsilon; ///< Tolerance for the recompression (using SVD)
  CompressionMethod compressionMethod; ///< Compression method
  int compressionMinLeafSize; ///< Force SVD compression if max(rows->n, cols->n) < compressionMinLeafSize
  /** \f$\eta\f$ in the admissiblity condition for two clusters \f$\sigma\f$ and \f$\tau\f$:
      \f[
      \min(diam(\sigma), diam(\tau)) < \eta \cdot d(\sigma, \tau)
      \f]
   */
  double admissibilityFactor;
  ClusteringType clustering; ///< Type of ClusterTree
  int maxLeafSize; ///< Maximum size of a leaf in a ClusterTree (and of a non-admissible block in an HMatrix)
  int maxParallelLeaves; ///< max(|L0|)
  int elementsPerBlock; ///< Maximum size of an admissible block. Should be size_t !
  bool useLu; ///< Use an LU decomposition
  bool useLdlt; ///< Use an LDL^t decomposition if possible
  bool cholesky;
  bool coarsening; ///< Coarsen the matrix structure after assembly.
  bool recompress; ////< Recompress the matrix after assembly.
  bool validateCompression; ///< Validate the rk-matrices after compression
  bool validationReRun; ///< For blocks above error threshold, re-run the compression algorithm
  bool validationDump; ///< For blocks above error threshold, dump the faulty block to disk
  double validationErrorThreshold; ///< Error threshold for the compression validation

private:
  /** This constructor sets the default values.
   */
  HMatSettings() : assemblyEpsilon(1e-4), recompressionEpsilon(1e-4),
                   compressionMethod(Svd),  compressionMinLeafSize(100),
                   admissibilityFactor(2.),
                   clustering(kMedian),
                   maxLeafSize(100),
                   maxParallelLeaves(5000),
                   elementsPerBlock(5000000),
                   useLu(true),
                   useLdlt(false),
                   cholesky(false),
                   coarsening(false),
                   recompress(true), validateCompression(false),
                   validationReRun(false), validationDump(false), validationErrorThreshold(0.) {}
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
  /** Output a textual representation of the settings to out.

      @param out The output stream, std::cout by default.
   */
  void printSettings(std::ostream& out = std::cout) const;

  virtual double getAdmissibilityFactor() const {
      return admissibilityFactor;
  }
  virtual int getMaxElementsPerBlock() const {
      return elementsPerBlock;
  }
};

/** Create a ClusterTree.

    The exact type of the returned ClusterTree depends on the global HMatrix
    library settings (\a HMatSettings::clustering).  Passing the returned
    ClusterTree pointer to the constructor of \a HMatInterface transfers its
    ownership to this instance. It is then automatically freed at the destrution
    of its owner.

    @note This is the only proper way to dispose of a ClusterTree instance.

    @param dls Array of DofCoordinate, of length n
    @param n number of elements in dls
    @return a ClusterTree instance.
 */
ClusterTree* createClusterTree(DofCoordinate* dls, int n);

/** C++ interface to the HMatrix library.

    This is the sole entry point to the HMatrix library.

    This interface is templated over the scalar type, T. This type has to be one
    of {S_t, D_t, C_t, Z_t}, using the standard BLAS notation. For the complex
    types, the C++ complex<> type is used. It is guaranteed to have the same
    layout as the equivalent FORTRAN types.

    The user code *has* to call \a HMatInterface<T>::init() before using any
    other function, and \a HMatInterface<T>::finalize() at the end.
*/
template<typename T, template <typename> class E = DefaultEngine>
class HMatInterface {
 private:
  static bool initialized; ///< True if the library has been initialized.

public:
  ClusterTree* rows; ///< Row ClusterTree
  ClusterTree* cols; ///< Column ClusterTree

private:
  E<T> engine;


public:
  /** Initialize the library.

      @warning This *must* be called before using the HMatrix library.
   */
  static int init();
  /** Finalize the library.

      @warning The library cannot be used after this has been called.
   */
  static void finalize();
  /** Build a new HMatrix from two cluster sets.

      @note The ownership of the two ClusterTree instances (which don't need to
      be different) is transfered to the returned HMatInterface instance, and
      will be disposed at destruction time.

      @param _rows The row ClusterTree instance, built with \a createClusterTree()
      @param _cols The column ClusterTree instance, built with \a createClusterTree()
      @param symmetric If kLowerSymmetric, only lower triangular structure is created
      @return a new HMatInterface instance.
   */
  HMatInterface(ClusterTree* _rows, ClusterTree* _cols, SymmetryFlag sym);
  /** Destroy an HMatInterface instance.

      @note This destructor is *not* virtual, as this class is not meant to be subclassed.
   */
  ~HMatInterface();
  /** Assemble an HMatrix.

      This builds an HMatrix using a provided AssemblyFunction. The compression
      method is determined by \a HMatSettings::compressionMethod, and the
      tolerance by \a HMatSettings::assemblyEpsilon. A recompression is done
      when \a HMatSettings::recompress is true.

      @param f The assembly function used to compute various matrix sub-parts
      @param sym If kLowerSymmetric, compute only the lower triangular matrix, and transpose
                 block to store upper counterpart.
      @param synchronize
   */
  void assemble(AssemblyFunction<T>& f, SymmetryFlag sym, bool synchronize=true);
  /** Compute a \f$LU\f$ or \f$LDL^T\f$ decomposition of the HMatrix, in place.

      An LDL^T decomposition is done if the HMatrix is symmetric and has been
      assembled as such (with sym = kLowerSymmetric in
      HMatInterface<T>::assemble()), and if HMatSettings::useLdlt is
      true. Otherwise an LU decomposition is done.
   */
  void factorize();

  /** Matrix-Vector product.

      This computes \f$ y \gets \alpha . op(A) x + \beta y\f$, with A = this, x
      and y FullMatrix<T>. If trans == 'N', then op(A) = A, if trans == 'T',
      then op(A) = A^T, as in BLAS.

      @param trans 'N' or 'T'
      @param alpha
      @param x
      @param beta
      @param y
   */
  void gemv(char trans, T alpha, FullMatrix<T>& x, T beta, FullMatrix<T>& y) const;
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
  void gemm(char transA, char transB, T alpha, const HMatInterface<T, E>* a, const HMatInterface<T, E>* b, T beta);
  /** Full <- Full x HMatrix product.

      This computes the product \f$ C_F \gets \alpha . op(A_F) \times op(B_H) +
      \beta C_F\f$, with \f$A_F\f$, \f$C_F\f$ two FullMatrix<T>, and \f$B_H\f$
      an HMatrixInterface<T> instance.

      The meaning of the arguments is as in \a HMatInterface<T>::gemm(), and in
      BLAS.
   */
  static void gemm(FullMatrix<T>& c, char transA, char transB, T alpha, FullMatrix<T>& a, const HMatInterface<T, E>& b, T beta);
  /** Return a new copy of this.
   */
  HMatInterface<T, E>* copy() const;
  /** Transpose this in place.
   */
  void transpose();
  /** Solve the system \f$A x = b\f$ in place, with A = this, and b a FullMatrix.

      @warning A has to be factored first with \a HMatInterface<T>::factorize().
   */
  void solve(FullMatrix<T>& b) const;
  /** Solve the system \f$A x = B\f$ in place, with A = this, and B a HMatInterface<T>.

      @warning A has to be factored first with \a HMatInterface<T>::factorize().
   */
  void solve(HMatInterface<T, E>& b) const;
  /** Return an approximation of the Frobenius norm of this.
   */
  double norm() const;
  /** this <- alpha * this
   */
  void scale(T alpha);
  /** Return the compression ratio of the HMatrix.

    @note This is only meaningful once the HMatrix has been assembled.

    @return The pair (element_stored, total_element).
  */
  std::pair<size_t, size_t> compressionRatio() const;
  /** Return the ratio between full and rk leaves of the HMatrix.

    @note This is only meaningful once the HMatrix has been assembled.

    @return The pair (full_leaves_size, rk_leaves_size).
  */
  std::pair<size_t, size_t> fullrkRatio() const;
  /** Create a Postscript file representing the HMatrix.

    The result .ps file shows the matrix structure and the compression ratio. In
    the output, red = full block, green = compressed. The darker the green, the
    worst the compression ration is. There is saturation at black when the block
    size is divided by less than 5.

    @param filename output filename.
   */
  void createPostcriptFile(const char* filename) const;
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
  void dumpTreeToFile(const char* filename) const;
  /** Return the number of block cluster tree nodes.
   */
  int nodesCount() const;

  typename E<T>::Settings & engineSettings() { return engine.settings; }
private:
  HMatInterface(HMatrix<T>* h);

private:
  /// Disallow the copy
  HMatInterface(const HMatInterface<T, E>& o);
};
#endif
