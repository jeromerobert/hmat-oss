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

/*! \file
  \ingroup HMatrix
  \brief Implementation of the HMatrix class.
*/
#ifndef _H_MATRIX_HPP
#define _H_MATRIX_HPP

#include "tree.hpp"
#include "assembly.hpp"
#include "data_types.hpp"
#include "full_matrix.hpp"
#include "cluster_tree.hpp"
#include "admissibility.hpp"
#include "recursion.hpp"
#include <cassert>
#include <fstream>
#include <iostream>


namespace hmat {

template<typename T> class Vector;
template<typename T> class RkMatrix;

/** Flag used to describe the symmetry of a matrix.
 */
enum SymmetryFlag {kNotSymmetric, kLowerSymmetric};

/** Default rank value for blocks that dont have an actual computed rank
   */
enum DefaultRank {UNINITIALIZED_BLOCK = -3, NONLEAF_BLOCK = -2, FULL_BLOCK = -1};

/** Settings global to a whole matrix */
struct MatrixSettings {
};

/** Settings local to a matrix bloc */
struct LocalSettings {
    const MatrixSettings * global;
    explicit LocalSettings(const MatrixSettings * s): global(s) {}
    //TODO add epsilons
};

/** Degrees of freedom permutation of a vector required in HMatrix context.

     In order that the subsets of rows and columns are
     contiguous in HMatrix, we must reorder the elements of the vector. This
     order is induced by the array of indices after the construction of
     ClusterTree, which must be passed as a parameter.

     \param v Vector to reorder.
     \param indices Array of indices after construction ClusterTree.

 */
template<typename T> void reorderVector(FullMatrix<T>* v, int* indices);

/** Inverse permutation of a vector.

     See \a reorderVector () for more details.

     \param v Vector to reorder of the problem.
     \param indices Array of indices after construction ClusterTree.

 */
template<typename T> void restoreVectorOrder(FullMatrix<T>* v, int *indices);

template<typename T> class HMatrix;
/** Class to write user defined data when dumping matrix onto disk.

    This class is used by dumpTreeToFile to write extra information into
    json file; the returned string must thus be a valid json fragment text.
 */
template<typename T> class HMatrixNodeDumper {
public:
  HMatrixNodeDumper() {}
  virtual std::string dumpExtraInfo(const HMatrix<T>& node, const std::string& prefix) const = 0;
};

template<typename T>
class HMatrixVoidNodeDumper : public HMatrixNodeDumper<T> {
public:
  virtual std::string dumpExtraInfo(const HMatrix<T>&, const std::string&) const { return ""; }
};

/** Class to truncate Rk matrices.
 */
template<typename T>
class EpsilonTruncate : public TreeProcedure<HMatrix<T> > {
private:
  double epsilon_;
public:
  EpsilonTruncate(double epsilon) : epsilon_(epsilon) {}
  void visit(HMatrix<T> * node, const Visit order) const;
};

/*! \brief The HMatrix class, representing a HMatrix.

  It is a tree of arity arity(ClusterTree)^2, 4 in most cases.
  An HMatrix is a tree-like structure that is:
    - a Leaf : in this case the node is either really a RkMatrix
      (compressed block), or a small dense block.
    - an internal node : in this case, it has 4 children that form a partition
      of the HMatrix Dofs, and the node doesn't carry data itself.
 */
template<typename T> class HMatrix : public Tree<HMatrix<T> >, public RecursionMatrix<T, HMatrix<T> > {
  friend class RkMatrix<T>;

  /// Rows of this HMatrix block
  const ClusterTree * rows_;
  /// Columns of this HMatrix block
  const ClusterTree * cols_;
  union {
   /// Compressed block, or NULL if the block is not a leaf or is full.
   RkMatrix<T> * rk_;
   /// Full block, or NULL if the block is not a leaf or is compressed.
   FullMatrix<T> * full_;
  };
  /// rank_ of the block for Rk matrices, or: UNINITIALIZED_BLOCK=-3 for an uninitialized matrix, NONLEAF_BLOCK=-2 for non leaf, FULL_BLOCK=-1 for full a matrix
  int rank_;
  void uncompatibleGemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>*b);
  void recursiveGemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>*b);
  void leafGemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>*b);
  HMatrix<T> * fullRkSubset(const IndexSet* subset, bool col) const;
  /*! \brief Auxiliary function used by HMatrix::dumpTreeToFile().
   */
  void dumpSubTree(std::ofstream& f, int depth, const HMatrixNodeDumper<T>& nodeDumper) const;
  /** Only used by internalCopy */
  HMatrix(const MatrixSettings * settings);
public:
  /*! \brief Create a HMatrix based on a row and column ClusterTree.

    \param _rows The row cluster tree
    \param _cols The column cluster tree
   */
  HMatrix(ClusterTree* _rows, ClusterTree* _cols, const MatrixSettings * settings,
       SymmetryFlag symmetryFlag = kNotSymmetric,
       AdmissibilityCondition * admissibilityCondition = &StandardAdmissibilityCondition::DEFAULT_ADMISSIBLITY);

  /*! \brief Create a copy of this matrix for internal use only.
   * Only copy this node, not the whole tree. The created matrix
   * is an uninitialized leaf with same rows and cols as this.
   */
  HMatrix<T> * internalCopy(bool temporary = false, bool withChildren = false) const;
  ~HMatrix();

  /*! \brief HMatrix coarsening.

     If all children are Rk leaves, then we try to merge them into a single Rk-leaf.
     This is done if the memory of the resulting leaf is less than the sum of the initial
     leaves. Note that this operation could be used hierarchically.
     \param upper the symmetric of 'this', when building a non-sym matrix with a sym content
   */
  void coarsen(HMatrix<T>* upper = NULL) ;
  /*! \brief HMatrix assembly.
   */
  void assemble(Assembly<T>& f, const AllocationObserver & = AllocationObserver());
  /*! \brief HMatrix assembly.

    \param f the assembly function
    \param upper the upper part of the matrix. If NULL, it is assumed
                 that upper=this (that is, the current block is on the diagonal)
    \param onlyLower if true, only assemble the lower part of the matrix, ie don't copy.
   */
  void assembleSymmetric(Assembly<T>& f,
     HMatrix<T>* upper=NULL, bool onlyLower=false,
     const AllocationObserver & = AllocationObserver());
  /*! \brief Evaluate the HMatrix, ie converts it to a full matrix.

    This conversion does the reorderng of the unknowns such that the resulting
    matrix can directly be used or compared with a full matrix assembled
    otherwise.

    \param result a FullMatrix that has to be preallocated at the same size than
    this.
   */
  void eval(FullMatrix<T>* result, bool renumber = true) const;
  /*! \brief Evaluate this as a subblock of the larger matrix result.

    _rows and _cols are the rows and columns of the result matrix. This has to
    be a subset of _rows and _cols, and it is put at its place inside the result
    matrix. This function does not do any reodering.

    \param result Result matrix, of size (_rows->n, _cols->n)
    \param _rows Rows of the result matrix
    \param _cols Columns of the result matrix
   */
  void evalPart(FullMatrix<T>* result, const IndexSet* _rows, const IndexSet* _cols) const;

  void info(hmat_info_t &);

  /** This *= alpha

      \param alpha scaling factor
   */
  void scale(T alpha);
  /** Compute y <- alpha * op(this) * x + beta * y.

      The arguments are similar to BLAS GEMV.
   */
  void gemv(char trans, T alpha, const Vector<T>* x, T beta, Vector<T>* y) const;
  /** Compute y <- alpha * op(this) * x + beta * y.

      The arguments are similar to BLAS GEMV.
   */
  void gemv(char trans, T alpha, const FullMatrix<T>* x, T beta, FullMatrix<T>* y) const;
  /*! \brief this <- alpha * op(A) * op(B) + beta * this

    \param transA 'N' or 'T', as in BLAS
    \param transB 'N' or 'T', as in BLAS
    \param alpha alpha
    \param a the matrix A
    \param b the matrix B
    \param beta beta
   */
  void gemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>*b, T beta);
  /*! \brief this <- this - M * D * M^T, where 'this' is symmetric (Lower stored),
      D diagonal

      \warning D has to be reduced in ldlt form with d->ldltDecomposition() before

      \param m M
      \param d D : only the diagonal of this matrix is considered
   */
  void mdmtProduct(const HMatrix<T> * m, const HMatrix<T> * d);

  /*! \brief this <- this - M * D * N^T with D diagonal

      \warning D has to be reduced in ldlt form with d->ldltDecomposition() before

      \param m M
      \param d D : only the diagonal of this matrix is considered
      \param n N
   */
  void mdntProduct(const HMatrix<T>* m, const HMatrix<T>* d, const HMatrix<T>* n);

  /** Create a matrix filled with 0s, with the same structure as H.

      \param h the model matrix,
      \return a 0 matrix with the same structure as H.
   */
  static HMatrix<T>* Zero(const HMatrix<T>* h);
  /** Create a matrix filled with 0s, based on 2 ClusterTree.

      \param rows the row ClusterTree.
      \param cols the column ClusterTree.
      \return a 0 HMatrix.
   */
  static HMatrix<T>* Zero(const ClusterTree* rows, const ClusterTree* cols,
                          const MatrixSettings * settings,
                          AdmissibilityCondition * admissibilityCondition =
                          &StandardAdmissibilityCondition::DEFAULT_ADMISSIBLITY);
  /*! \brief Create a Postscript file representing the HMatrix.

    The result .ps file shows the matrix structure and the compression ratio. In
    the output, red = full block, green = compressed. The darker the green, the
    worst the compression ration is. There is saturation at black when the block
    size is divided by less than 5.

    \param filename output filename.
   */
  void createPostcriptFile(const std::string& filename) const;
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
  void dumpTreeToFile(const std::string& filename, const HMatrixNodeDumper<T>& nodeDumper) const;
  /** Returns a copy of this (with all the structure and data)
       */
  HMatrix<T>* copy() const ;
  /** this <- o (copy)

      \param o The HMatrix to copy. 'this' must be allready created and have the right structure.
   */
  void copy(const HMatrix<T>* o);
  /** Copy the structure of an HMatrix without copying its content.

      \return an empty HMatrix (not even the Full leaves are
      allocated) mirroring the structure of this.
   */
  HMatrix<T>* copyStructure() const;
  /*! \brief Return square of the Frobenius norm of the matrix.
   */
  double normSqr() const;
  /*! \brief Return the Frobenius norm of the matrix.
   */
  double norm() const {
    return sqrt(normSqr());
  }
  /** Set a matrix to 0.
   */
  void clear();
  /** Inverse an HMatrix in place.

      \param tmp temporary HMatrix used in the inversion. If set, it must have
      the same structure as this. Otherwise, it is allocated at the start of the
      computation (and will be freed at the end).
      \param depth The depth, used for pretty printing purposes
   */
  void inverse();
  /*! \brief Transpose the H-matrix in place
   */
  void transpose();

  /**
   * Swap non diagonal blocks and cluster trees.
   * Only used internally.
   */
  void transposeNoRecurse();

  /*! \brief this <- o^t

    \param o
   */
  void copyAndTranspose(const HMatrix<T>* o);
  /*! \brief LU decomposition in place.

    \warning Do not use. Doesn't work
   */
  void luDecomposition();
  /* \brief LDL^t decomposition in place
     \warning this has to be created with the flag lower
     \warning this has to be assembled with assembleSymmetric with onlyLower = true
   */
  void ldltDecomposition();
  void lltDecomposition();

  /** This <- This + alpha * b

      \param alpha
      \param b
   */
  void axpy(T alpha, const HMatrix<T>* b);
  /** This <- This + alpha * b

      \param alpha
      \param b
   */
  void axpy(T alpha, const RkMatrix<T>* b);
  /** This <- This + alpha * b

      \param alpha
      \param b
      \param rows
      \param cols
   */
  void axpy(T alpha, const FullMatrix<T>* b, const IndexSet* rows, const IndexSet* cols);
  /** This <- This + alpha * Id

      \param alpha
   */
  void addIdentity(T alpha);
  /*! Return true if this is a full block.
   */
  inline bool isFullMatrix() const {
    return rank_ == FULL_BLOCK && full_ != NULL;
  }
  /* Return the full matrix corresponding to the current leaf
   */
  FullMatrix<T>* getFullMatrix() const {
    assert(isFullMatrix());
    return full_;
  }
  /*! Return true if this is a compressed block.
   */
  inline bool isRkMatrix() const {
    return rank_ >= 0;
  }
  /*! \brief Return F * H (F Full, H divided)
   */
  static FullMatrix<T>* multiplyFullH(char transM, char transH, const FullMatrix<T>* m, const HMatrix<T>* h);
  /*! \brief Return H * F (F Full, H divided)
   */
  static FullMatrix<T>* multiplyHFull(char transH, char transM, const HMatrix<T>* h, const FullMatrix<T>* m);
  /*! \brief Multiplication de deux HMatrix dont au moins une est une RkMatrix.

      Le resultat est alors une RkMatrix.

      \param transA 'T' ou 'N' selon si A est transposee ou non
      \param transB 'T' ou 'N' selon si B est transposee ou non
      \param a A
      \param b B
   */
  static RkMatrix<T>* multiplyRkMatrix(char transA, char transB, const HMatrix<T>* a, const HMatrix<T>* b);
  /** Multiplication de deux HMatrix dont au moins une est une matrice pleine,
      et aucune n'est une RkMatrix.

      Le resultat est alors une matrice pleine.
  */
  static FullMatrix<T>* multiplyFullMatrix(char transA, char transB, const HMatrix<T>* a, const HMatrix<T>* b);
  /*! \brief B <- B*D where B is this

    \warning D must a have been decomposed by LDLt
    \param d matrice D
    \param left run B <- D*B instead of B <- B*D
    \param inverse run B <- B * D^-1
  */
  void multiplyWithDiag(const HMatrix<T>* d, bool left = false, bool inverse = false) const;
  /*! \brief Resolution du systeme L X = B, avec this = L, et X = B.

    \param b la matrice B en entree, et X en sortie.
   */
  void solveLowerTriangularLeft(HMatrix<T>* b, bool unitriangular) const;
  /*! \brief Resolution du systeme L x = x, avec this = L, et x = b vecteur.

    B est un vecteur a plusieurs colonnes, donc une FullMatrix.

    \param b Le vecteur b en entree, et x en sortie.
   */
  void solveLowerTriangularLeft(FullMatrix<T>* b, bool unitriangular) const;
  /*! Resolution de X U = B, avec U = this, et X = B.

    \param b la matrice B en entree, X en sortie
   */
  void solveUpperTriangularRight(HMatrix<T>* b, bool unitriangular, bool lowerStored) const;
  /*! Resolution de U X = B, avec U = this, et X = B.

    \param b la matrice B en entree, X en sortie
   */
  void solveUpperTriangularLeft(HMatrix<T>* b, bool unitriangular, bool lowerStored) const;
  /*! Resolution de x U = b, avec U = this, et x = b.

    \warning b est un vecteur ligne et non colonne.

    \param b Le vecteur b en entree, x en sortie.
   */
  void solveUpperTriangularRight(FullMatrix<T>* b, bool unitriangular, bool lowerStored) const;
  /*! Resolution de U x = b, avec U = this, et x = b.
    U peut etre en fait L^T ou L est une matrice stockee inferieurement
    en precisant lowerStored = true

    \param b Le vecteur b en entree, x en sortie.
    \param indice les indices portes par le vecteur
    \param lowerStored indique le stockage de la matrice U ou L^T
  */
  void solveUpperTriangularLeft(FullMatrix<T>* b, bool unitriangular, bool lowerStored) const;
  /* Solve D x = b, in place with D a diagonal matrix.

     \param b Input: B, Output: X
   */
  void solveDiagonal(FullMatrix<T>* b) const;
  void solveDiagonal(HMatrix<T>* b) const;
  /*! Resolution de This * x = b.

    \warning This doit etre factorisee avec \a HMatrix::luDecomposition() avant.
   */
  void solve(FullMatrix<T>* b) const;
  /*! Resolution de This * X = b.

    \warning This doit etre factorisee avec \a HMatrix::luDecomposition() avant.
   */
  void solve(HMatrix<T>* b, hmat_factorization_t) const;
  /*! Resolution de This * x = b.

    \warning This doit etre factorisee avec \a HMatrix::ldltDecomposition() avant.
   */
  void solveLdlt(FullMatrix<T>* b) const ;
  /*! Resolution de This * x = b.

    \warning This doit etre factorisee avec \a HMatrix::lltDecomposition() avant.
   */
  void solveLlt(FullMatrix<T>* b) const ;
  /*! Triggers an assertion is the HMatrix contains any NaN.
   */
  void checkNan() const;
  /** Recursively set the isTriLower flag on this matrix */
  void setTriLower(bool value);

  const ClusterData* rows() const;
  const ClusterData* cols() const;

  /*! \brief Return the number of children in the row dimension.
    */
  inline int nrChildRow() const {
    return rows_->nrChild();
  }

  /*! \brief Return the number of children in the column dimension.
    */
  inline int nrChildCol() const {
    return cols_->nrChild();
  }
  /*! \brief Destroy the HMatrix.
    */
  void destroy() {
    delete this;
  }

  /*! Return the child (i, j) of this.

    \warning do not use on a leaf !

    \param i row
    \param j column
    \return the (i,j) child of this.
   */
  HMatrix<T>* get(int i, int j) const {
    assert(i>=0 && i<nrChildRow());
    assert(j>=0 && j<nrChildCol());
    return this->getChild(i + j * nrChildRow());
  }

  /*! Set the child (i, j) of this.

    \warning do not use on a leaf !

    \param i row index
    \param j column index
    \param child the child (i, j) of this.
   */
  using Tree<HMatrix<T> >::insertChild;
  void insertChild(int i, int j, HMatrix<T>* child) {
    insertChild(i+j*nrChildRow(), child) ;
  }

  void setClusterTrees(const ClusterTree* rows, const ClusterTree* cols);
  HMatrix<T> * subset(const IndexSet * rows, const IndexSet * cols) const;

  /* \brief Retrieve diagonal values.
  */
  void extractDiagonal(T* diag) const;

  /// Should try to coarsen the matrix at assembly
  static bool coarsening;
  /// Should recompress the matrix after assembly
  static bool recompress;
  /// Validate the rk-matrices after compression
  static bool validateCompression;
  /// For blocks above error threshold, re-run the compression algorithm
  static bool validationReRun;
  /// For blocks above error threshold, dump the faulty block to disk
  static bool validationDump;
  /// Error threshold for the compression validation
  static double validationErrorThreshold;
  char isUpper:1, isLower:1,       /// symmetric, upper or lower stored
       isTriUpper:1, isTriLower:1, /// upper/lower triangular
       admissible:1, temporary:1,
       ownClusterTree_:1;
  LocalSettings localSettings;

  int rank() const {
      assert(rank_ >= 0);
      return rank_;
  }

  RkMatrix<T> * rk() const {
      assert(rank_ >= 0);
      return rk_;
  }

  void rk(const FullMatrix<T> * a, const FullMatrix<T> * b, bool updateRank = true);

  void rk(RkMatrix<T> * m) {
      rk_ = m;
      rank_ = m == NULL ? 0 : m->rank();
  }

  FullMatrix<T> * full() const {
      assert(rank_ == FULL_BLOCK);
      return full_;
  }

  void full(FullMatrix<T> * m) {
      full_ = m;
      rank_ = FULL_BLOCK;
  }

  bool isNull() const {
      assert(rank_ >= FULL_BLOCK);
      return rank_ == 0 || (rank_ == FULL_BLOCK && full_ == NULL);
  }

  bool isAssembled() const {
      return rank_ > UNINITIALIZED_BLOCK;
  }

  /**
   * Tag a not leaf block as assembled.
   * Must only be called when all leaves of this block have been
   * assembled (no coherency check).
   */
  void assembled() {
      assert(!this->isLeaf());
      rank_ = NONLEAF_BLOCK;
  }

  /**
   * Tag an entire subtree (except the leaves) as assembled.
   * (with recursion and coherency check: the leaves *must* already be tagged as assembled).
   */
  void assembledRecurse() {
    if (!this->isLeaf()) {
      for (int i=0 ; i<this->nrChild() ; i++)
        if (this->getChild(i))
          this->getChild(i)->assembledRecurse();
      rank_ = NONLEAF_BLOCK;
    }
    assert(isAssembled());
  }

  const ClusterTree * rowsTree() const {
      return rows_;
  }

  const ClusterTree * colsTree() const {
      return cols_;
  }

  /**
   * Convert this HMatrix to a string for debug.
   * This is better than overriding << because it allows to use printf.
   */
  std::string toString() const;

  /*! \brief Return a short string describing the content of this HMatrix for debug (like: "HMatrix [320, 452]x[760, 890] norm=13.23442" or "uninitialized" if needed)
    */
  std::string description() const {
    std::ostringstream convert;
    convert << "HMatrix " << rows()->description() << "x" << cols()->description() ;
    if (isAssembled())
      convert << "norm=" << norm();
    else
      convert << "uninitialized";
    return convert.str();
  }

};

}  // end namespace hmat

#endif
