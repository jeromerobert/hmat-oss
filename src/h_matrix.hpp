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
#include "interaction.hpp"
#include "data_types.hpp"
#include "full_matrix.hpp"
#include "cluster_tree.hpp"
#include <cassert>
#include <fstream>
#include <iostream>


template<typename T> class Vector;
template<typename T> class RkMatrix;

/** Flag used to describe the symmetry of a matrix.
 */
enum SymmetryFlag {kNotSymmetric, kLowerSymmetric};

namespace hmat {
  /** Settings global to a whole matrix */
  struct MatrixSettings {
     virtual double getAdmissibilityFactor() const = 0;
     virtual int getMaxElementsPerBlock() const = 0;
  };

  /** Settings local to a matrix bloc */
  struct LocalSettings {
      const MatrixSettings * global;
      explicit LocalSettings(const MatrixSettings * s): global(s) {}
      //TODO add epsilons
  };
}

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


/*! \brief Data held by an HMatrix tree node.
 */
template<typename T> class HMatrixData {
public:
  /// Rows of this HMatrix block
  ClusterTree * rows;
  /// Columns of this HMatrix block
  ClusterTree * cols;
  /// Compressed block, or NULL if the block is not a leaf or is full.
  RkMatrix<T> *rk;
  /// Full block, or NULL if the block is not a leaf or is compressed.
  FullMatrix<T> *m;

public:
  HMatrixData() : rows(NULL), cols(NULL), rk(NULL), m(NULL) {}
  HMatrixData(ClusterTree* _rows, ClusterTree* _cols)
  : rows(_rows), cols(_cols), rk(NULL), m(NULL) {}
  ~HMatrixData();
  /*! \brief Return true if the block is admissible.
   */
  bool isAdmissibleLeaf(const hmat::MatrixSettings * settings) const;
};

/*! \brief The HMatrix class, representing a HMatrix.

  It is a tree of arity arity(ClusterTree)^2, 4 in this case.
  An HMatrix is a tree-like structure that is:
    - a Leaf : in this case the node is either really a RkMatrix
      (compressed block), or a small dense block.
    - an internal node : in this case, it has 4 children that form a partition
      of the HMatrix Dofs, and the node doesn't carry data itself.
 */
template<typename T> class HMatrix : public Tree<4> {
  friend class RkMatrix<T>;
public:
  /*! \brief Create a HMatrix based on a row and column ClusterTree.

    \param _rows The row cluster tree
    \param _cols The column cluster tree
   */
  HMatrix(ClusterTree* _rows, ClusterTree* _cols, const hmat::MatrixSettings * settings,
       SymmetryFlag symmetryFlag = kNotSymmetric);
  /*! \brief HMatrix assembly.
   */
  void assemble(const AssemblyFunction<T>& f);
  /*! \brief HMatrix assembly.

    \param f the assembly function
    \param upper the upper part of the matrix. If NULL, it is assumed
                 that upper=this (that is, the current block is on the diagonal)
    \param onlyLower if true, only assemble the lower part of the matrix, ie don't copy.
   */
  void assembleSymmetric(const AssemblyFunction<T>& f,
     HMatrix<T>* upper=NULL, bool onlyLower=false);
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
  void evalPart(FullMatrix<T>* result, const ClusterData* _rows, const ClusterData* _cols) const;
  /*! Compute the compression ratio of the HMatrix.

    \return the pair (elements_stored, total_elements).
   */
  std::pair<size_t, size_t> compressionRatio() const;
  /*! Compute the full/rk ratio of the HMatrix.

    \return the pair (full_elements, rk_elements).
   */
  std::pair<size_t, size_t> fullrkRatio() const;
  /** This *= alpha

      \param alpha scaling factor
   */
  void scale(T alpha);
  /** Compute y <- alpha * op(A) * y + beta * y.

      The arguments are similar to BLAS GEMV.
   */
  void gemv(char trans, T alpha, const Vector<T>* x, T beta, Vector<T>* y) const;
  /** Compute y <- alpha * op(A) * y + beta * y.

      The arguments are similar to BLAS GEMV.

      This version is in principle similar to GEMM(), but if you take
      enough drug, it is not the same thing at all.
   */
  void gemv(char trans, T alpha, const FullMatrix<T>* x, T beta, FullMatrix<T>* y) const;
  /*! \brief this <- alpha * A * B + beta * C

    \param transA 'N' or 'T', as in BLAS
    \param transB 'N' or 'T', as in BLAS
    \param alpha alpha
    \param a the matrix A
    \param b the matrix B
    \param beta beta
   */
  void gemm(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>*b, T beta, int depth=0);
  /*! \brief S <- S - M * D * M^T, where S is symmetric (Lower stored),
      D diagonal, this = S

      \warning D has to be reduced in ldlt form with d->ldltDecomposition() before

      \param m M
      \param d D : only the diagonal of this matrix is considered
   */
  void mdmtProduct(const HMatrix<T> * m, const HMatrix<T> * d);
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
  static HMatrix<T>* Zero(const ClusterTree* rows, const ClusterTree* cols, const hmat::MatrixSettings * settings);
  /*! \brief Create a Postscript file representing the HMatrix.

    The result .ps file shows the matrix structure and the compression ratio. In
    the output, red = full block, green = compressed. The darker the green, the
    worst the compression ration is. There is saturation at black when the block
    size is divided by less than 5.

    \param filename output filename.
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
  /** this <- o (copy)

      \param o The HMatrix t copy
   */
  void copy(const HMatrix<T>* o);
  /** Copy the structure of an HMatrix without copying its content.

      \return an empty HMatrix (not even the Full leaves are
      allocated) mirroring the structure of this.
   */
  HMatrix<T>* copyStructure() const;
  /*! \brief Return the Frobenius norm of the matrix.
   */
  double norm() const;
  /** Set a matrix to 0.
   */
  void clear();
  /** Inverse an HMatrix in place.

      \param tmp temporary HMatrix used in the inversion. If set, it must have
      the same structure as this. Otherwise, it is allocated at the start of the
      computation (and will be freed at the end).
      \param depth The depth, used for pretty printing purposes
   */
  void inverse(HMatrix<T>* tmp=NULL, int depth=0);
  /*! \brief Transpose the H-matrix in place
   */
  void transpose();
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
private:
  /*! \brief Auxiliary function used by HMatrix::dumpTreeToFile().
   */
  void dumpSubTree(std::ofstream& f, int depth) const;

public:
  /*! \brief Build a "fake" HMatrix for internal use only
   */
  HMatrix(const hmat::MatrixSettings * settings);
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
  void axpy(T alpha, const FullMatrix<T>* b, const ClusterData* rows, const ClusterData* cols);
  /*! Return true if this is a full block.
   */
  inline bool isFullMatrix() const {
    return isLeaf() && data.m;
  };
  /* Return the full matrix corresponding to the current leaf
   */
  FullMatrix<T>* getFullMatrix() const {
    assert(isFullMatrix());
    return data.m;
  }
  /*! Return true if this is a compressed block.
   */
  inline bool isRkMatrix() const {
    return isLeaf() && data.rk;
  };
  /*! Return true if this is not a leaf.
   */
  inline bool isHMatrix() const {
    return !isLeaf();
  };
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
  /*! \brief B <- B*D : ou B = this et D est en argument

    \warning D doit avoir ete decomposee en LDL^T avant
    \param d matrice D
  */
  void multiplyWithDiag(const HMatrix<T>* d, bool left = false);
  /*! \brief B <- B*D ou B <- B*D^-1  (ou idem a gauche) : ou B = this et D est en argument

     \param d matrice D
     \param inverse true : B<-B*D^-1, false B<-B*D
     \param left true : B<-D*B, false B<-B*D
  */
  void multiplyWithDiagOrDiagInv(const HMatrix<T>* d, bool inverse, bool left = false);
  /*! \brief Resolution du systeme L X = B, avec this = L, et X = B.

    \param b la matrice B en entree, et X en sortie.
   */
  void solveLowerTriangular(HMatrix<T>* b) const;
  /*! \brief Resolution du systeme L x = x, avec this = L, et x = b vecteur.

    B est un vecteur a plusieurs colonnes, donc une FullMatrix.

    \param b Le vecteur b en entree, et x en sortie.
   */
  void solveLowerTriangular(FullMatrix<T>* b) const;
  /*! Resolution de X U = B, avec U = this, et X = B.

    \param b la matrice B en entree, X en sortie
   */
  void solveUpperTriangular(HMatrix<T>* b, bool loweredStored = false) const;
  /*! Resolution de U X = B, avec U = this, et X = B.

    \param b la matrice B en entree, X en sortie
   */
  void solveUpperTriangularLeft(HMatrix<T>* b) const;
  /*! Resolution de x U = b, avec U = this, et x = b.

    \warning b est un vecteur ligne et non colonne.

    \param b Le vecteur b en entree, x en sortie.
   */
  void solveUpperTriangular(FullMatrix<T>* b, bool loweredStored = false) const;
  /*! Resolution de U x = b, avec U = this, et x = b.
    U peut etre en fait L^T ou L est une matrice stockee inferieurement
    en precisant lowerStored = true

    \param b Le vecteur b en entree, x en sortie.
    \param indice les indices portes par le vecteur
    \param lowerStored indique le stockage de la matrice U ou L^T
  */
  void solveUpperTriangularLeft(FullMatrix<T>* b, bool lowerStored=false) const;
  /* Solve D x = b, in place with D a diagonal matrix.

     \param b Input: B, Output: X
   */
  void solveDiagonal(FullMatrix<T>* b) const;
  /*! Resolution de This * x = b.

    \warning This doit etre factorisee avec \a HMatrix::luDecomposition() avant.
   */
  void solve(FullMatrix<T>* b) const;
  /*! Resolution de This * X = b.

    \warning This doit etre factorisee avec \a HMatrix::luDecomposition() avant.
   */
  void solve(HMatrix<T>* b) const;
  /*! Resolution de This * x = b.

    \warning This doit etre factorisee avec \a HMatrix::ldltDecomposition() avant.
   */
  void solveLdlt(FullMatrix<T>* b) const ;
  /*! Triggers an assertion is the HMatrix contains any NaN.
   */
  void checkNan() const;
  /** Recursively set the isTriLower flag on this matrix */
  void setTriLower(bool value);
public:
  const ClusterData* rows() const;
  const ClusterData* cols() const;
  /*! Return the child (i, j) of this.

    \warning do not use on a leaf !

    \param i row
    \param j column
    \return the (i,j) child of this.
   */
  inline HMatrix<T>* get(int i, int j) const {
    return static_cast<HMatrix<T>*>(getChild(i + j * 2));
  }

public:
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
  HMatrixData<T> data;
  bool isUpper, isLower;       /// symmetric, upper or lower stored
  bool isTriUpper, isTriLower; /// upper/lower triangular
  hmat::LocalSettings localSettings;

private:
  /* \brief Resolution de X * D = B, avec D = this (matrice dont on ne tient compte que de la diagonale)
     et B <- X

     \param b la HMatrix en entree qui contiendra la solution en sortie
  */
  void getDiag(Vector<T>* diag, int start=0) const;
#ifdef DEBUG_LDLT
  /*  \brief verifie que la matrice est bien Lower i.e. avec des fils NULL au-dessus
   */
  void assertLower();
  /*  \brief verifie que la matrice est bien Upper i.e. avec des fils NULL au-dessous
   */
  void assertUpper();
  /* \brief verifie que toutes les feuilles full de la hmatrice sont bien allouees
   */
  void assertAllocFull();
  /* \brief verifie juste que les matrices diagonales full ont bien une "diagonale" et sont isTriLower
   */
  bool assertLdlt() const;
  /* \brief fait le produit LDL^T pour verifier la decomposition
     Calcule la norme de L2 de deux matrices
     \param originalCopy hmatrice avant decomposition LDL^T
   */
  void testLdlt(HMatrix<T> * originalCopy) ;
#endif
};
#endif
