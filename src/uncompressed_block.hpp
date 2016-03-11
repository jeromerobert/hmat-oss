/*
  HMat-OSS (HMatrix library, open source software)

  Copyright (C) 2014-2016 Airbus Group SAS

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

#include "h_matrix.hpp"
#include "cluster_tree.hpp"
#include "rk_matrix.hpp"

namespace hmat {
/**
 * Extract an uncompressed block from the matrix not necessarly
 * matching a block of the h-matrix.
*/
template <typename T> class UncompressedBlock {
    const HMatrix<T> &matrix_;
    IndexSet rowIndexSet_, colIndexSet_;
    // TODO replace by a FullMatrix ?
    T *values_;
    int valuesLd;

    void getNullValues() {
        T *toFill = values_;
        for (int col = 0; col < colIndexSet_.size(); col++) {
            for (int row = 0; row < rowIndexSet_.size(); row++) {
                toFill[row] = Constants<T>::zero;
            }
            toFill += valuesLd;
        }
    }

    void getFullValues() {
        int nr = rowIndexSet_.size();
        int nc = colIndexSet_.size();
        FullMatrix<T> target(values_, nr, nc, valuesLd);
        int localRowOffset = rowIndexSet_.offset() - matrix_.rows()->offset();
        int localColOffset = colIndexSet_.offset() - matrix_.cols()->offset();
        assert(localRowOffset >= 0);
        assert(localColOffset >= 0);
        T *sa = matrix_.full()->m + localRowOffset +
                ((size_t)matrix_.full()->lda) * localColOffset;
        FullMatrix<T> source(sa, nr, nc, matrix_.full()->lda);
        target.copyMatrixAtOffset(&source, 0, 0);
    }

    void getRkValues() {
        // TODO may be we could ask the caller to provide a clean target array
        // to avoid this call.
        getNullValues();
        int nr = rowIndexSet_.size();
        int nc = colIndexSet_.size();
        int k = matrix_.rank();
        FullMatrix<T> result(values_, nr, nc, valuesLd);
        T *sa = matrix_.rk()->a->m + rowIndexSet_.offset() - matrix_.rows()->offset();
        FullMatrix<T> a(sa, nr, k, matrix_.rows()->size());
        T *sb = matrix_.rk()->b->m + colIndexSet_.offset() - matrix_.cols()->offset();
        FullMatrix<T> b(sb, nc, k, matrix_.cols()->size());
        result.gemm('N', 'T', Constants<T>::pone, &a, &b, Constants<T>::zero);
    }

    void getValues() {
        if (rowIndexSet_.size() == 0 || colIndexSet_.size() == 0)
            return;
        if (!matrix_.isLeaf()) {
            for (int i = 0; i < matrix_.nbChild(); i++) {
                UncompressedBlock view(*this, *matrix_.getChild(i));
            }
        } else if (matrix_.isNull()) {
            getNullValues();
        } else if (matrix_.isRkMatrix()) {
            getRkValues();
        } else if (matrix_.isFullMatrix()) {
            getFullValues();
        } else {
            assert(false);
        }
    }

    UncompressedBlock(const UncompressedBlock & o, const HMatrix<T> &matrix)
        : matrix_(matrix) {
        rowIndexSet_.intersection(o.rowIndexSet_, *matrix.rows());
        colIndexSet_.intersection(o.colIndexSet_, *matrix.cols());
        valuesLd = o.valuesLd;
        int drow = rowIndexSet_.offset() - o.rowIndexSet_.offset();
        int dcol = colIndexSet_.offset() - o.colIndexSet_.offset();
        values_ = o.values_ + drow + ((size_t)valuesLd) * dcol;
        getValues();
    }

  public:

    UncompressedBlock(const HMatrix<T> &matrix, const IndexSet & rows,
                     const IndexSet & cols, T *values, int ld = -1)
        : matrix_(matrix), rowIndexSet_(rows), colIndexSet_(cols),
          values_(values), valuesLd(ld) {
        assert(matrix.father == NULL);
        if(valuesLd == -1)
            valuesLd = rows.size();
        getValues();
    }

    int * colsNumbering() {
        return matrix_.cols()->indices() + colIndexSet_.offset();
    }

    int * rowsNumbering() {
        return matrix_.rows()->indices() + rowIndexSet_.offset();
    }

    void renumberRows() {
        HMAT_ASSERT_MSG(matrix_.father == NULL && rowIndexSet_ == *matrix_.rows(),
                        "Cannot renumber");
        FullMatrix<T> fm(values_, rowIndexSet_.size(), colIndexSet_.size(), valuesLd);
        restoreVectorOrder(&fm, rowsNumbering());
    }
};
}
