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
#pragma once
#include "h_matrix.hpp"
#include "cluster_tree.hpp"
#include <limits>
#include <algorithm>

namespace hmat {

/**
 * @brief Base class to extract uncompressed values from a matrix
 * Sub classes must implement getLeafValues()
 * T is the scalar type to extract
 * M is the matrix type
 * I is the current class type for CRTP
 * @see https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
 */
template <typename T, template <typename> class M, typename I> class UncompressedValuesBase {
  public:
    typedef std::vector<std::pair<int, int> >::const_iterator IndiceIt;
  protected:
    const M<T> * matrix_;
    T *values_;
    int valuesLd_;
    /**
     * The first element of the pair a number of element to get, using hmat numbering.
     * The second element of the pair is the position of the element in the original query.
     */
    IndiceIt rowStart_, rowEnd_, colStart_, colEnd_;

    /** CRTP accessor */
    I & me() {
        return *static_cast<I*>(this);
    }

    /** truncate the begin/end interval so it is included in the clusterData interval */
    static void compatibleQuery(const IndexSet & clusterData, IndiceIt & begin, IndiceIt & end) {
        int lb = clusterData.offset();
        int ub = lb + clusterData.size() - 1;
        std::pair<int, int> lbP(lb, 0);
        // use max int to ensure that upper_bound->first will be greater than ubP
        std::pair<int, int> ubP(ub, std::numeric_limits<int>::max());
        IndiceIt newBegin = std::lower_bound(begin, end, lbP);
        if(newBegin == end) {
            // empty intersection
            begin = newBegin;
            return;
        }
        assert(newBegin->first >= lb);
        IndiceIt newEnd = std::upper_bound(begin, end, ubP);
        begin = newBegin;
        end = newEnd;
    }

    /**
     * @brief createQuery Convert the C API query to a vector<pair<>> where each pair is
     * <hmat id, original query id>
     * @param query, querySize the C API query
     * @param indices the result
     */
    void createQuery(const ClusterData & clusterData, int * query, int querySize,
                     bool hmat_numbering, std::vector<std::pair<int, int> > & indices) {
        indices.resize(querySize);
        for(int i = 0; i < querySize; i++) {
            if(hmat_numbering)
                indices[i].first = query[i];
            else
                indices[i].first = clusterData.indices_rev()[query[i] - 1];
            indices[i].second = i;
        }
        std::sort(indices.begin(), indices.end());
    }

    void getValues() {
        if (rowStart_ == rowEnd_ || colStart_ == colEnd_)
            return;
        if (me().isLeaf()) {
            me().getLeafValues();
        } else {
            for (int i = 0; i < matrix_->nrChild(); i++) {
                I view;
                M<T> * child = matrix_->getChild(i);
                if (!child) continue;
                view.matrix_ = child;
                view.values_ = values_;
                view.valuesLd_ = valuesLd_;
                view.rowStart_ = rowStart_;
                view.colStart_ = colStart_;
                view.rowEnd_ = rowEnd_;
                view.colEnd_ = colEnd_;
                compatibleQuery(*view.matrix().rows(), view.rowStart_, view.rowEnd_);
                compatibleQuery(*view.matrix().cols(), view.colStart_, view.colEnd_);
                view.init(me());
                view.getValues();
            }
        }
    }

  public:
    /**
     * @brief uncompress Extract values from the matrix
     * The sub-matrix with rows and cols is extracted and copied to values.
     * @param matrix
     * @param rows The row ids to uncompress
     * @param rowSize The number of lines to uncompress
     * @param cols The column ids to uncompress
     * @param colSize The number of column to uncompress
     * @param values The target buffer where to store uncompressed values
     * @param ld The leading dimension of the target buffer
     * @param hmat_numbering true if rows and cols contains id using the hmat internal
     * numbering (from 0 to n-1), false if for natural numbering (from 1 to n)
     */
    void uncompress(const M<T> * matrix, int * rows, int rowSize, int * cols, int colSize, T *values,
                    int ld = -1, bool hmat_numbering = false)
    {
        matrix_ = matrix;
        values_ = values;
        valuesLd_ = ld == -1 ? rowSize : ld;
        std::vector<std::pair<int, int> > rowsIndices, colsIndices;
        createQuery(*me().matrix().rows(), rows, rowSize, hmat_numbering, rowsIndices);
        rowStart_ = rowsIndices.begin();
        rowEnd_ = rowsIndices.end();
        createQuery(*me().matrix().cols(), cols, colSize, hmat_numbering, colsIndices);
        colStart_ = colsIndices.begin();
        colEnd_ = colsIndices.end();
        me().init(me());
        getValues();
        me().finish();
    }
};

/** UncompressedValuesBase specialisation for HMatrix<T> */
template <typename T> class UncompressedValues: public UncompressedValuesBase<T, HMatrix, UncompressedValues<T> > {
public:
    typedef typename UncompressedValuesBase<T, HMatrix, UncompressedValues<T> >::IndiceIt IndiceIt;
private:
    friend class UncompressedValuesBase<T, HMatrix, UncompressedValues<T> >;
    void getValue(IndiceIt r, IndiceIt c, T v) {
        this->values_[r->second + ((size_t)this->valuesLd_) * c->second] = v;
    }

    void getNullValues() {
        for(IndiceIt r = this->rowStart_; r != this->rowEnd_; ++r) {
            for(IndiceIt c = this->colStart_; c != this->colEnd_; ++c) {
                getValue(r, c, Constants<T>::zero);
            }
        }
    }

    void getFullValues() {
        const HMatrix<T> & m = *this->matrix_;
        // Check for not supported cases
        assert(m.full()->pivots == NULL);
        assert(m.full()->diagonal == NULL);
        int ro = m.rows()->offset();
        int co = m.cols()->offset();
        for(IndiceIt r = this->rowStart_; r != this->rowEnd_; ++r) {
            for(IndiceIt c = this->colStart_; c != this->colEnd_; ++c) {
                getValue(r, c, m.full()->get(r->first - ro, c->first - co));
            }
        }
    }

    void getRkValues();

public:
    void getLeafValues() {
        if (this->matrix_->isNull()) {
            getNullValues();
        } else if (this->matrix_->isRkMatrix()) {
            getRkValues();
        } else if (this->matrix_->isFullMatrix()) {
            getFullValues();
        } else {
            assert(false);
        }
    }

    const HMatrix<T> & matrix() const {
        return *this->matrix_;
    }

    bool isLeaf() const {
        return matrix().isLeaf();
    }

    /** Init from parent */
    void init(UncompressedValues &) {}
    void finish(){}
};
}
