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
 * Extract an uncompressed set of values from the matrix
 */
template <typename T> class UncompressedValues {
    const HMatrix<T> &matrix_;
    T *values_;
    int valuesLd_;
    typedef std::vector<std::pair<int, int> >::iterator IndiceIt;
    /**
     * The first element of the pair a number of element to get, using hmat numbering.
     * The second element of the pair is the position of the element in the original query.
     */
    IndiceIt rowStart_, rowEnd_, colStart_, colEnd_;

    void getValue(IndiceIt r, IndiceIt c, T v) {
        values_[r->second + ((size_t)valuesLd_) * c->second] = v;
    }

    void getNullValues() {
        for(IndiceIt r = rowStart_; r != rowEnd_; ++r) {
            for(IndiceIt c = colStart_; c != colEnd_; ++c) {
                getValue(r, c, Constants<T>::zero);
            }
        }
    }

    void getFullValues() {
        // Not supported yet
        assert(matrix_.full()->pivots == NULL);
        assert(matrix_.full()->diagonal == NULL);
        int ro = matrix_.rows()->offset();
        int co = matrix_.cols()->offset();
        for(IndiceIt r = rowStart_; r != rowEnd_; ++r) {
            for(IndiceIt c = colStart_; c != colEnd_; ++c) {
                getValue(r, c, matrix_.full()->get(r->first - ro, c->first - co));
            }
        }
    }

    void getRkValues();

    void getValues() {
        if (rowStart_ == rowEnd_ || colStart_ == colEnd_)
            return;
        if (!matrix_.isLeaf()) {
            for (int i = 0; i < matrix_.nbChild(); i++) {
                UncompressedValues view(*this, *matrix_.getChild(i));
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

    void compatibleQuery(const IndexSet & clusterData, IndiceIt & begin, IndiceIt & end) {
        int lb = clusterData.offset();
        int ub = lb + clusterData.size() - 1;
        std::pair<int, int> lbP(lb, 0);
        std::pair<int, int> ubP(ub, 0);
        IndiceIt newBegin = std::lower_bound(begin, end, lbP);
        IndiceIt newEnd = std::upper_bound(begin, end, ubP);
        begin = newBegin;
        end = newEnd;
        assert(begin->first >= lb);
        assert(end->first > ub);
    }

    void createQuery(const ClusterData & clusterData, int * query, int querySize, std::vector<std::pair<int, int> > & indices) {
        indices.resize(querySize);
        for(int i = 0; i < querySize; i++) {
            indices[i].first = clusterData.indices_rev()[query[i] - 1];
            indices[i].second = i;
        }
        std::sort(indices.begin(), indices.end());
    }

    UncompressedValues(const UncompressedValues & o, const HMatrix<T> &matrix)
        : matrix_(matrix), values_(o.values_), valuesLd_(o.valuesLd_) {
        compatibleQuery(*matrix_.rows(), rowStart_, rowEnd_);
        compatibleQuery(*matrix_.cols(), colStart_, colEnd_);
        getValues();
    }

  public:

    UncompressedValues(const HMatrix<T> &matrix, int * rows, int rowSize, int * cols, int colSize, T *values, int ld = -1)
        : matrix_(matrix), values_(values), valuesLd_(ld)
    {
        assert(matrix.father == NULL);
        if(valuesLd_ == -1)
            valuesLd_ = rowSize;
        std::vector<std::pair<int, int> > rowsIndices, colsIndices;
        createQuery(*matrix.rows(), rows, rowSize, rowsIndices);
        rowStart_ = rowsIndices.begin();
        rowEnd_ = rowsIndices.end();
        createQuery(*matrix.cols(), cols, colSize, colsIndices);
        colStart_ = colsIndices.begin();
        colEnd_ = colsIndices.end();
        getValues();
    }
};
}
