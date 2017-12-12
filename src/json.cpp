/*
  HMat-OSS (HMatrix library, open source software)

  Copyright (C) 2014-2017 Airbus SAS

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

#include "json.hpp"
#include "cluster_tree.hpp"
#include "rk_matrix.hpp"

using namespace std;

namespace hmat {

void JSONDumper::dumpSubTree(int _depth) {
    string prefix("    ");
    for (int i = 0; i < _depth; i++) {
        prefix += "  ";
    }
    AxisAlignedBoundingBox rows_bbox(*rows_);
    AxisAlignedBoundingBox cols_bbox(*cols_);
    const int rows_dimension(rows_->coordinates()->dimension());
    const int cols_dimension(cols_->coordinates()->dimension());
    // TODO: remove isLeaf because it can be deduced form children
    out_ << prefix << "{\"isLeaf\": " << (nrChild_ == 0 ? "true" : "false") << "," << endl
      << prefix << " \"depth\": " << _depth << "," << endl
      << prefix << " \"rows\": "
      << "{\"offset\": " << rows_->offset() << ", \"n\": " << rows_->size() << ", "
      << "\"boundingBox\": [[" << rows_bbox.bbMin()[0];
    for (int dim = 1; dim < rows_dimension; ++dim) {
        out_ << ", " << rows_bbox.bbMin()[dim];
    }
    out_ << "], [" << rows_bbox.bbMax()[0];
    for (int dim = 1; dim < rows_dimension; ++dim) {
        out_ << ", " << rows_bbox.bbMax()[dim];
    }
    out_ << "]]}," << endl
      << prefix << " \"cols\": "
      << "{\"offset\": " << cols_->offset() << ", \"n\": " << cols_->size() << ", "
      << "\"boundingBox\": [[" << cols_bbox.bbMin()[0];
    for (int dim = 1; dim < cols_dimension; ++dim) {
        out_ << ", " << cols_bbox.bbMin()[dim];
    }
    out_ << "], [" << cols_bbox.bbMax()[0];
    for (int dim = 1; dim < cols_dimension; ++dim) {
        out_ << ", " << cols_bbox.bbMax()[dim];
    }
    out_ << "]]}";
    const std::string extra_info = nodeInfo_.str();
    if (!extra_info.empty()) {
        out_ << "," << endl << prefix << extra_info;
    }
    if (nrChild_ > 0) {
        out_ << "," << endl << prefix << " \"children\": [" << endl;
        loopOnChildren(_depth);
        out_ << endl << prefix << " ]";
    }
    out_ << "}";
}

void JSONDumper::nextChild(bool last) {
    if(!last)
        out_ << endl << ",";
    nodeInfo_.str("");
}

void JSONDumper::dumpPoints() {
    string delimiter;
    const DofCoordinates* points = rows_->coordinates();
    const int* indices = rows_->indices();
    const int dimension = points->dimension();
    out_ << "  \"points\": [" << endl;
    delimiter = "";
    for (int i = 0; i < points->numberOfDof(); i++) {
        out_ << "    " << delimiter << "[";
        if (dimension > 0) {
            out_ << points->spanCenter(i, 0);
            for (int dim = 1; dim < dimension; ++dim) {
                out_ << ", " << points->spanCenter(i, dim);
            }
        }
        out_ << "]" << endl;
        delimiter = " ,";
    }
    // Mapping
    out_ << "  ]," << endl
         << "  \"mapping\": [" << endl
         << "    ";
    delimiter = "";
    for (int i = 0; i < points->numberOfDof(); i++) {
        out_ << delimiter << indices[i];
        delimiter = " ,";
    }
    out_ << "]," << endl;
}

void JSONDumper::dump() {
    out_ << "{" << endl;
    dumpMeta();
    out_ << "  \"tree\":" << endl;
    dumpSubTree(0);
    out_ << "}" << endl;
}

template<typename T> void HMatrixJSONDumper<T>::dumpMeta() {
    dumpPoints();
}

template<typename T> void HMatrixJSONDumper<T>::update() {
    rows_ = current_->rows();
    cols_ = current_->cols();
    nrChild_ = current_->nrChild();
    if (current_->isFullMatrix()) {
        int zeros = current_->full()->storedZeros();
        double ratio = zeros / ((double) current_->full()->rows() * current_->full()->cols());
        nodeInfo_ << " \"leaf_type\": \"Full\", \"k\": " << 1-ratio;
    } else if (current_->isRkMatrix() && current_->rk()) {
        nodeInfo_ << " \"leaf_type\": \"Rk\", \"k\": " << current_->rank() << ",";
        nodeInfo_ << " \"method\": " << current_->rk()->method;
    }
}

template<typename T>
void HMatrixJSONDumper<T>::loopOnChildren(int depth) {
    HMatrix<T> * toLoopOn = current_;
    int last = toLoopOn->nrChild() - 1;
    for (int i = 0; i <= last; i++) {
        current_ = toLoopOn->getChild(i);
        if(current_ != NULL) {
            update();
            dumpSubTree(depth + 1);
            nextChild(i == last);
        }
    }
}

template<typename T>
HMatrixJSONDumper<T>::HMatrixJSONDumper(HMatrix<T> * m, std::ostream & out)
    : JSONDumper(out), current_(m) {
    update();
}

template class HMatrixJSONDumper<S_t>;
template class HMatrixJSONDumper<D_t>;
template class HMatrixJSONDumper<C_t>;
template class HMatrixJSONDumper<Z_t>;

}
