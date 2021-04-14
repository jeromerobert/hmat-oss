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

static void
dump_points(std::ostream& out, const std::string name, const DofCoordinates* points) {
    string delimiter;
    const int dimension = points->dimension();
    out << "  \"" << name << "\": [" << endl;
    delimiter = "";
    for (int i = 0; i < points->numberOfDof(); i++) {
        out << "    " << delimiter << "[";
        if (dimension > 0) {
            out << points->spanCenter(i, 0);
            for (int dim = 1; dim < dimension; ++dim) {
                out << ", " << points->spanCenter(i, dim);
            }
        }
        out << "]" << endl;
        delimiter = " ,";
    }
    out << "  ]," << endl;
}

static void
dump_mapping(std::ostream& out, const std::string name, int numberOfDof, const int* indices) {
    // Mapping
    string delimiter;
    out << "  \"" << name << "\": [" << endl
        << "    ";
    delimiter = "";
    for (int i = 0; i < numberOfDof; i++) {
        out << delimiter << indices[i];
        delimiter = " ,";
    }
    out << "]," << endl;
}

void JSONDumper::dumpPoints() {
    dump_points(out_, "points", rows_->coordinates());
    dump_mapping(out_, "mapping", rows_->coordinates()->numberOfDof(), rows_->indices());
    if (rows_ != cols_) {
        dump_points(out_, "points_cols", cols_->coordinates());
        dump_mapping(out_, "mapping_cols", cols_->coordinates()->numberOfDof(), cols_->indices());
    }
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
        nodeInfo_ << " \"leaf_type\": \"Full\"";
    } else if (current_->isRkMatrix()) {
        nodeInfo_ << " \"leaf_type\": \"Rk\", \"k\": " << current_->rank() << ",";
        nodeInfo_ << " \"epsilon\": " << current_->lowRankEpsilon();
    }
}

template<typename T>
void HMatrixJSONDumper<T>::loopOnChildren(int depth) {
    const HMatrix<T> * toLoopOn = current_;
    int last = toLoopOn->nrChild() - 1;
    while(last >= 0 && NULL == toLoopOn->getChild(last))
      --last;
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
HMatrixJSONDumper<T>::HMatrixJSONDumper(const HMatrix<T> * m, std::ostream & out)
    : JSONDumper(out), current_(m) {
    update();
}

template class HMatrixJSONDumper<S_t>;
template class HMatrixJSONDumper<D_t>;
template class HMatrixJSONDumper<C_t>;
template class HMatrixJSONDumper<Z_t>;

}
