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

#include "postscript.hpp"
#include "h_matrix.hpp"
#include "rk_matrix.hpp"
#include <fstream>
#include <utility>

using namespace std;
using namespace hmat;

static double writeHeader(ofstream & file, int maxDim)
{
    file << "%!" << endl << "% Fichier postscript representant une matrice"
         << endl;
    file << "/redrectangle {" << endl
         << "        newpath" << endl
         << "        moveto" << endl
         << "        rlineto" << endl
         << "        rlineto" << endl
         << "        rlineto" << endl
         << "        rlineto" << endl
         << "        closepath" << endl
         << "        1 0 0 setrgbcolor" << endl
         << "        fill"  << endl
         << "} def"  << endl
         << "/greenrectangle {" << endl
         << "        newpath" << endl
         << "        setrgbcolor" << endl
         << "        moveto" << endl
         << "        rlineto" << endl
         << "        rlineto" << endl
         << "        rlineto" << endl
         << "        rlineto" << endl
         << "        closepath" << endl
         << "        fill"  << endl
         << "} def"  << endl
         << ""  << endl
         << "/emptyrectangle {"  << endl
         << "	newpath"  << endl
         << "	setlinewidth"  << endl
         << "	0 0 0 setrgbcolor"  << endl
         << "	moveto"  << endl
         << "	rlineto"  << endl
         << "	rlineto"  << endl
         << "	rlineto"  << endl
         << "	rlineto"  << endl
         << "	closepath"  << endl
         << "	stroke"  << endl
         << "} def" << endl << endl
         << "/emptybluerectangle {"  << endl
         << "	newpath"  << endl
         << "	setlinewidth"  << endl
         << "	0 0 255 setrgbcolor"  << endl
         << "	moveto"  << endl
         << "	rlineto"  << endl
         << "	rlineto"  << endl
         << "	rlineto"  << endl
         << "	rlineto"  << endl
         << "	closepath"  << endl
         << "	stroke"  << endl
         << "} def"  << endl
         << "/cross {" << endl
         << " newpath" << endl
         << "setlinewidth" << endl
         << " 0 0 0 setrgbcolor" << endl
         << " moveto" << endl
         << " rlineto" << endl
         << " moveto" << endl
         << " rlineto" << endl
         << " stroke" << endl
         << "} def" << endl
         << "/showrank {" << endl
         << " /data exch def" << endl
         << " /xfont exch def" << endl
         << " /Times-Roman findfont" << endl
         << " xfont scalefont" << endl
         << " setfont" << endl
         << " 0 0 0 setrgbcolor" << endl
         << " /y1 exch def" << endl
         << " /x1 exch def" << endl
         << " x1 y1 moveto" << endl
         << " data show" << endl
         << "} def" << endl
         << " " << endl;
    double scale = (612. / maxDim) * 0.95;
    file << scale << " " << scale << " scale" << endl;
    file << maxDim / 40. << " " << maxDim / 40. << " translate" << endl;
    return scale;
}

namespace hmat {

template<typename T>
void PostscriptDumper<T>::write(const Tree<4> * tree, const std::string& filename) const {
    ofstream file;
    file.open(filename.c_str());
    const HMatrix<T> * m = cast(tree);
    double scale = writeHeader(file, max(m->rows()->size(), m->cols()->size()));
    recursiveDrawing(tree, file, 0, scale);
    file << "showpage" << endl;
}

template<typename T>
void PostscriptDumper<T>::recursiveDrawing(const Tree<4> * tree, ofstream& f, int depth, double scale) const {
    if (!tree->isLeaf()) {
        for (int i = 0; i < 4; i++) {
            const Tree<4>* child = tree->getChild(i);
            if (child) {
                recursiveDrawing(child, f, depth + 1, scale);
            }
        }
    }

    const HMatrix<T> * m = cast(tree);
    if (depth == 0) {
        int n = m->rows()->coordinates()->size();
        int startX = m->cols()->offset();
        int lengthX = m->cols()->size();
        int startY = n - m->rows()->offset();
        int lengthY = -m->rows()->size();
        f << 0 << " "<< -lengthY << " "
          << -lengthX << " " << 0 << " "
          << 0 << " " << lengthY << " "
          << lengthX << " " << 0 << " "
          << startX << " " << startY;
        f << " " << 30 - depth << " emptyrectangle" << endl;
    }
    drawMatrix(tree, m, f, depth, scale);
}

template<typename T>
const HMatrix<T> * PostscriptDumper<T>::cast(const Tree<4> * tree) const {
    return static_cast<const HMatrix<T> *>(tree);
}

template<typename T>
void PostscriptDumper<T>::drawMatrix(const Tree<4> *, const HMatrix<T> * m,
    ofstream& f, int depth, double, bool cross) const {
    int n = m->rows()->coordinates()->size();
    int startX = m->cols()->offset();
    int lengthX = m->cols()->size();
    int startY = n - m->rows()->offset();
    int lengthY = -m->rows()->size();

    if (m->isLeaf()) {
        if (m->isRkMatrix() && !m->isNull()) {
            double ratio = m->rk()->compressedSize() / m->rk()->uncompressedSize();
            double color = 0;
            if (ratio < .20) {
                color = 1 - 5 * ratio;
            }
            f << 0 << " "<< -lengthY << " "
              << -lengthX << " " << 0 << " "
              << 0 << " " << lengthY << " "
              << lengthX << " " << 0 << " "
              << startX << " " << startY;
            f << " 0 " << color << " 0 "
              << " greenrectangle" << endl;
            f << startX << " " << startY + (lengthY * .95) << " " << .7 * std::min(lengthX, - lengthY)
              << " (" << m->rank() << ") showrank" << endl;
        } else if (m->isFullMatrix()) {
            f << 0 << " "<< -lengthY << " "
              << -lengthX << " " << 0 << " "
              << 0 << " " << lengthY << " "
              << lengthX << " " << 0 << " "
              << startX << " " << startY;
            f << " redrectangle" << endl;
        }
    } else if(cross){
        int n = m->rows()->coordinates()->size();
        int startX = m->cols()->offset();
        int startY = m->rows()->offset();
        int colOffset = m->get(1, 1)->cols()->offset();
        int rowsCount = m->rows()->size();
        int rowOffset = m->get(1, 1)->rows()->offset();
        int colsCount = m->cols()->size();
        f << 0 << " " << -rowsCount << " "
          << colOffset << " " << n - startY << " "
          << colsCount << " " << 0 << " "
          << startX << " " << n - rowOffset << " "
          << 30 - depth << " cross" << endl;
    }
}

template class PostscriptDumper<S_t>;
template class PostscriptDumper<D_t>;
template class PostscriptDumper<C_t>;
template class PostscriptDumper<Z_t>;

}  // end namespace hmat
