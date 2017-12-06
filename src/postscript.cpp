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
    file << "%!PS-Adobe-" << endl;
    file << "%%BoundingBox: " << 0 << " " << 0 << " " << 615 << " " << 615 << endl;
    file << "%!" << endl << "% Fichier postscript representant une matrice"
         << endl;
    file << "/redrectangle {" << endl
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
         << "/grayrectangle {" << endl
         << "        newpath" << endl
         << "	     setlinewidth"  << endl
         << "        1 setgray" << endl
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
         << "	0 0 1 setrgbcolor"  << endl
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
         << " moveto" << endl
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
  void PostscriptDumper<T>::drawRectangle(const HMatrix<T> * m, ofstream& f, const std::string& name, int linewidth) const {
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
    f << " " << linewidth << " " << name << endl;
}

    template<typename T>
void PostscriptDumper<T>::write(const void * tree, const std::string& filename) const {
    ofstream file;
    file.open(filename.c_str());
    const HMatrix<T> * m = castToHMatrix(tree);
    writeHeader(file, max(m->rows()->size(), m->cols()->size()));
    recursiveDrawing(tree, file, 0);
    file << "showpage" << endl;
}

template<typename T>
void PostscriptDumper<T>::recursiveDrawing(const void * tree, ofstream& f, int depth) const {
  const HMatrix<T> * m = castToHMatrix(tree);

  if (depth == 0)
    drawRectangle(m, f, "grayrectangle", 0);

  if (!m->isLeaf()) {
        for (int i = 0; i < m->nrChild(); i++) {
            const HMatrix<T> * child = m->getChild(i);
            if (child) {
                recursiveDrawing(child, f, depth + 1);
            }
        }
    }

    if (depth == 0)
      drawRectangle(m, f, "emptyrectangle", 10-depth);
    drawMatrix(tree, f, depth);
}

template<typename T>
const HMatrix<T> * PostscriptDumper<T>::castToHMatrix(const void * tree) const {
    return static_cast<const HMatrix<T> *>(tree);
}

template<typename T>
void PostscriptDumper<T>::drawMatrix(const void *tree, ofstream& f, int depth, bool cross) const {
    const HMatrix<T> * m = castToHMatrix(tree);

    int n = m->rows()->coordinates()->size();
    int startX = m->cols()->offset();
    int lengthX = m->cols()->size();
    int startY = n - m->rows()->offset();
    int lengthY = -m->rows()->size();

    if (m->isLeaf()) {
        if (m->isRkMatrix() && !m->isNull() && m->rk() != NULL) {
            double ratio = m->rk()->compressedSize() / (double) m->rk()->uncompressedSize();
            double color = 0;
            double ratioMax = .8;
            double colorMin = .5;
            if (ratio >= 1) color = 0;
            else if (ratio < ratioMax) {
                color = 1 - (1-colorMin)/ratioMax * ratio;
            } else color = colorMin;
            f << 0 << " "<< -lengthY << " "
              << -lengthX << " " << 0 << " "
              << 0 << " " << lengthY << " "
              << lengthX << " " << 0 << " "
              << startX << " " << startY;
            f << " 0 " << color << " 0 "
              << " greenrectangle" << endl;
            // value to write: rank
            int value = m->rank();
            // size of the characters
            double size= (value>=100?.5:0.7) * min(-lengthY,lengthX);
            f << startX + 10 - depth << " " << startY + (lengthY * .75) << " " << size
              << " (" << value << ") showrank" << endl;
        } else if (m->isFullMatrix()) {
            int zeros = m->full()->storedZeros();
            double ratio = zeros / ((double) m->full()->rows() * m->full()->cols());
            double color = min(1-(1-ratio)*5,0.35);
            f << 0 << " "<< -lengthY << " "
              << -lengthX << " " << 0 << " "
              << 0 << " " << lengthY << " "
              << lengthX << " " << 0 << " "
              << startX << " " << startY;
            int value = (int) ceil(100 * (1-ratio));
            if (ratio < 0.8)
              f << " " << max((0.2+ratio),0.6) << " 0 0";
            else if (value == 0)
            f << " " << "1 1 0";
            else
              f << " " << "1 " << color << " " << color;
            f << " redrectangle" << endl;
            // value to write: percentage of non-zeros
            // size of the characters
            double size= (value>=100?.5:0.7) * min(-lengthY,lengthX);
            f << startX + 10 - depth << " " << startY + (lengthY * .75) << " " << size
            << " (" << value << ") showrank" << endl;
        } else {
           f << 0 << " "<< -lengthY << " "
             << -lengthX << " " << 0 << " "
             << 0 << " " << lengthY << " "
             << lengthX << " " << 0 << " "
             << startX << " " << startY;
             f << " .9 .9 .9"
             << " greenrectangle" << endl;
          //  f << startX << " " << startY + (lengthY * .95) << " " << .7 * std::min(lengthX, - lengthY)
          //    << " (" << 0 << ") showrank" << endl;
       }
    } else if(cross){ /* true pour une hmat, !(handle->position == kAboveL0) pour une HMatrixHandle */
        n = m->rows()->coordinates()->size();
        startX = m->cols()->offset();
        startY = m->rows()->offset();
        int rowsCount = m->rows()->size();
        int colsCount = m->cols()->size();
        /* On dessine la croix qui separe les sous-blocs dans la h-matrice.
           Dans le cas 2x2, on fait 1 croix. 1x1, 0 croix. 3x3, 2 croix.
           Dans les cas non carre, 2x3, on fait 2 croix, meme si un trait sera en double. */
        for (int k=1 ; k < std::max(m->nrChildRow(), m->nrChildCol()) ; k++) {
          int i = k>=m->nrChildRow() ? m->nrChildRow()-1 : k ;
          int j = k>=m->nrChildCol() ? m->nrChildCol()-1 : k ;
          int colOffset = m->cols()->offset();
          int rowOffset = m->rows()->offset();
          // to get colOffset, I try to find a non-null child in the same column
          for (int i2=0 ; i2<m->nrChildRow() ; i2++)
            if (m->get(i2, j))
              colOffset = m->get(i2, j)->cols()->offset();
          // to get rowOffset, I try to find a non-null child in the same row
          for (int j2=0 ; j2<m->nrChildCol() ; j2++)
            if (m->get(i, j2))
              rowOffset = m->get(i, j2)->rows()->offset();
          int thickness = max(1, min(10 - depth, (rowsCount + colsCount) / 100));
        f << 0 << " " << -rowsCount << " "
          << colOffset << " " << n - startY << " "
          << colsCount << " " << 0 << " "
          << startX << " " << n - rowOffset << " "
          << thickness << " cross" << endl;
    }
        /* La macro 'cross' est definie dans writeHeader() ci-dessus.
           Elle contient une serie de commande postscript (moveto, rlineto, etc.) qui vont depiler
           les valeurs ecrites ci-dessus. Les coordonnees Y sont toujours renversees ('n-..') pour avoir
           un postcript a l'endroit.
       /cross {
         newpath
         setlinewidth             (10-depth) trait plus epais en haut de l'arbre
         0 0 0 setrgbcolor        (noir)
         moveto                   (startX, n-rowoffset)
         rlineto                  (colsCount,0) trait horizontal
         moveto                   (colOffset, n-startY)
         rlineto                  (0, -rowsCount) trait vertical
         stroke
        } def
        */
    }
}

template class PostscriptDumper<S_t>;
template class PostscriptDumper<D_t>;
template class PostscriptDumper<C_t>;
template class PostscriptDumper<Z_t>;

}  // end namespace hmat
