#pragma once

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
#include <sstream>

namespace hmat {

class ClusterData;
template<typename T> class HMatrix;

class JSONDumper {
protected:
    const ClusterData * rows_, * cols_;
    std::ostringstream nodeInfo_;
    int nrChild_;
    std::ostream & out_;
    /** Dump the current object */
    void dumpSubTree(int _depth);
    /** Switch to next children in loopOnChildren */
    void nextChild(bool last = false);
    /**
     * Loop over current object children.
     * Must update and rows_, cols_, nodeInfo_, nrChild_ for
     * each child.
     */
    virtual void loopOnChildren(int depth) = 0;
    /**
     * Dump data which are global to the matrix and not
     * related to a single block
     */
    virtual void dumpMeta() = 0;
    /** To call from dumpMeta to dump the points of the matrix */
    void dumpPoints();
public:
    JSONDumper(std::ostream & out) : out_(out) {}
    void dump();
    virtual ~JSONDumper(){}
};

template<typename T>
class HMatrixJSONDumper: public JSONDumper {
    HMatrix<T> * current_;
    void update();
protected:
    virtual void loopOnChildren(int depth);
    virtual void dumpMeta();
public:
    HMatrixJSONDumper(HMatrix<T> * m, std::ostream & out);
};

}
