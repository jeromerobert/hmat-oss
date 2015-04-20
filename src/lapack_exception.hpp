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

#pragma once
#ifndef LAPACK_EXCEPTION_HPP
#define LAPACK_EXCEPTION_HPP

#include <stdexcept>
#include <sstream>

namespace hmat {

class LapackException: public std::runtime_error {
    std::string format_msg(const char * primitive, int info) {
        std::stringstream sstm;
        sstm << "Lapack error in "<< primitive << ", info=" << info;
        return sstm.str().c_str();
    }
    const char * primitive_;
    int info_;

public:
    LapackException()
        : runtime_error("Not an exception"), primitive_(NULL), info_(0)
    {}

    LapackException(const char * primitive, int info)
        : runtime_error(format_msg(primitive, info)),
          primitive_(primitive), info_(info)
    {}

    const char * primitive() const {
        return primitive_;
    }
    int info() const {
        return info_;
    }

    bool isError() {
        return info_ != 0;
    }
};

}  // end namespace hmat

#endif /* LAPACK_EXCEPTION_HPP */
