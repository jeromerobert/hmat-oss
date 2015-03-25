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
public:
    LapackException(const char * primitive, int info)
        : runtime_error("Lapack error"), primitive_(primitive), info_(info)
    {}

    const char * primitive() const {
        return primitive_;
    }
    int info() const {
        return info_;
    }
    virtual const char* what() const throw()
    {
        std::stringstream sstm;
        sstm << runtime_error::what() << " in "<< primitive_ << ", info=" << info_;
        return sstm.str().c_str();
    }

private:
    const char * primitive_;
    int info_;
};

}  // end namespace hmat

#endif /* LAPACK_EXCEPTION_HPP */
