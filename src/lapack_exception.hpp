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

class LapackException: public std::exception {
    const char * primitive_;
    int info_;
    std::string msg;
    LapackException() : primitive_(NULL), info_(0) {}
public:
    /** Singleton for no error */
    static LapackException & noError() {
        static LapackException instance;
        return instance;
    }
    LapackException(const char * primitive, int info)
        : primitive_(primitive), info_(info)
    {
        std::stringstream sstm;
        sstm << "Lapack error in "<< primitive_ << ", info=" << info_;
        msg = sstm.str();
    }

    const char * primitive() const {
        return primitive_;
    }
    int info() const {
        return info_;
    }

    bool isError() {
        return info_ != 0;
    }

    virtual const char* what() const throw() {
        return msg.c_str();
    }

    virtual ~LapackException() throw() {}
};

}  // end namespace hmat

#endif /* LAPACK_EXCEPTION_HPP */
