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

#ifndef _MY_ASSERT_H
#define _MY_ASSERT_H

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>


#if __GNUC__
#  define HMAT_FUNCTION __PRETTY_FUNCTION__
#elif defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#  define HMAT_FUNCTION __func__
#elif defined _MSC_VER
#  define HMAT_FUNCTION __FUNCTION__
#else
#  define HMAT_FUNCTION ((const char *) 0)
#endif

#if !defined (_WIN32) /* It's a UNIX system, I know this ! */
#include <execinfo.h>
#include <unistd.h>
inline static void hmat_backtrace(){
    void * stack[32];
    int n = backtrace(stack, 32);
    backtrace_symbols_fd(stack, n, STDERR_FILENO);
}
#else
  inline static void hmat_backtrace(){}
#endif

#ifdef __GNUC__
  #define HMAT_NORETURN __attribute__((noreturn))
#elif defined _MSC_VER
  #define HMAT_NORETURN __declspec(noreturn)
#else
  #define HMAT_NORETURN
#endif

HMAT_NORETURN inline static void hmat_assert(const char * format, ...) {
    va_list arglist;
    va_start(arglist, format);
    vfprintf(stderr, format, arglist);
    va_end(arglist);
    hmat_backtrace();
    fprintf(stderr, "\n");
    abort();
}

#if defined(__cplusplus) && __cplusplus >= 201103L

#include <stdexcept>
#include <string>

inline std::string hmat_build_message(const char * format, ...) {
    va_list arglist, argcopy;
    va_start(arglist, format);
    va_copy(argcopy, arglist);
    int n = std::vsnprintf(NULL, 0, format, arglist);
    va_end(arglist);
    if (n < 0)
        return std::string("Internal error: cannot build error message, bad format '%s' or wrong arguments", format);
    std::string message;
    message.resize(n + 1);
    std::vsnprintf(&message[0], n, format, argcopy);
    va_end(argcopy);
    return message;
}

#define HMAT_ASSERT(x) do { if (!(x)) { \
    hmat_backtrace(); \
    throw std::runtime_error(hmat_build_message("\n\n[hmat] assert failure %s at %s:%d %s\n", #x, __FILE__, __LINE__, HMAT_FUNCTION)); \
    }} while(0)

#define HMAT_ASSERT_MSG(x, format, ...) do { if (!(x)) { \
    hmat_backtrace(); \
    throw std::runtime_error(hmat_build_message("\n\n[hmat] assert failure %s at %s:%d %s, " format "\n", \
                #x, __FILE__, __LINE__, HMAT_FUNCTION, ## __VA_ARGS__)); \
    }} while(0)

#else

#define HMAT_ASSERT(x) do { if (!(x)) \
    hmat_assert("\n\n[hmat] assert failure %s at %s:%d %s\n", \
    #x, __FILE__, __LINE__, HMAT_FUNCTION); \
    } while(0)

#define HMAT_ASSERT_MSG(x, format, ...) do { if (!(x)) \
    hmat_assert("\n\n[hmat] assert failure %s at %s:%d %s, " format "\n", \
                #x, __FILE__, __LINE__, HMAT_FUNCTION, ## __VA_ARGS__); \
    } while(0)

#endif /* !__cplusplus */

#endif
