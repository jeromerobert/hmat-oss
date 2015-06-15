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

#ifndef _SYSTEM_TYPES_H
#define _SYSTEM_TYPES_H

#include "config.h"

#ifdef HAVE_STDINT_H
# include <stdint.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_TIME_H
# include <time.h>
#endif
  /* See what has to be done for WIN32... */
#if defined(_WIN32) && !defined(__MINGW32__)
# include <io.h>
# include <process.h>
# ifndef HAVE_MODE_T
#  define HAVE_MODE_T
   typedef int mode_t;
# endif

/* Windows defines the equivalent SSIZE_T in the platform SDK */
/* as the signed equivalent of size_t which is defined as long */
/* on WIN32 and long long/__int64 on WIN64 */
# ifdef _WIN64
   typedef __int64 off64_t;
   typedef __int64 ssize_t;
# else  /* _WIN64 */
#  ifdef __ICL
   /* Intel compiler and LARGE FILES on WIN32 */
    typedef __int64 off64_t;
#  else
    typedef off_t off64_t;
#  endif
   typedef long ssize_t;
# endif /* _WIN64 */

/* Intel compiler */
# ifdef __ICL
#  define lseek _lseeki64
# endif

#elif defined(HAVE_UNISTD_H)
   #include <unistd.h>
#endif


#endif /* _SYSTEM_TYPES_H */
