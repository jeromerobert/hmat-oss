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

#ifndef _CHRONO_H
#define _CHRONO_H

#include "config.h"
#include <stdint.h>
#include <math.h>

/* We use the realtime extension of libc */
#ifdef HAVE_TIME_H
#include <time.h>
#endif
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#ifdef HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#ifdef HAVE_MACH_MACH_TIME_H
#include <mach/mach_time.h>
#endif

#if defined(_WIN32) && !defined(__MINGW32__)
struct my_timespec {
	int64_t tv_sec;
	int64_t tv_nsec;
};
typedef struct my_timespec Time;
#else
typedef struct timespec Time;
#endif // Windows

inline static Time now() {
  Time result;
#ifdef _WIN32
  LARGE_INTEGER frequency;
  LARGE_INTEGER value;
  double dTime;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&value);
  dTime = ((double) value.QuadPart) / ((double) frequency.QuadPart);
  result.tv_sec = (int64_t) dTime;
  result.tv_nsec = (int64_t) (1.e9 * (dTime - result.tv_sec));
  if (result.tv_nsec >= 1000000000) {
      result.tv_sec += 1;
      result.tv_nsec = 0;
  }
#elif HAVE_LIBRT
  clock_gettime(CLOCK_MONOTONIC, &result);
#elif defined(HAVE_MACH_MACH_TIME_H) /* Version MacOS */
  static double timeConvert = 0.0;
  if ( timeConvert == 0.0 )
  {
    mach_timebase_info_data_t timeBase;
    (void)mach_timebase_info( &timeBase );
    timeConvert = (double)timeBase.numer /
        (double)timeBase.denom /
        1000000000.0;
  }
  double t = (double)mach_absolute_time( ) * timeConvert;
  result.tv_sec  = (long)floor(t) ; /* seconds */
  result.tv_nsec = (long)floor( (t-(double)result.tv_sec)*1.e9 ) ;  /* uSecs */
#else
  struct rusage temp;
  getrusage(RUSAGE_SELF, &temp);
  result.tv_sec  = (time_t) (temp.ru_utime.tv_sec); /* seconds */
  result.tv_nsec = (long) (1000 * temp.ru_utime.tv_usec);  /* uSecs */
#endif
  return result;
}

inline static int64_t time_diff_in_nanos(Time tick, Time tock) {
  return (tock.tv_sec - tick.tv_sec) * 1000000000 + (tock.tv_nsec - tick.tv_nsec);
}
inline static double time_diff(Time start, Time end) {
  return end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) * 1e-9 ;
}
#endif
