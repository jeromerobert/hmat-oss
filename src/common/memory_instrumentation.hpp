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

/*! \file
  \ingroup HMatrix
  \brief Memory Allocation tracking.
*/
#ifndef _MEMORY_INSTRUMENTATION_H
#define _MEMORY_INSTRUMENTATION_H
// We use things from C++11, so test for the standard level.
#if ((__cplusplus > 199711L) || defined(HAVE_CPP11)) && HAVE_MEM_INSTR
#include <string>
#include <stddef.h>
/*! \brief Memory Tracking.

  This system is only suited for the specific purpose of the \a HMatrix code.
 */
namespace mem_instr {
  /*! \brief Add an allocation to the tracking.

    \param size positive or negative integer, size in bytes.
   */
  void addAlloc(void* ptr, ptrdiff_t size, char type = 0);
  /*! \brief Dumps the data to filename.
   */
  void toFile(const std::string& filename);

  void enable();
  void disable();

  /**
   * Return the current time with the same reference as
   * memory instrumentation
   */
  size_t getNanoTime();
}

#define REGISTER_ALLOC(ptr, size) mem_instr::addAlloc(ptr, +(size))
#define REGISTER_FREE(ptr, size) mem_instr::addAlloc(ptr, -(size))
#define REGISTER_T_ALLOC(ptr, size, type) mem_instr::addAlloc(ptr, +(size), type)
#define REGISTER_T_FREE(ptr, size, type) mem_instr::addAlloc(ptr, -(size), type)
#define MEMORY_INSTRUMENTATION_TO_FILE(filename) mem_instr::toFile(filename)
#define MEMORY_INSTRUMENTATION_ENABLE mem_instr::enable()
#define MEMORY_INSTRUMENTATION_DISABLE mem_instr::disable()
#else
#define REGISTER_ALLOC(ptr, size) do {} while (0)
#define REGISTER_FREE(ptr, size) do { (void)ptr; (void)size; } while (0)
#define REGISTER_T_ALLOC(ptr, size, type) do {} while (0)
#define REGISTER_T_FREE(ptr, size, type) do { (void)ptr; (void)size; } while (0)
#define MEMORY_INSTRUMENTATION_TO_FILE(filename) do {} while (0)
#define MEMORY_INSTRUMENTATION_ENABLE do {} while (0)
#define MEMORY_INSTRUMENTATION_DISABLE do {} while (0)
#endif // __cplusplus > 199711L
#endif
