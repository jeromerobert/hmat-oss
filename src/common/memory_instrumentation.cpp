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
#if (__cplusplus > 199711L) || defined(HAVE_CPP11)
#include "memory_instrumentation.hpp"

#include "data_recorder.hpp"
#include "../system_types.h"

#include <algorithm>

namespace mem_instr {
  TimedDataRecorder<std::pair<void*, int64_t> > allocs;

  /// True if the memory tracking is enabled.
  static bool enabled = false;

  void addAlloc(void* ptr, int64_t size) {
    if (!enabled) {
      return;
    }
    allocs.recordSynchronized(std::make_pair(ptr, size));
  }

  void toFile(const std::string& filename) {
    allocs.toFile(filename.c_str());
  }

  void enable() {
    enabled = true;
  }

  void disable() {
    enabled = false;
  }
}
#endif // __cplusplus > 199711L
