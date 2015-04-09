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
#if ((__cplusplus > 199711L) || defined(HAVE_CPP11)) && HAVE_MEM_INSTR
#include "memory_instrumentation.hpp"

#include "data_recorder.hpp"
#include "../system_types.h"

#include <algorithm>

namespace mem_instr {

  static size_t get_res_mem()
  {
      size_t resident = 0;
#ifdef __linux__
      FILE * statm_file = fopen("/proc/self/statm", "r");
      fscanf(statm_file, "%*s %lu", &resident);
      fclose(statm_file);
#endif
      return resident;
  }
  struct Event
  {
      ptrdiff_t size;
      size_t rss;
      char type;
      Event(ptrdiff_t size, char type) : size(size), type(type)
      {
          rss = get_res_mem();
      }
  };

  std::ostream& operator<< (std::ostream& stream, const Event & event)
  {
      return stream << event.size << " " << event.rss << " " << (int)event.type;
  }

  TimedDataRecorder<Event> allocs;

  /// True if the memory tracking is enabled.
  static bool enabled = false;

  void addAlloc(void* ptr, ptrdiff_t size, char type) {
    if (!enabled) {
      return;
    }
    allocs.recordSynchronized(Event(size, type));
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

  size_t getNanoTime() {
    return allocs.getNanoTime();
  }
}
#endif // __cplusplus > 199711L
