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
#include <string>
#include <vector>
#include <stdio.h>
#include <stddef.h>
#include "common/chrono.h"
#ifndef HMAT_MEM_INSTR
#include "common/context.hpp"
#endif

/*! \brief Memory Tracking.

  This system is only suited for the specific purpose of the \a HMatrix code.
 */
namespace hmat {

class MemoryInstrumenter {
public:
    typedef ptrdiff_t mem_t ;
    typedef size_t (*HookFunction)(void*);
private:
    void allocImpl(mem_t size, char type);
    void freeImpl(mem_t size, char type);
    std::vector<std::string> labels_;
    std::vector<bool> cumulatives_;
    std::vector<HookFunction> hooks_;
    std::vector<void *> hookParams_;
    std::string filename_;
    FILE * output_;
    bool enabled_;
    Time start_;
    mem_t fullMatrixMem_;
public:
    static const char FULL_MATRIX = 1;
    static const char FIRST_AVAIL = 11;
    MemoryInstrumenter();
    ~MemoryInstrumenter();
    void setFile(const std::string & filename);
    char addType(const std::string & label, bool cumulative, HookFunction hook = NULL, void * param = NULL);
    void alloc(size_t size, char type) {
#ifdef HMAT_MEM_INSTR
        allocImpl(size, type);
#else
        ignore_unused_arg(size);
        ignore_unused_arg(type);
#endif
    }

    void free(size_t size, char type) {
#ifdef HMAT_MEM_INSTR
        freeImpl(size, type);
#else
        ignore_unused_arg(size);
        ignore_unused_arg(type);
#endif
    }

    void trig() {
#ifdef MEM_INSTR
        allocImpl(size, -1);
#endif
    }

    void enable();
    void disable();
    void finish();
    /**
       * Return the current time with the same reference as
       * memory instrumentation
       */
    size_t nanoTime();

    size_t fullMatrixMem() {
        return fullMatrixMem_;
    }

    static MemoryInstrumenter& instance()
    {
        static MemoryInstrumenter INSTANCE;
        return INSTANCE;
    }
};
}

#endif
