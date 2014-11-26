#  HMat-OSS (HMatrix library, open source software)
#
#  Copyright (C) 2014-2015 Airbus Group SAS
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
#  http://github.com/jeromerobert/hmat-oss

# - Find MKL installation.
# Try to find MKL. The following values are defined
#  MKL_FOUND   - True if MKL has been found in MKL
#  MKL_BLAS_FOUND   - True if blas has been found in MKL (should be always ok)
#  MKL_CBLAS_FOUND   - True if cblas has been found in MKL (OK for >=10.2)
#  MKL_LAPACKE_FOUND - True if lapacke has been found in MKL (OK for >=10.3)
#  MKL_BLACS_FOUND   - True if blacs has been found in libraries listed in ${MKL_LIBRARIES}
#  MKL_SCALAPACK_FOUND   - True if scalapack has been found in libraries listed in ${MKL_LIBRARIES}
#  MKL_DEFINITIONS  - list of compilation definition -DFOO
#  + many HAVE_XXX...

#See http://software.intel.com/sites/products/mkl/MKL_Link_Line_Advisor.html

#TODO add scalapack, sequential, ilp64, ...

# Allow to skip MKL detection even if MKLROOT is set in the environment.
option(MKL_DETECT "Try to detect and use MKL." ON)

if(MKL_DETECT)
    find_path(MKL_INCLUDE_DIRS NAMES mkl.h HINTS
        $ENV{MKLROOT}/include
        ${MKLROOT}/include)
    option(MKL_STATIC "Link with MKL statically." OFF)

    if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        set(MKL_ARCH "intel64")
        set(MKL_IL "lp64")
    else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        set(MKL_ARCH "ia32")
        set(MKL_IL "c")
    endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )

    find_library(MKL_CORE_LIBRARY NAMES mkl_core.lib mkl_core libmkl_core.so HINTS
        $ENV{MKLROOT}/lib
        ${MKLROOT}/lib
        PATH_SUFFIXES ${MKL_ARCH})

    get_filename_component(MKL_LIBRARY_DIR ${MKL_CORE_LIBRARY} PATH)

    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mkl=parallel" MKL_PARALLEL_COMPILER_FLAG)

    if(WIN32)
        if(MKL_STATIC)
            set(MKL_LINKER_FLAGS "mkl_intel_${MKL_IL}.lib mkl_core.lib mkl_intel_thread.lib")
            set(MKL_COMPILE_FLAGS "")
        else()
            set(MKL_LINKER_FLAGS "mkl_intel_${MKL_IL}_dll.lib mkl_core_dll.lib mkl_intel_thread_dll.lib")
            set(MKL_COMPILE_FLAGS "/Qmkl:parallel")
        endif()
    else()
        if(MKL_STATIC)
            set(MKL_LINKER_FLAGS "-Wl,--start-group ${MKL_LIBRARY_DIR}/libmkl_intel_${MKL_IL}.a ${MKL_LIBRARY_DIR}/libmkl_core.a ${MKL_LIBRARY_DIR}/libmkl_intel_thread.a -Wl,--end-group")
            set(MKL_COMPILE_FLAGS "")
        else()
            if(MKL_PARALLEL_COMPILER_FLAG)
                set(MKL_LINKER_FLAGS "-mkl=parallel")
                set(MKL_COMPILE_FLAGS "-mkl=parallel")
            elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                set(MKL_LINKER_FLAGS "-L${MKL_LIBRARY_DIR} -lmkl_intel_${MKL_IL} -lmkl_core -lmkl_gnu_thread")
                set(MKL_COMPILE_FLAGS "")
            endif()
        endif()
    endif()

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(MKL DEFAULT_MSG MKL_INCLUDE_DIRS)

    set(MKL_CBLAS_FOUND FALSE)
    set(MKL_BLAS_FOUND FALSE)
    set(MKL_LAPACKE_FOUND FALSE)
    set(MKL_BLACS_FOUND FALSE)
    set(MKL_SCALAPACK_FOUND FALSE)
endif(MKL_DETECT)

if (MKL_FOUND)
  include(CMakePushCheckState)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_FLAGS ${MKL_COMPILE_FLAGS})
  set(CMAKE_REQUIRED_INCLUDES ${MKL_INCLUDE_DIRS})
  # We want MKL link flags to be added at the end of the command line
  # else static compilation will fail, so we use CMAKE_REQUIRED_LIBRARIES
  # instead of CMAKE_REQUIRED_FLAGS
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${MKL_LINKER_FLAGS})
  include(CheckIncludeFile)
  check_include_file("mkl_cblas.h"   HAVE_MKL_CBLAS_H)
  check_include_file("mkl_lapacke.h" HAVE_MKL_LAPACKE_H)
  check_include_file("mkl.h"         HAVE_MKL_H)
  check_include_file("mkl_service.h" HAVE_MKL_SERVICE_H)
  check_include_file("mkl_blas.h"    HAVE_MKL_BLAS_H)
  check_include_file("mkl_types.h"   HAVE_MKL_TYPES_H)

  include(CheckFunctionExists)
  check_function_exists("MKLGetVersion"       HAVE_MKLGETVERSION)
  check_function_exists("dgemm"               HAVE_DGEMM)
  check_function_exists("cblas_dgemm"         HAVE_CBLAS_DGEMM)
  check_function_exists("LAPACKE_zgesdd"      HAVE_LAPACKE_ZGESDD)
  check_function_exists("LAPACKE_zgeqrf"      HAVE_LAPACKE_ZGEQRF)
  check_function_exists("zgemm3m"             HAVE_ZGEMM3M)
  check_function_exists("mkl_set_num_threads" HAVE_MKL_SET_NUM_THREADS)
  check_function_exists("mkl_get_max_threads" HAVE_MKL_GET_MAX_THREADS)
  check_function_exists("MKL_Get_Version"     HAVE_MKL_GET_VERSION)
  check_function_exists("mkl_simatcopy"       HAVE_MKL_IMATCOPY)

  get_property(_LANGUAGES_ GLOBAL PROPERTY ENABLED_LANGUAGES)
  if (_LANGUAGES_ MATCHES Fortran)
    include(CheckFortranFunctionExists)
    check_fortran_function_exists("psgesv"      HAVE_PSGESV)
    check_fortran_function_exists("dgesd2d"     HAVE_DGESD2D)
  endif ()
  cmake_pop_check_state()

  if (HAVE_MKL_CBLAS_H AND HAVE_CBLAS_DGEMM)
    set(MKL_CBLAS_FOUND TRUE)
  endif()

  if (HAVE_DGEMM)
    set(MKL_BLAS_FOUND TRUE)
  endif()

  if (HAVE_MKL_LAPACKE_H AND HAVE_LAPACKE_ZGESDD)
    set(MKL_LAPACKE_FOUND TRUE)
  endif()
  
  if(HAVE_DGESD2D)
     set(MKL_BLACS_FOUND TRUE)
  endif()

  if (HAVE_PSGESV)
    set(MKL_SCALAPACK_FOUND TRUE)
  endif()

  set(MKL_DEFINITIONS)
  foreach(arg_ HAVE_MKL_H HAVE_MKL_CBLAS_H HAVE_MKL_LAPACKE_H HAVE_MKL_SERVICE_H HAVE_MKL_BLAS_H HAVE_MKL_TYPES_H)
    if(${${arg_}})
      list(APPEND MKL_DEFINITIONS ${arg_})
    endif()
  endforeach(arg_ ${ARGN})
  message(STATUS "${MYPACK}_DEFINITIONS = ${${MYPACK}_DEFINITIONS}")
endif (MKL_FOUND)
