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

cmake_minimum_required(VERSION 2.8.9)

# Set CMAKE_BUILD_TYPE to Release by default.
# Must be done before calling project()
if(CMAKE_BUILD_TYPE MATCHES "^CMAKE_BUILD_TYPE$")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" Release Debug RelWithDebInfo MinSizeRel)
endif()

# Set BUILD_SHARED_LIBS to ON by default.
# Must be done before calling project()
if(BUILD_SHARED_LIBS MATCHES "^BUILD_SHARED_LIBS$")
    if(NOT WIN32)
        # __declspec(dllexport) are missing in hmat so it's currently not possible
        # to build it as a shared library on win32
        set(BUILD_SHARED_LIBS "ON" CACHE BOOL "Build shared libraries." FORCE)
    endif()
endif()

project(hmat-oss C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMake)

if(NOT HMAT_VERSION)
    include(GitVersion)
    git_version(HMAT 1.5.0)
endif()
set(HMAT_SO_VERSION 1)

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR     lib${LIB_SUFFIX} CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR     bin              CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include          CACHE PATH "Installation directory for header files")
set(INSTALL_DATA_DIR    share/hmat       CACHE PATH "Installation directory for data files")
set(INSTALL_CMAKE_DIR   ${INSTALL_LIB_DIR}/cmake/hmat CACHE PATH "Installation directory for cmake config files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE DATA CMAKE)
    set(var INSTALL_${p}_DIR)
    set(RELATIVE_INSTALL_${p}_DIR ${INSTALL_${p}_DIR})
    if(NOT IS_ABSOLUTE "${${var}}")
        set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
    endif()
endforeach()

# CMake 2.x does not define MSVC when using Intel compiler on Windows, see
# http://public.kitware.com/Bug/view.php?id=14476
if ( WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
  set(WINTEL TRUE)
  set(CMAKE_C_FLAGS "/Qstd=c99 ${CMAKE_C_FLAGS}")
endif()

include(CheckCXXCompilerFlag)
# needed for shared_ptr
check_cxx_compiler_flag("-std=c++98" stdcpp98)
if(stdcpp98)
    set(CMAKE_CXX_FLAGS "-std=c++98 ${CMAKE_CXX_FLAGS}")
endif()

# ========================
# C
# ========================
include(CheckIncludeFile)
check_include_file("stdint.h" HAVE_STDINT_H)
check_include_file("sys/types.h" HAVE_SYS_TYPES_H)
check_include_file("time.h" HAVE_TIME_H)
check_include_file("sys/resource.h" HAVE_SYS_RESOURCE_H)
check_include_file("unistd.h" HAVE_UNISTD_H)
check_include_file("mach/mach_time.h" HAVE_MACH_MACH_TIME_H)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Enable gcc vectorization
function(opt_flag flag)
  if(CMAKE_VERSION VERSION_GREATER 2.8.12)
    include(CheckCCompilerFlag)
    string(MAKE_C_IDENTIFIER ${flag} label)
    check_c_compiler_flag(${flag} ${label})
    if(${label})
       set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${flag}" PARENT_SCOPE)
       set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${flag}" PARENT_SCOPE)
    endif()
  endif()
endfunction()
opt_flag("-ffast-math")
opt_flag("-funsafe-math-optimizations")

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
   set(HMAT_32BITS TRUE)
endif()

# ========================
# SYSTEM & EXTERNAL LIBS
# ========================
set(EXTRA_LIBS "")

include(CheckLibraryExists)

check_library_exists("m" sqrt "" HAVE_LIBM)
if(HAVE_LIBM)
    set(M_LIBRARY m)
endif()

check_library_exists("rt" clock_gettime "" HAVE_LIBRT)
if(HAVE_LIBRT)
    set(RT_LIBRARY rt)
endif()

option(HMAT_DISABLE_OPENMP "Let HMat disable OpenMP (require OpenMP support)" ON)
if(HMAT_DISABLE_OPENMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
        set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}")
    endif()
endif()

# MKL
find_package(MKL)
if(MKL_FOUND)
  set(CMAKE_C_FLAGS "${MKL_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${MKL_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
  set(CBLAS_INCLUDE_DIR ${MKL_INCLUDE_DIRS})
endif()

# JeMalloc
option(HMAT_JEMALLOC "Use jemalloc for matrix blocks allocation." OFF)
if(HMAT_JEMALLOC)
    find_package(JeMalloc REQUIRED)
    if(JEMALLOC_FOUND)
        set(HAVE_JEMALLOC TRUE)
        include_directories(${JEMALLOC_INCLUDE_DIR})
    endif()
endif()

# Context timers
option(HMAT_CONTEXT "Use context timers." OFF)
if(HMAT_CONTEXT)
    message(STATUS "Use context timers")
    set(HAVE_CONTEXT TRUE)
endif()

option(HMAT_TIMELINE "Enable profiling timeline" OFF)

# Build date export through hmat_get_build_date()
option(HMAT_EXPORT_BUILD_DATE "Hmat exports build date." OFF)
if(HMAT_EXPORT_BUILD_DATE)
    message(STATUS "Exports build date")
endif(HMAT_EXPORT_BUILD_DATE)

option(USE_DEBIAN_OPENBLAS "On Debian, link to openblas instead of generic blas." ON)
# BLAS/LAPACK
if (NOT MKL_FOUND)
   if(USE_DEBIAN_OPENBLAS)
        get_filename_component(real_blas_path "/usr/lib/libblas.so" REALPATH)
        string(REGEX MATCH "/usr/lib/openblas-base/libblas.so" is_debian_openblas ${real_blas_path})
        if(is_debian_openblas)
            set(BLAS_FOUND ON)
            set(BLAS_LIBRARIES openblas)
            set(CBLAS_INCLUDE_DIR /usr/include/openblas)
            set(LAPACK_LIBRARIES openblas)
            unset(BLAS_LINKER_FLAGS)
        endif()
    endif()
    if(NOT is_debian_openblas)
        # workaround for cmake issue #0009976
        if (CMAKE_VERSION VERSION_LESS 2.8.4)
            enable_language(Fortran)
        endif ()
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
    endif()
    link_directories(${BLAS_LIBRARY_DIRS})
endif()

# CBLAS

# For backward compatibility
if(CBLAS_INCLUDE_DIRS)
    set(CBLAS_INCLUDE_DIR ${CBLAS_INCLUDE_DIRS})
endif()

include(CheckFunctionExists)
if (NOT MKL_CBLAS_FOUND)
    if (CMAKE_VERSION VERSION_GREATER 2.8.8)
        include(CMakePushCheckState)
        cmake_push_check_state()
    endif ()
    set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    check_function_exists("openblas_set_num_threads" HAVE_OPENBLAS_SET_NUM_THREADS)
    check_symbol_exists("openblas_set_num_threads" "cblas.h" HAVE_DECL_OPENBLAS_SET_NUM_THREADS)
    check_function_exists("goto_get_num_procs" HAVE_GOTO_GET_NUM_PROCS)
    check_function_exists("cblas_dgemm" CHECK_FUNCTION_EXISTS_CBLAS_DGEMM)
    if (CMAKE_VERSION VERSION_GREATER 2.8.8)
        cmake_pop_check_state()
    endif ()
endif()
if ((NOT MKL_CBLAS_FOUND) AND (NOT is_debian_openblas))
    # Functions may already be available via MKL or BLAS, but we need cblas.h
    if (CHECK_FUNCTION_EXISTS_CBLAS_DGEMM)
        find_path(CBLAS_INCLUDE_DIR NAMES cblas.h DOC "CBLAS include directory")
        if(CBLAS_INCLUDE_DIR)
            set(CMAKE_REQUIRED_INCLUDES ${CBLAS_INCLUDE_DIR})
            check_include_file("cblas.h" HAVE_CBLAS_H)
            if (NOT HAVE_CBLAS_H)
                message(FATAL_ERROR "cblas.h not found")
            endif()
        else(CBLAS_INCLUDE_DIR)
            message(FATAL_ERROR "cblas.h not found")
        endif(CBLAS_INCLUDE_DIR)
    else (CHECK_FUNCTION_EXISTS_CBLAS_DGEMM)
        find_package(CBLAS REQUIRED)
    endif()
endif()
include_directories(${CBLAS_INCLUDE_DIR})

# ========================
# Warning flags
# ========================
check_cxx_compiler_flag("-Werror -Wall -Wno-unused-result -Wno-sign-compare -Wno-literal-suffix" HAVE_GCC_WARNING_FLAGS)
if(HAVE_GCC_WARNING_FLAGS)
  SET(CMAKE_CXX_FLAGS "-Werror -Wall -Wno-unused-result -Wno-sign-compare -Wno-literal-suffix ${CMAKE_CXX_FLAGS}")
  SET(CMAKE_C_FLAGS "-Werror -Wall -Wno-sign-compare ${CMAKE_C_FLAGS}")
endif()
check_cxx_compiler_flag("-Wno-undefined-var-template" DISABLE_CLANG_FALSE_POSITIVES)
if(DISABLE_CLANG_FALSE_POSITIVES)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undefined-var-template")
endif()

# ========================
# Configuration file
# ========================
if(NOT BUILD_SHARED_LIBS)
    set(HMAT_STATIC ON)
endif()
configure_file("${PROJECT_SOURCE_DIR}/CMake/config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/config.h")
configure_file("${PROJECT_SOURCE_DIR}/CMake/hmat-config-oss.h.in" "${CMAKE_CURRENT_BINARY_DIR}/hmat/config-oss.h")
if(NOT HMAT_SKIP_CONFIG)
  configure_file("${PROJECT_SOURCE_DIR}/CMake/hmat-config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/hmat/config.h")
endif()
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# ============================
# Generation des bibliotheques
# ============================

if(HMAT_EXPORT_BUILD_DATE AND NOT WIN32)
  # Regle pour toujours recompiler les sources contenant __DATE__ et __TIME__
  add_custom_command(COMMAND ${CMAKE_COMMAND} -E touch ${PROJECT_BINARY_DIR}/forcedBuild.tmp
    OUTPUT ${PROJECT_BINARY_DIR}/forcedBuild.tmp)
  set_source_files_properties( src/c_default_interface.cpp PROPERTIES OBJECT_DEPENDS ${PROJECT_BINARY_DIR}/forcedBuild.tmp)
endif(HMAT_EXPORT_BUILD_DATE AND NOT WIN32)

# Sources
file(GLOB_RECURSE HMAT_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} src/*.[ch]pp include/*.h)
list(REMOVE_ITEM HMAT_SOURCES src/hmat_cpp_interface.cpp src/recursion.cpp) # because it is included in default_engine.cpp (and the other *_engine.cpp) and h_matrix.cpp

if(NOT HMAT_TIMELINE)
    list(REMOVE_ITEM HMAT_SOURCES src/common/timeline.cpp)
endif()

link_directories(${PROJECT_BINARY_DIR} ${BLAS_LIBRARY_DIRS})

if (MSVC)
    add_definitions(-D__func__=__FUNCTION__)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)
endif ()

# lib HMAT
if(HMAT_OSS_STATIC AND BUILD_SHARED_LIBS)
    add_library(${PROJECT_NAME} STATIC ${HMAT_SOURCES})
    set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
    add_library(${PROJECT_NAME} ${HMAT_SOURCES})
    install(TARGETS ${PROJECT_NAME}
        EXPORT    HMATLibraryDepends
        RUNTIME DESTINATION "${RELATIVE_INSTALL_BIN_DIR}" COMPONENT Runtime
        LIBRARY DESTINATION "${RELATIVE_INSTALL_LIB_DIR}" COMPONENT Runtime
        ARCHIVE DESTINATION "${RELATIVE_INSTALL_LIB_DIR}" COMPONENT Development
    )
endif()
if(BUILD_SHARED_LIBS)
    set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "HMAT_DLL_EXPORTS")
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "${LAPACK_LINKER_FLAGS} ${BLAS_LINKER_FLAGS} ${MPI_LINK_FLAGS}")

if(NOT HMAT_NO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${HMAT_VERSION} SOVERSION ${HMAT_SO_VERSION})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_NAME_DIR ${INSTALL_LIB_DIR}    )

# LINK_PRIVATE and LINK_PUBLIC keywords were introduced in 2.8.7
if(BUILD_SHARED_LIBS AND CMAKE_VERSION VERSION_GREATER 2.8.6)
    # Programs using hmat do not need to link directly with second level dependencies so we strip them all
    set(_LINK_PRIVATE LINK_PRIVATE)
    set(_LINK_PUBLIC LINK_PUBLIC)
endif()

target_link_libraries(${PROJECT_NAME}
            ${_LINK_PRIVATE}
            ${CBLAS_LIBRARIES}
            ${BLAS_LIBRARIES}
            ${LAPACK_LIBRARIES}
            ${MPI_C_LIBRARIES}
            ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES})
if(MKL_FOUND)
    # The start-group/end-group flags must be at the end of the
    # link command line, so we must use target_link_libraries not, CMAKE_SHARED_XXX_FLAGS
    target_link_libraries(${PROJECT_NAME} ${_LINK_PRIVATE} ${MKL_LINKER_FLAGS})
endif()

if(JEMALLOC_FOUND)
    target_link_libraries(${PROJECT_NAME} ${_LINK_PRIVATE} ${JEMALLOC_LIBRARIES})
endif()

if (HAVE_LIBRT)
    # rt is needed by my_chrono.hpp
    target_link_libraries(${PROJECT_NAME} ${_LINK_PUBLIC} ${RT_LIBRARY})
endif ()

if (HAVE_LIBM)
    target_link_libraries(${PROJECT_NAME} ${_LINK_PUBLIC} ${M_LIBRARY})
endif()

# Examples
include_directories(${PROJECT_SOURCE_DIR}/src)

# Install examples with RPATH
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${INSTALL_LIB_DIR}" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${INSTALL_LIB_DIR}")
endif("${isSystemDir}" STREQUAL "-1")

option(BUILD_EXAMPLES "build examples" OFF)
option(INSTALL_EXAMPLES "install examples" OFF)

macro(hmat_add_example name source)
  if (BUILD_EXAMPLES)
    if (MSVC OR WINTEL)
      set_source_files_properties("examples/${source}" PROPERTIES COMPILE_DEFINITIONS "_USE_MATH_DEFINES")
    endif ()
    if (MSVC)
      #  No complex.h on MSVC, compile with C++
      set_source_files_properties("examples/${source}" PROPERTIES LANGUAGE CXX)
    endif ()
    add_executable(${HMAT_PREFIX_EXAMPLE}${name} ${PROJECT_SOURCE_DIR}/examples/${source})
    target_link_libraries(${HMAT_PREFIX_EXAMPLE}${name} ${PROJECT_NAME})
    if (INSTALL_EXAMPLES)
      install(TARGETS ${HMAT_PREFIX_EXAMPLE}${name} DESTINATION "${RELATIVE_INSTALL_BIN_DIR}/examples" COMPONENT Runtime)
    endif ()
  endif ()
endmacro()

hmat_add_example(kriging kriging.cpp)
hmat_add_example(c-cylinder c-cylinder.c)
hmat_add_example(c-simple-cylinder c-simple-cylinder.c)
hmat_add_example(c-simple-kriging c-simple-kriging.c)
hmat_add_example(c-cholesky c-cholesky.c)

if (BUILD_EXAMPLES)
  enable_testing ()
  add_test (NAME cholesky COMMAND ${HMAT_PREFIX_EXAMPLE}c-cholesky 1000 S)
  add_test (NAME cylinder COMMAND ${HMAT_PREFIX_EXAMPLE}c-cylinder 1000 Z)
  add_test (NAME simple-cylinder COMMAND ${HMAT_PREFIX_EXAMPLE}c-simple-cylinder 1000 Z)
endif ()

install(DIRECTORY include/hmat DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT Development)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hmat/config-oss.h DESTINATION "${INSTALL_INCLUDE_DIR}/hmat" COMPONENT Development)
if(NOT HMAT_SKIP_CONFIG)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hmat/config.h DESTINATION "${INSTALL_INCLUDE_DIR}/hmat" COMPONENT Development)
endif()
set(CXX_HMAT_HEADERS "full_matrix;hmat_cpp_interface;compression;h_matrix;"
    "coordinates;clustering;admissibility;default_engine;cluster_tree;tree;assembly;data_types")
foreach(header ${CXX_HMAT_HEADERS})
    set(HMAT_HEADERS src/${header}.hpp;${HMAT_HEADERS})
endforeach()
install(FILES ${HMAT_HEADERS} DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT Development)
if(NOT DEFINED HMAT_LIBRARIES)
    set(HMAT_LIBRARIES ${PROJECT_NAME})
    #
    # BUILD-TREE
    #
    # Add all targets to the build-tree export set
    export(TARGETS ${PROJECT_NAME} FILE "${PROJECT_BINARY_DIR}/HMATLibraryDepends.cmake")

    # Create a HMATConfig.cmake file for use from the build tree
    set(HMAT_INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_BINARY_DIR}")
    set(HMAT_LIB_DIR "${PROJECT_BINARY_DIR}")
    set(HMAT_CMAKE_DIR "${PROJECT_BINARY_DIR}")
    set(HMAT_DEFINITIONS ${CBLAS_DEFINITIONS};${MKL_DEFINITIONS})
    configure_file(CMake/HMATConfig.cmake.in "${PROJECT_BINARY_DIR}/HMATConfig.cmake" @ONLY)
    configure_file(CMake/HMATConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/HMATConfigVersion.cmake" @ONLY)

    #
    # INSTALL-TREE
    #
    # Install the export set for use with the install-tree
    install(EXPORT HMATLibraryDepends DESTINATION
      "${RELATIVE_INSTALL_CMAKE_DIR}"
        COMPONENT    Development)

    # Create a HMATConfig.cmake file for the use from the install tree
    # and install it
    set(HMAT_CMAKE_DIR "${INSTALL_CMAKE_DIR}")
    file(RELATIVE_PATH rel_include_dir "${HMAT_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")
    list(APPEND RELATIVE_HMAT_INCLUDE_DIRS ${rel_include_dir})

    file(RELATIVE_PATH rel_lib_dir "${HMAT_CMAKE_DIR}" "${INSTALL_LIB_DIR}")
    list(APPEND RELATIVE_HMAT_LIB_DIR ${rel_lib_dir})

    configure_file(CMake/HMATConfig.cmake.in "${PROJECT_BINARY_DIR}/InstallFiles/HMATConfig.cmake" @ONLY)
    configure_file(CMake/HMATConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/InstallFiles/HMATConfigVersion.cmake" @ONLY)
    install(FILES
        "${PROJECT_BINARY_DIR}/InstallFiles/HMATConfig.cmake"
        "${PROJECT_BINARY_DIR}/InstallFiles/HMATConfigVersion.cmake"
        DESTINATION "${HMAT_CMAKE_DIR}" COMPONENT    Development)
else()
    # It's a mess to detect so we publish it up (temporary solution)
    set(HMAT_CBLAS_INCLUDE_DIR ${CBLAS_INCLUDE_DIR} PARENT_SCOPE)
    set(HMAT_CBLAS_LIBRARIES "${BLAS_LIBRARIES};${CBLAS_LIBRARIES}" PARENT_SCOPE)
endif()

# ========================
# final LOG
# ========================
if (CMAKE_VERSION VERSION_GREATER 2.8.5)
    include(FeatureSummary)
    feature_summary(WHAT ALL)
endif ()
