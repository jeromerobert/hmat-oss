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

cmake_minimum_required(VERSION 3.13)

# Set CMAKE_BUILD_TYPE to Release by default.
# Must be done before calling project()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
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

if(POLICY CMP0092)
    cmake_policy(SET CMP0092 NEW) # 3.15: MSVC warning flags are not in CMAKE_<LANG>_FLAGS by default.
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMake)

if(NOT HMAT_VERSION)
    include(GitVersion)
    git_version(HMAT 1.7.0)
endif()
set(HMAT_SO_VERSION 2)

if ( WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
    set(WINTEL TRUE)
endif()

# Offer the user the choice of overriding the installation directories
set(HMAT_INSTALL_LIB_DIR     lib${LIB_SUFFIX} CACHE PATH "Installation directory for libraries")
set(HMAT_INSTALL_BIN_DIR     bin              CACHE PATH "Installation directory for executables")
set(HMAT_INSTALL_INCLUDE_DIR include          CACHE PATH "Installation directory for header files")
set(HMAT_INSTALL_DATA_DIR    share/hmat       CACHE PATH "Installation directory for data files")
set(HMAT_INSTALL_CMAKE_DIR   ${HMAT_INSTALL_LIB_DIR}/cmake/hmat CACHE PATH "Installation directory for cmake config files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE DATA CMAKE)
    set(var HMAT_INSTALL_${p}_DIR)
    set(HMAT_RELATIVE_INSTALL_${p}_DIR ${HMAT_INSTALL_${p}_DIR} CACHE PATH "Installation path HMAT_INSTALL_${p}_DIR relative to CMAKE_INSTALL_PREFIX")
    mark_as_advanced(HMAT_RELATIVE_INSTALL_${p}_DIR)
    if(NOT IS_ABSOLUTE "${${var}}")
        set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
    endif()
endforeach()

# ========================
# Declare target
# ========================

if(NOT TARGET hmat)
    add_library(hmat)
endif()

# Options which modify the list of source files
option(HMAT_TIMELINE "Enable profiling timeline" OFF)

# Sources
file(GLOB_RECURSE HMAT_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} src/*.[ch]pp include/*.h)
list(REMOVE_ITEM HMAT_SOURCES src/recursion.cpp) # because it is included in h_matrix.cpp
if(NOT HMAT_TIMELINE)
    list(REMOVE_ITEM HMAT_SOURCES src/common/timeline.cpp)
endif()

target_sources(hmat PRIVATE ${HMAT_SOURCES})

if(BUILD_SHARED_LIBS)
    set_target_properties(hmat PROPERTIES DEFINE_SYMBOL HMAT_DLL_EXPORTS)
endif()

target_include_directories(hmat
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${HMAT_RELATIVE_INSTALL_INCLUDE_DIR}>)

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

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(HMAT_32BITS TRUE)
endif()

# ========================
# System & external libs
# ========================
if(NOT WIN32)
    include(CheckLibraryExists)
    check_library_exists("m" sqrt "" HAVE_LIBM)
    if(HAVE_LIBM)
        target_link_libraries(hmat PRIVATE m)
        # Examples will also have to be linked against libm
        set(LIBM_TARGET m)
    endif()

    check_library_exists("rt" clock_gettime "" HAVE_LIBRT)
    if(HAVE_LIBRT)
        # clock_gettime() function is called from src/common/chrono.h
        # and must thus be PUBLIC.
        # FIXME: link is mandatory only for GNU libc < 2.17.
        #        Maybe we can drop this line?
        target_link_libraries(hmat PUBLIC rt)
    endif()
endif(NOT WIN32)

option(HMAT_DISABLE_OPENMP "Let HMat disable OpenMP (require OpenMP support)" ON)
if(HMAT_DISABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)
        target_compile_options(hmat PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>$<$<COMPILE_LANGUAGE:C>:${OpenMP_C_FLAGS}>")
        target_link_libraries(hmat PRIVATE ${OpenMP_C_LIBRARIES})
    endif()
endif()

# JeMalloc
option(HMAT_JEMALLOC "Use jemalloc for matrix blocks allocation." OFF)
if(HMAT_JEMALLOC)
    find_package(JeMalloc REQUIRED)
    if(JEMALLOC_FOUND)
        set(HAVE_JEMALLOC TRUE)
        target_include_directories(hmat PRIVATE ${JEMALLOC_INCLUDE_DIR})
        target_link_libraries(hmat PRIVATE ${JEMALLOC_LIBRARIES})
    endif()
endif()

# Context timers
option(HMAT_CONTEXT "Use context timers." OFF)
if(HMAT_CONTEXT)
    message(STATUS "Use context timers")
    set(HAVE_CONTEXT TRUE)
endif()

# ========================
# BLAS/MKL
# ========================

find_package(MKL)
if(MKL_FOUND)
    target_compile_options(hmat PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${MKL_COMPILER_FLAGS}>$<$<COMPILE_LANGUAGE:C>:${MKL_COMPILER_FLAGS}>")
    set(CBLAS_INCLUDE_DIR ${MKL_INCLUDE_DIRS})
    set(CBLAS_LIBRARIES ${MKL_LINKER_FLAGS})
endif()

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
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
    endif()
endif()

# CBLAS

# For backward compatibility
if(CBLAS_INCLUDE_DIRS)
    set(CBLAS_INCLUDE_DIR ${CBLAS_INCLUDE_DIRS})
endif()

if (NOT MKL_CBLAS_FOUND)
    include(CMakePushCheckState)
    cmake_push_check_state()
    set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    include(CheckFunctionExists)
    check_function_exists("openblas_set_num_threads" HAVE_OPENBLAS_SET_NUM_THREADS)
    check_function_exists("goto_get_num_procs" HAVE_GOTO_GET_NUM_PROCS)
    check_function_exists("cblas_dgemm" CHECK_FUNCTION_EXISTS_CBLAS_DGEMM)
    cmake_pop_check_state()
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

target_include_directories(hmat PRIVATE ${CBLAS_INCLUDE_DIR})

if(BUILD_SHARED_LIBS)
    target_link_options(hmat PRIVATE "${LAPACK_LINKER_FLAGS};${BLAS_LINKER_FLAGS}")
else()
    set_target_properties(hmat PROPERTIES STATIC_LIBRARY_OPTIONS "${LAPACK_LINKER_FLAGS};${BLAS_LINKER_FLAGS}")
endif()

target_link_libraries(hmat
    PRIVATE
        ${LAPACK_LIBRARIES}
        ${CBLAS_LIBRARIES}
        ${BLAS_LIBRARIES}
)

# ========================
# Compiler flags
# ========================
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

function(hmat_set_compiler_flags _TARGET_NAME)
    check_cxx_compiler_flag("-Werror -Wall" HAVE_COMPILER_WARNING_FLAGS)
    if(HAVE_COMPILER_WARNING_FLAGS)
        target_compile_options(${_TARGET_NAME} PRIVATE -Werror -Wall)
        foreach(flag -Wno-sign-compare;-Wno-undefined-var-template;-Wno-unused-parameter)
            string(REPLACE "-" "_" varname ${flag})
            check_cxx_compiler_flag("${flag}" CXX${varname})
            if(CXX${varname})
                target_compile_options(${_TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${flag}>)
            endif()
            check_c_compiler_flag("${flag}" C${varname})
            if(C${varname})
                target_compile_options(${_TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C>:${flag}>)
            endif()
        endforeach()
    endif()
    check_cxx_compiler_flag("/W4" MSVC_COMPILER_WARNING_FLAGS)
    if(MSVC_COMPILER_WARNING_FLAGS)
        target_compile_options(${_TARGET_NAME} PRIVATE /W4 -wd869,1786,2557,3280)
    endif()

    # C99 required for complex numbers
    target_compile_features(${_TARGET_NAME} PRIVATE cxx_std_11;c_std_99)

    if(MSVC OR WINTEL)
        target_compile_definitions(${_TARGET_NAME} PRIVATE __func__=__FUNCTION__;_CRT_SECURE_NO_WARNINGS;NOMINMAX;_USE_MATH_DEFINES)
    endif()
endfunction(hmat_set_compiler_flags)

hmat_set_compiler_flags(hmat)

# Enable gcc vectorization
function(opt_flag flag)
    string(MAKE_C_IDENTIFIER ${flag} label)
    check_c_compiler_flag("${flag}" ${label})
    if(${label})
        target_compile_options(hmat PRIVATE $<$<NOT:$<CONFIG:Debug>>:${flag}>)
    endif()
endfunction()
opt_flag("-ffast-math")
opt_flag("-funsafe-math-optimizations")

# ========================
# Misc
# ========================

# Build date export through hmat_get_build_date()
option(HMAT_EXPORT_BUILD_DATE "Hmat exports build date." OFF)
if(HMAT_EXPORT_BUILD_DATE)
    message(STATUS "Exports build date")
endif()

if(HMAT_EXPORT_BUILD_DATE AND NOT WIN32)
  # Regle pour toujours recompiler les sources contenant __DATE__ et __TIME__
  add_custom_command(COMMAND ${CMAKE_COMMAND} -E touch ${PROJECT_BINARY_DIR}/forcedBuild.tmp
    OUTPUT ${PROJECT_BINARY_DIR}/forcedBuild.tmp)
  set_source_files_properties( src/c_default_interface.cpp PROPERTIES OBJECT_DEPENDS ${PROJECT_BINARY_DIR}/forcedBuild.tmp)
endif(HMAT_EXPORT_BUILD_DATE AND NOT WIN32)

if(NOT HMAT_NO_VERSION)
  set_target_properties(hmat PROPERTIES VERSION ${HMAT_VERSION} SOVERSION ${HMAT_SO_VERSION})
endif()

set_target_properties(hmat PROPERTIES INSTALL_NAME_DIR ${HMAT_INSTALL_LIB_DIR})

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

# ========================
# Examples
# ========================

option(BUILD_EXAMPLES "build examples" OFF)
option(INSTALL_EXAMPLES "install examples" OFF)

# Install examples with RPATH
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${HMAT_INSTALL_LIB_DIR}" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${HMAT_INSTALL_LIB_DIR}")
endif("${isSystemDir}" STREQUAL "-1")

function(hmat_add_example)
    if (BUILD_EXAMPLES)
        cmake_parse_arguments(HMAT_EXAMPLE "" "NAME;INSTALL_DIR" "SOURCES" ${ARGN})
        if(NOT HMAT_EXAMPLE_SOURCES)
            set(HMAT_EXAMPLE_SOURCES examples/${HMAT_EXAMPLE_NAME}.c)
        endif()
        if(NOT HMAT_INSTALL_DIR)
            set(HMAT_INSTALL_DIR "${HMAT_RELATIVE_INSTALL_BIN_DIR}/examples")
        endif()
        add_executable(${HMAT_PREFIX_EXAMPLE}${HMAT_EXAMPLE_NAME} ${HMAT_EXAMPLE_SOURCES})
        hmat_set_compiler_flags(${HMAT_PREFIX_EXAMPLE}${HMAT_EXAMPLE_NAME})
        target_link_libraries(${HMAT_PREFIX_EXAMPLE}${HMAT_EXAMPLE_NAME} PRIVATE hmat ${LIBM_TARGET})
        target_include_directories(${HMAT_PREFIX_EXAMPLE}${HMAT_EXAMPLE_NAME}
            PRIVATE
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        )
        if (INSTALL_EXAMPLES)
            install(TARGETS ${HMAT_PREFIX_EXAMPLE}${HMAT_EXAMPLE_NAME}
                    DESTINATION "${HMAT_INSTALL_DIR}"
                    COMPONENT HMAT_Runtime
            )
        endif ()
    endif ()
endfunction()

hmat_add_example(NAME kriging SOURCES examples/kriging.cpp)
hmat_add_example(NAME c-cylinder)
hmat_add_example(NAME c-simple-cylinder)
hmat_add_example(NAME c-simple-kriging)
hmat_add_example(NAME c-cholesky)

if (BUILD_EXAMPLES)
    enable_testing ()
    add_test (NAME cholesky COMMAND ${HMAT_PREFIX_EXAMPLE}c-cholesky 1000 S)
    add_test (NAME cylinder COMMAND ${HMAT_PREFIX_EXAMPLE}c-cylinder 1000 Z)
    add_test (NAME simple-cylinder COMMAND ${HMAT_PREFIX_EXAMPLE}c-simple-cylinder 1000 Z)
endif ()

# ========================
# Install
# ========================

install(DIRECTORY include/hmat DESTINATION "${HMAT_RELATIVE_INSTALL_INCLUDE_DIR}" COMPONENT HMAT_Development)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hmat/config-oss.h DESTINATION "${HMAT_RELATIVE_INSTALL_INCLUDE_DIR}/hmat" COMPONENT HMAT_Development)
if(NOT HMAT_SKIP_CONFIG)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hmat/config.h DESTINATION "${HMAT_RELATIVE_INSTALL_INCLUDE_DIR}/hmat" COMPONENT HMAT_Development)
endif()

# ========================
# Export
# ========================

# Allow using HMAT targets from add_subdirectory()
add_library(HMAT::hmat ALIAS hmat)

# We want users to link agains HMAT::hmat target
set_target_properties(hmat PROPERTIES EXPORT_NAME hmat)

# Define our exported target
install(TARGETS hmat
        EXPORT HMAT-targets
        RUNTIME
            DESTINATION "${HMAT_RELATIVE_INSTALL_BIN_DIR}"
            COMPONENT HMAT_Runtime
        LIBRARY
            DESTINATION "${HMAT_RELATIVE_INSTALL_LIB_DIR}"
            NAMELINK_COMPONENT HMAT_Development
            COMPONENT HMAT_Runtime
        ARCHIVE
            DESTINATION "${HMAT_RELATIVE_INSTALL_LIB_DIR}"
            COMPONENT HMAT_Development
        INCLUDES
            DESTINATION "${HMAT_RELATIVE_INSTALL_INCLUDE_DIR}"
)

# Allow using HMATTargets.cmake from the build tree
export(EXPORT HMAT-targets NAMESPACE "HMAT::" FILE HMATTargets.cmake)

# Create HMATConfig.cmake and HMATConfigVersion.cmake files
if(NOT HMATCONFIGCMAKE)
    set(HMATCONFIGCMAKE ${PROJECT_SOURCE_DIR}/CMake/HMATConfig.cmake.in)
endif()
configure_file(${HMATCONFIGCMAKE} "${PROJECT_BINARY_DIR}/HMATConfig.cmake" @ONLY)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/HMATConfigVersion.cmake"
    VERSION ${HMAT_VERSION} COMPATIBILITY AnyNewerVersion)
install(FILES
      ${PROJECT_BINARY_DIR}/HMATConfig.cmake
      ${PROJECT_BINARY_DIR}/HMATConfigVersion.cmake
    DESTINATION ${HMAT_RELATIVE_INSTALL_CMAKE_DIR})

install(EXPORT HMAT-targets
        NAMESPACE "HMAT::"
        FILE HMATTargets.cmake
        DESTINATION ${HMAT_RELATIVE_INSTALL_CMAKE_DIR}
        COMPONENT HMAT_Development)

# It's a mess to detect so we publish it up (temporary solution)
if(NOT ${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    set(HMAT_CBLAS_INCLUDE_DIR ${CBLAS_INCLUDE_DIR} PARENT_SCOPE)
    set(HMAT_CBLAS_LIBRARIES "${BLAS_LIBRARIES};${CBLAS_LIBRARIES}" PARENT_SCOPE)
    set(HMAT_LIBM_TARGET "${LIBM_TARGET}" PARENT_SCOPE)
endif()

# ========================
# final LOG
# ========================
include(FeatureSummary)
feature_summary(WHAT ALL)
