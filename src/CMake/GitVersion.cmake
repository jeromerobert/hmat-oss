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

# Set label_VERSION to the git version

function(git_version label default_version)
    option(${label}_GIT_VERSION "Get the version string from git describe" ON)
    if(${label}_GIT_VERSION)
        find_package(Git)
        if(GIT_FOUND)
            execute_process(COMMAND "${GIT_EXECUTABLE}"
                describe --dirty=-dirty --always --tags
                OUTPUT_VARIABLE _GIT_DESCRIBE ERROR_QUIET)
            if(_GIT_DESCRIBE)
                string(STRIP ${_GIT_DESCRIBE} ${label}_VERSION)
                set(${label}_VERSION ${${label}_VERSION} PARENT_SCOPE)
            endif()
        endif()
    endif()
    if(NOT ${label}_VERSION)
        set(${label}_VERSION ${default_version} PARENT_SCOPE)
    endif()
    message(STATUS "Version string is ${${label}_VERSION}")
endfunction()
