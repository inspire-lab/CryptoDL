#!/bin/sh
##############################
# This is part of old build system.
# It is not mainted anymore.
# Here for archaeological purposes 
###############################
CMAKE_VERSION=3.10.2
GMP_VERSION=6.2.0
NTL_VERSION=11.4.3
HELIB_COMMIT=286e4bc0f585dc9ace754fd755ad09d91a4648e7


# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
DEPENDENCIES_DIR=$(dirname "$SCRIPT")
# this works only in bash
# DEPENDENCIES_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#directories
# install dirs
GMP_DIR="${DEPENDENCIES_DIR}"/install/gmp-${GMP_VERSION}
NTL_DIR="${DEPENDENCIES_DIR}"/install/ntl-${NTL_VERSION}
HELIB_DIR="${DEPENDENCIES_DIR}"/install/helib

# include dirs 
NTL_INCLUDE_DIR="${NTL_DIR}"/include 
GMP_INCLUDE_DIR="${GMP_DIR}"/include 
HELIB_INCLUDE_DIR="${HELIB_DIR}"/include 

# generate a makefile that can be included
MAKEFILE="${DEPENDENCIES_DIR}"/makefile.versions
echo "# this file is auto generated. edit version.sh instead" > "${MAKEFILE}"
echo "DEP_INCLUDES := -I${GMP_INCLUDE_DIR} -I${NTL_INCLUDE_DIR} -I${HELIB_INCLUDE_DIR}" >> "${MAKEFILE}"
echo "DEP_LIBS := ${HELIB_DIR}/fhe.a -L${NTL_DIR}/lib -lntl -L${GMP_DIR}/lib -lgmp -lboost_filesystem -lpthread -lboost_system -ljpeg" >> "${MAKEFILE}"
echo "DEP_RPATH := -Wl,-rpath=${NTL_DIR}/lib" >> "${MAKEFILE}"
