#!/bin/sh
GMP_VERSION=6.1.2
NTL_VERSION=11.3.2
HELIB_COMMIT=ac0308715e5ae6bf5e750e8701e736d855550fc8


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
