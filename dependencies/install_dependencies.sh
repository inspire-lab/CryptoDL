#!/bin/bash
source versions.sh

# libraries
GMP_URL=https://gmplib.org/download/gmp/gmp-${GMP_VERSION}.tar.bz2
NTL_URL=https://shoup.net/ntl/ntl-${NTL_VERSION}.tar.gz
HELIB_GIT=https://github.com/homenc/HElib.git


# helpers
WD="${DEPENDENCIES_DIR}"

# create dirs
mkdir -p "${GMP_DIR}"
mkdir -p "${NTL_DIR}"

# download
if [ ! -f gmp-${GMP_VERSION}.tar.bz2 ]; then
    wget ${GMP_URL}
fi
if [ ! -f ntl-${NTL_VERSION}.tar.gz ]; then
    wget ${NTL_URL}
fi
git clone ${HELIB_GIT}

# start building
echo "Building ${GMP_VERSION}"
tar xf gmp-${GMP_VERSION}.tar.bz2
cd gmp-${GMP_VERSION}
./configure --prefix="${GMP_DIR}" --disable-shared
make
make install
cd "${WD}"

echo "Building ${NTL_VERSION}"
tar xf ntl-${NTL_VERSION}.tar.gz
cd ntl-${NTL_VERSION}/src
./configure NTL_GMP_LIP=on PREFIX="${NTL_DIR}" GMP_PREFIX="${GMP_DIR}"
make
make install
cd "${WD}"

# build HElib
echo "Building HElib"
cd HElib
git checkout ${HELIB_COMMIT}
cd src
make INC_NTL=-I"${NTL_INCLUDE_DIR}" INC_GMP=-I"${GMP_INCLUDE_DIR}" LIB_NTL="${NTL_DIR}" LIB_GMP="${GMP_DIR}"
mkdir -p "${HELIB_INCLUDE_DIR}"/helib
cp *.h "${HELIB_INCLUDE_DIR}"/helib
cp fhe.a "${HELIB_DIR}"