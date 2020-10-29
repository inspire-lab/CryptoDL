#!/bin/bash
source versions.sh

# libraries
GMP_URL=https://gmplib.org/download/gmp/gmp-${GMP_VERSION}.tar.bz2
NTL_URL=https://shoup.net/ntl/ntl-${NTL_VERSION}.tar.gz
HELIB_GIT=https://github.com/homenc/HElib.git


# helpers
WD="${DEPENDENCIES_DIR}"

# check if cmake is installed
cmake --version
if [ $? -ne 0 ]; then
    echo "installing cmake  ${CMAKE_VERSION}"
    wget https://github.com/Kitware/CMake/releases/download/v$(echo $CMAKE_VERSION | cut -d'-' -f2)/cmake-$CMAKE_VERSION.tar.gz
    tar -zxvf cmake-$CMAKE_VERSION.tar.gz > /dev/null
    rm cmake-$CMAKE_VERSION.tar.gz
    # mv $CMAKE_VERSION cmake_$CMAKE_VERSION
    cd cmake-$CMAKE_VERSION
    ./bootstrap
    make; make install
    cd "${DEPENDENCIES_DIR}"
fi 

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

# # start building
# echo "Building ${GMP_VERSION}"
# tar xf gmp-${GMP_VERSION}.tar.bz2
# cd gmp-${GMP_VERSION}
# ./configure --prefix="${GMP_DIR}"
# make -j16
# make install
# cd "${WD}"

# echo "Building ${NTL_VERSION}"
# tar xf ntl-${NTL_VERSION}.tar.gz
# cd ntl-${NTL_VERSION}/src
# ./configure NTL_GMP_LIP=on PREFIX="${NTL_DIR}" GMP_PREFIX="${GMP_DIR}" SHARED=on  NTL_THREADS=on NTL_THREAD_BOOST=on
# make -j16
# make install
# cd "${WD}"

# build HElib
echo "Building HElib"
cd HElib
git checkout ${HELIB_COMMIT}
rm -rf build
mkdir build
cd build
cmake -DGMP_DIR="${GMP_DIR}" -DNTL_DIR="${NTL_DIR}" ..
make -j16
mkdir -p "${HELIB_INCLUDE_DIR}"
cp -r ../include/helib "${HELIB_INCLUDE_DIR}"
cp lib/libhelib.a "${HELIB_DIR}"/fhe.a