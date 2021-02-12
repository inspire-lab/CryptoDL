#!/bin/bash
###############################
# this scripts install packages and tools into the system.
# requires root
###############################
# read required version
source versions.sh

# check if user is root
if (( $EUID != 0 )); then
    echo "Please run as root"
    exit 1
fi

# read args
CLEAN_UP=0
for arg in "$@"
do
    case $arg in
        -c|--cleanup)
        CLEAN_UP=1
        shift # Remove
        ;;
        *)
        shift # Remove generic argument from processing
        ;;
    esac
done
# read user information to change ownership later
USER=$(stat -c '%U' .)
GROUP=$(stat -c '%G' .)

# install packages using apt
apt-get update && DEBIAN_FRONTEND="noninteractive" apt install -y m4 libarmadillo-dev libboost-all-dev libjpeg-dev build-essential wget libcurl4-openssl-dev patchelf

# install cmake 
# helpers
WD="${DEPENDENCIES_DIR}"

# check if cmake is installed
cmake --version
# not installed? install it
if [ $? -ne 0 ]; then
    echo "installing cmake  ${CMAKE_VERSION}"
    wget -nv https://github.com/Kitware/CMake/releases/download/v$(echo $CMAKE_VERSION | cut -d'-' -f2)/cmake-$CMAKE_VERSION.tar.gz
    tar -zxvf cmake-$CMAKE_VERSION.tar.gz > /dev/null
    rm cmake-$CMAKE_VERSION.tar.gz
    # mv $CMAKE_VERSION cmake_$CMAKE_VERSION
    cd cmake-$CMAKE_VERSION
    ./bootstrap --system-curl
    make -j16; make install
    cd "${WD}"
    if [ $CLEAN_UP -eq 1 ]; then
        rm -rf cmake-$CMAKE_VERSION
    else
        chown -R $USER:$GROUP cmake-$CMAKE_VERSION
    fi
fi 

# need to make the version file accessable to normal users
chown -R $USER:$GROUP makefile.versions
