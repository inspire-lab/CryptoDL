#!/bin/bash
source versions.sh

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

# libraries
HELIB_GIT=https://github.com/homenc/HElib.git




git clone ${HELIB_GIT}

# start building
# build HElib
echo "Building HElib"
cd HElib
git checkout ${HELIB_COMMIT}
rm -rf build
mkdir build
cd build
cmake -DPACKAGE_BUILD=ON -DCMAKE_INSTALL_PREFIX=${HELIB_DIR} ..
make -j16
make install
cp ${HELIB_DIR}/lib/libhelib.a ${HELIB_DIR}/lib/fhe.a
cd "${WD}"
# clean up
if [ $CLEAN_UP -eq 1 ]; then
    rm -rf HElib
fi