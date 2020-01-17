#!/bin/bash
#enable compiler optimatization and disable debug symbols

find Debug -type f -name *.mk -exec sed -i 's/-O0/-O3/g' {} \;
find Debug -type f -name *.mk -exec sed -i 's/-g3/-g0/g' {} \;

