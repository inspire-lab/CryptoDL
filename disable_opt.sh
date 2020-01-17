#!/bin/bash
#disable compiler optimization and enable debug symbols

find Debug -type f -name *.mk -exec sed -i 's/-O3/-O0/g' {} \;
find Debug -type f -name *.mk -exec sed -i 's/-g0/-g3/g' {} \;

