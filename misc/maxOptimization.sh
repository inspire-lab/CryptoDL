#!/bin/bash

echo 's/-'$1'/-'$2'/g'
find ../Debug -name '*.mk' -exec sed -i -e 's/-'$1'/-'$2'/g' {} \;
