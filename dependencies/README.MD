# Installing dependencies

CryptoDL requires a number of dependencies which need to be installed on the system. Install these first.

Packages and tools needed:
- m4
- libarmadillo-dev 
- libboost-all-dev
- libjpeg-dev
- build-essential
- wget
- libcurl4-openssl-dev
- cmake 3.10.2

A number of them can be installed using the ubunutu package manager:
```
sudo apt install -y m4 libarmadillo-dev libboost-all-dev libjpeg-dev build-essential wget libcurl4-openssl-dev
```
CMake needs to be built from source:
https://cmake.org/install/


Alternatively you can use the `config_system.sh` script to install all the above. Needs to be run as `root`


After that run `install_dependencies.sh` in this directory. 

## Automating the process

If you want to install everyhting, for example a new system run the following:
```
sudo ./config_system.sh && ./install_dependencies.sh
```