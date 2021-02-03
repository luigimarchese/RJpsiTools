# Standalone installation at PSI

WORK IN PROGRESS: sucessfully installed in my area, now trying to recollect all the steps

1. Use `conda` to create an independent environment that contains all the dependencies, in a consistent way https://conda.io/projects/conda/en/latest/user-guide/install/linux.html
2. create a new conda environment USE YML FILE
3. apply this patch to Hammer source code https://gitlab.com/mpapucci/Hammer/-/issues/68
4. compile
```
cmake -DCMAKE_INSTALL_PREFIX=../Hammer-install \
-DWITH_PYTHON=ON \
-DWITH_ROOT=ON \
-DWITH_EXAMPLES=ON \
-DWITH_EXAMPLES_EXTRA=ON \
-DFORCE_YAMLCPP_INSTALL=ON \
-DFORCE_HEPMC_INSTALL=ON \
-DCMAKE_CXX_FLAGS=`root-config --cflags` \
../Hammer-1.1.0-Source

make

make install
```
5. add hammer's python libraries to python path, do it automatically whenever the conda environment is activated
