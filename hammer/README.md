# Install Conda

1. Get the installer from https://docs.conda.io/en/latest/miniconda.html

2. here is for Python 3.8, linux 64 bit. Change if needed  
`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`

3. execute the script  
`bash Miniconda3-latest-Linux-x86_64.sh`

4. follow the guided procedure
    1. read and accept the licence
    2. install in a preferred location. Notice I'll use `work` because some package can sometimes (e.g. `ROOT`) be large and `$HOME` space might be tight
    ```Miniconda3 will now be installed into this location:
    /afs/cern.ch/user/m/manzoni/miniconda3
    
    - Press ENTER to confirm the location
    - Press CTRL-C to abort the installation
    - Or specify a different location below
    
    [/afs/cern.ch/user/m/manzoni/miniconda3] >>> /afs/cern.ch/work/m/manzoni/miniconda3
    ```

5. **If you're using `bash`**, just go ahead and let the installer initialise `conda`.
It will modify your `.bashrc` login script accordingly.  

6. **If you're not using `bash`** *do not initialise conda from within the installer* and exit the installation
Initialise `conda` passing your shell type as an argument. Notice, the full path to *your* conda. If you don't do so, in some systems, it'd point to the default installation of `conda`, which will most likely mess things up  
`/afs/cern.ch/work/m/manzoni/miniconda3/conda init tcsh`

7. edit/create your `$HOME/.condarc` file to include the `conda-forge` repo (channel) and to instruct it to look for environments in your installation postion. Notice that the base conda environment is deactivated by default to avoid interference with CMSSW etc...
```[manzoni@t3ui02 ~]$ more .condarc
auto_activate_base: false
channels:
  - conda-forge
  - defaults
envs_dirs:
  - /afs/cern.ch/work/m/manzoni/miniconda3
```

### now `conda` is ready!

## Create the `hammer3p8` environment

Create a `conda` environment that contains all the packages needed to compile Hammer, including optional ones, e.g. `ROOT`

```
# get the yml file
wget https://raw.githubusercontent.com/rmanzoni/RJpsiTools/main/hammer/conda_env_hammer3p8.yml

# create environment (notice the path)
conda env create -p /afs/cern.ch/work/m/manzoni/miniconda3/envs/hammer3p8 --file=conda_env_hammer3p8.yml
```

activate the environment

```
conda activate hammer3p8
```

## Download and install Hammer

https://hammer.physics.lbl.gov/

```
cd /afs/cern.ch/work/m/manzoni/

git clone https://gitlab.com/mpapucci/Hammer.git
cd Hammer

# move to version 1.1.0
git checkout v1.1.0

# apply a patch to avoid a compile crash
git checkout 30a91bdc include/Hammer/Math/Utils.hh

# create a build directory
cd ..
mkdir Hammer-build
cd Hamemr-build

# Here we decide to build Hammer with ROOT, Python, examples but no doxygen docs.
cmake -DCMAKE_INSTALL_PREFIX=../Hammer-install \
-DWITH_PYTHON=ON \
-DWITH_ROOT=ON \
-DWITH_EXAMPLES=ON \
-DWITH_EXAMPLES_EXTRA=ON \
-DFORCE_YAMLCPP_INSTALL=ON \
-DFORCE_HEPMC_INSTALL=ON \
-DCMAKE_CXX_FLAGS=`root-config --cflags` \
../Hammer

make

# if you get a crash related to 

make install
```







==============================================================================
==============================================================================
==============================================================================
==============================================================================




WORK IN PROGRESS: sucessfully installed in my area, now trying to recollect all the steps

1. Use `conda` to create an independent environment that contains all the dependencies, in a consistent way https://conda.io/projects/conda/en/latest/user-guide/install/linux.html
1.1 install conda in your work area, rather than default `$HOME` because some packages can be alrge and eat up space
1.2 fiddle with conda config file. Add conda-forge channel, add your work installation as a place where to look for environments
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
