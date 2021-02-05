# Hammer

## Install Conda

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
8. Logout from lxplus and login again to apply the changes.

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
All the dependencies are taken care of by the conda environment.

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
cd Hammer-build

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

make install
```

if you've made it thus far, you're all set and Hammer is installed.


## Make Hammer available in python

Hammer's python libraries are installed, but they need to be added to `PYTHONPATH` to make them easily available.
Following the idea described here, when the `hammer3p8` environment is activated, `PYTHONPATH` is set automatically

https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux


Example `env_vars.csh` to be put in `YOURPATHTOCONDA/miniconda3/envs/hammer3p8/etc/conda/activate.d`

```
echo "Making Hammer python libraries known to python via settin  PYTHONPATH"
echo ''
if (! $?PYTHONPATH) then
  setenv PYTHONPATH "/afs/cern.ch/work/m/manzoni/Hammer-install/lib64/python3.8/site-packages"
  echo "PYTHONPATH was undefined, setting it to $PYTHONPATH"
else
  if ("$PYTHONPATH" == "")  then
      setenv PYTHONPATH "/afs/cern.ch/work/m/manzoni/Hammer-install/lib64/python3.8/site-packages"
      echo "PYTHONPATH was empty, setting it to $PYTHONPATH"
  else
      echo "PYTHONPATH contains $PYTHONPATH, appending..."
      setenv OLDPYTHONPATH $PYTHONPATH
      setenv PYTHONPATH "/afs/cern.ch/work/m/manzoni/Hammer-install/lib64/python3.8/site-packages:$PYTHONPATH"
      echo "now PYTHONPATH is $PYTHONPATH"
  endif
endif
```

`env_vars.csh` example
```
#!/usr/local/bin/bash

if [[ ! -v PYTHONPATH ]]; then
    echo "PYTHONPATH is not set"
    export PYTHONPATH="/Users/manzoni/Documents/hammer/Hammer-install/lib/python3.8/site-packages"
    echo "PYTHONPATH has now the value: $PYTHONPATH"
elif [[ -z "$PYTHONPATH" ]]; then
    echo "PYTHONPATH is set to the empty string"
    export PYTHONPATH="/Users/manzoni/Documents/hammer/Hammer-install/lib/python3.8/site-packages"
    echo "PYTHONPATH has now the value: $PYTHONPATH"
else
    echo "PYTHONPATH has the value: $PYTHONPATH"
    export PYTHONPATH="/Users/manzoni/Documents/hammer/Hammer-install/lib/python3.8/site-packages":$PYTHONPATH
    echo "appending..."
    echo "PYTHONPATH has now the value: $PYTHONPATH"
fi
```


Example `env_vars.csh` to be put in `YOURPATHTOCONDA/miniconda3/envs/hammer3p8/etc/conda/deactivate.d`

```
echo "removing Hammer libs from PYTHONPATH"
echo ''

if ($?OLDPYTHONPATH) then
    echo "tranferring OLDPYTHONPATH to PYTHONPATH"
    setenv PYTHONPATH $OLDPYTHONPATH
    echo "unsetenv OLDPYTHONPATH"
    unsetenv OLDPYTHONPATH
else
    echo "unsetenv PYTHONPATH"
    unsetenv PYTHONPATH
endif
```
