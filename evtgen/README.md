# Instructions 

## common
```
# use a more recent 106X release that includes https://github.com/cms-sw/cmssw/pull/32450#event-4108272397
cmsrel CMSSW_10_6_20
cd CMSSW_10_6_20/src/
cmsenv

git-cms-addpkg GeneratorInterface/ExternalDecays

# merge the MultiAncestor filter bugfix
git-cms-merge-topic rmanzoni:from-CMSSW_10_6_X_2020-12-02-2300

```

## Jpsi + X background

```
# get the EvtGen .dec file
curl -s --insecure https://raw.githubusercontent.com/rmanzoni/RJpsiTools/main/evtgen/HbToJpsiMuMuInclusive.dec -o GeneratorInterface/ExternalDecays/data/HbToJpsiMuMuInclusive.dec

# get the gen fragments
# Jpsi + X
curl -s --insecure https://raw.githubusercontent.com/rmanzoni/RJpsiTools/main/evtgen/RJpsi-HbToJpsiMuMu-RunIISummer19UL18-fragment.py --create-dirs -o Configuration/GenProduction/python/RJpsi-HbToJpsiMuMu-RunIISummer19UL18-fragment.py
# Jpsi + mu, mass(mu,mu,mu)<10
curl -s --insecure https://raw.githubusercontent.com/rmanzoni/RJpsiTools/main/evtgen/RJpsi-HbToJpsiMuMu-3MuFilter-RunIISummer19UL18-fragment.py --create-dirs -o Configuration/GenProduction/python/RJpsi-HbToJpsiMuMu-3MuFilter-RunIISummer19UL18-fragment.py

# compile
scram b -rj16

# get the cmsdriver command for the JPsi(mm) + X sample
cmsDriver.py Configuration/GenProduction/python/RJpsi-HbToJpsiMuMu-RunIISummer19UL18-fragment.py \
--fileout file:RJpsi-HbToJpsiMuMu-RunIISummer19UL18GEN.root \
--mc \
--eventcontent RAWSIM \
--datatier GEN \
--conditions 106X_upgrade2018_realistic_v11_L1v1 \
--beamspot Realistic25ns13TeVEarly2018Collision \
--step GEN \
--geometry DB:Extended \
--era Run2_2018 \
--python_filename RJpsi-HbToJpsiMuMu-RunIISummer19UL18GEN_cfg.py \
--no_exec \
--customise Configuration/DataProcessing/Utils.addMonitoring \
-n -1

# now run
cmsRun RJpsi-HbToJpsiMuMu-RunIISummer19UL18GEN_cfg.py

# in case you are submitting the jobs, don't forget to add the following lines at the end of the configuration file 
from IOMC.RandomEngine.RandomServiceHelper import  RandomNumberServiceHelper
randHelper =  RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randHelper.populate()
process.RandomNumberGeneratorService.saveFileName =  cms.untracked.string("RandomEngineState.log")


# get the cmsdriver command for the JPsi(mm) + mu, mass(mu,mu,mu)<10 sample
cmsDriver.py Configuration/GenProduction/python/RJpsi-HbToJpsiMuMu-3MuFilter-RunIISummer19UL18-fragment.py \
--fileout file:RJpsi-HbToJpsiMuMu-3MuFilter-RunIISummer19UL18GEN.root \
--mc \
--eventcontent RAWSIM \
--datatier GEN \
--conditions 106X_upgrade2018_realistic_v11_L1v1 \
--beamspot Realistic25ns13TeVEarly2018Collision \
--step GEN \
--geometry DB:Extended \
--era Run2_2018 \
--python_filename RJpsi-HbToJpsiMuMu-3MuFilter-RunIISummer19UL18GEN_cfg.py \
--no_exec \
--customise Configuration/DataProcessing/Utils.addMonitoring \
-n -1

# now run
cmsRun RJpsi-HbToJpsiMuMu-3MuFilter-RunIISummer19UL18GEN_cfg.py

# in case you are submitting the jobs, don't forget to add the following lines at the end of the configuration file 
from IOMC.RandomEngine.RandomServiceHelper import  RandomNumberServiceHelper
randHelper =  RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randHelper.populate()
process.RandomNumberGeneratorService.saveFileName =  cms.untracked.string("RandomEngineState.log")

# and inspect the file you produced
ipython -i -- inspector.py --verbose


```

## Bc Jpsi X inclusive sample

```
# get the EvtGen .dec file
curl -s --insecure https://raw.githubusercontent.com/rmanzoni/RJpsiTools/main/evtgen/BcToJpsiMuMuInclusive.dec -o GeneratorInterface/ExternalDecays/data/BcToJpsiMuMuInclusive.dec

# get the gen fragment
curl -s --insecure https://raw.githubusercontent.com/rmanzoni/RJpsiTools/main/evtgen/RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN-fragment.py --create-dirs -o Configuration/GenProduction/python/RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN-fragment.py

# compile
scram b -rj16

# get the cmsdriver command for the JPsi(mm) + X sample
cmsDriver.py Configuration/GenProduction/python/RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN-fragment.py \
--filein file:/eos/home-m/manzoni/RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN_step1_72.root \
--fileout file:RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN.root \
--mc \
--eventcontent RAWSIM \
--datatier GEN \
--conditions 106X_upgrade2018_realistic_v11_L1v1 \
--beamspot Realistic25ns13TeVEarly2018Collision \
--step GEN \
--geometry DB:Extended \
--era Run2_2018 \
--python_filename RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN_cfg.py \
--no_exec \
--customise Configuration/DataProcessing/Utils.addMonitoring \
-n -1


# now run
cmsRun RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN_cfg.py

```

## LHE file for Bc sample stored at T2_CSCS
```
uberftp -ls gsiftp://storage01.lcg.cscs.ch/pnfs/lcg.cscs.ch/cms/trivcat/store/user/cgalloni/Bc_LHE_600M_Nov2020
```
