# Instructions

```
# use a more recent 106X release that includes https://github.com/cms-sw/cmssw/pull/32450#event-4108272397
cmsrel CMSSW_10_6_19_patch3
cd CMSSW_10_6_19_patch3/src/
cmsenv

git-cms-addpkg GeneratorInterface/ExternalDecays

# merge the MultiAncestor filter bugfix
git-cms-merge-topic rmanzoni:from-CMSSW_10_6_X_2020-12-02-2300

# get the EvtGen .dec file
curl -s --insecure https://raw.githubusercontent.com/rmanzoni/RJpsiTools/main/evtgen/HbToJpsiMuMuInclusive.dec -o GeneratorInterface/ExternalDecays/data/HbToJpsiMuMuInclusive.dec

# get the gen fragment
curl -s --insecure https://raw.githubusercontent.com/rmanzoni/RJpsiTools/main/evtgen/RJpsi-HbToJpsiMuMu-RunIISummer19UL18-fragment.py --create-dirs -o Configuration/GenProduction/python/RJpsi-HbToJpsiMuMu-RunIISummer19UL18-fragment.py

# compile
scram b -rj16

# get the cmsdriver command
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


# and inspect the file you produced
ipython -i -- inspector.py --verbose


```
