# Instructions to compute the hammer weights from a Bc MC sample without filters

## Production of the Gen MC

### Set the right environment to produce the Bc Jpsi X sample without filters on the final particles
```
# use a more recent 106X release that includes https://github.com/cms-sw/cmssw/pull/32450#event-4108272397
cmsrel CMSSW_10_6_20
cd CMSSW_10_6_20/src/
cmsenv

git-cms-addpkg GeneratorInterface/ExternalDecays

# merge the MultiAncestor filter bugfix
git-cms-merge-topic rmanzoni:from-CMSSW_10_6_X_2020-12-02-2300

```

### Bc Jpsi X inclusive sample

```
# get the EvtGen .dec file
curl -s --insecure https://raw.githubusercontent.com/rmanzoni/RJpsiTools/main/evtgen/BcToJpsiMuMuInclusive.dec -o GeneratorInterface/ExternalDecays/data/BcToJpsiMuMuInclusive.dec
```
```
# to run in local (cfg file already without the filters)
cmsRun RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN_cfg.py
```
```
# to run in bash (it uses the `RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN_TEMPLATE_cfg.py`)
python submitter_gen.py
```

### LHE file for Bc sample stored at T2_CSCS (already in the config file)
```
uberftp -ls gsiftp://storage01.lcg.cscs.ch/pnfs/lcg.cscs.ch/cms/trivcat/store/user/cgalloni/Bc_LHE_600M_Nov2020
```

## Inspect the root files to save a flat tree with the important branches (one for tau and one for mu)
```
# in local
python inspector_tau.py
python inspector_mu.py
```

```
# in bash
python submitter_inspector_tau.py
python submitter_inspector_mu.py
```

## Compute the hammer weights
```
# in local
python hammer_tau.py
python hammer_mu.py
```

```
# in bash
python submitter_hammer_tau.py
python submitter_hammer_mu.py
```

## Compute the total final weights (after merging the final files all in the same file)
```
python compute_yield_weights.py
```

Result:
```
Average hammer weight for mu:  0.6034201195348217  +-  5.259941680788111e-05
Average hammer weight for tau:  0.5542931776945014  +-  7.252682537366594e-05
```