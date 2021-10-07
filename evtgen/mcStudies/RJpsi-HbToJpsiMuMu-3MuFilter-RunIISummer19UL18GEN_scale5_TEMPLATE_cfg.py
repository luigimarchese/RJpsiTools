# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Configuration/GenProduction/python/jpsimu_alternative.py --fileout file:RJpsi-HbToJpsiMuMu-3MuFilter-RunIISummer19UL18GEN.root --mc --eventcontent RAWSIM --datatier GEN --conditions 106X_upgrade2018_realistic_v11_L1v1 --beamspot Realistic25ns13TeVEarly2018Collision --step GEN --geometry DB:Extended --era Run2_2018 --python_filename RJpsi-HbToJpsiMuMu-3MuFilter-RunIISummer19UL18GEN_cfg.py --no_exec --customise Configuration/DataProcessing/Utils.addMonitoring -n -1
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

process = cms.Process('GEN',Run2_2018)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13TeVEarly2018Collision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(HOOK_MAX_EVENTS)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('QCD bbbar production, Jpsi from any b-hadron (either directly or feeddown), Jpsi->mumu, mu pt>2.8, mu |eta|<2.52, additional mu with pt>2. and |eta|<2.52, invariannt mass(Jpsi, mu)<10, 13 TeV, TuneCP5'),
    name = cms.untracked.string('\\$Source$'),
    version = cms.untracked.string('\\$Revision$')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(1),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(20971520),
    fileName = cms.untracked.string('file:HOOK_FILE_OUT'),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '106X_upgrade2018_realistic_v11_L1v1', '')

process.jpsi_from_b_hadron_filter = cms.EDFilter("PythiaFilterMultiAncestor",
    DaughterIDs = cms.untracked.vint32(-13, 13),
    DaughterMaxEtas = cms.untracked.vdouble(2.52, 2.52),
    DaughterMaxPts = cms.untracked.vdouble(1000000.0, 1000000.0),
    DaughterMinEtas = cms.untracked.vdouble(-2.52, -2.52),
    DaughterMinPts = cms.untracked.vdouble(3.5, 3.5),
    MaxEta = cms.untracked.double(3.0),
    MinEta = cms.untracked.double(-3.0),
    MinPt = cms.untracked.double(6.0),
    MotherIDs = cms.untracked.vint32(5),
    ParticleID = cms.untracked.int32(443)
)


process.three_mu_filter = cms.EDFilter("MCMultiParticleFilter",
    AcceptMore = cms.bool(True),
    EtaMax = cms.vdouble(2.52, 2.52, 2.52),
    NumRequired = cms.int32(3),
    ParticleID = cms.vint32(13, 13, 13),
    PtMin = cms.vdouble(2.8, 2.8, 2.0),
    Status = cms.vint32(1, 1, 1)
)


process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
            convertPythiaCodes = cms.untracked.bool(False),
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2014_NOLONGLIFE.DEC'),
            list_forced_decays = cms.vstring(
                'MyB+', 
                'MyB-', 
                'MyBc+', 
                'MyBc-', 
                'Myanti-B0', 
                'MyB0', 
                'Myanti-Bs', 
                'MyBs', 
                'MyLambda_b0', 
                'MyXi_b-', 
                'Myanti-Xi_b+', 
                'MyXi_b0', 
                'Myanti-Xi_b0', 
                'MyOmega_b-', 
                'Myanti-Omega_b+'
            ),
            operates_on_particles = cms.vint32(),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt_2014.pdl'),
            user_decay_embedded = cms.vstring("\nAlias      MyJ/psi          J/psi\nAlias      Mypsi(2S)        psi(2S)\nAlias      Mypsi(3770)      psi(3770)\nAlias      Mychi_c0         chi_c0\nAlias      Mychi_c1         chi_c1\nAlias      Mychi_c2         chi_c2\nAlias      Myh_c            h_c\nAlias      MyB+             B+\nAlias      MyB-             B-\nAlias      Myanti-B0        anti-B0\nAlias      MyB0             B0\nAlias      Myanti-Bs        anti-B_s0\nAlias      MyBs             B_s0\nAlias      MyBc+            B_c+\nAlias      MyBc-            B_c-\nAlias      MyB*+            B*+\nAlias      MyB*-            B*-\nAlias      MyB*0            B*0\nAlias      Myanti-B*0       anti-B*0\nAlias      MyBs*            B_s*0\nAlias      Myanti-Bs*       anti-B_s*0\nAlias      MyLambda_b0      Lambda_b0\nAlias      MyXi_b-          Xi_b-\nAlias      Myanti-Xi_b+     anti-Xi_b+\nAlias      MyXi_b0          Xi_b0\nAlias      Myanti-Xi_b0     anti-Xi_b0\nAlias      MyOmega_b-       Omega_b-\nAlias      Myanti-Omega_b+  anti-Omega_b+\nChargeConj MyB-             MyB+\nChargeConj Myanti-B0        MyB0\nChargeConj Myanti-Bs        MyBs\nChargeConj MyBc-            MyBc+\nChargeConj MyB*-            MyB*+\nChargeConj MyB*0            Myanti-B*0\nChargeConj MyBs*            Myanti-Bs*\nChargeConj MyXi_b-          Myanti-Xi_b+\nChargeConj MyXi_b0          Myanti-Xi_b0\nChargeConj MyOmega_b-       Myanti-Omega_b+\n\n\nDecay MyJ/psi  # original total forced BR = 0.05930000\n1.00000000 mu+ mu- PHOTOS  VLL;\nEnddecay\n\n\nDecay Mychi_c0  # original total forced BR = 0.01160000\n1.00000000 gamma MyJ/psi PHSP;\nEnddecay\n\n\nDecay Mychi_c1  # original total forced BR = 0.34400000\n1.00000000 MyJ/psi gamma VVP 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;\nEnddecay\n\n\nDecay Mychi_c2  # original total forced BR = 0.19500000\n1.00000000 gamma MyJ/psi PHSP;\nEnddecay\n\n\nDecay Mypsi(2S)  # original total forced BR = 0.82380000\n0.56261153 MyJ/psi pi+ pi- VVPIPI;\n0.29687805 MyJ/psi pi0 pi0 VVPIPI;\n0.05492160 MyJ/psi eta PARTWAVE 0.0 0.0 1.0 0.0 0.0 0.0;\n0.00217677 MyJ/psi pi0 PARTWAVE 0.0 0.0 1.0 0.0 0.0 0.0;\n0.00186854 gamma Mychi_c0 PHSP;\n0.05299265 gamma Mychi_c1 PHSP;\n0.02853747 gamma Mychi_c2 PHSP;\n0.00001340 Myh_c gamma PHSP;\nEnddecay\n\n\nDecay Mypsi(3770)  # original total forced BR = 0.00363000\n0.53168044 MyJ/psi pi+ pi- PHSP;\n0.22038567 MyJ/psi pi0 pi0 PHSP;\n0.24793388 MyJ/psi eta PHSP;\nEnddecay\n\n\nDecay Myh_c  # original total forced BR = 0.01000000\n1.00000000 MyJ/psi pi0 PHSP;\nEnddecay\n\n\nDecay MyB+  # original total forced BR = 0.01756960\n0.09774322 MyJ/psi K+ SVS;\n0.13784300 MyJ/psi K*+ SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;\n0.00472329 MyJ/psi pi+ SVS;\n0.00481969 MyJ/psi rho+ SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;\n0.01927874 MyJ/psi K0 pi+ PHSP;\n0.00963937 MyJ/psi K+ pi0 PHSP;\n0.00963937 MyJ/psi K\'_1+ SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0;\n0.04819685 MyJ/psi K_2*+ PHSP;\n0.17350868 MyJ/psi K_1+ SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0;\n0.00501247 MyJ/psi phi K+ PHSP;\n0.10314127 MyJ/psi K+ pi+ pi- PHSP;\n0.01041052 MyJ/psi eta K+ PHSP;\n0.03373780 MyJ/psi omega K+ PHSP;\n0.00113745 MyJ/psi p+ anti-Lambda0 PHSP;\n0.03718877 Mypsi(2S) K+ SVS;\n0.03569201 Mypsi(2S) K*+ SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;\n0.02302710 Mypsi(2S) K0 pi+ PHSP;\n0.01151355 Mypsi(2S) K+ pi0 PHSP;\n0.10937875 Mypsi(2S) K+ pi- pi+ PHSP;\n0.00575678 Mypsi(2S) K+ pi0 pi0 PHSP;\n0.00575678 Mypsi(2S) K0 pi+ pi0 PHSP;\n0.02302710 Mypsi(2S) K_1+ PHSP;\n0.00148525 Mypsi(2S) pi+ PHSP;\n0.00017146 Mypsi(3770) K+ SVS;\n0.00017495 Mypsi(3770) K*+ PHSP;\n0.00010497 Mypsi(3770) K0 pi+ PHSP;\n0.00006998 Mypsi(3770) K+ pi0 PHSP;\n0.00006998 Mypsi(3770) K+ pi- pi+ PHSP;\n0.00003499 Mypsi(3770) K+ pi0 pi0 PHSP;\n0.00003499 Mypsi(3770) K0 pi+ pi0 PHSP;\n0.00010497 Mypsi(3770) K_1+ PHSP;\n0.00014872 Mychi_c0 K+ PHSP;\n0.00044727 K*+ Mychi_c0 SVS;\n0.00022363 Mychi_c0 K0 pi+ PHSP;\n0.00011182 Mychi_c0 K+ pi0 PHSP;\n0.00022363 Mychi_c0 K+ pi- pi+ PHSP;\n0.00011182 Mychi_c0 K+ pi0 pi0 PHSP;\n0.00011182 Mychi_c0 K0 pi+ pi0 PHSP;\n0.01525334 Mychi_c1 K+ SVS;\n0.00994783 Mychi_c1 K*+ SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;\n0.01326377 Mychi_c1 K0 pi+ PHSP;\n0.00663189 Mychi_c1 K+ pi0 PHSP;\n0.01326377 Mychi_c1 K+ pi- pi+ PHSP;\n0.00663189 Mychi_c1 K+ pi0 pi0 PHSP;\n0.00663189 Mychi_c1 K0 pi+ pi0 PHSP;\n0.00066319 Mychi_c1 pi+ PHSP;\n0.00037594 Mychi_c2 K+ STS;\n0.00037594 Mychi_c2 K*+ PHSP;\n0.00375935 Mychi_c2 K0 pi+ PHSP;\n0.00187968 Mychi_c2 K+ pi0 PHSP;\n0.00375935 Mychi_c2 K+ pi- pi+ PHSP;\n0.00187968 Mychi_c2 K+ pi0 pi0 PHSP;\n0.00187968 Mychi_c2 K0 pi+ pi0 PHSP;\nEnddecay\nCDecay MyB-\n\n\nDecay Myanti-B0  # original total forced BR = 0.01778030\n0.07883916 MyJ/psi anti-K0 PHSP;\n0.02805986 MyJ/psi omega anti-K0 PHSP;\n0.00085990 MyJ/psi eta PHSP;\n0.00171980 MyJ/psi pi- pi+ PHSP;\n0.04163721 MyJ/psi anti-K0 pi- pi+ PHSP;\n0.04887847 MyJ/psi anti-K0 rho0 PHSP;\n0.07241254 MyJ/psi K*- pi+ PHSP;\n0.05974035 MyJ/psi anti-K*0 pi- pi+ PHSP;\n0.03941958 MyJ/psi K_S0 SVS;\n0.03941958 MyJ/psi K_L0 SVS;\n0.12038585 MyJ/psi anti-K*0 SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus;\n0.00159308 MyJ/psi pi0 SVS;\n0.00244392 MyJ/psi rho0 SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus;\n0.00271547 MyJ/psi omega SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus;\n0.00000000 MyJ/psi K- pi+ PHSP;\n0.00905157 MyJ/psi anti-K0 pi0 PHSP;\n0.11767038 MyJ/psi anti-K_10 SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0;\n0.00905157 MyJ/psi anti-K\'_10 SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0;\n0.04525784 MyJ/psi anti-K_2*0 PHSP;\n0.00850847 MyJ/psi phi anti-K0 PHSP;\n0.03351554 Mypsi(2S) anti-K0 PHSP;\n0.01675777 Mypsi(2S) K_S0 SVS;\n0.01675777 Mypsi(2S) K_L0 SVS;\n0.03297496 Mypsi(2S) anti-K*0 SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus;\n0.02162293 Mypsi(2S) K- pi+ PHSP;\n0.01081146 Mypsi(2S) anti-K0 pi0 PHSP;\n0.01081146 Mypsi(2S) anti-K0 pi+ pi- PHSP;\n0.00540573 Mypsi(2S) anti-K0 pi0 pi0 PHSP;\n0.00540573 Mypsi(2S) K- pi+ pi0 PHSP;\n0.02162293 Mypsi(2S) anti-K_10 PHSP;\n0.00007886 Mypsi(3770) K_S0 SVS;\n0.00007886 Mypsi(3770) K_L0 SVS;\n0.00015771 Mypsi(3770) anti-K*0 SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;\n0.00004600 Mypsi(3770) K- pi+ PHSP;\n0.00004600 Mypsi(3770) anti-K0 pi0 PHSP;\n0.00004600 Mypsi(3770) anti-K0 pi+ pi- PHSP;\n0.00002300 Mypsi(3770) anti-K0 pi0 pi0 PHSP;\n0.00002300 Mypsi(3770) K- pi+ pi0 PHSP;\n0.00009529 Mypsi(3770) anti-K_10 PHSP;\n0.00007350 Mychi_c0 K_S0 PHSP;\n0.00007350 Mychi_c0 K_L0 PHSP;\n0.00031499 anti-K*0 Mychi_c0 SVS;\n0.00021000 Mychi_c0 K- pi+ PHSP;\n0.00010500 Mychi_c0 anti-K0 pi0 PHSP;\n0.00021000 Mychi_c0 anti-K0 pi+ pi- PHSP;\n0.00010500 Mychi_c0 anti-K0 pi0 pi0 PHSP;\n0.00010500 Mychi_c0 K- pi+ pi0 PHSP;\n0.00014700 Mychi_c0 anti-K0 PHSP;\n0.00607179 Mychi_c1 K_S0 SVS;\n0.00607179 Mychi_c1 K_L0 SVS;\n0.00691250 Mychi_c1 anti-K*0 SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus;\n0.01245496 Mychi_c1 K- pi+ PHSP;\n0.00622748 Mychi_c1 anti-K0 pi0 PHSP;\n0.01245496 Mychi_c1 anti-K0 pi+ pi- PHSP;\n0.00622748 Mychi_c1 anti-K0 pi0 pi0 PHSP;\n0.00622748 Mychi_c1 K- pi+ pi0 PHSP;\n0.00034874 Mychi_c1 pi0 PHSP;\n0.01214358 Mychi_c1 anti-K0 PHSP;\n0.00491971 Mychi_c1 K+ pi- PHSP;\n0.00088253 Mychi_c2 K_S0 STS;\n0.00088253 Mychi_c2 K_L0 STS;\n0.00052952 Mychi_c2 anti-K*0 PHSP;\n0.00353011 Mychi_c2 K- pi+ PHSP;\n0.00176506 Mychi_c2 anti-K0 pi0 PHSP;\n0.00353011 Mychi_c2 anti-K0 pi+ pi- PHSP;\n0.00176506 Mychi_c2 anti-K0 pi0 pi0 PHSP;\n0.00176506 Mychi_c2 K- pi+ pi0 PHSP;\nEnddecay\nCDecay MyB0\n\n\nDecay MyBs  # original total forced BR = 0.02298000\n0.04783498 MyJ/psi eta\' SVS;\n0.02391749 MyJ/psi eta SVS;\n0.09716481 MyJ/psi phi SVV_HELAMP  1.0 0.0 1.0 0.0 1.0 0.0;\n0.00597937 MyJ/psi K0 SVS;\n0.05231951 MyJ/psi K- K+ PHSP;\n0.05231951 MyJ/psi anti-K0 K0 PHSP;\n0.05231951 MyJ/psi K0 K- pi+ PHSP;\n0.05231951 MyJ/psi anti-K0 K0 pi0 PHSP;\n0.05231951 MyJ/psi K- K+ pi0 PHSP;\n0.02914944 MyJ/psi phi pi+ pi- PHSP;\n0.02914944 MyJ/psi phi pi0 pi0 PHSP;\n0.01494843 MyJ/psi eta pi+ pi- PHSP;\n0.01494843 MyJ/psi eta pi0 pi0 PHSP;\n0.02989687 MyJ/psi eta\' pi+ pi- PHSP;\n0.02989687 MyJ/psi eta\' pi0 pi0 PHSP;\n0.01494843 MyJ/psi pi+ pi- PHSP;\n0.01494843 MyJ/psi pi0 pi0 PHSP;\n0.02075627 Mypsi(2S) eta\' SVS;\n0.01048973 Mypsi(2S) eta SVS;\n0.03035325 Mypsi(2S) phi SVV_HELAMP  1.0 0.0 1.0 0.0 1.0 0.0;\n0.01339114 Mypsi(2S) K- K+ PHSP;\n0.01339114 Mypsi(2S) anti-K0 K0 PHSP;\n0.01339114 Mypsi(2S) K0 K- pi+ PHSP;\n0.01339114 Mypsi(2S) anti-K0 K0 pi0 PHSP;\n0.01339114 Mypsi(2S) K- K+ pi0 PHSP;\n0.01517663 Mypsi(2S) phi pi+ pi- PHSP;\n0.01517663 Mypsi(2S) phi pi0 pi0 PHSP;\n0.00892743 Mypsi(2S) eta pi+ pi- PHSP;\n0.00892743 Mypsi(2S) eta pi0 pi0 PHSP;\n0.01785485 Mypsi(2S) eta\' pi+ pi- PHSP;\n0.01785485 Mypsi(2S) eta\' pi0 pi0 PHSP;\n0.00892743 Mypsi(2S) pi+ pi- PHSP;\n0.00892743 Mypsi(2S) pi0 pi0 PHSP;\n0.00008670 Mychi_c0 eta\' PHSP;\n0.00004335 Mychi_c0 eta PHSP;\n0.00017340 phi Mychi_c0 SVS;\n0.00002601 Mychi_c0 K- K+ PHSP;\n0.00002601 Mychi_c0 anti-K0 K0 PHSP;\n0.00002601 Mychi_c0 K0 K- pi+ PHSP;\n0.00002601 Mychi_c0 anti-K0 K0 pi0 PHSP;\n0.00002601 Mychi_c0 K- K+ pi0 PHSP;\n0.01799791 Mychi_c1 eta\' SVS;\n0.00771339 Mychi_c1 eta SVS;\n0.03599583 Mychi_c1 phi SVV_HELAMP  1.0 0.0 1.0 0.0 1.0 0.0;\n0.00668494 Mychi_c1 K- K+ PHSP;\n0.00668494 Mychi_c1 anti-K0 K0 PHSP;\n0.00668494 Mychi_c1 K0 K- pi+ PHSP;\n0.00668494 Mychi_c1 anti-K0 K0 pi0 PHSP;\n0.00668494 Mychi_c1 K- K+ pi0 PHSP;\n0.01028452 Mychi_c1 phi pi+ pi- PHSP;\n0.01028452 Mychi_c1 phi pi0 pi0 PHSP;\n0.00257113 Mychi_c1 eta pi+ pi- PHSP;\n0.00257113 Mychi_c1 eta pi0 pi0 PHSP;\n0.00514226 Mychi_c1 eta\' pi+ pi- PHSP;\n0.00514226 Mychi_c1 eta\' pi0 pi0 PHSP;\n0.00677725 Mychi_c2 eta\' STS;\n0.00342506 Mychi_c2 eta STS;\n0.00233196 Mychi_c2 K- K+ PHSP;\n0.00233196 Mychi_c2 anti-K0 K0 PHSP;\n0.00233196 Mychi_c2 K0 K- pi+ PHSP;\n0.00233196 Mychi_c2 anti-K0 K0 pi0 PHSP;\n0.00233196 Mychi_c2 K- K+ pi0 PHSP;\n0.00034755 Myh_c eta\' SVS;\n0.00017564 Myh_c eta SVS;\n0.00074742 Myh_c phi SVV_HELAMP  1.0 0.0 1.0 0.0 1.0 0.0;\n0.00011959 Myh_c K- K+ PHSP;\n0.00011959 Myh_c anti-K0 K0 PHSP;\n0.00011959 Myh_c K0 K- pi+ PHSP;\n0.00011959 Myh_c anti-K0 K0 pi0 PHSP;\n0.00011959 Myh_c K- K+ pi0 PHSP;\nEnddecay\nCDecay Myanti-Bs\n\n\nDecay MyBc+  # original total forced BR = 0.91097820\n0.00232078 MyJ/psi mu+ nu_mu PHOTOS BC_VMN 1;\n0.00006857 Mypsi(2S) mu+ nu_mu PHOTOS BC_VMN 1;\n0.00000292 Mychi_c0 mu+ nu_mu PHOTOS BC_SMN 3;\n0.00008646 Mychi_c1 mu+ nu_mu PHOTOS BC_VMN 3;\n0.00004901 Mychi_c2 mu+ nu_mu PHOTOS BC_TMN 3;\n0.00000357 Myh_c mu+ nu_mu PHOTOS PHSP;\n0.00006586 MyBs mu+ nu_mu PHOTOS PHSP;\n0.10422587 MyBs* mu+ nu_mu PHOTOS PHSP;\n0.00700332 MyB0 mu+ nu_mu PHOTOS PHSP;\n0.01194684 MyB*0 mu+ nu_mu PHOTOS PHSP;\n0.00232078 MyJ/psi e+ nu_e PHOTOS BC_VMN 1;\n0.00006857 Mypsi(2S) e+ nu_e PHOTOS BC_VMN 1;\n0.00000292 Mychi_c0 e+ nu_e PHOTOS BC_SMN 3;\n0.00008646 Mychi_c1 e+ nu_e PHOTOS BC_VMN 3;\n0.00004901 Mychi_c2 e+ nu_e PHOTOS BC_TMN 3;\n0.00000357 Myh_c e+ nu_e PHOTOS PHSP;\n0.00006586 MyBs e+ nu_e PHOTOS PHSP;\n0.10422587 MyBs* e+ nu_e PHOTOS PHSP;\n0.00700332 MyB0 e+ nu_e PHOTOS PHSP;\n0.01194684 MyB*0 e+ nu_e PHOTOS PHSP;\n0.00058630 MyJ/psi tau+ nu_tau PHOTOS BC_VMN 1;\n0.00000584 Mypsi(2S) tau+ nu_tau PHOTOS BC_VMN 1;\n0.00015879 MyJ/psi pi+ SVS;\n0.00048858 MyJ/psi rho+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;\n0.00001344 MyJ/psi K+ SVS;\n0.00002687 MyJ/psi K*+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;\n0.00020765 MyJ/psi D_s+ SVS;\n0.00081838 MyJ/psi D_s*+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;\n0.00001099 MyJ/psi D+ SVS;\n0.00003420 MyJ/psi D*+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;\n0.00026801 MyBs pi+ PHSP;\n0.00011766 rho+ MyBs SVS;\n0.00001732 MyBs K+ PHSP;\n0.00000000 K*+ MyBs SVS;\n0.13388699 MyBs* pi+ SVS;\n0.41607956 MyBs* rho+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;\n0.00762126 MyBs* K+ SVS;\n0.00000000 MyBs* K*+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;\n0.02183388 MyB0 pi+ PHSP;\n0.01977408 rho+ MyB0 SVS;\n0.00144186 MyB0 K+ PHSP;\n0.00030897 K*+ MyB0 SVS;\n0.01956810 MyB*0 pi+ SVS;\n0.05293685 MyB*0 rho+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;\n0.00113289 MyB*0 K+ SVS;\n0.00119468 MyB*0 K*+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;\n0.00000047 MyB+ pi0 PHSP;\n0.00000043 rho0 MyB+ SVS;\n0.00002509 MyB+ anti-K0 PHSP;\n0.00000545 K*0 MyB+ SVS;\n0.00067973 MyB*+ pi0 SVS;\n0.00185382 MyB*+ rho0 SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;\n0.03295680 MyB*+ anti-K0 SVS;\n0.03439866 MyB*+ K*0 SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;\nEnddecay\nCDecay MyBc-\n\n\nDecay MyB*+  # original total forced BR = 1.00000000\n1.00000000 MyB+ gamma VSP_PWAVE;\nEnddecay\nCDecay MyB*-\n\n\nDecay MyB*0  # original total forced BR = 1.00000000\n1.00000000 MyB0 gamma VSP_PWAVE;\nEnddecay\nCDecay Myanti-B*0\n\n\nDecay MyBs*  # original total forced BR = 1.00000000\n1.00000000 MyBs gamma VSP_PWAVE;\nEnddecay\nCDecay Myanti-Bs*\n\n\nDecay MyLambda_b0  # original total forced BR = 0.00085000\n0.67437494 Lambda0 MyJ/psi PHSP;\n0.32562506 Lambda0 Mypsi(2S) PHSP;\nEnddecay\n\n\nDecay MyXi_b-  # original total forced BR = 0.00085000\n0.67437494 Xi- MyJ/psi PHSP;\n0.32562506 Xi- Mypsi(2S) PHSP;\nEnddecay\nCDecay Myanti-Xi_b+\n\n\nDecay MyXi_b0  # original total forced BR = 0.00047000\n1.00000000 Xi0 MyJ/psi PHSP;\nEnddecay\nCDecay Myanti-Xi_b0\n\n\nDecay MyOmega_b-  # original total forced BR = 0.00085000\n0.67437494 Omega- MyJ/psi PHSP;\n0.32562506 Omega- Mypsi(2S) PHSP;\nEnddecay\nCDecay Myanti-Omega_b+\n\nEnd\n")
        ),
        parameterSets = cms.vstring('EvtGen130')
    ),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring(
            'pythia8CommonSettings', 
            'pythia8CP5Settings', 
            'processParameters'
        ),
        processParameters = cms.vstring(
            'SoftQCD:nonDiffractive = on', 
            'PTFilter:filter = on', 
            'PTFilter:quarkToFilter = 5', 
            'PTFilter:scaleToFilter = 5.0', 
            '300553:new = 300553 -300553 1 0 0 1.0579400e+01 2.0500001e-02 10.5584 10.6819 0.0000000e+00', 
            '100313:new = 100313 -100313 1 0 0 1.4140000e+00 2.3199996e-01 0.254 2.574 0.0000000e+00', 
            '100323:new = 100323 -100323 1 1 0 1.4140000e+00 2.3199996e-01 0.254 2.574 0.0000000e+00', 
            '30343:new = 30343 -30343 1 0 0 1.6000000e+00 0.0000000e+00 1.6 1.6 0.0000000e+00', 
            '30353:new = 30353 -30353 1 1 0 1.6000000e+00 0.0000000e+00 1.6 1.6 0.0000000e+00', 
            '30363:new = 30363 -30363 1 0 0 1.8000000e+00 0.0000000e+00 1.8 1.8 0.0000000e+00', 
            '9020221:new = 9020221 -9020221 0 0 0 1.4089000e+00 5.1100000e-02 1.1534 1.6644 3.8616000e-12', 
            '9000443:new = 9000443 -9000443 1 0 0 4.0390000e+00 8.0000005e-02 3.639 4.439 0.0000000e+00', 
            '9010443:new = 9010443 -9010443 1 0 0 4.1530000e+00 7.8000000e-02 3.763 4.543 0.0000000e+00', 
            '9020443:new = 9020443 -9020443 1 0 0 4.4210000e+00 6.1999976e-02 4.111 4.731 0.0000000e+00', 
            '110551:new = 110551 -110551 0 0 0 1.0232500e+01 0.0000000e+00 10.2325 10.2325 0.0000000e+00', 
            '120553:new = 120553 -120553 1 0 0 1.0255500e+01 0.0000000e+00 10.2555 10.2555 0.0000000e+00', 
            '100555:new = 100555 -100555 2 0 0 1.0268600e+01 0.0000000e+00 10.2686 10.2686 0.0000000e+00', 
            '210551:new = 210551 -210551 0 0 0 1.0500700e+01 0.0000000e+00 10.5007 10.5007 0.0000000e+00', 
            '220553:new = 220553 -220553 1 0 0 1.0516000e+01 0.0000000e+00 10.516 10.516 0.0000000e+00', 
            '200555:new = 200555 -200555 2 0 0 1.0526400e+01 0.0000000e+00 10.5264 10.5264 0.0000000e+00', 
            '130553:new = 130553 -130553 1 0 0 1.0434900e+01 0.0000000e+00 10.4349 10.4349 0.0000000e+00', 
            '30553:new = 30553 -30553 1 0 0 1.0150100e+01 0.0000000e+00 10.1501 10.1501 0.0000000e+00', 
            '20555:new = 20555 -20555 2 0 0 1.0156200e+01 0.0000000e+00 10.1562 10.1562 0.0000000e+00', 
            '120555:new = 120555 -120555 2 0 0 1.0440600e+01 0.0000000e+00 10.4406 10.4406 0.0000000e+00', 
            '557:new = 557 -557 3 0 0 1.0159900e+01 0.0000000e+00 10.1599 10.1599 0.0000000e+00', 
            '100557:new = 100557 -100557 3 0 0 1.0444300e+01 0.0000000e+00 10.4443 10.4443 0.0000000e+00', 
            '110553:new = 110553 -110553 1 0 0 1.0255000e+01 0.0000000e+00 10.255 10.255 0.0000000e+00', 
            '210553:new = 210553 -210553 1 0 0 1.0516000e+01 0.0000000e+00 10.516 10.516 0.0000000e+00', 
            '110555:new = 110555 -110555 2 0 0 1.0441000e+01 0.0000000e+00 10.441 10.441 0.0000000e+00', 
            '10555:new = 10555 -10555 2 0 0 1.0157000e+01 0.0000000e+00 10.157 10.157 0.0000000e+00', 
            '13124:new = 13124 -13124 1.5 0 0 1.6900000e+00 6.0000018e-02 1.39 1.99 0.0000000e+00', 
            '43122:new = 43122 -43122 0.5 0 0 1.8000000e+00 2.9999996e-01 0.3 3.3 0.0000000e+00', 
            '53122:new = 53122 -53122 0.5 0 0 1.8100000e+00 1.5000001e-01 1.06 2.56 0.0000000e+00', 
            '13126:new = 13126 -13126 2.5 0 0 1.8300000e+00 9.5000007e-02 1.355 2.305 0.0000000e+00', 
            '13212:new = 13212 -13212 0.5 0 0 1.6600000e+00 1.0000000e-01 1.16 2.16 0.0000000e+00', 
            '3126:new = 3126 -3126 2.5 0 0 1.8200000e+00 7.9999995e-02 1.42 2.22 0.0000000e+00', 
            '3216:new = 3216 -3216 2.5 0 0 1.7750000e+00 1.1999999e-01 1.175 2.375 0.0000000e+00', 
            '14124:new = 14124 -14124 2.5 1 0 2.626600 0 2.626600 2.626600 0.0000000e+00'
        ),
        pythia8CP5Settings = cms.vstring(
            'Tune:pp 14', 
            'Tune:ee 7', 
            'MultipartonInteractions:ecmPow=0.03344', 
            'MultipartonInteractions:bProfile=2', 
            'MultipartonInteractions:pT0Ref=1.41', 
            'MultipartonInteractions:coreRadius=0.7634', 
            'MultipartonInteractions:coreFraction=0.63', 
            'ColourReconnection:range=5.176', 
            'SigmaTotal:zeroAXB=off', 
            'SpaceShower:alphaSorder=2', 
            'SpaceShower:alphaSvalue=0.118', 
            'SigmaProcess:alphaSvalue=0.118', 
            'SigmaProcess:alphaSorder=2', 
            'MultipartonInteractions:alphaSvalue=0.118', 
            'MultipartonInteractions:alphaSorder=2', 
            'TimeShower:alphaSorder=2', 
            'TimeShower:alphaSvalue=0.118', 
            'SigmaTotal:mode = 0', 
            'SigmaTotal:sigmaEl = 21.89', 
            'SigmaTotal:sigmaTot = 100.309', 
            'PDF:pSet=LHAPDF6:NNPDF31_nnlo_as_0118'
        ),
        pythia8CommonSettings = cms.vstring(
            'Tune:preferLHAPDF = 2', 
            'Main:timesAllowErrors = 10000', 
            'Check:epTolErr = 0.01', 
            'Beams:setProductionScalesFromLHEF = off', 
            'SLHA:keepSM = on', 
            'SLHA:minMassSM = 1000.', 
            'ParticleDecays:limitTau0 = on', 
            'ParticleDecays:tau0Max = 10', 
            'ParticleDecays:allowPhotonRadiation = on'
        )
    ),
    comEnergy = cms.double(13000.0),
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    pythiaPylistVerbosity = cms.untracked.int32(1)
)


process.mu_mu_same_charge_filter = cms.EDFilter("MCParticlePairFilter",
    MaxEta = cms.untracked.vdouble(2.52, 2.52),
    MaxInvMass = cms.untracked.double(10.0),
    MinEta = cms.untracked.vdouble(-2.52, -2.52),
    MinPt = cms.untracked.vdouble(2.8, 2.0),
    ParticleCharge = cms.untracked.int32(-1),
    Status = cms.untracked.vint32(1, 1),
    particleID1 = cms.untracked.vint32(13),
    particleID2 = cms.untracked.vint32(13)
)


process.ProductionFilterSequence = cms.Sequence(process.generator+process.jpsi_from_b_hadron_filter+process.three_mu_filter+process.mu_mu_same_charge_filter)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.endjob_step,process.RAWSIMoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path).insert(0, process.ProductionFilterSequence)

# customisation of the process.

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring 

#call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)

# End of customisation functions

# Customisation from command line


# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

from IOMC.RandomEngine.RandomServiceHelper import  RandomNumberServiceHelper
randHelper =  RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randHelper.populate()
process.RandomNumberGeneratorService.saveFileName =  cms.untracked.string("RandomEngineState.log")
