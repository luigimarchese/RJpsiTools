import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    comEnergy = cms.double(13000.0),

    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            'SoftQCD:nonDiffractive = on',
			'PTFilter:filter = on', # this turn on the filter
            'PTFilter:quarkToFilter = 5', # PDG id of q quark
            'PTFilter:scaleToFilter = 1.0',
        ),
        parameterSets = cms.vstring(
            'pythia8CommonSettings',
            'pythia8CP5Settings',
            'processParameters',
        ),
    ),

    ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
            decay_table            = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2014_NOLONGLIFE.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt_2014.pdl'),
            list_forced_decays     = cms.vstring(
                'MyB+'           ,
                'MyB-'           ,
            ),        
            operates_on_particles  = cms.vint32(),    
            convertPythiaCodes     = cms.untracked.bool(False),
            user_decay_file         = cms.vstring('GeneratorInterface/ExternalDecays/data/BToJpsiMuMuInclusive.dec'),
        ),
        parameterSets = cms.vstring('EvtGen130'),
    ),

)

jpsi_from_b_hadron_filter = cms.EDFilter(
    "PythiaFilterMultiAncestor",
    ParticleID      = cms.untracked.int32 (443),
    MinPt           = cms.untracked.double(-1.),
    MinEta          = cms.untracked.double(-1.e4),
    MaxEta          = cms.untracked.double( 1.e4),
    MotherIDs       = cms.untracked.vint32([5]),
    DaughterIDs     = cms.untracked.vint32([-13, 13]),
    DaughterMinPts  = cms.untracked.vdouble([-1.  , -1.  ]),
    DaughterMaxPts  = cms.untracked.vdouble([ 1.e6,  1.e6]),
    DaughterMinEtas = cms.untracked.vdouble([-1.e4, -1.e4]),
    DaughterMaxEtas = cms.untracked.vdouble([ 1.e4,  1.e4]),
)

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('\$Revision$'),
    name = cms.untracked.string('\$Source$'),
    annotation = cms.untracked.string(
        'QCD bbbar production, '\
        'Jpsi (no kin cuts) from B+ hadron (either directly or feeddown), '\
        'Jpsi->mumu (no  kin cuts on muons), '\
        '13 TeV, '\
        'TuneCP5'
    )
)

ProductionFilterSequence = cms.Sequence(generator*jpsi_from_b_hadron_filter)
