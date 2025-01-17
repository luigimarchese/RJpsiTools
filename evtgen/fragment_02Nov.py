import FWCore.ParameterSet.Config as cms

from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *
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
            'PTFilter:scaleToFilter = 2.0', 
        ),
        parameterSets = cms.vstring(
            'pythia8CommonSettings',
            'pythia8CP5Settings',
            'processParameters',
        ),
    ),

    ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2014_NOLONGLIFE.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt_2014.pdl'),
            list_forced_decays     = cms.vstring(
                'MyB+', #521
                'MyB-',
                'MyBc+', #541
                'MyBc-', 
                'Myanti-B0', #511
                'MyB0',
                'Myanti-Bs', #531
                'MyBs',
                'MyLambda_b0', #5122
                'MyXi_b-',     #5132
                'Myanti-Xi_b+',
                'MyXi_b0',     #5232
                'Myanti-Xi_b0',
                'MyOmega_b-',  #5332
                'Myanti-Omega_b+',
                'MyBs*', #533
                'Myanti-Bs*',
                'MyB*0', #513
                'Myanti-B*0',
                'MyB*+', #523
                'MyB*-'
            ),        
            operates_on_particles = cms.vint32(521,-521,541,-541,511,-511,531,-531,5122,5132,-5132,5232,-5232,5332,-5332,533,-533,513,-513,523,-523),
            convertPythiaCodes = cms.untracked.bool(False),

            user_decay_embedded = cms.vstring(['Alias      MyJ/psi          J/psi', 'Alias      Mypsi(2S)        psi(2S)', 'Alias      Mypsi(3770)      psi(3770)', 'Alias      Mychi_c0         chi_c0', 'Alias      Mychi_c1         chi_c1', 'Alias      Mychi_c2         chi_c2', 'Alias      Myh_c            h_c', 'Alias      MyB+             B+', 'Alias      MyB-             B-', 'Alias      Myanti-B0        anti-B0', 'Alias      MyB0             B0', 'Alias      Myanti-Bs        anti-B_s0', 'Alias      MyBs             B_s0', 'Alias      MyBc+            B_c+', 'Alias      MyBc-            B_c-', 'Alias      MyB*+            B*+', 'Alias      MyB*-            B*-', 'Alias      MyB*0            B*0', 'Alias      Myanti-B*0       anti-B*0', 'Alias      MyBs*            B_s*0', 'Alias      Myanti-Bs*       anti-B_s*0', 'Alias      MyLambda_b0      Lambda_b0', 'Alias      MyXi_b-          Xi_b-', 'Alias      Myanti-Xi_b+     anti-Xi_b+', 'Alias      MyXi_b0          Xi_b0', 'Alias      Myanti-Xi_b0     anti-Xi_b0', 'Alias      MyOmega_b-       Omega_b-', 'Alias      Myanti-Omega_b+  anti-Omega_b+', 'ChargeConj MyB-             MyB+', 'ChargeConj Myanti-B0        MyB0', 'ChargeConj Myanti-Bs        MyBs', 'ChargeConj MyBc-            MyBc+', 'ChargeConj MyB*-            MyB*+', 'ChargeConj MyB*0            Myanti-B*0', 'ChargeConj MyBs*            Myanti-Bs*', 'ChargeConj MyXi_b-          Myanti-Xi_b+', 'ChargeConj MyXi_b0          Myanti-Xi_b0', 'ChargeConj MyOmega_b-       Myanti-Omega_b+', '', '', 'Decay MyJ/psi  # original total forced BR = 0.05960000', '1.00000000 mu+ mu- PHOTOS  VLL;', 'Enddecay', '', '', 'Decay Mychi_c0  # original total forced BR = 0.01400000', '1.00000000 gamma MyJ/psi PHSP;', 'Enddecay', '', '', 'Decay Mychi_c1  # original total forced BR = 0.34400000', '1.00000000 MyJ/psi gamma VVP 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;', 'Enddecay', '', '', 'Decay Mychi_c2  # original total forced BR = 0.19500000', '1.00000000 gamma MyJ/psi PHSP;', 'Enddecay', '', '', 'Decay Mypsi(2S)  # original total forced BR = 0.85560000', '0.56145341 MyJ/psi pi+ pi- VVPIPI;', '0.29529729 MyJ/psi pi0 pi0 VVPIPI;', '0.05455877 MyJ/psi eta PARTWAVE 0.0 0.0 1.0 0.0 0.0 0.0;', '0.00210464 MyJ/psi pi0 PARTWAVE 0.0 0.0 1.0 0.0 0.0 0.0;', '0.00221894 gamma Mychi_c0 PHSP;', '0.05429973 gamma Mychi_c1 PHSP;', '0.03005427 gamma Mychi_c2 PHSP;', '0.00001295 Myh_c gamma PHSP;', 'Enddecay', '', '', 'Decay Mypsi(3770)  # original total forced BR = 0.01302000', '0.42110683 MyJ/psi pi+ pi- PHSP;', '0.17455206 MyJ/psi pi0 pi0 PHSP;', '0.19637106 MyJ/psi eta PHSP;', '0.00000000 MyJ/psi pi0 PARTWAVE 0.0 0.0 1.0 0.0 0.0 0.0;', '0.02107716 Mychi_c0 gamma PHSP;', '0.18689289 Mychi_c1 gamma VVP 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;', '0.00000000 Mychi_c2 gamma PHSP;', 'Enddecay', '', '', 'Decay Myh_c  # original total forced BR = 0.01000000', '1.00000000 MyJ/psi pi0 PHSP;', 'Enddecay', '', '', 'Decay MyB+  # original total forced BR = 0.01814100', '0.09387850 MyJ/psi K+ SVS;', '0.13161398 MyJ/psi K*+ SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;', '0.00360788 MyJ/psi pi+ SVS;', '0.00377355 MyJ/psi rho+ SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;', '0.10492303 MyJ/psi K0 pi+ PHSP;', '0.00920377 MyJ/psi K+ pi0 PHSP;', "0.00920377 MyJ/psi K'_1+ SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0;", '0.04601887 MyJ/psi K_2*+ PHSP;', '0.16566794 MyJ/psi K_1+ SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0;', '0.00460189 MyJ/psi phi K+ PHSP;', '0.00460189 MyJ/psi phi K*+ PHSP;', '0.07455057 MyJ/psi K+ pi+ pi- PHSP;', '0.01141268 MyJ/psi eta K+ PHSP;', '0.02945208 MyJ/psi omega K+ PHSP;', '0.00134375 MyJ/psi p+ anti-Lambda0 PHSP;', '0.00310167 MyJ/psi K+ K- K+ PHSP;', '0.00151862 MyJ/psi K+ pi0 pi0 PHSP;', '0.00423374 MyJ/psi K+ pi0 eta PHSP;', "0.00097560 MyJ/psi K+ pi0 eta' PHSP;", "0.00023009 MyJ/psi K+ eta eta' PHSP;", '0.00083478 MyJ/psi K0 pi+ eta PHSP;', "0.00190518 MyJ/psi K0 pi+ eta' PHSP;", '0.03547447 Mypsi(2S) K+ SVS;', '0.03808958 Mypsi(2S) K*+ SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;', '0.02274005 Mypsi(2S) K0 pi+ PHSP;', '0.01137002 Mypsi(2S) K+ pi0 PHSP;', '0.02444555 Mypsi(2S) K+ pi- pi+ PHSP;', '0.00568501 Mypsi(2S) K+ pi0 pi0 PHSP;', '0.00568501 Mypsi(2S) K0 pi+ pi0 PHSP;', '0.02274005 Mypsi(2S) K_1+ PHSP;', '0.00138714 Mypsi(2S) pi+ PHSP;', "0.00773162 Mypsi(2S) K'_1+ SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0;", '0.00682201 Mypsi(2S) K_2*+ PHSP;', '0.00084138 Mypsi(2S) eta K+ PHSP;', '0.00049971 Mypsi(2S) omega K+ PHSP;', '0.00022740 Mypsi(2S) phi K+ PHSP;', '0.00132461 Mypsi(2S) rho+ SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;', '0.00018138 Mypsi(3770) K+ SVS;', '0.00021091 Mypsi(3770) K*+ PHSP;', '0.00012655 Mypsi(3770) K0 pi+ PHSP;', '0.00008436 Mypsi(3770) K+ pi0 PHSP;', '0.00008436 Mypsi(3770) K+ pi- pi+ PHSP;', '0.00004218 Mypsi(3770) K+ pi0 pi0 PHSP;', '0.00004218 Mypsi(3770) K0 pi+ pi0 PHSP;', '0.00012655 Mypsi(3770) K_1+ PHSP;', '0.00019457 Mychi_c0 K+ PHSP;', '0.00051541 K*+ Mychi_c0 SVS;', '0.00025771 Mychi_c0 K0 pi+ PHSP;', '0.00012885 Mychi_c0 K+ pi0 PHSP;', '0.00025771 Mychi_c0 K+ pi- pi+ PHSP;', '0.00012885 Mychi_c0 K+ pi0 pi0 PHSP;', '0.00012885 Mychi_c0 K0 pi+ pi0 PHSP;', '0.00030538 Mychi_c0 K_1+ PHSP;', '0.01500731 Mychi_c1 K+ SVS;', '0.00949830 Mychi_c1 K*+ SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;', '0.01836337 Mychi_c1 K0 pi+ PHSP;', '0.01041646 Mychi_c1 K+ pi0 PHSP;', '0.01184121 Mychi_c1 K+ pi- pi+ PHSP;', '0.00633220 Mychi_c1 K+ pi0 pi0 PHSP;', '0.00633220 Mychi_c1 K0 pi+ pi0 PHSP;', '0.00069654 Mychi_c1 pi+ PHSP;', '0.01073307 Mychi_c1 K_1+ SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0;', "0.00236824 Mychi_c1 K'_1+ SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0;", '0.00217511 Mychi_c1 K_2*+ PHSP;', '0.00033244 Mychi_c1 K+ eta PHSP;', '0.00027513 Mychi_c1 K+ omega PHSP;', '0.00004591 Mychi_c1 K+ phi PHSP;', '0.00032927 Mychi_c1 rho+ SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;', '0.00019742 Mychi_c2 K+ STS;', '0.00035895 Mychi_c2 K*+ PHSP;', '0.00208189 Mychi_c2 K0 pi+ PHSP;', '0.00179474 Mychi_c2 K+ pi0 PHSP;', '0.00240495 Mychi_c2 K+ pi- pi+ PHSP;', '0.00179474 Mychi_c2 K+ pi0 pi0 PHSP;', '0.00179474 Mychi_c2 K0 pi+ pi0 PHSP;', '0.00029218 Mychi_c2 K_1+ PHSP;', '0.00000036 Mychi_c2 K+ phi PHSP;', 'Enddecay', 'CDecay MyB-', '', '', 'Decay Myanti-B0  # original total forced BR = 0.02562818', '0.05341780 MyJ/psi anti-K0 PHSP;', '0.01378911 MyJ/psi omega anti-K0 PHSP;', '0.00064749 MyJ/psi eta PHSP;', '0.00239811 MyJ/psi pi- pi+ PHSP;', '0.02697869 MyJ/psi anti-K0 pi- pi+ PHSP;', '0.03237442 MyJ/psi anti-K0 rho0 PHSP;', '0.04796211 MyJ/psi K*- pi+ PHSP;', '0.03956874 MyJ/psi anti-K*0 pi- pi+ PHSP;', '0.02610937 MyJ/psi K_S0 SVS;', '0.02610937 MyJ/psi K_L0 SVS;', '0.07613985 MyJ/psi anti-K*0 SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus;', '0.00099521 MyJ/psi pi0 SVS;', '0.00064749 MyJ/psi rho0 SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus;', '0.00107915 MyJ/psi omega SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus;', '0.06894553 MyJ/psi K- pi+ PHSP;', '0.00599526 MyJ/psi anti-K0 pi0 PHSP;', '0.07793842 MyJ/psi anti-K_10 SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0;', "0.00599526 MyJ/psi anti-K'_10 SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0;", '0.02997632 MyJ/psi anti-K_2*0 PHSP;', '0.00293768 MyJ/psi phi anti-K0 PHSP;', '0.00293768 MyJ/psi phi anti-K*0 PHSP;', '0.06810619 MyJ/psi K- rho+ PHSP;', '0.00959242 MyJ/psi K0 eta PHSP;', '0.01378911 MyJ/psi K0 omega PHSP;', '0.00038969 MyJ/psi sigma_0 SVS;', '0.00025180 MyJ/psi f_2 PHSP;', '0.00012590 MyJ/psi rho(2S)0 PHSP;', '0.00317149 MyJ/psi anti-K0 K- K+ PHSP;', '0.00086931 MyJ/psi anti-K0 pi0 pi0 PHSP;', '0.00164870 MyJ/psi anti-K0 anti-K0 K0 PHSP;', '0.00022782 MyJ/psi anti-K0 pi0 eta PHSP;', "0.00052758 MyJ/psi anti-K0 pi0 eta' PHSP;", '0.00013190 MyJ/psi anti-K0 eta eta PHSP;', '0.00046763 MyJ/psi K- pi+ eta PHSP;', "0.00107915 MyJ/psi K- pi+ eta' PHSP;", '0.02147839 Mypsi(2S) anti-K0 PHSP;', '0.01147983 Mypsi(2S) K_S0 SVS;', '0.01147983 Mypsi(2S) K_L0 SVS;', '0.02184870 Mypsi(2S) anti-K*0 SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus;', '0.02147839 Mypsi(2S) K- pi+ PHSP;', '0.00740634 Mypsi(2S) anti-K0 pi0 PHSP;', '0.00740634 Mypsi(2S) anti-K0 pi+ pi- PHSP;', '0.00370317 Mypsi(2S) anti-K0 pi0 pi0 PHSP;', '0.00370317 Mypsi(2S) K- pi+ pi0 PHSP;', '0.01481268 Mypsi(2S) anti-K_10 PHSP;', '0.03003271 Mypsi(2S) K*- pi+ PHSP;', '0.02573703 Mypsi(2S) anti-K*0 pi+ pi- PHSP;', "0.01140576 Mypsi(2S) anti-K'_10 SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0 ;", '0.01066513 Mypsi(2S) anti-K_2*0 PHSP ;', '0.02103401 Mypsi(2S) anti-K0 rho0 PHSP;', '0.04206801 Mypsi(2S) K- rho+ PHSP;', '0.00592507 Mypsi(2S) anti-K0 eta PHSP;', '0.00014813 Mypsi(2S) anti-K0 phi PHSP;', '0.00062954 Mypsi(2S) pi0 PHSP;', '0.00035180 Mypsi(2S) eta PHSP;', '0.00099986 Mypsi(2S) rho0 SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;', '0.00024071 Mypsi(2S) sigma_0 SVS;', '0.00015553 Mypsi(2S) f_2 PHSP;', '0.00006595 Mypsi(3770) K_S0 SVS;', '0.00006595 Mypsi(3770) K_L0 SVS;', '0.00013189 Mypsi(3770) anti-K*0 SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;', '0.00003847 Mypsi(3770) K- pi+ PHSP;', '0.00003847 Mypsi(3770) anti-K0 pi0 PHSP;', '0.00003847 Mypsi(3770) anti-K0 pi+ pi- PHSP;', '0.00001923 Mypsi(3770) anti-K0 pi0 pi0 PHSP;', '0.00001923 Mypsi(3770) K- pi+ pi0 PHSP;', '0.00007968 Mypsi(3770) anti-K_10 PHSP;', '0.00005875 Mychi_c0 K_S0 PHSP;', '0.00005875 Mychi_c0 K_L0 PHSP;', '0.00025180 anti-K*0 Mychi_c0 SVS;', '0.00016787 Mychi_c0 K- pi+ PHSP;', '0.00008393 Mychi_c0 anti-K0 pi0 PHSP;', '0.00016787 Mychi_c0 anti-K0 pi+ pi- PHSP;', '0.00008393 Mychi_c0 anti-K0 pi0 pi0 PHSP;', '0.00008393 Mychi_c0 K- pi+ pi0 PHSP;', '0.00011751 Mychi_c0 anti-K0 PHSP;', '0.00012590 Mychi_c0 anti-K_10 PHSP;', '0.00402162 Mychi_c1 K_S0 SVS;', '0.00402162 Mychi_c1 K_L0 SVS;', '0.00490844 Mychi_c1 anti-K*0 SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus;', '0.01024998 Mychi_c1 K- pi+ PHSP;', '0.00412474 Mychi_c1 anti-K0 pi0 PHSP;', '0.00659959 Mychi_c1 anti-K0 pi+ pi- PHSP;', '0.00721830 Mychi_c1 anti-K0 pi0 pi0 PHSP;', '0.00412474 Mychi_c1 K- pi+ pi0 PHSP;', '0.00023099 Mychi_c1 pi0 PHSP;', '0.00804325 Mychi_c1 anti-K0 PHSP;', '0.00325855 Mychi_c1 K+ pi- PHSP;', '0.00125392 Mychi_c1 K*- pi+ PHSP;', '0.00038154 Mychi_c1 anti-K*0 pi+ pi- PHSP;', '0.00398038 Mychi_c1 anti-K_10 SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0 ;', "0.00398038 Mychi_c1 anti-K'_10 SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0 ;", '0.00081257 Mychi_c1 anti-K_2*0 PHSP ;', '0.00052797 Mychi_c1 anti-K0 rho0 PHSP ;', '0.00106418 Mychi_c1 K- rho+ PHSP ;', '0.00255734 Mychi_c1 anti-K0 phi PHSP ;', '0.00026605 Mychi_c1 anti-K0 omega PHSP ;', '0.00009239 Mychi_c1 rho0 SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;', '0.00058454 Mychi_c2 K_S0 STS;', '0.00058454 Mychi_c2 K_L0 STS;', '0.00057285 Mychi_c2 anti-K*0 PHSP;', '0.00084173 Mychi_c2 K- pi+ PHSP;', '0.00116908 Mychi_c2 anti-K0 pi0 PHSP;', '0.00233815 Mychi_c2 anti-K0 pi+ pi- PHSP;', '0.00116908 Mychi_c2 anti-K0 pi0 pi0 PHSP;', '0.00116908 Mychi_c2 K- pi+ pi0 PHSP;', '0.00018705 Mychi_c2 K*- pi+ PHSP;', '0.00006664 Mychi_c2 anti-K_10 PHSP;', "0.00014380 Mychi_c2 anti-K'_10 PHSP;", '0.00013094 Mychi_c2 anti-K_2*0 PHSP;', '0.00007131 Mychi_c2 anti-K0 rho0 PHSP;', '0.00001403 Mychi_c2 K- rho+ PHSP;', '0.00002923 Mychi_c2 anti-K0 phi PHSP ;', 'Enddecay', 'CDecay MyB0', '', '', 'Decay MyBs  # original total forced BR = 0.02191459', "0.02478605 MyJ/psi eta' SVS;", '0.03004370 MyJ/psi eta SVS;', '0.08111799 MyJ/psi phi SVV_HELAMP  1.0 0.0 1.0 0.0 1.0 0.0;', '0.00093135 MyJ/psi phi phi PHSP;', '0.00600874 MyJ/psi K0 SVS;', '0.05933631 MyJ/psi K- K+ PHSP;', '0.05257647 MyJ/psi anti-K0 K0 PHSP;', '0.07135378 MyJ/psi K0 K- pi+ PHSP;', '0.05257647 MyJ/psi anti-K0 K0 pi0 PHSP;', '0.05257647 MyJ/psi K- K+ pi0 PHSP;', '0.02929261 MyJ/psi phi pi+ pi- PHSP;', '0.02929261 MyJ/psi phi pi0 pi0 PHSP;', '0.01502185 MyJ/psi eta pi+ pi- PHSP;', '0.01502185 MyJ/psi eta pi0 pi0 PHSP;', "0.03004370 MyJ/psi eta' pi+ pi- PHSP;", "0.03004370 MyJ/psi eta' pi0 pi0 PHSP;", '0.01569783 MyJ/psi pi+ pi- PHSP;', '0.01502185 MyJ/psi pi0 pi0 PHSP;', '0.00345503 MyJ/psi K*0 SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;', '0.01021486 MyJ/psi f_0 SVS;', "0.00307948 MyJ/psi f'_0 SVS;", '0.00007811 MyJ/psi f_2 PHSP;', "0.02057993 MyJ/psi f'_2 PHSP;", '0.00039733 MyJ/psi anti-K0 K+ pi- PHSP;', "0.00598478 Mypsi(2S) eta' SVS;", '0.01530991 Mypsi(2S) eta SVS;', '0.02505258 Mypsi(2S) phi SVV_HELAMP  1.0 0.0 1.0 0.0 1.0 0.0;', '0.01391810 Mypsi(2S) K- K+ PHSP;', '0.01391810 Mypsi(2S) anti-K0 K0 PHSP;', '0.01391810 Mypsi(2S) K0 K- pi+ PHSP;', '0.01391810 Mypsi(2S) anti-K0 K0 pi0 PHSP;', '0.01391810 Mypsi(2S) K- K+ pi0 PHSP;', '0.01577385 Mypsi(2S) phi pi+ pi- PHSP;', '0.01577385 Mypsi(2S) phi pi0 pi0 PHSP;', '0.00927874 Mypsi(2S) eta pi+ pi- PHSP;', '0.00927874 Mypsi(2S) eta pi0 pi0 PHSP;', "0.01855747 Mypsi(2S) eta' pi+ pi- PHSP;", "0.01855747 Mypsi(2S) eta' pi0 pi0 PHSP;", '0.00329395 Mypsi(2S) pi+ pi- PHSP;', '0.00927874 Mypsi(2S) pi0 pi0 PHSP;', '0.00113665 Mypsi(2S) K0 SVS;', '0.00111809 Mypsi(2S) K*0 SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus;', '0.00320116 Mypsi(2S) f_0 SVS;', "0.00078405 Mypsi(2S) f'_0 SVS;", '0.00002320 Mypsi(2S) f_2 PHSP;', "0.00409099 Mypsi(2S) f'_2 PHSP;", '0.00012434 Mypsi(2S) anti-K0 K+ pi- PHSP;', "0.00010515 Mychi_c0 eta' PHSP;", '0.00005258 Mychi_c0 eta PHSP;', '0.00021031 phi Mychi_c0 SVS;', '0.00003155 Mychi_c0 K- K+ PHSP;', '0.00003155 Mychi_c0 anti-K0 K0 PHSP;', '0.00003155 Mychi_c0 K0 K- pi+ PHSP;', '0.00003155 Mychi_c0 anti-K0 K0 pi0 PHSP;', '0.00003155 Mychi_c0 K- K+ pi0 PHSP;', "0.01808631 Mychi_c1 eta' SVS;", '0.00775127 Mychi_c1 eta SVS;', '0.00527087 Mychi_c1 phi SVV_HELAMP  1.0 0.0 1.0 0.0 1.0 0.0;', '0.00671777 Mychi_c1 K- K+ PHSP;', '0.00671777 Mychi_c1 anti-K0 K0 PHSP;', '0.00671777 Mychi_c1 K0 K- pi+ PHSP;', '0.00671777 Mychi_c1 anti-K0 K0 pi0 PHSP;', '0.00671777 Mychi_c1 K- K+ pi0 PHSP;', '0.01033503 Mychi_c1 phi pi+ pi- PHSP;', '0.01033503 Mychi_c1 phi pi0 pi0 PHSP;', '0.00258376 Mychi_c1 eta pi+ pi- PHSP;', '0.00258376 Mychi_c1 eta pi0 pi0 PHSP;', "0.00516752 Mychi_c1 eta' pi+ pi- PHSP;", "0.00516752 Mychi_c1 eta' pi0 pi0 PHSP;", "0.00681053 Mychi_c2 eta' STS;", '0.00344188 Mychi_c2 eta STS;', '0.00234341 Mychi_c2 K- K+ PHSP;', '0.00234341 Mychi_c2 anti-K0 K0 PHSP;', '0.00234341 Mychi_c2 K0 K- pi+ PHSP;', '0.00234341 Mychi_c2 anti-K0 K0 pi0 PHSP;', '0.00234341 Mychi_c2 K- K+ pi0 PHSP;', "0.00034926 Myh_c eta' SVS;", '0.00017651 Myh_c eta SVS;', '0.00075109 Myh_c phi SVV_HELAMP  1.0 0.0 1.0 0.0 1.0 0.0;', '0.00012017 Myh_c K- K+ PHSP;', '0.00012017 Myh_c anti-K0 K0 PHSP;', '0.00012017 Myh_c K0 K- pi+ PHSP;', '0.00012017 Myh_c anti-K0 K0 pi0 PHSP;', '0.00012017 Myh_c K- K+ pi0 PHSP;', 'Enddecay', 'CDecay Myanti-Bs', '', '', 'Decay MyB*+  # original total forced BR = 1.00000000', '1.00000000 MyB+ gamma VSP_PWAVE;', 'Enddecay', 'CDecay MyB*-', '', '', 'Decay MyB*0  # original total forced BR = 1.00000000', '1.00000000 MyB0 gamma VSP_PWAVE;', 'Enddecay', 'CDecay Myanti-B*0', '', '', 'Decay MyBs*  # original total forced BR = 1.00000000', '1.00000000 MyBs gamma VSP_PWAVE;', 'Enddecay', 'CDecay Myanti-Bs*', '', '', 'Decay MyBc+  # original total forced BR = 0.91150320', '0.01415519 MyJ/psi mu+ nu_mu PHOTOS BC_VMN 1;', '0.00043257 Mypsi(2S) mu+ nu_mu PHOTOS BC_VMN 1;', '0.00002146 Mychi_c0 mu+ nu_mu PHOTOS BC_SMN 3;', '0.00052735 Mychi_c1 mu+ nu_mu PHOTOS BC_VMN 3;', '0.00029894 Mychi_c2 mu+ nu_mu PHOTOS BC_TMN 3;', '0.00002180 Myh_c mu+ nu_mu PHOTOS PHSP;', '0.00039974 MyBs mu+ nu_mu PHOTOS PHSP;', '0.00050190 MyBs* mu+ nu_mu PHOTOS PHSP;', '0.04250056 MyB0 mu+ nu_mu PHOTOS PHSP;', '0.07250095 MyB*0  mu+ nu_mu PHOTOS PHSP;', '0.01415519 MyJ/psi e+ nu_e PHOTOS BC_VMN 1;', '0.00043257 Mypsi(2S) e+ nu_e PHOTOS BC_VMN 1;', '0.00002146 Mychi_c0 e+ nu_e PHOTOS BC_SMN 3;', '0.00052735 Mychi_c1 e+ nu_e PHOTOS BC_VMN 3;', '0.00029894 Mychi_c2 e+ nu_e PHOTOS BC_TMN 3;', '0.00002180 Myh_c e+ nu_e PHOTOS PHSP;', '0.00039974 MyBs e+ nu_e PHOTOS PHSP;', '0.00050190 MyBs* e+ nu_e PHOTOS PHSP;', '0.04250056 MyB0 e+ nu_e PHOTOS PHSP;', '0.07250095 MyB*0  e+ nu_e PHOTOS PHSP;', '0.00357605 MyJ/psi tau+ nu_tau PHOTOS BC_VMN 1;', '0.00003681 Mypsi(2S) tau+ nu_tau PHOTOS BC_VMN 1;', '0.00000183 Mychi_c0 tau+ nu_tau PHOTOS BC_SMN 3;', '0.00004485 Mychi_c1 tau+ nu_tau PHOTOS BC_VMN 3;', '0.00002542 Mychi_c2 tau+ nu_tau PHOTOS BC_TMN 3;', '0.00096851 MyJ/psi pi+ SVS;', '0.00298004 MyJ/psi rho+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;', '0.00008195 MyJ/psi K+ SVS;', '0.00016390 MyJ/psi K*+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;', '0.00126652 MyJ/psi D_s+ SVS;', '0.00499157 MyJ/psi D_s*+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;', '0.00006705 MyJ/psi D+ SVS;', '0.00020860 MyJ/psi D*+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;', '0.00162672 MyBs pi+ PHSP;', '0.00071417 rho+ MyBs SVS;', '0.00010514 MyBs K+ PHSP;', '0.00000000 K*+ MyBs SVS;', '0.00064474 MyBs* pi+ SVS;', '0.00200364 MyBs* rho+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;', '0.00003670 MyBs* K+ SVS;', '0.00000000 MyBs* K*+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;', '0.13250174 MyB0 pi+ PHSP;', '0.12000157 rho+ MyB0 SVS;', '0.00875011 MyB0 K+ PHSP;', '0.00187502 K*+ MyB0 SVS;', '0.11875156 MyB*0  pi+ SVS;', '0.32125421 MyB*0  rho+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;', '0.00687509 MyB*0  K+ SVS;', '0.00725010 MyB*0  K*+ SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;', '0.00000300 MyB+ pi0 PHSP;', '0.00000275 rho0 MyB+ SVS;', '0.00016027 MyB+ anti-K0 PHSP;', '0.00003481 K*0 MyB+ SVS;', '0.00000267 MyB*+ pi0 SVS;', '0.00000729 MyB*+ rho0 SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;', '0.00012951 MyB*+ anti-K0 SVS;', '0.00013518 MyB*+ K*0 SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0;', 'Enddecay', 'CDecay MyBc-', '', '', 'Decay MyLambda_b0  # original total forced BR = 0.00260980', '0.23443612 Lambda0 MyJ/psi PHSP;', '0.11707809 Lambda0 Mypsi(2S) PHSP;', '0.15961608 MyJ/psi p+ K- PHSP;', '0.23443612 Lambda(1520)0 MyJ/psi PHSP;', '0.11707809 Lambda(1520)0 Mypsi(2S) PHSP;', '0.01566233 Lambda0 MyJ/psi phi PHSP;', '0.01316833 MyJ/psi p+ pi- PHSP;', '0.03331986 MyJ/psi p+ K- pi- pi+ PHSP;', '0.02039624 Mypsi(2S) p+ K- PHSP;', '0.00231075 Mypsi(2S) p+ pi- PHSP;', '0.01312643 Mychi_c1 p+ K- PHSP;', '0.00778128 Mychi_c2 p+ K- PHSP;', '0.02016151 Mychi_c1 Lambda0 PHSP;', '0.01142876 Mychi_c2 Lambda0 PHSP;', 'Enddecay', '', '', 'Decay MyXi_b-  # original total forced BR = 0.00165500', '0.39190936 Xi- MyJ/psi PHSP;', '0.19572069 Xi- Mypsi(2S) PHSP;', '0.31686289 Lambda0 K- MyJ/psi PHSP;', '0.02725021 Lambda0 K- Mychi_c1 PHSP;', '0.01544707 Lambda0 K- Mychi_c2 PHSP;', '0.03370420 Xi- Mychi_c1 PHSP;', '0.01910558 Xi- Mychi_c2 PHSP;', 'Enddecay', 'CDecay Myanti-Xi_b+', '', '', 'Decay MyXi_b0  # original total forced BR = 0.00129250', '0.49727338 Xi0 MyJ/psi PHSP;', '0.15357856 Xi0 Mypsi(2S) PHSP;', '0.24863669 Sigma+ K- MyJ/psi PHSP;', '0.02138276 Sigma+ K- Mychi_c1 PHSP;', '0.01212104 Sigma+ K- Mychi_c2 PHSP;', '0.04276551 Xi0 Mychi_c1 PHSP;', '0.02424208 Xi0 Mychi_c2 PHSP;', 'Enddecay', 'CDecay Myanti-Xi_b0', '', '', 'Decay MyOmega_b-  # original total forced BR = 0.00108500', '0.61193782 Omega- MyJ/psi PHSP;', '0.30560356 Omega- Mypsi(2S) PHSP;', '0.05262665 Omega- Mychi_c1 PHSP;', '0.02983197 Omega- Mychi_c2 PHSP;', 'Enddecay', 'CDecay Myanti-Omega_b+', '', 'End']),
        ),
        parameterSets = cms.vstring('EvtGen130'),
    ),

)

generator.PythiaParameters.processParameters.extend(EvtGenExtraParticles)

jpsi_from_b_hadron_filter = cms.EDFilter(
    "PythiaFilterMultiAncestor",
    ParticleID      = cms.untracked.int32 (443),
    MinPt           = cms.untracked.double(0.),
    MinEta          = cms.untracked.double(-100.),
    MaxEta          = cms.untracked.double( 100.),
    MotherIDs       = cms.untracked.vint32([5]),
    DaughterIDs     = cms.untracked.vint32([-13, 13]),
    DaughterMinPts  = cms.untracked.vdouble([ 3.2 , 3.2  ]), 
    DaughterMaxPts  = cms.untracked.vdouble([ 1.e6,  1.e6]),
    DaughterMinEtas = cms.untracked.vdouble([-2.52 , -2.52 ]),
    DaughterMaxEtas = cms.untracked.vdouble([ 2.52 ,  2.52 ]),
)

three_mu_filter = cms.EDFilter(
    "MCMultiParticleFilter",
    NumRequired = cms.int32(3),
    AcceptMore  = cms.bool(True),
    ParticleID  = cms.vint32(13,13,13),
    PtMin       = cms.vdouble(3.2, 3.2, 3.2),
    EtaMax      = cms.vdouble(2.52, 2.52, 2.52),
    Status      = cms.vint32(1, 1, 1),
)


# two OS muons make the Jpsi, so, if  there's an additional muon around
# it must have the same charge as that of to one of the Jpsi muons.
# In addition, the invariant mass of these two can't be too large 
mu_mu_same_charge_filter = cms.EDFilter(
    "MCParticlePairFilter",
    ParticleID1    = cms.untracked.vint32(13), # mu
    ParticleID2    = cms.untracked.vint32(13), # mu
    ParticleCharge = cms.untracked.int32(1), # same charge
    MaxInvMass     = cms.untracked.double(10.),
    MinPt          = cms.untracked.vdouble(3.2, 3.2), 
    MinEta         = cms.untracked.vdouble(-2.52, -2.52),
    MaxEta         = cms.untracked.vdouble( 2.52,  2.52),
    Status         = cms.untracked.vint32(1, 1),
)

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('\$Revision$'),
    name = cms.untracked.string('\$Source$'),
    annotation = cms.untracked.string(
        'QCD bbbar production, '\
        'Jpsi from any b-hadron (either directly or feeddown), '\
        'Jpsi->mumu, mu pt>3.2, mu |eta|<2.52, '\
        'additional mu with pt>3.2 and |eta|<2.52, '\
        'invariant mass(Jpsi, mu)<10, '\
        '13 TeV, '\
        'TuneCP5'
    )
)

ProductionFilterSequence = cms.Sequence(generator*jpsi_from_b_hadron_filter*three_mu_filter*mu_mu_same_charge_filter)
