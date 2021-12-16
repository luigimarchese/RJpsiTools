'''
Difference from v6:
- addition of division High mass and low mass
'''
#from nanoAOD root files, to flat ntuples
from mybatch import *
from coffea.analysis_objects import JaggedCandidateArray
import awkward as awk
import numpy as np
import uproot
from nanoframe import NanoFrame
import os
import particle
import pandas as pd
import uproot_methods
import ROOT
from pdb import set_trace
from root_pandas import to_root
from uproot_methods import TLorentzVectorArray
from uproot_methods import TVector3Array
from uproot_methods import TLorentzVector
from uproot_methods import TVector3
from scipy.constants import c as speed_of_light
import uproot
import math

#hammer
from bgl_variations import variations
from itertools import product
from hammer.hammerlib import Hammer, IOBuffer, Particle, Process, FourMomentum, WTerm
from hammer import hepmc, pdg

maxEvents = -1
checkDoubles = True

nMaxFiles = REPLACE_MAX_FILES
skipFiles = REPLACE_SKIP_FILES

#Compute hammer
flag_hammer_mu  = True
flag_hammer_tau = True

#Add also pu weight
flag_pu_weight = False

## lifetime weights ##
def weight_to_new_ctau(old_ctau, new_ctau, ct):
    '''
    Returns an event weight based on the ratio of the normalised lifetime distributions.
    old_ctau: ctau used for the sample production
    new_ctau: target ctau
    ct      : per-event lifetime
    '''
    weight = old_ctau/new_ctau * np.exp( (1./old_ctau - 1./new_ctau) * ct )
    return weight
    
def lifetime_weight(pf, fake = True):
    #print("Adding lifetime weight branch...")
    if fake:
        ctau_weight_central = np.ones(len(pf))
        ctau_weight_up = np.ones(len(pf))
        ctau_weight_down = np.ones(len(pf))
        pf['ctau_weight_central'] = ctau_weight_central
        pf['ctau_weight_up'] = ctau_weight_up
        pf['ctau_weight_down'] = ctau_weight_down
        return pf
    else:
        Bc_mass = 6.274
        ctau_pdg    = 0.510e-12 * speed_of_light * 1000. # in mm
        ctau_actual = 0.1358
        ctau_up     = (0.510+0.009)*1e-12 * speed_of_light * 1000. # in mm
        ctau_down   = (0.510-0.009)*1e-12 * speed_of_light * 1000. # in mm
        
        ctau_weight_central = []
        ctau_weight_up = []
        ctau_weight_down = []

        for i in range(len(pf)):
            flag = 0
            #jpsi vertex
            if( abs(pf.mu1_mother_pdgId[i]) == 443 ):
                jpsi_vertex = TVector3(pf.mu1_mother_vx[i],pf.mu1_mother_vy[i],pf.mu1_mother_vz[i])
            elif( abs(pf.mu2_mother_pdgId[i]) == 443 ):
                jpsi_vertex = TVector3(pf.mu2_mother_vx[i],pf.mu2_mother_vy[i],pf.mu2_mother_vz[i])
             
            else: 
                flag = 1
        
            #Bc vertex
            if(abs(pf.mu1_grandmother_pdgId[i]) == 541):
                Bc_vertex = TVector3(pf.mu1_grandmother_vx[i],pf.mu1_grandmother_vy[i],pf.mu1_grandmother_vz[i])
                Bc_p4 = TLorentzVector.from_ptetaphim(pf.mu1_grandmother_pt[i],pf.mu1_grandmother_eta[i],pf.mu1_grandmother_phi[i],Bc_mass)
            elif(abs(pf.mu2_grandmother_pdgId[i]) == 541):
                Bc_vertex = TVector3(pf.mu2_grandmother_vx[i],pf.mu2_grandmother_vy[i],pf.mu2_grandmother_vz[i])
                Bc_p4 = TLorentzVector.from_ptetaphim(pf.mu2_grandmother_pt[i],pf.mu2_grandmother_eta[i],pf.mu2_grandmother_phi[i],Bc_mass)

            else:
                flag = 1
        
            if(flag == 1):
                ctau_weight_central.append(1)
                ctau_weight_up.append (1)
                ctau_weight_down.append(1)
       
            else:
                # distance
                lxyz = (jpsi_vertex - Bc_vertex).mag
                beta = Bc_p4.beta
                gamma = Bc_p4.gamma
                ct = lxyz/(beta * gamma)
                #print(lxyz,beta,gamma,ct)
                ctau_weight_central.append( weight_to_new_ctau(ctau_actual, ctau_pdg , ct*10.))
                ctau_weight_up.append (weight_to_new_ctau(ctau_actual, ctau_up  , ct*10.))
                ctau_weight_down.append(weight_to_new_ctau(ctau_actual, ctau_down, ct*10.))

        pf['ctau_weight_central'] = ctau_weight_central
        pf['ctau_weight_up'] = ctau_weight_up
        pf['ctau_weight_down'] = ctau_weight_down
        return pf
## end lifetime weights ##

def mcor(pf):
    #https://cds.cern.ch/record/2697350/files/1910.13404.pdf
    #only for bto3mu and bto2mutrk 
    #print("Adding mcor variable...")
    b_dir_vec = TVector3Array(pf.jpsivtx_vtx_x - pf.pv_x,pf.jpsivtx_vtx_y - pf.pv_y,pf.jpsivtx_vtx_z - pf.pv_z )
    b_dir = b_dir_vec/np.sqrt(b_dir_vec.mag2)
    jpsimu_p4 = TLorentzVectorArray.from_ptetaphim(pf.Bpt,pf.Beta,pf.Bphi,pf.Bmass)
    jpsimu_p3 = jpsimu_p4.p3
    p_parallel = jpsimu_p3.dot(b_dir) 
    p_perp = np.sqrt(jpsimu_p3.mag2 - p_parallel * p_parallel)
    mcor = np.sqrt(pf.Bmass * pf.Bmass + p_perp* p_perp) + p_perp
    pf['mcor'] = mcor
    return pf

def DR_jpsimu(pf):
    #print("Adding DR between jpsi and mu branch...")
    mu1_p4 = TLorentzVectorArray.from_ptetaphim(pf.mu1pt,pf.mu1eta,pf.mu1phi,pf.mu1mass)
    mu2_p4 = TLorentzVectorArray.from_ptetaphim(pf.mu2pt,pf.mu2eta,pf.mu2phi,pf.mu2mass)
    jpsi_p4= mu1_p4 + mu2_p4 
    mu_p4 = TLorentzVectorArray.from_ptetaphim(pf.kpt,pf.keta,pf.kphi,pf.kmass)
    pf['DR_jpsimu'] = jpsi_p4.delta_r(mu_p4)
    return pf

def dr13(pf):
    mu1_p4 = TLorentzVectorArray.from_ptetaphim(pf.mu1pt,pf.mu1eta,pf.mu1phi,pf.mu1mass)
    mu_p4 = TLorentzVectorArray.from_ptetaphim(pf.kpt,pf.keta,pf.kphi,pf.kmass)
    pf['dr13'] = mu_p4.delta_r(mu1_p4)
    return pf

def dr23(pf):
    mu2_p4 = TLorentzVectorArray.from_ptetaphim(pf.mu2pt,pf.mu2eta,pf.mu2phi,pf.mu2mass)
    mu_p4 = TLorentzVectorArray.from_ptetaphim(pf.kpt,pf.keta,pf.kphi,pf.kmass)
    pf['dr23'] = mu_p4.delta_r(mu2_p4)
    return pf

def dr12(pf):
    mu1_p4 = TLorentzVectorArray.from_ptetaphim(pf.mu1pt,pf.mu1eta,pf.mu1phi,pf.mu1mass)
    mu2_p4 = TLorentzVectorArray.from_ptetaphim(pf.mu2pt,pf.mu2eta,pf.mu2phi,pf.mu2mass)
    pf['dr12'] = mu2_p4.delta_r(mu1_p4)
    return pf

def jpsi_branches(pf):
    mu1_p4 = TLorentzVectorArray.from_ptetaphim(pf.mu1pt,pf.mu1eta,pf.mu1phi,pf.mu1mass)
    mu2_p4 = TLorentzVectorArray.from_ptetaphim(pf.mu2pt,pf.mu2eta,pf.mu2phi,pf.mu2mass)
    jpsi_p4= mu1_p4 + mu2_p4
    
    pf['jpsi_pt'] = jpsi_p4.pt
    pf['jpsi_eta'] = jpsi_p4.eta
    pf['jpsi_phi'] = jpsi_p4.phi
    pf['jpsi_mass'] = jpsi_p4.mass
    return pf
    
def decaytime(pf):
    beamspot_pos = TVector3Array(pf.beamspot_x, pf.beamspot_y, pf.beamspot_z)
    PV_pos = TVector3Array(pf.pv_x,pf.pv_y,pf.pv_z)
    jpsiVertex_pos = TVector3Array(pf.jpsivtx_vtx_x,pf.jpsivtx_vtx_y,pf.jpsivtx_vtx_z)
    bcVertex_pos = TVector3Array(pf.bvtx_vtx_x,pf.bvtx_vtx_y,pf.bvtx_vtx_z)

    dist_pv_jpsi = (PV_pos - jpsiVertex_pos).mag
    dist_pv_bc = (PV_pos - bcVertex_pos).mag
    dist_beamspot_jpsi = (beamspot_pos - jpsiVertex_pos).mag
    dist_beamspot_bc = (beamspot_pos - bcVertex_pos).mag

    if(len(PV_pos)): 
        decay_time_pv_jpsi = dist_pv_jpsi * 6.276 / (pf.Bpt_reco * 2.998e+10)
        pf['decay_time_pv_jpsi'] = decay_time_pv_jpsi
        decay_time_pv_bc = dist_pv_bc * 6.276 / (pf.Bpt_reco * 2.998e+10)
        pf['decay_time_pv_bc'] = decay_time_pv_bc
        decay_time_beamspot_jpsi = dist_beamspot_jpsi * 6.276 / (pf.Bpt_reco * 2.998e+10)
        pf['decay_time_beamspot_jpsi'] = decay_time_beamspot_jpsi
        decay_time_beamspot_bc = dist_beamspot_bc * 6.276 / (pf.Bpt_reco * 2.998e+10)
        pf['decay_time_beamspot_bc'] = decay_time_beamspot_bc

    # when there are 0 events, just to save the branch (empty)
    else:
        pf['decay_time_pv_jpsi'] = pf.pv_x #it's NaN anyway
        pf['decay_time_pv_bc'] = pf.pv_x #it's NaN anyway
        pf['decay_time_beamspot_jpsi'] = pf.pv_x #it's NaN anyway
        pf['decay_time_beamspot_bc'] = pf.pv_x #it's NaN anyway
    return pf

def bp4_lhcb(pf):
    PV_pos = TVector3Array(pf.pv_x,pf.pv_y,pf.pv_z)
    SV_pos = TVector3Array(pf.jpsivtx_vtx_x,pf.jpsivtx_vtx_y,pf.jpsivtx_vtx_z)

    phi_pv_sv = (PV_pos - SV_pos).phi
    theta_pv_sv = (PV_pos - SV_pos).theta
    eta_pv_sv = -np.log( np.tan (abs(theta_pv_sv)/2) ) * np.sign(theta_pv_sv)
    real_mass = [6.276 for i in phi_pv_sv]
    Bpt = [item for item in pf.Bpt]

    mu1_p4 = TLorentzVectorArray.from_ptetaphim(pf.mu1pt,pf.mu1eta,pf.mu1phi,pf.mu1mass)
    mu2_p4 = TLorentzVectorArray.from_ptetaphim(pf.mu2pt,pf.mu2eta,pf.mu2phi,pf.mu2mass)
    jpsi_p4= mu1_p4 + mu2_p4 
    mu_p4 = TLorentzVectorArray.from_ptetaphim(pf.kpt,pf.keta,pf.kphi,pf.kmass)
    b_p4 = TLorentzVectorArray.from_ptetaphim(Bpt, eta_pv_sv,phi_pv_sv, real_mass)

    if(len(PV_pos)): 
        
        pf['Bpt_reco_b'] = b_p4.pt
        pf['Q_sq_b'] = (b_p4 - jpsi_p4)*(b_p4 - jpsi_p4)
        pf['pt_miss_b'] = (b_p4 - jpsi_p4 - mu_p4).pt
        pf['pt_var_b'] = jpsi_p4.pt - mu_p4.pt
        mu_beta_lab = TVector3Array(b_p4.x/b_p4.t,b_p4.y/b_p4.t,b_p4.z/b_p4.t)
        mu_p4.boost(-mu_beta_lab)
        pf['E_mu_star_b'] = mu_p4.energy

    # when there are 0 events, just to save the branch (empty)
    else:
        pf['Bpt_reco_b'] = pf.pv_x #it's NaN anyway
        pf['Q_sq_b'] = pf.pv_x #it's NaN anyway
        pf['pt_miss_b'] = pf.pv_x #it's NaN anyway
        pf['pt_var_b'] = pf.pv_x #it's NaN anyway
        pf['E_mu_star_b'] = pf.pv_x #it's NaN anyway

    return pf


#for the rho corrected iso branch
def getAreaEff( eta, drcone ):
    aeff_dic = { '03' :
              [ (1.000, 0.13),
              (1.479, 0.14),
              (2.000, 0.07),
              (2.200, 0.09),
              (2.300, 0.11),
              (2.400, 0.11),
              (2.500, 0.14) ],
           '04' :
              [ (1.000, 0.208),
              (1.479, 0.209),
              (2.000, 0.115),
              (2.200, 0.143),
              (2.300, 0.183),
              (2.400, 0.194),
              (2.500, 0.261) ],
    }
    aeff = []
    for ieta in eta:
        for i,eta_loop in enumerate(aeff_dic[drcone]):
            if i == 0 and ieta<=eta_loop[0]:
                aeff.append(aeff_dic[drcone][0][1])
                break
            if ieta>eta_loop[0]:
                aeff.append(aeff_dic[drcone][i-1][1])
                break
    return aeff

def rho_corr_iso(df):
    #unpaired muon
    aEff = getAreaEff(df.keta,'03')
    zeros = pd.Series({'zero':[0 for i in range(len(aEff))]})
    df['k_raw_rho_corr_iso03'] = df.k_raw_ch_pfiso03 + pd.concat([df.k_raw_n_pfiso03+df.k_raw_pho_pfiso03 + df.fixedGridRhoFastjetAll * aEff,zeros],axis=1).max(axis=1)
    df['k_raw_rho_corr_iso03_rel'] = (df.k_raw_ch_pfiso03 + pd.concat([df.k_raw_n_pfiso03+df.k_raw_pho_pfiso03 + df.fixedGridRhoFastjetAll * aEff,zeros],axis=1).max(axis=1))/df.kpt

    aEff = getAreaEff(df.keta,'04')
    df['k_raw_rho_corr_iso04'] = df.k_raw_ch_pfiso04 + pd.concat([df.k_raw_n_pfiso04+df.k_raw_pho_pfiso04 + df.fixedGridRhoFastjetAll * aEff,zeros],axis=1).max(axis=1)
    df['k_raw_rho_corr_iso04_rel'] = (df.k_raw_ch_pfiso04 + pd.concat([df.k_raw_n_pfiso04+df.k_raw_pho_pfiso04 + df.fixedGridRhoFastjetAll * aEff,zeros],axis=1).max(axis=1))/df.kpt

    #mu1
    aEff = getAreaEff(df.mu1eta,'03')
    zeros = pd.Series({'zero':[0 for i in range(len(aEff))]})
    df['mu1_raw_rho_corr_iso03'] = df.mu1_raw_ch_pfiso03 + pd.concat([df.mu1_raw_n_pfiso03+df.mu1_raw_pho_pfiso03 + df.fixedGridRhoFastjetAll * aEff,zeros],axis=1).max(axis=1)
    df['mu1_raw_rho_corr_iso03_rel'] = (df.mu1_raw_ch_pfiso03 + pd.concat([df.mu1_raw_n_pfiso03+df.mu1_raw_pho_pfiso03 + df.fixedGridRhoFastjetAll * aEff,zeros],axis=1).max(axis=1))/df.kpt

    aEff = getAreaEff(df.mu1eta,'04')
    df['mu1_raw_rho_corr_iso04'] = df.mu1_raw_ch_pfiso04 + pd.concat([df.mu1_raw_n_pfiso04+df.mu1_raw_pho_pfiso04 + df.fixedGridRhoFastjetAll * aEff,zeros],axis=1).max(axis=1)
    df['mu1_raw_rho_corr_iso04_rel'] = (df.mu1_raw_ch_pfiso04 + pd.concat([df.mu1_raw_n_pfiso04+df.mu1_raw_pho_pfiso04 + df.fixedGridRhoFastjetAll * aEff,zeros],axis=1).max(axis=1))/df.kpt

    #mu2
    aEff = getAreaEff(df.mu2eta,'03')
    zeros = pd.Series({'zero':[0 for i in range(len(aEff))]})
    df['mu2_raw_rho_corr_iso03'] = df.mu2_raw_ch_pfiso03 + pd.concat([df.mu2_raw_n_pfiso03+df.mu2_raw_pho_pfiso03 + df.fixedGridRhoFastjetAll * aEff,zeros],axis=1).max(axis=1)
    df['mu2_raw_rho_corr_iso03_rel'] = (df.mu2_raw_ch_pfiso03 + pd.concat([df.mu2_raw_n_pfiso03+df.mu2_raw_pho_pfiso03 + df.fixedGridRhoFastjetAll * aEff,zeros],axis=1).max(axis=1))/df.kpt

    aEff = getAreaEff(df.mu2eta,'04')
    df['mu2_raw_rho_corr_iso04'] = df.mu2_raw_ch_pfiso04 + pd.concat([df.mu2_raw_n_pfiso04+df.mu2_raw_pho_pfiso04 + df.fixedGridRhoFastjetAll * aEff,zeros],axis=1).max(axis=1)
    df['mu2_raw_rho_corr_iso04_rel'] = (df.mu2_raw_ch_pfiso04 + pd.concat([df.mu2_raw_n_pfiso04+df.mu2_raw_pho_pfiso04 + df.fixedGridRhoFastjetAll * aEff,zeros],axis=1).max(axis=1))/df.kpt

    return df

# Compute the form factor weights for the mu sample
def hammer_weights_mu(df,ham):
    ham.include_decay("BcJpsiMuNu")
    ff_input_scheme = dict()
    ff_input_scheme["BcJpsi"] = "Kiselev"
    ham.set_ff_input_scheme(ff_input_scheme)
    ff_schemes  = dict()
    ff_schemes['bglvar' ] = {'BcJpsi':'BGLVar' }
    for i, j in product(range(11), ['up', 'down']):
        unc = 'e%d%s'%(i,j)
        ff_schemes['bglvar_%s'%unc] = {'BcJpsi':'BGLVar_%s'%unc  }
                        
    for k, v in ff_schemes.items():
        ham.add_ff_scheme(k, v)
    ham.set_units("GeV")
    ham.init_run()
    for i, j in product(range(11), ['up', 'down']):
        unc = 'e%d%s'%(i,j)
        ham.set_ff_eigenvectors('BctoJpsi', 'BGLVar_%s'%unc, variations['e%d'%i][j])
    pids=[]
    weights = dict()
    for k in ff_schemes.keys():
        weights[k] = []
    for i in range(len(df)): #loop on the events
        ham.init_event()
        thebc_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(df.bc_gen_pt[i],df.bc_gen_eta[i] , df.bc_gen_phi[i], df.bc_gen_mass[i])
        themu_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(df.mu3_gen_pt[i] , df.mu3_gen_eta[i],df.mu3_gen_phi[i] , df.mu3_gen_mass[i])
        thejpsi_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(df.jpsi_gen_pt[i] , df.jpsi_gen_eta[i],df.jpsi_gen_phi[i] , df.jpsi_gen_mass[i])
        thenu_p4   = thebc_p4 - themu_p4 - thejpsi_p4
                        
        thebc   = Particle(FourMomentum(thebc_p4.e()  , thebc_p4.px()  , thebc_p4.py()  , thebc_p4.pz()  ), 541)
        themu   = Particle(FourMomentum(themu_p4.e()  , themu_p4.px()  , themu_p4.py()  , themu_p4.pz()  ), -13          )
        thejpsi = Particle(FourMomentum(thejpsi_p4.e(), thejpsi_p4.px(), thejpsi_p4.py(), thejpsi_p4.pz()), 443          )
        thenu   = Particle(FourMomentum(thenu_p4.e()  , thenu_p4.px()  , thenu_p4.py()  , thenu_p4.pz())  , 14           )
        
        Bc2JpsiLNu = Process()
        
        thebc_idx   = Bc2JpsiLNu.add_particle(thebc  )
        themu_idx   = Bc2JpsiLNu.add_particle(themu  )
        thejpsi_idx = Bc2JpsiLNu.add_particle(thejpsi)
        thenu_idx   = Bc2JpsiLNu.add_particle(thenu  )
        Bc2JpsiLNu.add_vertex(thebc_idx, [thejpsi_idx, themu_idx, thenu_idx])
        pid = ham.add_process(Bc2JpsiLNu)
        pids.append(pid)
        ham.process_event()
        for k in ff_schemes.keys():
            weights[k].append(ham.get_weight(k))
    for k in ff_schemes.keys():
        #save the nan as 1
        weights_clean = [ham if (not np.isnan(ham)) else 1. for ham in weights[k]]
        df["hammer_"+k] = weights_clean
    return df

# Compute the form factor weights for the tau sample
def hammer_weights_tau(df,ham):
    ham.include_decay(["BcJpsiTauNu"])
    ff_input_scheme = dict()
    ff_input_scheme["BcJpsi"] = "Kiselev"
    ham.set_ff_input_scheme(ff_input_scheme)
    ff_schemes  = dict()
    ff_schemes['bglvar' ] = {'BcJpsi':'BGLVar' }
    for i, j in product(range(11), ['up', 'down']):
        unc = 'e%d%s'%(i,j)
        ff_schemes['bglvar_%s'%unc] = {'BcJpsi':'BGLVar_%s'%unc  }
                        
    for k, v in ff_schemes.items():
        ham.add_ff_scheme(k, v)
    ham.set_units("GeV")
    ham.init_run()
    for i, j in product(range(11), ['up', 'down']):
        unc = 'e%d%s'%(i,j)
        ham.set_ff_eigenvectors('BctoJpsi', 'BGLVar_%s'%unc, variations['e%d'%i][j])
    pids=[]
    weights = dict()
    for k in ff_schemes.keys():
        weights[k] = []
    for i in range(len(df)): #loop on the events
        ham.init_event()

        thebc_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(df.bc_gen_pt[i],df.bc_gen_eta[i] , df.bc_gen_phi[i],  df.bc_gen_mass[i])
        thetau_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(df.tau_gen_pt[i] , df.tau_gen_eta[i],df.tau_gen_phi[i] , df.tau_gen_mass[i])
        thejpsi_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(df.jpsi_gen_pt[i] , df.jpsi_gen_eta[i],df.jpsi_gen_phi[i] , df.jpsi_gen_mass[i])

        thenutau_p4   = thebc_p4 - thetau_p4 - thejpsi_p4

        thebc   = Particle(FourMomentum(thebc_p4.e()  , thebc_p4.px()  , thebc_p4.py()  , thebc_p4.pz()  ), 541)
        thetau   = Particle(FourMomentum(thetau_p4.e()  , thetau_p4.px()  , thetau_p4.py()  , thetau_p4.pz()  ), -15          )
        thejpsi = Particle(FourMomentum(thejpsi_p4.e(), thejpsi_p4.px(), thejpsi_p4.py(), thejpsi_p4.pz()), 443          )
        thenutau   = Particle(FourMomentum(thenutau_p4.e()  , thenutau_p4.px()  , thenutau_p4.py()  , thenutau_p4.pz())  , 16)
        
        Bc2JpsiLNu = Process()
        
        thebc_idx   = Bc2JpsiLNu.add_particle(thebc  )
        thetau_idx   = Bc2JpsiLNu.add_particle(thetau  )
        thejpsi_idx = Bc2JpsiLNu.add_particle(thejpsi)
        thenutau_idx   = Bc2JpsiLNu.add_particle(thenutau  )
        Bc2JpsiLNu.add_vertex(thebc_idx, [thejpsi_idx, thetau_idx, thenutau_idx])
        pid = ham.add_process(Bc2JpsiLNu)
        pids.append(pid)
        ham.process_event()
        
        for k in ff_schemes.keys():
            weights[k].append(ham.get_weight(k))

    for k in ff_schemes.keys():
        #save the nan as 1
        weights_clean = [ham if (not np.isnan(ham)) else 1. for ham in weights[k]]
        df["hammer_"+k] = weights_clean
    return df
    
# To DO : optimize this!
def HighMassLowMassDivision(df):
    hmlm = nf['HighMassLowMassFlags']
    hmlm['index'] = [[i for subitem in item] for i,item in enumerate(hmlm['mu1_idx'])]
    ancestors_flag = []
    for ind2 in range(len(df['index'])): #index of dataframe
        for ind1 in range(len(hmlm['index'])): #index of hmmlm
            if hmlm['index'][ind1][0][0] == df['index'][ind2][0]: # if we are checking the same event
                flag = 0
                for jnd1 in range(len(hmlm['index'][ind1])): #loop over the different possibilities in hmlm for that event
                    # when the third muon from the candidate is not a real muon (not really a problem, in the analysis I don't use these events)
                    if (abs(df['k_genpdgId'][ind2]) != 13):
                        flag = 1
                        ancestors_flag.append(-1) 
                        break 
                        
                    # One of the muons of the jpsi is not a real muon (I still compute the value from one of the cases)
                    elif ( abs(df['mu1_genpdgId'][ind2]) != 13 or abs(df['mu2_genpdgId'][ind2]) != 13):
                        flag = 1
                        ancestors_flag.append(int(hmlm['jpsi_ancestor_idx'][ind1][jnd1] == hmlm['mu3_ancestor_idx'][ind1][jnd1]))
                        break 

                    # the jpsi has a perfect match
                    elif ( (hmlm['mu1_idx'][ind1][jnd1] == df['mu1_genPartIdx'][ind2] or hmlm['mu1_idx'][ind1][jnd1] == df['mu2_genPartIdx'][ind2]) and (hmlm['mu2_idx'][ind1][jnd1] == df['mu1_genPartIdx'][ind2] or hmlm['mu2_idx'][ind1][jnd1] == df['mu2_genPartIdx'][ind2]) ):
                        
                        #perfect match also for the third muon
                        if hmlm['mu3_idx'][ind1][jnd1] == df['k_genPartIdx'][ind2]:
                            flag = 1
                            ancestors_flag.append(int(hmlm['jpsi_ancestor_idx'][ind1][jnd1] == hmlm['mu3_ancestor_idx'][ind1][jnd1]))
                            break 
                     
                        # third muon is not a perfect match but it's the last choice, so I take it
                        elif hmlm['mu3_idx'][ind1][jnd1] != df['k_genPartIdx'][ind2] and jnd1 == len(hmlm['index'][ind1])-1:
                            flag = 1
                            ancestors_flag.append(int(hmlm['jpsi_ancestor_idx'][ind1][jnd1] == hmlm['mu3_ancestor_idx'][ind1][jnd1]))
                            print("diff mu3 but same jpsi",ind2,ind1,jnd1,hmlm['mu1_idx'][ind1][jnd1],df['mu1_genPartIdx'][ind2],df['mu1_genpdgId'][ind2],hmlm['mu2_idx'][ind1][jnd1],df['mu2_genPartIdx'][ind2],df['mu2_genpdgId'][ind2],hmlm['mu3_idx'][ind1][jnd1],df['k_genPartIdx'][ind2],df['k_genpdgId'][ind2])
                            break 

                    # the jpsi has not a perfect match
                    elif((hmlm['mu1_idx'][ind1][jnd1] == df['mu1_genPartIdx'][ind2] or hmlm['mu1_idx'][ind1][jnd1] == df['mu2_genPartIdx'][ind2]) or (hmlm['mu2_idx'][ind1][jnd1] == df['mu1_genPartIdx'][ind2] or hmlm['mu2_idx'][ind1][jnd1] == df['mu2_genPartIdx'][ind2])):

                        #the third muon has perfect match (than it's ok)
                        if hmlm['mu3_idx'][ind1][jnd1] == df['k_genPartIdx'][ind2]:
                            flag = 1
                            ancestors_flag.append(int(hmlm['jpsi_ancestor_idx'][ind1][jnd1] == hmlm['mu3_ancestor_idx'][ind1][jnd1]))
                            break
                        
                        #third muon doesn't have a perfect match, but it's the last choice
                        elif hmlm['mu3_idx'][ind1][jnd1] != df['k_genPartIdx'][ind2] and jnd1 == len(hmlm['index'][ind1])-1:
                            flag = 1
                            ancestors_flag.append(int(hmlm['jpsi_ancestor_idx'][ind1][jnd1] == hmlm['mu3_ancestor_idx'][ind1][jnd1]))
                            print("jpsi not perfect match and different mu3",ind2,ind1,jnd1,len(hmlm['index'][ind1]),hmlm['mu1_idx'][ind1][jnd1],df['mu1_genPartIdx'][ind2],df['mu1_genpdgId'][ind2],hmlm['mu2_idx'][ind1][jnd1],df['mu2_genPartIdx'][ind2],df['mu2_genpdgId'][ind2],hmlm['mu3_idx'][ind1][jnd1],df['k_genPartIdx'][ind2],df['k_genpdgId'][ind2])
                            break

                    # the jpsi has no match at all and it's the last event you are checking
                    elif(not((hmlm['mu1_idx'][ind1][jnd1] == df['mu1_genPartIdx'][ind2] or hmlm['mu1_idx'][ind1][jnd1] == df['mu2_genPartIdx'][ind2]) or (hmlm['mu2_idx'][ind1][jnd1] == df['mu1_genPartIdx'][ind2] or hmlm['mu2_idx'][ind1][jnd1] == df['mu2_genPartIdx'][ind2])) and jnd1 == len(hmlm['index'][ind1])-1):
                        flag = 1
                        ancestors_flag.append(int(hmlm['jpsi_ancestor_idx'][ind1][jnd1] == hmlm['mu3_ancestor_idx'][ind1][jnd1]))
                        print("jpsi no match",ind2,ind1,jnd1,len(hmlm['index'][ind1]),hmlm['mu1_idx'][ind1][jnd1],df['mu1_genPartIdx'][ind2],df['mu1_genpdgId'][ind2],hmlm['mu2_idx'][ind1][jnd1],df['mu2_genPartIdx'][ind2],df['mu2_genpdgId'][ind2],hmlm['mu3_idx'][ind1][jnd1],df['k_genPartIdx'][ind2],df['k_genpdgId'][ind2])
                        break
                                                                        
                if flag ==0:
                    print(ind2,hmlm['mu1_idx'][ind1][jnd1],df['mu1_genPartIdx'][ind2],df['mu1_genpdgId'][ind2],hmlm['mu2_idx'][ind1][jnd1],df['mu2_genPartIdx'][ind2],df['mu2_genpdgId'][ind2],hmlm['mu3_idx'][ind1][jnd1],df['k_genPartIdx'][ind2],df['k_genpdgId'][ind2])
                    raise ValueError("Error in the Division of the Hb MC into High Mass and Low Mass contributions")
    if len(ancestors_flag)!=len(df['index']):
        raise ValueError("Error in the Division of the Hb MC into High Mass and Low Mass contributions")
    df['hmlm_flag'] = ancestors_flag
    return df
#######################################################################################
######  STARTING THE SCRIPT ##########################################################
#######################################################################################

nprocessedAll = 0
channels = REPLACE_CHANNELS

#loop on input datasets
for dataset in [args.data,args.mc_mu,args.mc_tau,args.mc_bc,args.mc_hb,args.mc_onia,args.mc_gen]: 
    if(dataset==''): 
        continue
    print(" ")
    
    print("Opening file", dataset)
    f=open(dataset,"r")
    paths = f.readlines()

    ###################
    # MC BcToXToJpsi #
    ###################
    if(dataset == args.mc_bc):

        final_dfs_mmm = {
            'is_jpsi_mu' : pd.DataFrame(),
            'is_jpsi_tau' : pd.DataFrame(),
            'is_jpsi_pi' : pd.DataFrame(),
            'is_psi2s_mu' :pd.DataFrame(),
            'is_chic0_mu' : pd.DataFrame(),
            'is_chic1_mu' : pd.DataFrame(),
            'is_chic2_mu' : pd.DataFrame(),
            'is_hc_mu' : pd.DataFrame(),
            'is_psi2s_tau' : pd.DataFrame(),
            'is_jpsi_3pi' : pd.DataFrame(),
            'is_jpsi_hc' : pd.DataFrame(),
        }
        
        final_dfs_pmm = {
            'is_jpsi_mu' : pd.DataFrame(),
            'is_jpsi_tau' : pd.DataFrame(),
            'is_jpsi_pi' : pd.DataFrame(),
            'is_psi2s_mu' :pd.DataFrame(),
            'is_chic0_mu' : pd.DataFrame(),
            'is_chic1_mu' : pd.DataFrame(),
            'is_chic2_mu' : pd.DataFrame(),
            'is_hc_mu' : pd.DataFrame(),
            'is_psi2s_tau' : pd.DataFrame(),
            'is_jpsi_3pi' : pd.DataFrame(),
            'is_jpsi_hc' : pd.DataFrame(),
        }
        
        final_dfs_kmm = {
            'is_jpsi_mu' : pd.DataFrame(),
            'is_jpsi_tau' : pd.DataFrame(),
            'is_jpsi_pi' : pd.DataFrame(),
            'is_psi2s_mu' :pd.DataFrame(),
            'is_chic0_mu' : pd.DataFrame(),
            'is_chic1_mu' : pd.DataFrame(),
            'is_chic2_mu' : pd.DataFrame(),
            'is_hc_mu' : pd.DataFrame(),
            'is_psi2s_tau' : pd.DataFrame(),
            'is_jpsi_3pi' : pd.DataFrame(),
            'is_jpsi_hc' : pd.DataFrame(),
        }

        final_dfs_2m3p = {
            'is_jpsi_mu' : pd.DataFrame(),
            'is_jpsi_tau' : pd.DataFrame(),
            'is_jpsi_pi' : pd.DataFrame(),
            'is_psi2s_mu' :pd.DataFrame(),
            'is_chic0_mu' : pd.DataFrame(),
            'is_chic1_mu' : pd.DataFrame(),
            'is_chic2_mu' : pd.DataFrame(),
            'is_hc_mu' : pd.DataFrame(),
            'is_psi2s_tau' : pd.DataFrame(),
            'is_jpsi_3pi' : pd.DataFrame(),
            'is_jpsi_hc' : pd.DataFrame(),
        }
        

    # For the rest of the samples
    else:
        final_dfs_mmm = {
            'ptmax' : pd.DataFrame(),
        }

        final_dfs_pmm = {
            'ptmax' : pd.DataFrame(),
        }

        final_dfs_kmm = {
            'ptmax' : pd.DataFrame(),
        }

        final_dfs_2m3p = {
            'ptmax' : pd.DataFrame(),
        }

    
    nprocessedDataset = 0
    nFiles = 0
    for i,fname in enumerate(paths):
        fname= fname.strip('\n')
        
        if(i < skipFiles): # if I want to skip 1 file, I want to skip i=0 -> i+1
            print("Skipping the file...")
            continue
           
        if(nFiles >= nMaxFiles and nMaxFiles != -1):
            print("No! Max number of files reached!")
            break
        nFiles +=1
        print("Processing file ", fname)
       
        # Create nf before the loop on the channels (because it reopens the file)
        nf = NanoFrame(fname, )
        for channel in channels:
            print("In channel "+channel)
            # Load the needed collections, NanoFrame is just an empty shell until we call the collections
            evt = nf['event']
            muons = nf['Muon']
            bcands = nf[channel]
            hlt = nf['HLT']
            gen= nf['GenPart']
            bcands['event'] = nf['event']
            bcands['run'] = nf['run']
            bcands['luminosityBlock'] = nf['luminosityBlock']    
            

            #WHAT about this?
            #if(len(nf[channel]) == 0):
            #    print("Empty channel!")
            #    continue
            bcands['l_xy_sig'] = bcands.bodies3_l_xy / np.sqrt(bcands.bodies3_l_xy_unc)
            bcands['fixedGridRhoFastjetAll'] = nf['fixedGridRhoFastjetAll']
            bcands['fixedGridRhoFastjetCentral'] = nf['fixedGridRhoFastjetCentral']
            bcands['fixedGridRhoFastjetCentralCalo'] = nf['fixedGridRhoFastjetCentralCalo']
            bcands['fixedGridRhoFastjetCentralChargedPileUp'] = nf['fixedGridRhoFastjetCentralChargedPileUp']
            bcands['fixedGridRhoFastjetCentralNeutral'] = nf['fixedGridRhoFastjetCentralNeutral']
            
            # Bc MC sample type flag
            if(dataset == args.mc_bc):
                bcands['is_jpsi_tau'] = nf['DecayFlag_is_jpsi_tau']
                bcands['is_jpsi_mu'] = nf['DecayFlag_is_jpsi_mu']
                bcands['is_jpsi_pi'] = nf['DecayFlag_is_jpsi_pi']
                bcands['is_psi2s_mu'] = nf['DecayFlag_is_psi2s_mu']
                bcands['is_psi2s_tau'] = nf['DecayFlag_is_psi2s_tau']
                bcands['is_chic0_mu'] = nf['DecayFlag_is_chic0_mu']
                bcands['is_chic1_mu'] = nf['DecayFlag_is_chic1_mu']
                bcands['is_chic2_mu'] = nf['DecayFlag_is_chic2_mu']
                bcands['is_hc_mu'] = nf['DecayFlag_is_hc_mu']
                bcands['is_jpsi_3pi'] = nf['DecayFlag_is_jpsi_3pi']
                bcands['is_jpsi_hc'] = nf['DecayFlag_is_jpsi_hc']

                #bc gen info
                bcands['BcGenInfo_bc_gen_pt'] = nf['BcGenInfo_bc_gen_pt']
                bcands['BcGenInfo_bc_gen_eta'] = nf['BcGenInfo_bc_gen_eta']
                bcands['BcGenInfo_bc_gen_phi'] = nf['BcGenInfo_bc_gen_phi']
                bcands['BcGenInfo_bc_gen_mass'] = nf['BcGenInfo_bc_gen_mass']
                
                bcands['BcGenInfo_jpsi_gen_pt'] = nf['BcGenInfo_jpsi_gen_pt']
                bcands['BcGenInfo_jpsi_gen_eta'] = nf['BcGenInfo_jpsi_gen_eta']
                bcands['BcGenInfo_jpsi_gen_phi'] = nf['BcGenInfo_jpsi_gen_phi']
                bcands['BcGenInfo_jpsi_gen_mass'] = nf['BcGenInfo_jpsi_gen_mass']
                

                bcands['BcGenInfo_tau_gen_pt'] = nf['BcGenInfo_tau_gen_pt']
                bcands['BcGenInfo_tau_gen_eta'] = nf['BcGenInfo_tau_gen_eta']
                bcands['BcGenInfo_tau_gen_phi'] = nf['BcGenInfo_tau_gen_phi']
                bcands['BcGenInfo_tau_gen_mass'] = nf['BcGenInfo_tau_gen_mass']
                
                bcands['BcGenInfo_mu3_gen_pt'] = nf['BcGenInfo_mu3_gen_pt']
                bcands['BcGenInfo_mu3_gen_eta'] = nf['BcGenInfo_mu3_gen_eta']
                bcands['BcGenInfo_mu3_gen_phi'] = nf['BcGenInfo_mu3_gen_phi']
                bcands['BcGenInfo_mu3_gen_mass'] = nf['BcGenInfo_mu3_gen_mass']

            ###########################################
            ###### GEN weights for HbToJpsiMu MC ######
            ###########################################

            if(dataset == args.mc_hb):
                #useful for splitting Hb into highmass and lowmass
                bcands['index'] = [[i for subitem in item] for i,item in enumerate(bcands['event'])]
                #jpsi mother division
                bcands['jpsimother_bzero'] = nf['JpsiMotherFlag_bzero']
                bcands['jpsimother_bplus'] = nf['JpsiMotherFlag_bplus']
                bcands['jpsimother_bplus_c'] = nf['JpsiMotherFlag_bplus_c']
                bcands['jpsimother_bzero_s'] = nf['JpsiMotherFlag_bzero_s']
                bcands['jpsimother_sigmaminus_b'] = nf['JpsiMotherFlag_sigmaminus_b']
                bcands['jpsimother_lambdazero_b'] = nf['JpsiMotherFlag_lambdazero_b']
                bcands['jpsimother_ximinus_b'] = nf['JpsiMotherFlag_ximinus_b']
                bcands['jpsimother_sigmazero_b'] = nf['JpsiMotherFlag_sigmazero_b']
                bcands['jpsimother_xizero_b'] = nf['JpsiMotherFlag_xizero_b']
                bcands['jpsimother_other'] = nf['JpsiMotherFlag_other']
            
                #import weights from file
                f = ROOT.TFile.Open('decay_weight.root','r')
                histo = f.Get('weight')
                weights_jpsimother = {
                    'other': histo.GetBinContent(1),
                    'bzero': histo.GetBinContent(2),
                    'bplus': histo.GetBinContent(3),
                    'bzero_s': histo.GetBinContent(4),
                    'bplus_c': histo.GetBinContent(5),
                    'sigmaminus_b': histo.GetBinContent(6),
                    'lambdazero_b': histo.GetBinContent(7),
                    'ximinus_b': histo.GetBinContent(8),
                    'sigmazero_b': histo.GetBinContent(9),
                    'xizero_b': histo.GetBinContent(10),
                }
                weights_jpsim_tmp = 0.
                check_jpsimoth = 0.
                for b_had in weights_jpsimother:
                    weights_jpsim_tmp = weights_jpsim_tmp+ bcands['jpsimother_'+b_had] * weights_jpsimother[b_had]
                    check_jpsimoth = check_jpsimoth + bcands['jpsimother_'+b_had]

                # Check if there is more than 1 mother associated with the same jpsi (it shouldn't be)
                if any(check_jpsimoth.any())>=2.:
                    print('EROOR: more than one mother associated with jpsi')
                bcands['jpsimother_weight'] = weights_jpsim_tmp
                
                
            #number of events processed
            nprocessedDataset += hlt.shape[0]
            nprocessedAll+=hlt.shape[0]

            #add muon infos        
            mu1 = JaggedCandidateArray.zip({n: muons[bcands['mu1Idx']][n] for n in muons[bcands['mu1Idx']].columns})
            mu2 = JaggedCandidateArray.zip({n: muons[bcands['mu2Idx']][n] for n in muons[bcands['mu2Idx']].columns})
            if channel == 'BTo3Mu':
                k = JaggedCandidateArray.zip({n: muons[bcands['kIdx']][n] for n in muons[bcands['kIdx']].columns})
            else:
                tracks = nf['ProbeTracks']
                if channel == 'BTo2Mu3P':
                    pi1 = JaggedCandidateArray.zip({n: tracks[bcands['pi1Idx']][n] for n in tracks[bcands['pi1Idx']].columns})
                    pi2 = JaggedCandidateArray.zip({n: tracks[bcands['pi2Idx']][n] for n in tracks[bcands['pi2Idx']].columns})
                    pi3 = JaggedCandidateArray.zip({n: tracks[bcands['pi3Idx']][n] for n in tracks[bcands['pi3Idx']].columns})
                if channel == 'BTo2MuP' or channel == 'BTo2MuK':
                    k = JaggedCandidateArray.zip({n: tracks[bcands['kIdx']][n] for n in tracks[bcands['kIdx']].columns})
                

            # MC Onia sometimes has genc ollection empty, so we need to create a mask
            if(dataset==args.mc_onia and channel!='BTo2Mu3P'):
                mask = (k.genPartIdx != -1)
                mu1_new = mu1[mask]
                mu2_new = mu2[mask]
                k_new = k[mask]
                mu1 = JaggedCandidateArray.zip({n: mu1_new[n] for n in mu1_new.columns})
                mu2 = JaggedCandidateArray.zip({n: mu2_new[n] for n in mu2_new.columns})
                k = JaggedCandidateArray.zip({n: k_new[n] for n in k_new.columns})
                bcands = JaggedCandidateArray.zip({n: bcands[mask][n] for n in bcands[mask].columns})

            # add gen info as a column of the muon
            if (dataset!=args.data):
                #pile up weights only for mc and if flag ==True
                if flag_pu_weight:
                    bcands['puWeight'] = nf['puWeight']
                    bcands['puWeightUp'] = nf['puWeight_up']
                    bcands['puWeightDown'] = nf['puWeight_down']

                mu1['gen'] = gen[mu1.genPartIdx]
                mu2['gen'] = gen[mu2.genPartIdx]
                mu1['mother'] = gen[gen[mu1.genPartIdx].genPartIdxMother]
                mu2['mother'] = gen[gen[mu2.genPartIdx].genPartIdxMother]
                mu1['grandmother'] = gen[gen[gen[mu1.genPartIdx].genPartIdxMother].genPartIdxMother]
                mu2['grandmother'] = gen[gen[gen[mu2.genPartIdx].genPartIdxMother].genPartIdxMother]

                if(channel == 'BTo2Mu3P'):
                    pi1['gen'] = gen[pi1.genPartIdx]
                    pi2['gen'] = gen[pi2.genPartIdx]
                    pi3['gen'] = gen[pi3.genPartIdx]
                    pi1['mother'] = gen[gen[pi1.genPartIdx].genPartIdxMother]
                    pi2['mother'] = gen[gen[pi2.genPartIdx].genPartIdxMother]
                    pi3['mother'] = gen[gen[pi3.genPartIdx].genPartIdxMother]
                    pi1['grandmother'] = gen[gen[gen[pi1.genPartIdx].genPartIdxMother].genPartIdxMother]
                    pi2['grandmother'] = gen[gen[gen[pi2.genPartIdx].genPartIdxMother].genPartIdxMother]
                    pi3['grandmother'] = gen[gen[gen[pi3.genPartIdx].genPartIdxMother].genPartIdxMother]
                else:
                    k['gen'] = gen[k.genPartIdx]
                    k['mother'] = gen[gen[k.genPartIdx].genPartIdxMother]
                    k['grandmother'] = gen[gen[gen[k.genPartIdx].genPartIdxMother].genPartIdxMother]
                
                if (dataset!=args.mc_mu):
                    mu1['grandgrandmother'] =gen[gen[gen[gen[mu1.genPartIdx].genPartIdxMother].genPartIdxMother].genPartIdxMother]
                    mu2['grandgrandmother'] =gen[gen[gen[gen[mu2.genPartIdx].genPartIdxMother].genPartIdxMother].genPartIdxMother]
                    if(channel == 'BTo2Mu3P'):
                        pi1['grandgrandmother'] =gen[gen[gen[gen[pi1.genPartIdx].genPartIdxMother].genPartIdxMother].genPartIdxMother]          
                        pi2['grandgrandmother'] =gen[gen[gen[gen[pi2.genPartIdx].genPartIdxMother].genPartIdxMother].genPartIdxMother]         
                        pi3['grandgrandmother'] =gen[gen[gen[gen[pi3.genPartIdx].genPartIdxMother].genPartIdxMother].genPartIdxMother]   
                    else:
                        k['grandgrandmother'] =gen[gen[gen[gen[k.genPartIdx].genPartIdxMother].genPartIdxMother].genPartIdxMother]                
                        
            bcands['mu1']= mu1
            bcands['mu2'] = mu2
            if(channel == 'BTo2Mu3P'):
                bcands['pi1'] = pi1
                bcands['pi2'] = pi2
                bcands['pi3'] = pi3
            else:
                bcands['k'] = k
        
            # very loose selection here
            b_selection = ((bcands.p4.mass < 10 ) & (bcands.bodies3_svprob > 1e-7))
            x_selection= (bcands.p4.pt > -99)

            #Delete the signal from the JpsiX MC
            if (dataset==args.mc_onia or dataset==args.mc_hb):
                if(channel == 'BTo2Mu3P'):
                    x_selection= ~ ((bcands.pi1.genPartIdx>=0) & ( bcands.mu1.genPartIdx>=0) & (bcands.mu2.genPartIdx>=0) & (abs(bcands.mu1.mother.pdgId) == 443) & (abs(bcands.mu2.mother.pdgId) == 443) & (abs(bcands.pi1.grandmother.pdgId) == 541) & (abs(bcands.mu2.grandmother.pdgId) == 541) & ( (abs(bcands.pi1.mother.pdgId)==541) | ( (abs(bcands.pi1.mother.pdgId)==15) & (abs(bcands.pi1.grandmother.pdgId)== 541))))

                else:
                    x_selection= ~ ((bcands.k.genPartIdx>=0) & ( bcands.mu1.genPartIdx>=0) & (bcands.mu2.genPartIdx>=0) & (abs(bcands.mu1.mother.pdgId) == 443) & (abs(bcands.mu2.mother.pdgId) == 443) & (abs(bcands.mu1.grandmother.pdgId) == 541) & (abs(bcands.mu2.grandmother.pdgId) == 541) & ( (abs(bcands.k.mother.pdgId)==541) | ( (abs(bcands.k.mother.pdgId)==15) & (abs(bcands.k.grandmother.pdgId)== 541))))

            if channel == 'BTo2MuP':
                x_selection = x_selection & ((bcands.mu2.p4.pt>1) & (bcands.mu1.p4.pt>1) & (bcands.k.p4.pt>1) & (bcands.p4.mass<10) & (bcands.mu2.p4.eta<2.5) & (bcands.mu1.p4.eta<2.5) & (bcands.k.p4.eta<2.5))

            #######################################
            ###### MC Bc types ###################
            #######################################
            if(dataset == args.mc_bc):
                jpsi_tau_sel = (bcands.is_jpsi_tau == 1)
                jpsi_mu_sel = (bcands.is_jpsi_mu == 1)
                jpsi_pi_sel = (bcands.is_jpsi_pi == 1)
                psi2s_mu_sel = (bcands.is_psi2s_mu == 1)
                chic0_mu_sel = (bcands.is_chic0_mu == 1)
                chic1_mu_sel = (bcands.is_chic1_mu == 1)
                chic2_mu_sel = (bcands.is_chic2_mu == 1)
                hc_mu_sel = (bcands.is_hc_mu == 1)
                psi2s_tau_sel = (bcands.is_psi2s_tau == 1)
                jpsi_3pi_sel = (bcands.is_jpsi_3pi == 1)
                jpsi_hc_sel = (bcands.is_jpsi_hc == 1)
                flag_selection= [jpsi_tau_sel,jpsi_mu_sel,jpsi_pi_sel,psi2s_mu_sel,chic0_mu_sel,chic1_mu_sel,chic2_mu_sel,hc_mu_sel,psi2s_tau_sel,jpsi_3pi_sel,jpsi_hc_sel]
                flag_names = ['is_jpsi_tau','is_jpsi_mu','is_jpsi_pi','is_psi2s_mu','is_chic0_mu','is_chic1_mu','is_chic2_mu','is_hc_mu','is_psi2s_tau','is_jpsi_3pi','is_jpsi_hc']
            else:
                flag_selection = [(bcands.p4.pt>-99)]
                flag_names = ['ptmax']

            for selection,name in zip(flag_selection, flag_names):
                if(dataset == args.mc_bc):
                    print("Processing ",name)
                best_pf_cand_pt = bcands[b_selection & x_selection & selection ].p4.pt.argmax() #B with higher pt
                bcands_flag = (bcands[b_selection & x_selection & selection][best_pf_cand_pt]).flatten()

                
                ###########################################################################
                ##########  Saving all the useful branches for the flat ntuples ###########
                ###########################################################################
                dfs = {}            
                for chan, tab, sel in [
                        (channel, bcands_flag, b_selection & x_selection & selection), 
                ]:
                    dfs[name] = pd.DataFrame()
                    df = dfs[name]
                    df['event'] = tab['event']
                    if len(df['event']) == 0:
                        continue
                    df['run'] = tab['run']
                    df['luminosityBlock'] = tab['luminosityBlock']
                    if dataset == args.mc_hb:
                        df['index'] = tab['index']
                    #Bc Vertex properties
                    df['bvtx_chi2'] = tab.bodies3_chi2
                    df['bvtx_svprob'] = tab.bodies3_svprob
                    df['bvtx_lxy_sig'] = (tab.bodies3_l_xy / tab.bodies3_l_xy_unc)
                    df['bvtx_lxy'] = tab.bodies3_l_xy
                    df['bvtx_lxy_unc'] = tab.bodies3_l_xy_unc
                    df['bvtx_vtx_x'] = tab.bodies3_vtx_x
                    df['bvtx_vtx_y'] = tab.bodies3_vtx_y
                    df['bvtx_vtx_z'] = tab.bodies3_vtx_z
                    df['bvtx_vtx_ex'] = tab.bodies3_vtx_ex
                    df['bvtx_vtx_ey'] = tab.bodies3_vtx_ey
                    df['bvtx_vtx_ez'] = tab.bodies3_vtx_ez
                    df['bvtx_cos2D'] = tab.bodies3_cos2D

                    #jpsi vertex properties
                    df['jpsivtx_chi2'] = tab.jpsivtx_chi2
                    df['jpsivtx_svprob'] = tab.jpsivtx_svprob
                    df['jpsivtx_lxy_sig'] = (tab.jpsivtx_l_xy / tab.jpsivtx_l_xy_unc)
                    df['jpsivtx_lxy'] = tab.jpsivtx_l_xy
                    df['jpsivtx_lxy_unc'] = tab.jpsivtx_l_xy_unc
                    df['jpsivtx_vtx_x'] = tab.jpsivtx_vtx_x
                    df['jpsivtx_vtx_y'] = tab.jpsivtx_vtx_y
                    df['jpsivtx_vtx_z'] = tab.jpsivtx_vtx_z
                    df['jpsivtx_vtx_ex'] = tab.jpsivtx_vtx_ex
                    df['jpsivtx_vtx_ey'] = tab.jpsivtx_vtx_ey
                    df['jpsivtx_vtx_ez'] = tab.jpsivtx_vtx_ez
                    df['jpsivtx_cos2D'] = tab.jpsivtx_cos2D
                    

                    #postfit 3 partc vertex
                    df['bvtx_fit_mass'] = tab.bodies3_fit_mass
                    df['bvtx_fit_massErr'] = tab.bodies3_fit_massErr
                    df['bvtx_fit_pt'] = tab.bodies3_fit_pt
                    df['bvtx_fit_eta'] = tab.bodies3_fit_eta
                    df['bvtx_fit_phi'] = tab.bodies3_fit_phi
                    df['bvtx_fit_mu1_pt'] = tab.bodies3_fit_mu1_pt
                    df['bvtx_fit_mu1_eta'] = tab.bodies3_fit_mu1_eta
                    df['bvtx_fit_mu1_phi'] = tab.bodies3_fit_mu1_phi
                    df['bvtx_fit_mu2_pt'] = tab.bodies3_fit_mu2_pt
                    df['bvtx_fit_mu2_eta'] = tab.bodies3_fit_mu2_pt
                    df['bvtx_fit_mu2_phi'] = tab.bodies3_fit_mu2_pt
                    df['bvtx_fit_cos2D'] = tab.bodies3_fit_cos2D

                    #postfit 2 part vertex
                    df['jpsivtx_fit_mass'] = tab.jpsivtx_fit_mass
                    df['jpsivtx_fit_massErr'] = tab.jpsivtx_fit_massErr
                    df['jpsivtx_fit_pt'] = tab.jpsivtx_fit_pt
                    df['jpsivtx_fit_eta'] = tab.jpsivtx_fit_eta
                    df['jpsivtx_fit_phi'] = tab.jpsivtx_fit_phi
                    df['jpsivtx_fit_mu1_pt'] = tab.jpsivtx_fit_mu1_pt
                    df['jpsivtx_fit_mu1_eta'] = tab.jpsivtx_fit_mu1_eta
                    df['jpsivtx_fit_mu1_phi'] = tab.jpsivtx_fit_mu1_phi
                    df['jpsivtx_fit_mu2_pt'] = tab.jpsivtx_fit_mu2_pt
                    df['jpsivtx_fit_mu2_eta'] = tab.jpsivtx_fit_mu2_pt
                    df['jpsivtx_fit_mu2_phi'] = tab.jpsivtx_fit_mu2_pt
                    df['jpsivtx_fit_cos2D'] = tab.jpsivtx_fit_cos2D
                
                    #iso
                    df['b_iso03'] = tab.b_iso03
                    df['b_iso04'] = tab.b_iso04
                    df['mu1_iso03'] = tab.mu1_iso03
                    df['mu1_iso04'] = tab.mu1_iso04
                    df['mu2_iso03'] = tab.mu2_iso03
                    df['mu2_iso04'] = tab.mu2_iso04
                    
                    #other iso for mu1 and mu2
                    df['mu1_raw_db_corr_iso03'] = tab.mu1.db_corr_iso03
                    df['mu1_raw_db_corr_iso03_rel'] = tab.mu1.db_corr_iso03_rel
                    df['mu1_raw_db_corr_iso04'] = tab.mu1.db_corr_iso04
                    df['mu1_raw_db_corr_iso04_rel'] = tab.mu1.db_corr_iso04_rel
                    df['mu1_raw_ch_pfiso03'] = tab.mu1.raw_ch_pfiso03
                    df['mu1_raw_ch_pfiso03_rel'] = tab.mu1.raw_ch_pfiso03_rel
                    df['mu1_raw_ch_pfiso04'] = tab.mu1.raw_ch_pfiso04
                    df['mu1_raw_ch_pfiso04_rel'] = tab.mu1.raw_ch_pfiso04_rel
                    df['mu1_raw_n_pfiso03'] = tab.mu1.raw_n_pfiso03
                    df['mu1_raw_n_pfiso03_rel'] = tab.mu1.raw_n_pfiso03_rel
                    df['mu1_raw_n_pfiso04'] = tab.mu1.raw_n_pfiso04
                    df['mu1_raw_n_pfiso04_rel'] = tab.mu1.raw_n_pfiso04_rel
                    df['mu1_raw_pho_pfiso03'] = tab.mu1.raw_pho_pfiso03
                    df['mu1_raw_pho_pfiso03_rel'] = tab.mu1.raw_pho_pfiso03_rel
                    df['mu1_raw_pho_pfiso04'] = tab.mu1.raw_pho_pfiso04
                    df['mu1_raw_pho_pfiso04_rel'] = tab.mu1.raw_pho_pfiso04_rel
                    df['mu1_raw_pu_pfiso03'] = tab.mu1.raw_pu_pfiso03
                    df['mu1_raw_pu_pfiso03_rel'] = tab.mu1.raw_pu_pfiso03_rel
                    df['mu1_raw_pu_pfiso04'] = tab.mu1.raw_pu_pfiso04
                    df['mu1_raw_pu_pfiso04_rel'] = tab.mu1.raw_pu_pfiso04_rel
                    df['mu1_raw_trk_iso03'] = tab.mu1.raw_trk_iso03
                    df['mu1_raw_trk_iso03_rel'] = tab.mu1.raw_trk_iso03_rel
                    df['mu1_raw_trk_iso05'] = tab.mu1.raw_trk_iso05
                    df['mu1_raw_trk_iso05_rel'] = tab.mu1.raw_trk_iso05_rel

                    df['mu2_raw_db_corr_iso03'] = tab.mu2.db_corr_iso03
                    df['mu2_raw_db_corr_iso03_rel'] = tab.mu2.db_corr_iso03_rel
                    df['mu2_raw_db_corr_iso04'] = tab.mu2.db_corr_iso04
                    df['mu2_raw_db_corr_iso04_rel'] = tab.mu2.db_corr_iso04_rel
                    df['mu2_raw_ch_pfiso03'] = tab.mu2.raw_ch_pfiso03
                    df['mu2_raw_ch_pfiso03_rel'] = tab.mu2.raw_ch_pfiso03_rel
                    df['mu2_raw_ch_pfiso04'] = tab.mu2.raw_ch_pfiso04
                    df['mu2_raw_ch_pfiso04_rel'] = tab.mu2.raw_ch_pfiso04_rel
                    df['mu2_raw_n_pfiso03'] = tab.mu2.raw_n_pfiso03
                    df['mu2_raw_n_pfiso03_rel'] = tab.mu2.raw_n_pfiso03_rel
                    df['mu2_raw_n_pfiso04'] = tab.mu2.raw_n_pfiso04
                    df['mu2_raw_n_pfiso04_rel'] = tab.mu2.raw_n_pfiso04_rel
                    df['mu2_raw_pho_pfiso03'] = tab.mu2.raw_pho_pfiso03
                    df['mu2_raw_pho_pfiso03_rel'] = tab.mu2.raw_pho_pfiso03_rel
                    df['mu2_raw_pho_pfiso04'] = tab.mu2.raw_pho_pfiso04
                    df['mu2_raw_pho_pfiso04_rel'] = tab.mu2.raw_pho_pfiso04_rel
                    df['mu2_raw_pu_pfiso03'] = tab.mu2.raw_pu_pfiso03
                    df['mu2_raw_pu_pfiso03_rel'] = tab.mu2.raw_pu_pfiso03_rel
                    df['mu2_raw_pu_pfiso04'] = tab.mu2.raw_pu_pfiso04
                    df['mu2_raw_pu_pfiso04_rel'] = tab.mu2.raw_pu_pfiso04_rel
                    df['mu2_raw_trk_iso03'] = tab.mu2.raw_trk_iso03
                    df['mu2_raw_trk_iso03_rel'] = tab.mu2.raw_trk_iso03_rel
                    df['mu2_raw_trk_iso05'] = tab.mu2.raw_trk_iso05
                    df['mu2_raw_trk_iso05_rel'] = tab.mu2.raw_trk_iso05_rel

                    #other iso for k
                    if(channel == 'BTo3Mu'):
                        df['k_raw_db_corr_iso03'] = tab.k.db_corr_iso03
                        df['k_raw_db_corr_iso03_rel'] = tab.k.db_corr_iso03_rel
                        df['k_raw_db_corr_iso04'] = tab.k.db_corr_iso04
                        df['k_raw_db_corr_iso04_rel'] = tab.k.db_corr_iso04_rel
                        df['k_raw_ch_pfiso03'] = tab.k.raw_ch_pfiso03
                        df['k_raw_ch_pfiso03_rel'] = tab.k.raw_ch_pfiso03_rel
                        df['k_raw_ch_pfiso04'] = tab.k.raw_ch_pfiso04
                        df['k_raw_ch_pfiso04_rel'] = tab.k.raw_ch_pfiso04_rel
                        df['k_raw_n_pfiso03'] = tab.k.raw_n_pfiso03
                        df['k_raw_n_pfiso03_rel'] = tab.k.raw_n_pfiso03_rel
                        df['k_raw_n_pfiso04'] = tab.k.raw_n_pfiso04
                        df['k_raw_n_pfiso04_rel'] = tab.k.raw_n_pfiso04_rel
                        df['k_raw_pho_pfiso03'] = tab.k.raw_pho_pfiso03
                        df['k_raw_pho_pfiso03_rel'] = tab.k.raw_pho_pfiso03_rel
                        df['k_raw_pho_pfiso04'] = tab.k.raw_pho_pfiso04
                        df['k_raw_pho_pfiso04_rel'] = tab.k.raw_pho_pfiso04_rel
                        df['k_raw_pu_pfiso03'] = tab.k.raw_pu_pfiso03
                        df['k_raw_pu_pfiso03_rel'] = tab.k.raw_pu_pfiso03_rel
                        df['k_raw_pu_pfiso04'] = tab.k.raw_pu_pfiso04
                        df['k_raw_pu_pfiso04_rel'] = tab.k.raw_pu_pfiso04_rel
                        df['k_raw_trk_iso03'] = tab.k.raw_trk_iso03
                        df['k_raw_trk_iso03_rel'] = tab.k.raw_trk_iso03_rel
                        df['k_raw_trk_iso05'] = tab.k.raw_trk_iso05
                        df['k_raw_trk_iso05_rel'] = tab.k.raw_trk_iso05_rel

                    #trigger of the muon
                    if(channel == 'BTo3Mu'):
                        df['mu1_isFromJpsi_MuT'] = tab.mu1.isMuonFromJpsi_dimuon0Trg
                        df['mu1_isFromJpsi_TrkPsiPT'] = tab.mu1.isMuonFromJpsi_jpsiTrk_PsiPrimeTrg
                        df['mu1_isFromJpsi_TrkT'] = tab.mu1.isMuonFromJpsi_jpsiTrkTrg
                        df['mu1_isFromJpsi_TrkNResT'] = tab.mu1.isMuonFromJpsi_jpsiTrk_NonResonantTrg
                        df['mu1_isFromMuT'] = tab.mu1.isDimuon0Trg
                        df['mu1_isFromTrkT'] = tab.mu1.isJpsiTrkTrg
                        df['mu1_isFromTrkPsiPT'] = tab.mu1.isJpsiTrk_PsiPrimeTrg
                        df['mu1_isFromTrkNResT'] = tab.mu1.isJpsiTrk_NonResonantTrg

                        df['mu2_isFromJpsi_MuT'] = tab.mu2.isMuonFromJpsi_dimuon0Trg
                        df['mu2_isFromJpsi_TrkPsiPT'] = tab.mu2.isMuonFromJpsi_jpsiTrk_PsiPrimeTrg
                        df['mu2_isFromJpsi_TrkT'] = tab.mu2.isMuonFromJpsi_jpsiTrkTrg
                        df['mu2_isFromJpsi_TrkNResT'] = tab.mu2.isMuonFromJpsi_jpsiTrk_NonResonantTrg
                        df['mu2_isFromMuT'] = tab.mu2.isDimuon0Trg
                        df['mu2_isFromTrkT'] = tab.mu2.isJpsiTrkTrg
                        df['mu2_isFromTrkPsiPT'] = tab.mu2.isJpsiTrk_PsiPrimeTrg
                        df['mu2_isFromTrkNResT'] = tab.mu2.isJpsiTrk_NonResonantTrg

                        df['k_isFromJpsi_MuT'] = tab.k.isMuonFromJpsi_dimuon0Trg
                        df['k_isFromJpsi_TrkPsiPT'] = tab.k.isMuonFromJpsi_jpsiTrk_PsiPrimeTrg
                        df['k_isFromJpsi_TrkT'] = tab.k.isMuonFromJpsi_jpsiTrkTrg
                        df['k_isFromJpsi_TrkNResT'] = tab.k.isMuonFromJpsi_jpsiTrk_NonResonantTrg
                        df['k_isFromMuT'] = tab.k.isDimuon0Trg
                        df['k_isFromTrkT'] = tab.k.isJpsiTrkTrg
                        df['k_isFromTrkPsiPT'] = tab.k.isJpsiTrk_PsiPrimeTrg
                        df['k_isFromTrkNResT'] = tab.k.isJpsiTrk_NonResonantTrg

                    #rho
                    df['fixedGridRhoFastjetAll'] = tab.fixedGridRhoFastjetAll
                    df['fixedGridRhoFastjetCentral'] = tab.fixedGridRhoFastjetCentral
                    df['fixedGridRhoFastjetCentralCalo'] = tab.fixedGridRhoFastjetCentralCalo
                    df['fixedGridRhoFastjetCentralChargedPileUp'] = tab.fixedGridRhoFastjetCentralChargedPileUp
                    df['fixedGridRhoFastjetCentralNeutral'] = tab.fixedGridRhoFastjetCentralNeutral
              
                    #beamspot
                    df['beamspot_x'] = tab.beamspot_x
                    df['beamspot_y'] = tab.beamspot_y
                    df['beamspot_z'] = tab.beamspot_z

                    #our variables
                    df['m_miss_sq'] = tab.m_miss_sq
                    df['Q_sq']=tab.Q_sq
                    df['pt_var']=tab.pt_var
                    df['pt_miss_vec']=tab.pt_miss_vec
                    df['pt_miss_scal'] = tab.pt_miss
                    df['DR_mu1mu2']=tab.DR 
                                        
                    #Kinematic
                    df['mu1pt'] = tab.mu1.p4.pt
                    df['mu2pt'] = tab.mu2.p4.pt
                    df['mu1mass'] = tab.mu1.p4.mass
                    df['mu2mass'] = tab.mu2.p4.mass
                    df['mu1phi'] = tab.mu1.p4.phi
                    df['mu2phi'] = tab.mu2.p4.phi
                    df['mu1eta'] = tab.mu1.p4.eta
                    df['mu2eta'] = tab.mu2.p4.eta
                    df['mu1charge'] = tab.mu1.charge
                    df['mu2charge'] = tab.mu2.charge
                    

                    df['Bpt'] = tab.p4.pt
                    df['Bmass'] = tab.p4.mass
                    df['Beta'] = tab.p4.eta
                    df['Bphi'] = tab.p4.phi
                    df['Bcharge'] = tab.charge
                    df['Bpt_reco'] = (tab.p4.pt * 6.275 / tab.p4.mass)

                    df['mu1_dxy'] = tab.mu1_dxy
                    df['mu2_dxy'] = tab.mu2_dxy
                    df['mu1_dxyErr'] = tab.mu1_dxyErr
                    df['mu2_dxyErr'] = tab.mu2_dxyErr
                
                    df['mu1_dz'] = tab.mu1_dz
                    df['mu2_dz'] = tab.mu2_dz
                    df['mu1_dzErr'] = tab.mu1_dzErr
                    df['mu2_dzErr'] = tab.mu2_dzErr             
                    df['nPV'] = tab.nPrimaryVertices
                
                    #PV position
                    df['pv_x'] = tab.pv_x
                    df['pv_y'] = tab.pv_y
                    df['pv_z'] = tab.pv_z
                    
                    df['mu1_mediumID']= tab.mu1.mediumId
                    df['mu1_mediumID']= df['mu1_mediumID'].astype(int)
                    df['mu2_mediumID']= tab.mu2.mediumId
                    df['mu2_mediumID']= df['mu2_mediumID'].astype(int)
                    df['mu1_tightID']= tab.mu1.tightId
                    df['mu1_tightID']= df['mu1_tightID'].astype(int)
                    df['mu2_tightID']= tab.mu2.tightId
                    df['mu2_tightID']= df['mu2_tightID'].astype(int)
                    df['mu1_softID']= tab.mu1.softId
                    df['mu1_softID']= df['mu1_softID'].astype(int)
                    df['mu2_softID']= tab.mu2.softId
                    df['mu2_softID']= df['mu2_softID'].astype(int)

                    if(chan == 'BTo3Mu'):
                        df['k_tightID']= tab.k.tightId
                        df['k_tightID']= df['k_tightID'].astype(int)
                        df['k_mediumID']=tab.k.mediumId
                        df['k_mediumID']= df['k_mediumID'].astype(int)
                        df['k_softID']=tab.k.softId
                        df['k_softID']= df['k_softID'].astype(int)
                    
                    #other muon Ids for mu1
                    df['mu1_looseId'] = tab.mu1.looseId
                    df['mu1_looseId'] = df['mu1_looseId'].astype(int)
                    df['mu1_mediumpromptId'] = tab.mu1.mediumpromptId
                    df['mu1_mediumpromptId'] = df['mu1_mediumpromptId'].astype(int)
                    df['mu1_globalHighPtId'] = tab.mu1.globalHighPtId
                    df['mu1_globalHighPtId'] = df['mu1_globalHighPtId'].astype(int)
                    df['mu1_trkHighPtId'] = tab.mu1.trkHighPtId
                    df['mu1_trkHighPtId'] = df['mu1_trkHighPtId'].astype(int)
                    df['mu1_pfIsoVeryLooseId'] = tab.mu1.pfIsoVeryLooseId
                    df['mu1_pfIsoVeryLooseId'] = df['mu1_pfIsoVeryLooseId'].astype(int)
                    df['mu1_pfIsoLooseId'] = tab.mu1.pfIsoLooseId
                    df['mu1_pfIsoLooseId'] = df['mu1_pfIsoLooseId'].astype(int)
                    df['mu1_pfIsoMediumId'] = tab.mu1.pfIsoMediumId
                    df['mu1_pfIsoMediumId'] = df['mu1_pfIsoMediumId'].astype(int)
                    df['mu1_pfIsoTightId'] = tab.mu1.pfIsoTightId
                    df['mu1_pfIsoTightId'] = df['mu1_pfIsoTightId'].astype(int)
                    df['mu1_pfIsoVeryTightId'] = tab.mu1.pfIsoVeryTightId
                    df['mu1_pfIsoVeryTightId'] = df['mu1_pfIsoVeryTightId'].astype(int)
                    df['mu1_pfIsoVeryVeryTightId'] = tab.mu1.pfIsoVeryVeryTightId
                    df['mu1_pfIsoVeryVeryTightId'] = df['mu1_pfIsoVeryVeryTightId'].astype(int)
                    df['mu1_tkIsoLooseId'] = tab.mu1.tkIsoLooseId
                    df['mu1_tkIsoLooseId'] = df['mu1_tkIsoLooseId'].astype(int)
                    df['mu1_tkIsoTightId'] = tab.mu1.tkIsoTightId
                    df['mu1_tkIsoTightId'] = df['mu1_tkIsoTightId'].astype(int)
                    df['mu1_softMvaId'] = tab.mu1.softMvaId
                    df['mu1_softMvaId'] = df['mu1_softMvaId'].astype(int)
                    df['mu1_mvaLooseId'] = tab.mu1.mvaLooseId
                    df['mu1_mvaLooseId'] = df['mu1_mvaLooseId'].astype(int)
                    df['mu1_mvaTightId'] = tab.mu1.mvaTightId
                    df['mu1_mvaTightId'] = df['mu1_mvaTightId'].astype(int)
                    df['mu1_mvaMediumId'] = tab.mu1.mvaMediumId
                    df['mu1_mvaMediumId'] = df['mu1_mvaMediumId'].astype(int)
                    df['mu1_miniIsoLooseId'] = tab.mu1.miniIsoLooseId
                    df['mu1_miniIsoLooseId'] = df['mu1_miniIsoLooseId'].astype(int)
                    df['mu1_miniIsoMediumId'] = tab.mu1.miniIsoMediumId
                    df['mu1_miniIsoMediumId'] = df['mu1_miniIsoMediumId'].astype(int)
                    df['mu1_miniIsoTightId'] = tab.mu1.miniIsoTightId
                    df['mu1_miniIsoTightId'] = df['mu1_miniIsoTightId'].astype(int)
                    df['mu1_miniIsoVeryTightId'] = tab.mu1.miniIsoVeryTightId
                    df['mu1_miniIsoVeryTightId'] = df['mu1_miniIsoVeryTightId'].astype(int)
                    df['mu1_triggerLooseId'] = tab.mu1.triggerLooseId
                    df['mu1_triggerLooseId'] = df['mu1_triggerLooseId'].astype(int)
                    df['mu1_inTimeMuonId'] = tab.mu1.inTimeMuonId
                    df['mu1_inTimeMuonId'] = df['mu1_inTimeMuonId'].astype(int)
                    df['mu1_multiIsoLooseId'] = tab.mu1.multiIsoLooseId
                    df['mu1_multiIsoLooseId'] = df['mu1_multiIsoLooseId'].astype(int)
                    df['mu1_multiIsoMediumId'] = tab.mu1.multiIsoMediumId
                    df['mu1_multiIsoMediumId'] = df['mu1_multiIsoMediumId'].astype(int)
                    df['mu1_puppiIsoLooseId'] = tab.mu1.puppiIsoLooseId
                    df['mu1_puppiIsoLooseId'] = df['mu1_puppiIsoLooseId'].astype(int)
                    df['mu1_puppiIsoMediumId'] = tab.mu1.puppiIsoMediumId
                    df['mu1_puppiIsoMediumId'] = df['mu1_puppiIsoMediumId'].astype(int)
                    df['mu1_puppiIsoTightId'] = tab.mu1.puppiIsoTightId
                    df['mu1_puppiIsoTightId'] = df['mu1_puppiIsoTightId'].astype(int)
                    df['mu1_mvaVTightId'] = tab.mu1.mvaVTightId
                    df['mu1_mvaVTightId'] = df['mu1_mvaVTightId'].astype(int)
                    df['mu1_mvaVVTightId'] = tab.mu1.mvaVVTightId
                    df['mu1_mvaVVTightId'] = df['mu1_mvaVVTightId'].astype(int)
                    df['mu1_lowPtMvaLooseId'] = tab.mu1.lowPtMvaLooseId
                    df['mu1_lowPtMvaLooseId'] = df['mu1_lowPtMvaLooseId'].astype(int)
                    df['mu1_lowPtMvaMediumId'] = tab.mu1.lowPtMvaMediumId
                    df['mu1_lowPtMvaMediumId'] = df['mu1_lowPtMvaMediumId'].astype(int)

                    #other muon Ids for mu2
                    df['mu2_looseId'] = tab.mu2.looseId
                    df['mu2_looseId'] = df['mu2_looseId'].astype(int)
                    df['mu2_mediumpromptId'] = tab.mu2.mediumpromptId
                    df['mu2_mediumpromptId'] = df['mu2_mediumpromptId'].astype(int)
                    df['mu2_globalHighPtId'] = tab.mu2.globalHighPtId
                    df['mu2_globalHighPtId'] = df['mu2_globalHighPtId'].astype(int)
                    df['mu2_trkHighPtId'] = tab.mu2.trkHighPtId
                    df['mu2_trkHighPtId'] = df['mu2_trkHighPtId'].astype(int)
                    df['mu2_pfIsoVeryLooseId'] = tab.mu2.pfIsoVeryLooseId
                    df['mu2_pfIsoVeryLooseId'] = df['mu2_pfIsoVeryLooseId'].astype(int)
                    df['mu2_pfIsoLooseId'] = tab.mu2.pfIsoLooseId
                    df['mu2_pfIsoLooseId'] = df['mu2_pfIsoLooseId'].astype(int)
                    df['mu2_pfIsoMediumId'] = tab.mu2.pfIsoMediumId
                    df['mu2_pfIsoMediumId'] = df['mu2_pfIsoMediumId'].astype(int)
                    df['mu2_pfIsoTightId'] = tab.mu2.pfIsoTightId
                    df['mu2_pfIsoTightId'] = df['mu2_pfIsoTightId'].astype(int)
                    df['mu2_pfIsoVeryTightId'] = tab.mu2.pfIsoVeryTightId
                    df['mu2_pfIsoVeryTightId'] = df['mu2_pfIsoVeryTightId'].astype(int)
                    df['mu2_pfIsoVeryVeryTightId'] = tab.mu2.pfIsoVeryVeryTightId
                    df['mu2_pfIsoVeryVeryTightId'] = df['mu2_pfIsoVeryVeryTightId'].astype(int)
                    df['mu2_tkIsoLooseId'] = tab.mu2.tkIsoLooseId
                    df['mu2_tkIsoLooseId'] = df['mu2_tkIsoLooseId'].astype(int)
                    df['mu2_tkIsoTightId'] = tab.mu2.tkIsoTightId
                    df['mu2_tkIsoTightId'] = df['mu2_tkIsoTightId'].astype(int)
                    df['mu2_softMvaId'] = tab.mu2.softMvaId
                    df['mu2_softMvaId'] = df['mu2_softMvaId'].astype(int)
                    df['mu2_mvaLooseId'] = tab.mu2.mvaLooseId
                    df['mu2_mvaLooseId'] = df['mu2_mvaLooseId'].astype(int)
                    df['mu2_mvaTightId'] = tab.mu2.mvaTightId
                    df['mu2_mvaTightId'] = df['mu2_mvaTightId'].astype(int)
                    df['mu2_mvaMediumId'] = tab.mu2.mvaMediumId
                    df['mu2_mvaMediumId'] = df['mu2_mvaMediumId'].astype(int)
                    df['mu2_miniIsoLooseId'] = tab.mu2.miniIsoLooseId
                    df['mu2_miniIsoLooseId'] = df['mu2_miniIsoLooseId'].astype(int)
                    df['mu2_miniIsoMediumId'] = tab.mu2.miniIsoMediumId
                    df['mu2_miniIsoMediumId'] = df['mu2_miniIsoMediumId'].astype(int)
                    df['mu2_miniIsoTightId'] = tab.mu2.miniIsoTightId
                    df['mu2_miniIsoTightId'] = df['mu2_miniIsoTightId'].astype(int)
                    df['mu2_miniIsoVeryTightId'] = tab.mu2.miniIsoVeryTightId
                    df['mu2_miniIsoVeryTightId'] = df['mu2_miniIsoVeryTightId'].astype(int)
                    df['mu2_triggerLooseId'] = tab.mu2.triggerLooseId
                    df['mu2_triggerLooseId'] = df['mu2_triggerLooseId'].astype(int)
                    df['mu2_inTimeMuonId'] = tab.mu2.inTimeMuonId
                    df['mu2_inTimeMuonId'] = df['mu2_inTimeMuonId'].astype(int)
                    df['mu2_multiIsoLooseId'] = tab.mu2.multiIsoLooseId
                    df['mu2_multiIsoLooseId'] = df['mu2_multiIsoLooseId'].astype(int)
                    df['mu2_multiIsoMediumId'] = tab.mu2.multiIsoMediumId
                    df['mu2_multiIsoMediumId'] = df['mu2_multiIsoMediumId'].astype(int)
                    df['mu2_puppiIsoLooseId'] = tab.mu2.puppiIsoLooseId
                    df['mu2_puppiIsoLooseId'] = df['mu2_puppiIsoLooseId'].astype(int)
                    df['mu2_puppiIsoMediumId'] = tab.mu2.puppiIsoMediumId
                    df['mu2_puppiIsoMediumId'] = df['mu2_puppiIsoMediumId'].astype(int)
                    df['mu2_puppiIsoTightId'] = tab.mu2.puppiIsoTightId
                    df['mu2_puppiIsoTightId'] = df['mu2_puppiIsoTightId'].astype(int)
                    df['mu2_mvaVTightId'] = tab.mu2.mvaVTightId
                    df['mu2_mvaVTightId'] = df['mu2_mvaVTightId'].astype(int)
                    df['mu2_mvaVVTightId'] = tab.mu2.mvaVVTightId
                    df['mu2_mvaVVTightId'] = df['mu2_mvaVVTightId'].astype(int)
                    df['mu2_lowPtMvaLooseId'] = tab.mu2.lowPtMvaLooseId
                    df['mu2_lowPtMvaLooseId'] = df['mu2_lowPtMvaLooseId'].astype(int)
                    df['mu2_lowPtMvaMediumId'] = tab.mu2.lowPtMvaMediumId
                    df['mu2_lowPtMvaMediumId'] = df['mu2_lowPtMvaMediumId'].astype(int)

                    #other muon Ids for k
                    if(chan == 'BTo3Mu'):
                        df['k_looseId'] = tab.k.looseId
                        df['k_looseId'] = df['k_looseId'].astype(int)
                        df['k_mediumpromptId'] = tab.k.mediumpromptId
                        df['k_mediumpromptId'] = df['k_mediumpromptId'].astype(int)
                        df['k_globalHighPtId'] = tab.k.globalHighPtId
                        df['k_globalHighPtId'] = df['k_globalHighPtId'].astype(int)
                        df['k_trkHighPtId'] = tab.k.trkHighPtId
                        df['k_trkHighPtId'] = df['k_trkHighPtId'].astype(int)
                        df['k_pfIsoVeryLooseId'] = tab.k.pfIsoVeryLooseId
                        df['k_pfIsoVeryLooseId'] = df['k_pfIsoVeryLooseId'].astype(int)
                        df['k_pfIsoLooseId'] = tab.k.pfIsoLooseId
                        df['k_pfIsoLooseId'] = df['k_pfIsoLooseId'].astype(int)
                        df['k_pfIsoMediumId'] = tab.k.pfIsoMediumId
                        df['k_pfIsoMediumId'] = df['k_pfIsoMediumId'].astype(int)
                        df['k_pfIsoTightId'] = tab.k.pfIsoTightId
                        df['k_pfIsoTightId'] = df['k_pfIsoTightId'].astype(int)
                        df['k_pfIsoVeryTightId'] = tab.k.pfIsoVeryTightId
                        df['k_pfIsoVeryTightId'] = df['k_pfIsoVeryTightId'].astype(int)
                        df['k_pfIsoVeryVeryTightId'] = tab.k.pfIsoVeryVeryTightId
                        df['k_pfIsoVeryVeryTightId'] = df['k_pfIsoVeryVeryTightId'].astype(int)
                        df['k_tkIsoLooseId'] = tab.k.tkIsoLooseId
                        df['k_tkIsoLooseId'] = df['k_tkIsoLooseId'].astype(int)
                        df['k_tkIsoTightId'] = tab.k.tkIsoTightId
                        df['k_tkIsoTightId'] = df['k_tkIsoTightId'].astype(int)
                        df['k_softMvaId'] = tab.k.softMvaId
                        df['k_softMvaId'] = df['k_softMvaId'].astype(int)
                        df['k_mvaLooseId'] = tab.k.mvaLooseId
                        df['k_mvaLooseId'] = df['k_mvaLooseId'].astype(int)
                        df['k_mvaTightId'] = tab.k.mvaTightId
                        df['k_mvaTightId'] = df['k_mvaTightId'].astype(int)
                        df['k_mvaMediumId'] = tab.k.mvaMediumId
                        df['k_mvaMediumId'] = df['k_mvaMediumId'].astype(int)
                        df['k_miniIsoLooseId'] = tab.k.miniIsoLooseId
                        df['k_miniIsoLooseId'] = df['k_miniIsoLooseId'].astype(int)
                        df['k_miniIsoMediumId'] = tab.k.miniIsoMediumId
                        df['k_miniIsoMediumId'] = df['k_miniIsoMediumId'].astype(int)
                        df['k_miniIsoTightId'] = tab.k.miniIsoTightId
                        df['k_miniIsoTightId'] = df['k_miniIsoTightId'].astype(int)
                        df['k_miniIsoVeryTightId'] = tab.k.miniIsoVeryTightId
                        df['k_miniIsoVeryTightId'] = df['k_miniIsoVeryTightId'].astype(int)
                        df['k_triggerLooseId'] = tab.k.triggerLooseId
                        df['k_triggerLooseId'] = df['k_triggerLooseId'].astype(int)
                        df['k_inTimeMuonId'] = tab.k.inTimeMuonId
                        df['k_inTimeMuonId'] = df['k_inTimeMuonId'].astype(int)
                        df['k_multiIsoLooseId'] = tab.k.multiIsoLooseId
                        df['k_multiIsoLooseId'] = df['k_multiIsoLooseId'].astype(int)
                        df['k_multiIsoMediumId'] = tab.k.multiIsoMediumId
                        df['k_multiIsoMediumId'] = df['k_multiIsoMediumId'].astype(int)
                        df['k_puppiIsoLooseId'] = tab.k.puppiIsoLooseId
                        df['k_puppiIsoLooseId'] = df['k_puppiIsoLooseId'].astype(int)
                        df['k_puppiIsoMediumId'] = tab.k.puppiIsoMediumId
                        df['k_puppiIsoMediumId'] = df['k_puppiIsoMediumId'].astype(int)
                        df['k_puppiIsoTightId'] = tab.k.puppiIsoTightId
                        df['k_puppiIsoTightId'] = df['k_puppiIsoTightId'].astype(int)
                        df['k_mvaVTightId'] = tab.k.mvaVTightId
                        df['k_mvaVTightId'] = df['k_mvaVTightId'].astype(int)
                        df['k_mvaVVTightId'] = tab.k.mvaVVTightId
                        df['k_mvaVVTightId'] = df['k_mvaVVTightId'].astype(int)
                        df['k_lowPtMvaLooseId'] = tab.k.lowPtMvaLooseId
                        df['k_lowPtMvaLooseId'] = df['k_lowPtMvaLooseId'].astype(int)
                        df['k_lowPtMvaMediumId'] = tab.k.lowPtMvaMediumId
                        df['k_lowPtMvaMediumId'] = df['k_lowPtMvaMediumId'].astype(int)

                    #is PF ?
                    df['mu1_isPF'] = tab.mu1.isPFcand
                    df['mu1_isPF'] = df['mu1_isPF'].astype(int)
                    df['mu2_isPF'] = tab.mu2.isPFcand
                    df['mu2_isPF'] = df['mu2_isPF'].astype(int)
                    if(chan == 'BTo3Mu'):
                        df['k_isPF'] = tab.k.isPFcand
                        df['k_isPF'] = df['k_isPF'].astype(int)
                        
                    df['nB'] = sel.sum()[sel.sum() != 0]

                    if(channel != 'BTo2Mu3P'):
                        df['ip3d'] = tab.ip3D                    
                        df['ip3d_e'] = tab.ip3D_e
                        df['E_mu_star']=tab.E_mu_star
                        df['E_mu_canc']=tab.E_mu_canc
                        df['k_iso03'] = tab.k_iso03
                        df['k_iso04'] = tab.k_iso04
                        
                        df['bvtx_fit_k_pt'] = tab.bodies3_fit_k_pt
                        df['bvtx_fit_k_phi'] = tab.bodies3_fit_k_phi
                        df['bvtx_fit_k_eta'] = tab.bodies3_fit_k_eta
                        
                        df['k_dxyErr'] = tab.k_dxyErr
                        df['k_dzErr'] = tab.k_dzErr
                        
                        df['kpt'] = tab.k.p4.pt
                        df['kmass'] = tab.k.p4.mass
                        df['kphi'] = tab.k.p4.phi
                        df['keta'] = tab.k.p4.eta
                        df['k_dxy'] = tab.k_dxy
                        df['k_dz'] = tab.k_dz
                        df['kcharge'] = tab.charge
                
                    else:
                        df['pi1_iso03'] = tab.pi1_iso03
                        df['pi2_iso03'] = tab.pi2_iso03
                        df['pi3_iso03'] = tab.pi3_iso03
                        df['pi1_iso04'] = tab.pi1_iso04
                        df['pi2_iso04'] = tab.pi2_iso04
                        df['pi3_iso04'] = tab.pi3_iso04
                        df['bvtx_fit_pi1_pt'] = tab.bodies3_fit_pi1_pt
                        df['bvtx_fit_pi1_eta'] = tab.bodies3_fit_pi1_eta
                        df['bvtx_fit_pi1_phi'] = tab.bodies3_fit_pi1_phi
                        df['bvtx_fit_pi2_pt'] = tab.bodies3_fit_pi2_pt
                        df['bvtx_fit_pi2_eta'] = tab.bodies3_fit_pi2_eta
                        df['bvtx_fit_pi2_phi'] = tab.bodies3_fit_pi2_phi
                        df['bvtx_fit_pi3_pt'] = tab.bodies3_fit_pi3_pt
                        df['bvtx_fit_pi3_eta'] = tab.bodies3_fit_pi3_eta
                        df['bvtx_fit_pi3_phi'] = tab.bodies3_fit_pi3_phi

                        #impact parameter
                        df['pi1_dxyErr'] = tab.pi1_dxyErr
                        df['pi1_dzErr'] = tab.pi1_dzErr
                        df['pi1_dxy'] = tab.pi1_dxy
                        df['pi1_dz'] = tab.pi1_dz
                        df['pi2_dxyErr'] = tab.pi2_dxyErr
                        df['pi2_dzErr'] = tab.pi2_dzErr
                        df['pi2_dxy'] = tab.pi2_dxy
                        df['pi2_dz'] = tab.pi2_dz
                        df['pi3_dxyErr'] = tab.pi3_dxyErr
                        df['pi3_dzErr'] = tab.pi3_dzErr
                        df['pi3_dxy'] = tab.pi3_dxy
                        df['pi3_dz'] = tab.pi3_dz
                        
                        df['pi1pt'] = tab.pi1.p4.pt
                        df['pi1mass'] = tab.pi1.p4.mass
                        df['pi1phi'] = tab.pi1.p4.phi
                        df['pi1eta'] = tab.pi1.p4.eta
                        df['pi2pt'] = tab.pi2.p4.pt
                        df['pi2mass'] = tab.pi2.p4.mass
                        df['pi2phi'] = tab.pi2.p4.phi
                        df['pi2eta'] = tab.pi2.p4.eta
                        df['pi3pt'] = tab.pi3.p4.pt
                        df['pi3mass'] = tab.pi3.p4.mass
                        df['pi3phi'] = tab.pi3.p4.phi
                        df['pi3eta'] = tab.pi3.p4.eta
                        df['pi1charge'] = tab.pi1.charge
                        df['pi2charge'] = tab.pi2.charge
                        df['pi3charge'] = tab.pi3.charge

                        df['pi1_vy'] = tab.pi1.vy
                        df['pi1_vx'] = tab.pi1.vx
                        df['pi1_vz'] = tab.pi1.vz
                        df['pi2_vy'] = tab.pi2.vy
                        df['pi2_vx'] = tab.pi2.vx
                        df['pi2_vz'] = tab.pi2.vz
                        df['pi3_vy'] = tab.pi3.vy
                        df['pi3_vx'] = tab.pi3.vx
                        df['pi3_vz'] = tab.pi3.vz

                    if(dataset!=args.data):
                        #PU weight
                        if flag_pu_weight:
                            df['puWeight'] = tab.puWeight
                            df['puWeightUp'] = tab.puWeightUp
                            df['puWeightDown'] = tab.puWeightDown
                        if(dataset == args.mc_bc):    
                            # branches of the Bc GEN info (for hammer)
                            df['bc_gen_pt'] = tab.BcGenInfo_bc_gen_pt
                            df['bc_gen_eta'] = tab.BcGenInfo_bc_gen_eta
                            df['bc_gen_phi'] = tab.BcGenInfo_bc_gen_phi
                            df['bc_gen_mass'] = tab.BcGenInfo_bc_gen_mass
                            
                            df['jpsi_gen_pt'] = tab.BcGenInfo_jpsi_gen_pt
                            df['jpsi_gen_eta'] = tab.BcGenInfo_jpsi_gen_eta
                            df['jpsi_gen_phi'] = tab.BcGenInfo_jpsi_gen_phi
                            df['jpsi_gen_mass'] = tab.BcGenInfo_jpsi_gen_mass
                            
                            df['tau_gen_pt'] = tab.BcGenInfo_tau_gen_pt
                            df['tau_gen_eta'] = tab.BcGenInfo_tau_gen_eta
                            df['tau_gen_phi'] = tab.BcGenInfo_tau_gen_phi
                            df['tau_gen_mass'] = tab.BcGenInfo_tau_gen_mass
                            
                            df['mu3_gen_pt'] = tab.BcGenInfo_mu3_gen_pt
                            df['mu3_gen_eta'] = tab.BcGenInfo_mu3_gen_eta
                            df['mu3_gen_phi'] = tab.BcGenInfo_mu3_gen_phi
                            df['mu3_gen_mass'] = tab.BcGenInfo_mu3_gen_mass

                        #gen Part Flavour e gen Part Idx  -> if I need to access the gen info, this values tell me is it is a valid info or not
                        df['mu1_genPartFlav'] = tab.mu1.genPartFlav
                        df['mu2_genPartFlav'] = tab.mu2.genPartFlav
                                          
                        df['mu1_genPartIdx'] = tab.mu1.genPartIdx
                        df['mu2_genPartIdx'] = tab.mu2.genPartIdx
      
                        #lifetime (gen info)
                        df['mu1_gen_vx'] = tab.mu1.gen.vx
                        df['mu2_gen_vx'] = tab.mu2.gen.vx
                        
                        df['mu1_gen_vy'] = tab.mu1.gen.vy
                        df['mu2_gen_vy'] = tab.mu2.gen.vy
                    
                        df['mu1_gen_vz'] = tab.mu1.gen.vz
                        df['mu2_gen_vz'] = tab.mu2.gen.vz
                        
                        #pdgId
                        df['mu1_pdgId'] = tab.mu1.pdgId
                        df['mu2_pdgId'] = tab.mu2.pdgId
                        
                        df['mu1_genpdgId'] = tab.mu1.gen.pdgId
                        df['mu2_genpdgId'] = tab.mu2.gen.pdgId
                        
                        #particele gen info
                        df['mu1_gen_pt'] = tab.mu1.gen.p4.pt
                        df['mu2_gen_pt'] = tab.mu2.gen.p4.pt

                        df['mu1_gen_eta'] = tab.mu1.gen.p4.eta
                        df['mu2_gen_eta'] = tab.mu2.gen.p4.eta

                        df['mu1_gen_phi'] = tab.mu1.gen.p4.phi
                        df['mu2_gen_phi'] = tab.mu2.gen.p4.phi

                        #mother info
                        df['mu1_mother_pdgId'] = tab.mu1.mother.pdgId
                        df['mu2_mother_pdgId'] = tab.mu2.mother.pdgId
                        
                        df['mu1_mother_pt'] = tab.mu1.mother.p4.pt
                        df['mu2_mother_pt'] = tab.mu2.mother.p4.pt
                        
                        df['mu1_mother_eta'] = tab.mu1.mother.p4.eta
                        df['mu2_mother_eta'] = tab.mu2.mother.p4.eta
                        
                        df['mu1_mother_phi'] = tab.mu1.mother.p4.phi
                        df['mu2_mother_phi'] = tab.mu2.mother.p4.phi
                        
                        df['mu1_mother_vx'] = tab.mu1.mother.vx
                        df['mu2_mother_vx'] = tab.mu2.mother.vx

                        df['mu1_mother_vy'] = tab.mu1.mother.vy
                        df['mu2_mother_vy'] = tab.mu2.mother.vy
                        df['mu1_mother_vz'] = tab.mu1.mother.vz
                        df['mu2_mother_vz'] = tab.mu2.mother.vz

                        #grandmother info
                        df['mu1_grandmother_pdgId'] = tab.mu1.grandmother.pdgId
                        df['mu2_grandmother_pdgId'] = tab.mu2.grandmother.pdgId
                        
                        df['mu1_grandmother_pt'] = tab.mu1.grandmother.p4.pt
                        df['mu2_grandmother_pt'] = tab.mu2.grandmother.p4.pt

                        
                        df['mu1_grandmother_eta'] = tab.mu1.grandmother.p4.eta
                        df['mu2_grandmother_eta'] = tab.mu2.grandmother.p4.eta
                        
                        df['mu1_grandmother_phi'] = tab.mu1.grandmother.p4.phi
                        df['mu2_grandmother_phi'] = tab.mu2.grandmother.p4.phi

                        df['mu1_grandmother_vx'] = tab.mu1.grandmother.vx
                        df['mu2_grandmother_vx'] = tab.mu2.grandmother.vx
                        df['mu1_grandmother_vy'] = tab.mu1.grandmother.vy
                        df['mu2_grandmother_vy'] = tab.mu2.grandmother.vy
                        df['mu1_grandmother_vz'] = tab.mu1.grandmother.vz
                        df['mu2_grandmother_vz'] = tab.mu2.grandmother.vz

                        
                        if(channel != 'BTo2Mu3P'):
                            df['k_genpdgId'] = tab.k.gen.pdgId
                            df['k_pdgId'] = tab.k.pdgId
                            df['k_gen_vz'] = tab.k.gen.vz
                            df['k_genPartIdx'] = tab.k.genPartIdx
                            df['k_genPartFlav'] = tab.k.genPartFlav
                            df['k_gen_vx'] = tab.k.gen.vx
                            df['k_gen_vy'] = tab.k.gen.vy

                            df['k_gen_pt'] = tab.k.gen.p4.pt
                            df['k_gen_eta'] = tab.k.gen.p4.eta
                            df['k_gen_phi'] = tab.k.gen.p4.phi

                            df['k_mother_pdgId'] = tab.k.mother.pdgId
                            df['k_mother_pt'] = tab.k.mother.p4.pt
                            df['k_mother_eta'] = tab.k.mother.p4.eta
                            df['k_mother_phi'] = tab.k.mother.p4.phi
                            df['k_mother_vx'] = tab.k.mother.vx
                            df['k_mother_vy'] = tab.k.mother.vy
                            df['k_mother_vz'] = tab.k.mother.vz
                            
                            df['k_grandmother_pdgId'] = tab.k.grandmother.pdgId
                            df['k_grandmother_pt'] = tab.k.grandmother.p4.pt
                            df['k_grandmother_eta'] = tab.k.grandmother.p4.eta
                            df['k_grandmother_phi'] = tab.k.grandmother.p4.phi                        
                            df['k_grandmother_vx'] = tab.k.grandmother.vx
                            df['k_grandmother_vy'] = tab.k.grandmother.vy
                            df['k_grandmother_vz'] = tab.k.grandmother.vz

                        else:
                            df['pi1_genpdgId'] = tab.pi1.gen.pdgId
                            df['pi1_pdgId'] = tab.pi1.pdgId
                            df['pi1_gen_vz'] = tab.pi1.gen.vz
                            df['pi1_genPartIdx'] = tab.pi1.genPartIdx
                            df['pi1_genPartFlav'] = tab.pi1.genPartFlav
                            df['pi1_gen_vx'] = tab.pi1.gen.vx
                            df['pi1_gen_vy'] = tab.pi1.gen.vy
                            df['pi1_mother_pdgId'] = tab.pi1.mother.pdgId
                            df['pi1_mother_pt'] = tab.pi1.mother.p4.pt
                            df['pi1_mother_eta'] = tab.pi1.mother.p4.eta
                            df['pi1_mother_phi'] = tab.pi1.mother.p4.phi
                            df['pi1_mother_vx'] = tab.pi1.mother.vx
                            df['pi1_mother_vy'] = tab.pi1.mother.vy
                            df['pi1_mother_vz'] = tab.pi1.mother.vz
                            
                            df['pi1_grandmother_pdgId'] = tab.pi1.grandmother.pdgId
                            df['pi1_grandmother_pt'] = tab.pi1.grandmother.p4.pt
                            df['pi1_grandmother_eta'] = tab.pi1.grandmother.p4.eta
                            df['pi1_grandmother_phi'] = tab.pi1.grandmother.p4.phi                        
                            df['pi1_grandmother_vx'] = tab.pi1.grandmother.vx
                            df['pi1_grandmother_vy'] = tab.pi1.grandmother.vy
                            df['pi1_grandmother_vz'] = tab.pi1.grandmother.vz

                            df['pi2_genpdgId'] = tab.pi2.gen.pdgId
                            df['pi2_pdgId'] = tab.pi2.pdgId
                            df['pi2_gen_vz'] = tab.pi2.gen.vz
                            df['pi2_genPartIdx'] = tab.pi2.genPartIdx
                            df['pi2_genPartFlav'] = tab.pi2.genPartFlav
                            df['pi2_gen_vx'] = tab.pi2.gen.vx
                            df['pi2_gen_vy'] = tab.pi2.gen.vy
                            df['pi2_mother_pdgId'] = tab.pi2.mother.pdgId
                            df['pi2_mother_pt'] = tab.pi2.mother.p4.pt
                            df['pi2_mother_eta'] = tab.pi2.mother.p4.eta
                            df['pi2_mother_phi'] = tab.pi2.mother.p4.phi
                            df['pi2_mother_vx'] = tab.pi2.mother.vx
                            df['pi2_mother_vy'] = tab.pi2.mother.vy
                            df['pi2_mother_vz'] = tab.pi2.mother.vz
                            
                            df['pi2_grandmother_pdgId'] = tab.pi2.grandmother.pdgId
                            df['pi2_grandmother_pt'] = tab.pi2.grandmother.p4.pt
                            df['pi2_grandmother_eta'] = tab.pi2.grandmother.p4.eta
                            df['pi2_grandmother_phi'] = tab.pi2.grandmother.p4.phi                        
                            df['pi2_grandmother_vx'] = tab.pi2.grandmother.vx
                            df['pi2_grandmother_vy'] = tab.pi2.grandmother.vy
                            df['pi2_grandmother_vz'] = tab.pi2.grandmother.vz

                            df['pi3_genpdgId'] = tab.pi3.gen.pdgId
                            df['pi3_pdgId'] = tab.pi3.pdgId
                            df['pi3_gen_vz'] = tab.pi3.gen.vz
                            df['pi3_genPartIdx'] = tab.pi3.genPartIdx
                            df['pi3_genPartFlav'] = tab.pi3.genPartFlav
                            df['pi3_gen_vx'] = tab.pi3.gen.vx
                            df['pi3_gen_vy'] = tab.pi3.gen.vy
                            df['pi3_mother_pdgId'] = tab.pi3.mother.pdgId
                            df['pi3_mother_pt'] = tab.pi3.mother.p4.pt
                            df['pi3_mother_eta'] = tab.pi3.mother.p4.eta
                            df['pi3_mother_phi'] = tab.pi3.mother.p4.phi
                            df['pi3_mother_vx'] = tab.pi3.mother.vx
                            df['pi3_mother_vy'] = tab.pi3.mother.vy
                            df['pi3_mother_vz'] = tab.pi3.mother.vz
                            
                            df['pi3_grandmother_pdgId'] = tab.pi3.grandmother.pdgId
                            df['pi3_grandmother_pt'] = tab.pi3.grandmother.p4.pt
                            df['pi3_grandmother_eta'] = tab.pi3.grandmother.p4.eta
                            df['pi3_grandmother_phi'] = tab.pi3.grandmother.p4.phi                        
                            df['pi3_grandmother_vx'] = tab.pi3.grandmother.vx
                            df['pi3_grandmother_vy'] = tab.pi3.grandmother.vy
                            df['pi3_grandmother_vz'] = tab.pi3.grandmother.vz

                        
                        if (dataset == args.mc_hb):
                            df['jpsimother_bzero'] = tab.jpsimother_bzero
                            df['jpsimother_bplus'] = tab.jpsimother_bplus
                            df['jpsimother_bplus_c'] = tab.jpsimother_bplus_c
                            df['jpsimother_bzero_s'] = tab.jpsimother_bzero_s
                            df['jpsimother_sigmaminus_b'] = tab.jpsimother_sigmaminus_b
                            df['jpsimother_lambdazero_b'] = tab.jpsimother_lambdazero_b
                            df['jpsimother_ximinus_b'] = tab.jpsimother_ximinus_b
                            df['jpsimother_sigmazero_b'] = tab.jpsimother_sigmazero_b
                            df['jpsimother_xizero_b'] = tab.jpsimother_xizero_b
                            df['jpsimother_other'] = tab.jpsimother_other
                            df['jpsimother_weight'] = tab.jpsimother_weight

                        if (dataset!=args.mc_mu):
                            #grand grand mother info
                            df['mu1_grandgrandmother_pdgId'] = tab.mu1.grandgrandmother.pdgId
                            df['mu2_grandgrandmother_pdgId'] = tab.mu2.grandgrandmother.pdgId
                            
                            df['mu1_grandgrandmother_pt'] = tab.mu1.grandgrandmother.p4.pt
                            df['mu2_grandgrandmother_pt'] = tab.mu2.grandgrandmother.p4.pt
                            
                            df['mu1_grandgrandmother_eta'] = tab.mu1.grandgrandmother.p4.eta
                            df['mu2_grandgrandmother_eta'] = tab.mu2.grandgrandmother.p4.eta

                            
                            df['mu1_grandgrandmother_phi'] = tab.mu1.grandgrandmother.p4.phi
                            df['mu2_grandgrandmother_phi'] = tab.mu2.grandgrandmother.p4.phi

                            
                            df['mu1_grandgrandmother_vx'] = tab.mu1.grandgrandmother.vx
                            df['mu2_grandgrandmother_vx'] = tab.mu2.grandgrandmother.vx

                            df['mu1_grandgrandmother_vy'] = tab.mu1.grandgrandmother.vy
                            df['mu2_grandgrandmother_vy'] = tab.mu2.grandgrandmother.vy
                            df['mu1_grandgrandmother_vz'] = tab.mu1.grandgrandmother.vz
                            df['mu2_grandgrandmother_vz'] = tab.mu2.grandgrandmother.vz

                            if(channel != 'BTo2Mu3P'):
                                df['k_grandgrandmother_pdgId'] = tab.k.grandgrandmother.pdgId
                                df['k_grandgrandmother_pt'] = tab.k.grandgrandmother.p4.pt
                                df['k_grandgrandmother_eta'] = tab.k.grandgrandmother.p4.eta  
                                df['k_grandgrandmother_phi'] = tab.k.grandgrandmother.p4.phi
                                df['k_grandgrandmother_vx'] = tab.k.grandgrandmother.vx
                                df['k_grandgrandmother_vy'] = tab.k.grandgrandmother.vy
                                df['k_grandgrandmother_vz'] = tab.k.grandgrandmother.vz

                            else:
                                df['pi1_grandgrandmother_pdgId'] = tab.pi1.grandgrandmother.pdgId
                                df['pi1_grandgrandmother_pt'] = tab.pi1.grandgrandmother.p4.pt
                                df['pi1_grandgrandmother_eta'] = tab.pi1.grandgrandmother.p4.eta  
                                df['pi1_grandgrandmother_phi'] = tab.pi1.grandgrandmother.p4.phi
                                df['pi1_grandgrandmother_vx'] = tab.pi1.grandgrandmother.vx
                                df['pi1_grandgrandmother_vy'] = tab.pi1.grandgrandmother.vy
                                df['pi1_grandgrandmother_vz'] = tab.pi1.grandgrandmother.vz

                                df['pi2_grandgrandmother_pdgId'] = tab.pi2.grandgrandmother.pdgId
                                df['pi2_grandgrandmother_pt'] = tab.pi2.grandgrandmother.p4.pt
                                df['pi2_grandgrandmother_eta'] = tab.pi2.grandgrandmother.p4.eta  
                                df['pi2_grandgrandmother_phi'] = tab.pi2.grandgrandmother.p4.phi
                                df['pi2_grandgrandmother_vx'] = tab.pi2.grandgrandmother.vx
                                df['pi2_grandgrandmother_vy'] = tab.pi2.grandgrandmother.vy
                                df['pi2_grandgrandmother_vz'] = tab.pi2.grandgrandmother.vz
                                
                                df['pi3_grandgrandmother_pdgId'] = tab.pi3.grandgrandmother.pdgId
                                df['pi3_grandgrandmother_pt'] = tab.pi3.grandgrandmother.p4.pt
                                df['pi3_grandgrandmother_eta'] = tab.pi3.grandgrandmother.p4.eta  
                                df['pi3_grandgrandmother_phi'] = tab.pi3.grandgrandmother.p4.phi
                                df['pi3_grandgrandmother_vx'] = tab.pi3.grandgrandmother.vx
                                df['pi3_grandgrandmother_vy'] = tab.pi3.grandgrandmother.vy
                                df['pi3_grandgrandmother_vz'] = tab.pi3.grandgrandmother.vz

                    # if the dataframe is empty, it will fill the branches with NaN
                    if(dataset == args.mc_mu or dataset == args.mc_tau or dataset == args.mc_bc):
                        df = lifetime_weight(df, fake = False)
                    else:
                        df = lifetime_weight(df)
                    df = jpsi_branches(df)
                    if channel != 'BTo2Mu3P':
                        df = DR_jpsimu(df)
                        df = mcor(df)
                        df = dr12(df)
                        df = dr23(df)
                        df = dr13(df)
                        if channel == 'BTo3Mu':
                            df = rho_corr_iso(df)
                    df = decaytime(df)
                    if channel == 'BTo3Mu':
                        df = bp4_lhcb(df)
                        if (dataset == args.mc_hb):
                            df = HighMassLowMassDivision(df)
                                        
                    flag_init_hammer = 0
                    #print("dataset:",dataset," channel:", channel)
                    if((dataset == args.mc_mu or (dataset == args.mc_bc and name == 'is_jpsi_mu')) and flag_hammer_mu and channel =='BTo3Mu'):
                        if not flag_init_hammer:
                            ham = Hammer()
                            fbBuffer = IOBuffer
                            flag_init_hammer =1
                        df = hammer_weights_mu(df,ham)

                    if((dataset == args.mc_tau or (dataset == args.mc_bc and name == 'is_jpsi_tau')) and flag_hammer_tau and channel =='BTo3Mu'):
                        if not flag_init_hammer:
                            ham = Hammer()
                            fbBuffer = IOBuffer
                            flag_init_hammer =1
                        df = hammer_weights_tau(df,ham)
                    ##########################################################
                    ##### Concatenate the dataframe to the total one #########
                    ##########################################################
                    if(channel=='BTo3Mu'):
                        final_dfs_mmm[name] = pd.concat((final_dfs_mmm[name], dfs[name])) 
                    elif(channel == 'BTo2MuP'):
                        final_dfs_pmm[name] = pd.concat((final_dfs_pmm[name], dfs[name])) 
                    elif(channel == 'BTo2MuK'):
                        final_dfs_kmm[name] = pd.concat((final_dfs_kmm[name], dfs[name])) 
                    elif(channel == 'BTo2Mu3P'):
                        final_dfs_2m3p[name] = pd.concat((final_dfs_2m3p[name], dfs[name])) 
                    if(nprocessedDataset > maxEvents and maxEvents != -1):
                        break
    
    ######################################
    ####### Save  ########################
    ######################################
    dataset=dataset.strip('.txt')
    name=dataset.split('/')
    d=name[len(name)-1].split('_')
    adj='_v7_'
  
    for flag in flag_names:
        for channel in channels:
            if channel == 'BTo3Mu':
                final_dfs_mmm[flag].to_root('REPLACE_FILE_OUT'+'_'+flag+'.root', key=channel)
            elif (channel == 'BTo2MuP'):
                final_dfs_pmm[flag].to_root('REPLACE_FILE_OUT'+'_'+flag+'.root', key=channel, mode = 'a')
            elif (channel == 'BTo2MuK'):
                final_dfs_kmm[flag].to_root('REPLACE_FILE_OUT'+'_'+flag+'.root', key=channel, mode = 'a')
            elif (channel == 'BTo2Mu3P'):
                final_dfs_2m3p[flag].to_root('REPLACE_FILE_OUT'+'_'+flag+'.root', key=channel, mode = 'a')
        print("Saved file "+ 'REPLACE_FILE_OUT'+'_'+flag+'.root')

print('DONE! Processed events: ', nprocessedAll)
