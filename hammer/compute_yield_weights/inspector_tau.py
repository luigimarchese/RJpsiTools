'''
Script that takes a ntuple that is from GEN up to miniAOD (in the last case add option --miniaod 1)
and gives as an output a flat ntupla that contains all the gen infos.
It looks for events of the kind Bc-> jpsi TAU (saves the info of the mu3 and of tau)
'''
from __future__ import print_function
import ROOT
import argparse
import numpy as np
from time import time
from datetime import datetime, timedelta
from array import array
from glob import glob
from collections import OrderedDict
from scipy.constants import c as speed_of_light
from DataFormats.FWLite import Events, Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi
# https://pypi.org/project/particle/
from particle import Particle


parser = argparse.ArgumentParser(description='')
parser.add_argument('--files_per_job', dest='files_per_job', default=2    , type=int)
parser.add_argument('--jobid'        , dest='jobid'        , default=0    , type=int)
parser.add_argument('--verbose'      , dest='verbose'      , default=0    , type=int)
parser.add_argument('--destination'  , dest='destination'  , default='.'  , type=str)
parser.add_argument('--maxevents'    , dest='maxevents'    , default=-1   , type=int)
parser.add_argument('--miniaod'      , dest='is_miniaod' , default=1, type=int)
args = parser.parse_args()

files_per_job = args.files_per_job
jobid         = args.jobid
verbose       = args.verbose
destination   = args.destination
maxevents     = args.maxevents
is_miniaod    = args.is_miniaod


diquarks = [
    1103,
    2101,
    2103,
    2203,
    3101,
    3103,
    3201,
    3203,
    3303,
    4101,
    4103,
    4201,
    4203,
    4301,
    4303,
    4403,
    5101,
    5103,
    5201,
    5203,
    5301,
    5303,
    5401,
    5403,
    5503,
]

# Bc lifetime
# https://pdglive.lbl.gov/DataBlock.action?node=S091T&home=MXXX049
ctau_pdg    = 0.510e-12 * speed_of_light * 1000. # in mm
ctau_actual = 0.507e-12 * speed_of_light * 1000. # in mm
ctau_up     = (0.510+0.009)*1e-12 * speed_of_light * 1000. # in mm
ctau_down   = (0.510-0.009)*1e-12 * speed_of_light * 1000. # in mm

def weight_to_new_ctau(old_ctau, new_ctau, ct):
    '''
    Returns an event weight based on the ratio of the normalised lifetime distributions.
    old_ctau: ctau used for the sample production
    new_ctau: target ctau
    ct      : per-event lifetime
    '''
    weight = old_ctau/new_ctau * np.exp( (1./old_ctau - 1./new_ctau) * ct )
    return weight

def isAncestor(a, p):
    if a == p :
        return True
    for i in xrange(0,p.numberOfMothers()):
        if isAncestor(a,p.mother(i)):
            return True
    return False

def printAncestors(particle, ancestors=[], verbose=True):
    for i in xrange(0, particle.numberOfMothers()):
        mum = particle.mother(i)
        if abs(mum.pdgId())<8 or abs(mum.pdgId())==21: continue
        if not mum.isLastCopy(): continue
        try:
            if verbose: print(' <-- ', Particle.from_pdgid(mum.pdgId()).name, end = '')
            ancestors.append(mum)
            printAncestors(mum, ancestors=ancestors, verbose=verbose)
        except:
            if verbose: print(' <-- ', 'pdgid', mum.pdgId(), end = '')
            ancestors.append(mum)
            printAncestors(mum, ancestors=ancestors, verbose=verbose)
        else:
            pass
    
def makePackedLinks(packedCollection):
    links = {}
    
    for particle in packedCollection:
        # packedGenParticles only have one mother and it may be themselves
        mum = particle.lastPrunedRef().get()
        
        if mum.status() == 1:
            # we are in prunedGenParticles and mother is us, no need for link
            continue
        
        # use address, as hashing the particles does not work properly
        links.setdefault(ROOT.addressof(mum), []).append(particle)
    
    return links


handles = OrderedDict()
handles['genp'   ] = ('genParticles', Handle('std::vector<reco::GenParticle>'))
handles['genInfo'] = ('generator'   , Handle('GenEventInfoProduct'           ))

if is_miniaod:
    handles['genp'   ] = ('prunedGenParticles', Handle('std::vector<reco::GenParticle>'))
    handles['pgp'   ] = ('packedGenParticles', Handle('std::vector<pat::PackedGenParticle>'))

files = ['root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAOD/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v11_L1v1-v2/00000/AF312F62-A71A-B948-9155-7FF158EA9A5A.root']

events = Events(files)

branches = [
    'run',
    'lumi',
    'event',

    'qscale',

    'min_bq_pt',
    'max_bq_eta',

    'mu1_pt',
    'mu1_eta',
    'mu1_y',
    'mu1_phi',
    'mu1_q',

    'mu2_pt',
    'mu2_eta',
    'mu2_y',
    'mu2_phi',
    'mu2_q',

    'jpsi_pt',
    'jpsi_eta',
    'jpsi_y',
    'jpsi_phi',
    'jpsi_m',
    'jpsi_status',

    'bhad_pt',
    'bhad_eta',
    'bhad_y',
    'bhad_phi',
    'bhad_m',
    'bhad_q',
    'bhad_pdgid',

    'tau_pt',
    'tau_eta',
    'tau_y',
    'tau_phi',
    'tau_q',
    'tau_pdgid',
    'tau_m',

    'nutau_pt',
    'nutau_eta',
    'nutau_phi',
    'nutau_y',
    'nutau_pdgid',
    'nutau_q',


    'mmm_pt',
    'mmm_eta',
    'mmm_y',
    'mmm_phi',
    'mmm_m',
    'mmm_q',

    'm2_miss',
    'pt_miss_sca',
    'pt_miss_vec',
    'q2',
    'e_star_mu3',
    'e_hash_mu3',
    'ptvar',
    'm2_miss_reco',
    'pt_miss_sca_reco',
    'pt_miss_vec_reco',
    'q2_reco',
    'e_star_mu3_reco',
    
    'ismu3fromtau',
    'mu3_pt',
    'mu3_eta',
    'mu3_y',
    'mu3_phi',
    'mu3_q',

    'dr_jpsi_m',

    'is3m',

    'is_jpsi_mu'  ,
    'is_psi2s_mu' ,
    'is_chic0_mu' ,
    'is_chic1_mu' ,
    'is_chic2_mu' ,
    'is_hc_mu'    ,
    'is_jpsi_tau' ,
    'is_psi2s_tau',
    'is_jpsi_pi'  ,
    'is_jpsi_k'   ,
    'is_jpsi_3pi' ,
    'is_jpsi_hc'  ,
    'is_jpsi_pppi',

    'pv_x'  ,
    'pv_y'  ,
    'pv_z'  ,

    'sv_x'  ,
    'sv_y'  ,
    'sv_z'  ,

    'lxyz'  ,
    'beta'  ,
    'gamma' ,

    'n_extra_mu',
    'n_jpsi',
    
    'weight',
]

#output file
fout = ROOT.TFile('%s/inspector_output_tau_v1.root' %(destination), 'recreate')
ntuple = ROOT.TNtuple('tree', 'tree', ':'.join(branches))
tofill = OrderedDict(zip(branches, [np.nan]*len(branches)))

start = time()
maxevents = maxevents if maxevents>=0 else events.size() # total number of events in the files

for i, event in enumerate(events):
    if (i+1)>maxevents:
        break
        
    if i%1000==0:
        percentage = float(i)/maxevents*100.
        speed = float(i)/(time()-start)
        eta = datetime.now() + timedelta(seconds=(maxevents-i) / max(0.1, speed))
        print('\t===> processing %d / %d event \t completed %.1f%s \t %.1f ev/s \t ETA %s s' %(i, maxevents, percentage, '%', speed, eta.strftime('%Y-%m-%d %H:%M:%S')))
    # access the handles
    for k, v in handles.iteritems():
        event.getByLabel(v[0], v[1])
        setattr(event, k, v[1].product())

    event.qscale = event.genInfo.qScale()
   
    if verbose: print('=========>')
    
    jpsis = [ip for ip in event.genp if abs(ip.pdgId())==443]    
    bs =  [ip for ip in event.genp if (abs(ip.pdgId())>500 and abs(ip.pdgId())<600) or (abs(ip.pdgId())>5000 and abs(ip.pdgId())<6000)]
    muons =  [ip for ip in event.genp if abs(ip.pdgId())==13 and ip.status()==1]
    if is_miniaod:
        # this should not be needed, but we never know
        packed_muons = [ip for ip in event.pgp if abs(ip.pdgId())==13 and ip.lastPrunedRef().status() != 1]
    else:
        packed_muons = []
    muons = sorted(muons + packed_muons, key = lambda ip: ip.pt(), reverse = True)
    
    packedLinks = None
    if is_miniaod:
        packedLinks = makePackedLinks(event.pgp)


    bq = [ip for ip in event.genp if abs(ip.pdgId())==5 and ip.isHardProcess()]

    #find the jpsi
    for jpsi in jpsis:
        
        for k, v in tofill.items(): 
            tofill[k] = np.nan

        tofill['min_bq_pt' ] = min([ip.pt() for ip in bq]) if len(bq) else np.nan
        tofill['max_bq_eta'] = max([abs(ip.eta()) for ip in bq]) if len(bq) else np.nan

        tofill['n_jpsi'] = len(jpsis)

        #find the daughters of the jpsi
        daus = [jpsi.daughter(idau).pdgId() for idau in range(jpsi.numberOfDaughters())]
        if verbose: print('\t%s %s pt %3.2f,\t genealogy: ' %(Particle.from_pdgid(jpsi.pdgId()), str(daus), jpsi.pt()), end='')

        #jpsi ancestors
        ancestors = []
        if verbose: print('\t', printAncestors(jpsi, ancestors, verbose=True))
        else: printAncestors(jpsi, ancestors, verbose=False)
        
        # only save jpsi->mumu
        if sum([abs(dau)==13 for dau in daus])<2: continue
                
        if len(ancestors)==0: continue
        
        first_ancestor = ancestors[-1]
        if first_ancestor.pdgId() in diquarks:
            first_ancestor = ancestors[-2]
        
        # compute the distance between primary and secondary vtx
        sv = jpsi.vertex()
        pv = first_ancestor.vertex()
        lxyz = np.sqrt((sv.x()-pv.x())**2 + (sv.y()-pv.y())**2 + (sv.z()-pv.z())**2) # in [cm]
        
        # lorentz boost of the B
        beta = first_ancestor.p4().Beta()
        gamma = first_ancestor.p4().Gamma()

        # now, lifetime L = beta * gamma * c * t ===> t = (L)/(beta*gamma*c)
        ct = lxyz / (beta * gamma)
        
        tofill['run'        ] = event.eventAuxiliary().run()
        tofill['lumi'       ] = event.eventAuxiliary().luminosityBlock()
        tofill['event'      ] = event.eventAuxiliary().event()
        tofill['qscale'     ] = event.qscale
        tofill['lxyz'       ] = lxyz
        tofill['beta'       ] = beta
        tofill['gamma'      ] = gamma
        tofill['pv_x'       ] = pv.x()
        tofill['pv_y'       ] = pv.y()
        tofill['pv_z'       ] = pv.z()
        tofill['sv_x'       ] = sv.x()
        tofill['sv_y'       ] = sv.y()
        tofill['sv_z'       ] = sv.z()
        tofill['jpsi_pt'    ] = jpsi.pt()
        tofill['jpsi_eta'   ] = jpsi.eta()        
        tofill['jpsi_y'     ] = jpsi.y()
        tofill['jpsi_phi'   ] = jpsi.phi()
        tofill['jpsi_m'     ] = jpsi.mass()
        tofill['jpsi_status'] = jpsi.status()
        tofill['bhad_pt'    ] = first_ancestor.pt()
        tofill['bhad_eta'   ] = first_ancestor.eta()
        tofill['bhad_y'     ] = first_ancestor.y()
        tofill['bhad_phi'   ] = first_ancestor.phi()
        tofill['bhad_m'     ] = first_ancestor.mass()
        tofill['bhad_q'     ] = first_ancestor.charge()
        tofill['bhad_pdgid' ] = first_ancestor.pdgId()
                
        final_state_muons = [ip for ip in event.genp if abs(ip.pdgId())==13 and ip.status()==1]
        tofill['is3m'      ] = len(final_state_muons)>=3

        jpsi_muons = [imu for imu in final_state_muons if isAncestor(jpsi, imu)]
        jpsi_muons.sort(key = lambda x : x.pt(), reverse = True)
        if len(jpsi_muons)<2:
            continue

        tofill['mu1_pt' ] = jpsi_muons[0].pt()
        tofill['mu1_eta'] = jpsi_muons[0].eta()
        tofill['mu1_y'  ] = jpsi_muons[0].y()
        tofill['mu1_phi'] = jpsi_muons[0].phi()
        tofill['mu1_q'  ] = jpsi_muons[0].charge()
        tofill['mu2_pt' ] = jpsi_muons[1].pt()
        tofill['mu2_eta'] = jpsi_muons[1].eta()
        tofill['mu2_y'  ] = jpsi_muons[1].y()
        tofill['mu2_phi'] = jpsi_muons[1].phi()
        tofill['mu2_q'  ] = jpsi_muons[1].charge()

        if abs(first_ancestor.pdgId()) in [541, 543]:
            final_daus = []
            if abs(jpsi.mother(0).pdgId())==541:
                the_b = jpsi.mother(0)
                its_a_b = True
            else:    
                the_b = jpsi.mother(0).mother(0)
                its_a_b = False

            for ii in range(the_b.numberOfDaughters()): 
                if the_b.daughter(ii).pdgId() not in [22, 541]:
                    final_daus.append(abs(the_b.daughter(ii).pdgId()))
                        
            final_daus.sort()
    
            tofill['is_jpsi_mu'  ] = (final_daus==[13, 14, 443       ]) 
            tofill['is_psi2s_mu' ] = (final_daus==[13, 14, 100443    ]) 
            tofill['is_chic0_mu' ] = (final_daus==[13, 14, 10441     ]) 
            tofill['is_chic1_mu' ] = (final_daus==[13, 14, 20443     ]) 
            tofill['is_chic2_mu' ] = (final_daus==[13, 14, 445       ]) 
            tofill['is_hc_mu'    ] = (final_daus==[13, 14, 10443     ]) 
            tofill['is_jpsi_tau' ] = (final_daus==[15, 16, 443       ]) 
            tofill['is_psi2s_tau'] = (final_daus==[15, 16, 100443    ]) 
            tofill['is_jpsi_pi'  ] = (final_daus==[211, 443          ]) 
            tofill['is_jpsi_k'   ] = (final_daus==[321, 443          ]) 
            tofill['is_jpsi_3pi' ] = (final_daus==[211, 211, 211, 443]) 
            tofill['is_jpsi_hc'  ] = ((final_daus==[431, 443])      or \
                                      (final_daus==[433, 443])      or \
                                      (final_daus==[411, 443])      or \
                                      (final_daus==[313, 413, 443]) or \
                                      (final_daus==[321, 423, 443]) or \
                                      (final_daus==[413, 443])      or \
                                      (final_daus==[321, 421, 443]) or \
                                      (final_daus==[313, 411, 443]) )
            tofill['is_jpsi_pppi'] = (final_daus==[211, 443, 2212, 2212]) 
            
            known_decay = 0
            known_decay = tofill['is_jpsi_mu'  ] + \
                tofill['is_psi2s_mu' ] + \
                tofill['is_chic0_mu' ] + \
                tofill['is_chic1_mu' ] + \
                tofill['is_chic2_mu' ] + \
                tofill['is_hc_mu'    ] + \
                tofill['is_jpsi_tau' ] + \
                tofill['is_psi2s_tau'] + \
                tofill['is_jpsi_pi'  ] + \
                tofill['is_jpsi_k'   ] + \
                tofill['is_jpsi_3pi' ] + \
                tofill['is_jpsi_hc'  ] + \
                tofill['is_jpsi_pppi'] 
            if not known_decay:
                print('unknown decay', final_daus)
            
            # weight branch is outdated
            if   tofill['is_jpsi_mu'  ]: 
                tofill['weight'] = 1.
            elif tofill['is_psi2s_mu' ]: 
                tofill['weight'] = 0.5474 # Psi(2S) -> J/Psi X BR, forced decay at generation
            elif tofill['is_chic0_mu' ]: 
                tofill['weight'] = 0.0116
            elif tofill['is_chic1_mu' ]: 
                tofill['weight'] = 0.3440
            elif tofill['is_chic2_mu' ]: 
                tofill['weight'] = 0.1950
            elif tofill['is_hc_mu'    ]: 
                tofill['weight'] = 0.01
            elif tofill['is_jpsi_tau' ]: 
                tofill['weight'] = 1. 
            elif tofill['is_psi2s_tau']: 
                tofill['weight'] = 0.5474
            elif tofill['is_jpsi_pi'  ]: 
                tofill['weight'] = 1.
            elif tofill['is_jpsi_k'   ]: 
                tofill['weight'] = 1.
            elif tofill['is_jpsi_3pi' ]: 
                tofill['weight'] = 1.
            elif tofill['is_jpsi_hc'  ]: 
                tofill['weight'] = 1.
            elif tofill['is_jpsi_pppi']: 
                tofill['weight'] = 1.
            else                       : 
                tofill['weight'] = -1.

            if len(final_state_muons)>=3:
                final_state_muons_non_jpsi = [imu for imu in final_state_muons if imu not in jpsi_muons]
                final_state_muons_non_jpsi.sort(key = lambda x : x.pt(), reverse = True)
            
                extra_mu = final_state_muons_non_jpsi[0]

                #looking for the tau, daighter of the Bc
                if(final_daus==[15, 16, 443]):
                    taus =  [ip for ip in event.genp if abs(ip.pdgId())==15 and ip.status()==2]
                    nutaus = [ip for ip in event.genp if abs(ip.pdgId())==16]# and ip.status()==1]
                    tau_frombc =[itau for itau in taus if isAncestor(first_ancestor,itau)]
                    if(len(tau_frombc)>1):
                        #I take the one with highest pt
                        tau_frombc.sort(key = lambda x : x.pt(), reverse = True)
                    extra_mu_fromtau = [imu for imu in final_state_muons_non_jpsi if isAncestor(tau_frombc[0],imu)]
                    nutau_fromtau = [inu for inu in nutaus if isAncestor(tau_frombc[0],inu)]

                    tofill['tau_pt' ] = tau_frombc[0].pt()
                    tofill['tau_eta'] = tau_frombc[0].eta()
                    tofill['tau_y'  ] = tau_frombc[0].y()
                    tofill['tau_phi'] = tau_frombc[0].phi()
                    tofill['tau_q'  ] = tau_frombc[0].charge()
                    tofill['tau_pdgid'  ] = tau_frombc[0].pdgId()
                    tofill['tau_m'  ] = tau_frombc[0].mass()
                    tofill['nutau_pt' ] = nutau_fromtau[0].pt()
                    tofill['nutau_eta'] = nutau_fromtau[0].eta()
                    tofill['nutau_y'  ] = nutau_fromtau[0].y()
                    tofill['nutau_phi'] = nutau_fromtau[0].phi()
                    tofill['nutau_q'  ] = nutau_fromtau[0].charge()
                    tofill['nutau_pdgid'  ] = nutau_fromtau[0].pdgId()
                    
                    #it finds the tau, parent of the extra mu
                    if(len(extra_mu_fromtau)!=0):
                        extra_mu = extra_mu_fromtau[0]
                    tofill['ismu3fromtau'] = len(extra_mu_fromtau)!=0
                else:
                    tofill['ismu3fromtau'] = 0
                tofill['mu3_pt' ] = extra_mu.pt()
                tofill['mu3_eta'] = extra_mu.eta()
                tofill['mu3_y'  ] = extra_mu.y()
                tofill['mu3_phi'] = extra_mu.phi()
                tofill['mu3_q'  ] = extra_mu.charge()
                
                three_mu_p4 = extra_mu.p4() + jpsi_muons[0].p4() + jpsi_muons[1].p4()
                b_scaled_p4 = three_mu_p4 * (6.275/three_mu_p4.mass())
                
                tofill['mmm_pt' ] = three_mu_p4.pt()
                tofill['mmm_eta'] = three_mu_p4.eta()
                tofill['mmm_y'  ] = three_mu_p4.y()
                tofill['mmm_phi'] = three_mu_p4.phi()
                tofill['mmm_m'  ] = three_mu_p4.mass()
                tofill['mmm_q'  ] = extra_mu.charge()
                
                tofill['dr_jpsi_m'] = deltaR(extra_mu, jpsi)
                
                tofill['n_extra_mu'] = len(final_state_muons_non_jpsi)
                
                # how the hell does one boost LorentzVectors back and forth?!
                # using TLorentzVectors of course... shit.
                three_mu_p4_tlv = ROOT.TLorentzVector() ; three_mu_p4_tlv.SetPtEtaPhiE(three_mu_p4.pt(), three_mu_p4.eta(), three_mu_p4.phi(), three_mu_p4.energy())
                b_scaled_p4_tlv = ROOT.TLorentzVector() ; b_scaled_p4_tlv.SetPtEtaPhiE(b_scaled_p4.pt(), b_scaled_p4.eta(), b_scaled_p4.phi(), b_scaled_p4.energy())
                jpsi_p4_tlv     = ROOT.TLorentzVector() ; jpsi_p4_tlv    .SetPtEtaPhiE(jpsi.p4()  .pt(), jpsi.p4()  .eta(), jpsi.p4()  .phi(), jpsi.p4()  .energy())
                extra_mu_p4_tlv = ROOT.TLorentzVector() ; extra_mu_p4_tlv.SetPtEtaPhiE(extra_mu   .pt(), extra_mu   .eta(), extra_mu   .phi(), extra_mu   .energy())
                b_gen_p4_tlv = ROOT.TLorentzVector() ; b_gen_p4_tlv.SetPtEtaPhiM(first_ancestor.pt(),first_ancestor.eta(),first_ancestor.phi(),first_ancestor.mass())
                
                three_mu_p4_boost = three_mu_p4_tlv.BoostVector()
                b_scaled_p4_boost = b_scaled_p4_tlv.BoostVector()
                b_gen_p4_boost = b_gen_p4_tlv.BoostVector()
                jpsi_p4_boost     = jpsi_p4_tlv    .BoostVector()
                
                extra_mu_p4_in_b_rf    = extra_mu_p4_tlv.Clone(); extra_mu_p4_in_b_rf   .Boost(-b_scaled_p4_boost)
                extra_mu_p4_in_bgen_rf    = extra_mu_p4_tlv.Clone(); extra_mu_p4_in_bgen_rf   .Boost(-b_gen_p4_boost)
                extra_mu_p4_in_jpsi_rf = extra_mu_p4_tlv.Clone(); extra_mu_p4_in_jpsi_rf.Boost(-jpsi_p4_boost    )
                
                tofill['m2_miss'    ] = (b_gen_p4_tlv - jpsi_p4_tlv - extra_mu_p4_tlv).M2()
                tofill['m2_miss_reco'    ] = (b_scaled_p4 - extra_mu.p4() - jpsi_muons[0].p4() - jpsi_muons[1].p4()).mass2()
                tofill['pt_miss_sca'] = b_gen_p4_tlv.Pt() - extra_mu_p4_tlv.Pt() - jpsi_p4_tlv.Pt()
                tofill['pt_miss_sca_reco'] = b_scaled_p4.pt() - extra_mu.pt() - jpsi_muons[0].pt() - jpsi_muons[1].pt()
                tofill['pt_miss_vec'] = (b_gen_p4_tlv - extra_mu_p4_tlv - jpsi_p4_tlv).Pt()
                tofill['pt_miss_vec_reco'] = (b_scaled_p4 - extra_mu.p4() - jpsi_muons[0].p4() - jpsi_muons[1].p4()).pt()
                tofill['q2'] = (b_gen_p4_tlv - jpsi_p4_tlv).M2()
                tofill['q2_reco'         ] = (b_scaled_p4 - jpsi.p4()).mass2()
                
                tofill['e_star_mu3' ] = extra_mu_p4_in_bgen_rf   .E()
                tofill['e_star_mu3_reco' ] = extra_mu_p4_in_b_rf   .E()
                tofill['e_hash_mu3' ] = extra_mu_p4_in_jpsi_rf.E()
                tofill['ptvar'      ] = jpsi.pt() - extra_mu.pt()
            
        # fill only if it comes from  a Bc
        if abs(first_ancestor.pdgId()) in [541, 543]:
            ntuple.Fill(array('f', tofill.values()))

    if verbose:
        if len(bs)>1: # and len(jpsis)>1:
            print('--> b hadrons')
            for aa, ib in enumerate(bs):
                try:
                    print('---->', Particle.from_pdgid(ib.pdgId()))
                except:
                    print('---->', ib.pdgId())
                for idau in range(ib.numberOfDaughters()):
                    try: 
                        print('\t\t %d]-th B\t'%aa, Particle.from_pdgid(ib.daughter(idau).pdgId()))
                    except: 
                        print('\t\t %d]-th B\t'%aa, ib.daughter(idau).pdgId())
        if len(muons)<3:
            print('--> muons')
        if len(jpsis)>1:
            print('--> jpsis')
            for ij in jpsis:
                if abs(ij.daughter(0).pdgId())!=13:
                    pass
    
fout.cd()
ntuple.Write()
fout.Close()
    
