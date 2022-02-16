'''
Script that computes the form factor weights for the Bc-> jpsi tau process
Starting from Kiselev FF (standard MC FF), this script reweights to BGL FF (updated)
It takes as input a flat ntupla (like output of inspector)
N.B. The NAN probles has been solved: all the hammer weights should be different from NaN
'''
import ROOT
from itertools import product
from time import time
from datetime import datetime, timedelta
from hammer.hammerlib import Hammer, IOBuffer, Particle, Process, FourMomentum, WTerm
from hammer import hepmc, pdg
from root_pandas import read_root, to_root
import numpy as np
from bgl_variations import variations

ham = Hammer()
fbBuffer = IOBuffer

# We don't add the tau decay, becasue that doesn't change the FF of the Bc
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

fname = 'inspector_output_tau_v1.root'
fin = ROOT.TFile.Open(fname)
tree = fin.Get('tree')
maxevents = 1000
tree_df = read_root(fname, 'tree', where='is_jpsi_tau & is3m & ismu3fromtau & bhad_pdgid == 541')

pids = []
weights = dict()

for k in ff_schemes.keys():
        weights[k] = []

start = time()
maxevents = maxevents if maxevents>=0 else tree.GetEntries() # total number of events in the files


for i, ev in enumerate(tree):

    if (i+1)>maxevents:
        break
        
    if i%1000==0:
        percentage = float(i)/maxevents*100.
        speed = float(i)/(time()-start)
        eta = datetime.now() + timedelta(seconds=(maxevents-i) / max(0.1, speed))
        print('\t===> processing %d / %d event \t completed %.1f%s \t %.1f ev/s \t ETA %s s' %(i, maxevents, percentage, '%', speed, eta.strftime('%Y-%m-%d %H:%M:%S')))

    if not ev.is_jpsi_tau: continue
    if not ev.is3m: continue
    if not ev.ismu3fromtau: continue
    if not ev.bhad_pdgid == 541: continue
    
    ham.init_event()
    
    thebc_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.bhad_pt, ev.bhad_eta, ev.bhad_phi, ev.bhad_m)
    thejpsi_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.jpsi_pt, ev.jpsi_eta, ev.jpsi_phi, ev.jpsi_m)
    thetau_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.tau_pt, ev.tau_eta, ev.tau_phi, ev.tau_m)

    thenutau_p4   = thebc_p4 - thetau_p4 - thejpsi_p4


    thebc   = Particle(FourMomentum(thebc_p4.e()  , thebc_p4.px()  , thebc_p4.py()  , thebc_p4.pz()  ), ev.bhad_pdgid)
    thejpsi = Particle(FourMomentum(thejpsi_p4.e(), thejpsi_p4.px(), thejpsi_p4.py(), thejpsi_p4.pz()), 443          )
    thetau = Particle(FourMomentum(thetau_p4.e(), thetau_p4.px(), thetau_p4.py(), thetau_p4.pz()), -15          )
    thenutau   = Particle(FourMomentum(thenutau_p4.e()  , thenutau_p4.px()  , thenutau_p4.py()  , thenutau_p4.pz())  , 16     )

    Bc2JpsiLNu = Process()
    
    # each of these add_particle operations returns an index, needed to define vertices 
    thebc_idx   = Bc2JpsiLNu.add_particle(thebc  )
    thejpsi_idx = Bc2JpsiLNu.add_particle(thejpsi)
    thetau_idx   = Bc2JpsiLNu.add_particle(thetau  )
    thenutau_idx   = Bc2JpsiLNu.add_particle(thenutau  )

    # define decay vertex
    Bc2JpsiLNu.add_vertex(thebc_idx, [thejpsi_idx, thetau_idx, thenutau_idx])

    # save process id to later retrieve the per-event weight
    pid = ham.add_process(Bc2JpsiLNu)
    pids.append(pid)

    ham.process_event()
    for k in ff_schemes.keys():
            weights[k].append(ham.get_weight(k))
    
    if i>maxevents: break

reduced_tree = tree_df[:len(weights[k])].copy()
for k in ff_schemes.keys():
        reduced_tree['hammer_'+k] = np.nan_to_num(np.array(weights[k])) 
to_root(reduced_tree, 'hammer_output_tau_v2.root', key='tree')

