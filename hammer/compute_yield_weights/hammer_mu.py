'''
Script that computes the form factor weights for the Bc-> jpsi mu process
Starting from Kiselev FF (standard MC FF), this script reweights to BGL FF (updated)
It takes as input a flat ntupla (like output of inspector)
N.B. The NAN probles has been solved: all the hammer weights should be different from NaN
'''
import ROOT
from time import time
from datetime import datetime, timedelta
from hammer.hammerlib import Hammer, IOBuffer, Particle, Process, FourMomentum, WTerm
from hammer import hepmc, pdg
from root_pandas import read_root, to_root
import numpy as np
import math

ham = Hammer()
fbBuffer = IOBuffer
ham.include_decay("BcJpsiMuNu")

ff_input_scheme = dict()
ff_input_scheme["BcJpsi"] = "Kiselev"
ham.set_ff_input_scheme(ff_input_scheme)

ff_scheme  = dict()

ff_scheme['BcJpsi'] = 'BGLVar'
ham.add_ff_scheme('BGL', ff_scheme)
ham.add_total_sum_of_weights() # adds "Total Sum of Weights" histo with auto bin filling
ham.set_units("GeV")
ham.init_run()

#input (Kiselev)
fname = 'inspector_output_mu_v1.root'

fin = ROOT.TFile.Open(fname)
tree = fin.Get('tree')

maxevents = -1

tree_df = read_root(fname, 'tree', where='is_jpsi_mu & is3m')

pids = []
weights = []

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

    if not ev.is_jpsi_mu: continue
    if not ev.is3m: continue

    ham.init_event()
    
    thebc_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.bhad_pt, ev.bhad_eta, ev.bhad_phi, ev.bhad_m)
    themu_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.mu3_pt , ev.mu3_eta , ev.mu3_phi , 0.1056   )
    thejpsi_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.jpsi_pt, ev.jpsi_eta, ev.jpsi_phi, ev.jpsi_m)

    thenu_p4   = thebc_p4 - themu_p4 - thejpsi_p4
        
    thebc   = Particle(FourMomentum(thebc_p4.e()  , thebc_p4.px()  , thebc_p4.py()  , thebc_p4.pz()  ), ev.bhad_pdgid)
    themu   = Particle(FourMomentum(themu_p4.e()  , themu_p4.px()  , themu_p4.py()  , themu_p4.pz()  ), -13          )
    thejpsi = Particle(FourMomentum(thejpsi_p4.e(), thejpsi_p4.px(), thejpsi_p4.py(), thejpsi_p4.pz()), 443          )
    thenu   = Particle(FourMomentum(thenu_p4.e()  , thenu_p4.px()  , thenu_p4.py()  , thenu_p4.pz())  , 14           )

    Bc2JpsiLNu = Process()
    
    # each of these add_particle operations returns an index, needed to define vertices 
    thebc_idx   = Bc2JpsiLNu.add_particle(thebc  )
    themu_idx   = Bc2JpsiLNu.add_particle(themu  )
    thejpsi_idx = Bc2JpsiLNu.add_particle(thejpsi)
    thenu_idx   = Bc2JpsiLNu.add_particle(thenu  )


    # define decay vertex
    Bc2JpsiLNu.add_vertex(thebc_idx, [thejpsi_idx, themu_idx, thenu_idx])

    # save process id to later retrieve the per-event weight
    pid = ham.add_process(Bc2JpsiLNu)
    pids.append(pid)

    ham.process_event()
    weights.append(ham.get_weight('BGL'))
    if i>maxevents: break

reduced_tree = tree_df[:len(weights)].copy()

#it shouldn't be needed anymore: nan problem solved
reduced_tree['hammer'] = np.nan_to_num(np.array(weights)) 
#output file
to_root(reduced_tree, 'hammer_output_mu_v1.root', key='tree')

