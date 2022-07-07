'''
Script that computes the form factor weights for the Bc-> jpsi mu process
Starting from Kiselev FF (standard MC FF), this script reweights to BGL FF (updated)
It takes as input a flat ntupla (like output of inspector)
N.B. The NAN probles has been solved: all the hammer weights should be different from NaN
'''
import argparse
import ROOT
from time import time
from datetime import datetime, timedelta
from hammer.hammerlib import Hammer, IOBuffer, Particle, Process, FourMomentum, WTerm
from hammer import hepmc, pdg
from root_pandas import read_root, to_root
import numpy as np
import math
from glob import glob

ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--files_per_job', dest='files_per_job', default=2    , type=int)
parser.add_argument('--jobid'        , dest='jobid'        , default=0    , type=int)
parser.add_argument('--verbose'      , dest='verbose'      , default=0    , type=int)
parser.add_argument('--destination'  , dest='destination'  , default='.'  , type=str)
parser.add_argument('--maxevents'    , dest='maxevents'    , default=-1   , type=int)
args = parser.parse_args()

files_per_job = 1
jobid         = args.jobid
verbose       = args.verbose
destination   = args.destination
maxevents     = args.maxevents

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

files = glob('/pnfs/psi.ch/cms/trivcat/store/user/friti/HOOK_INPUT/*.root')
files.sort()
fname = files[(jobid)*files_per_job:(jobid+1)*files_per_job]
fname = fname[0]
print("files: ",fname)

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
to_root(reduced_tree, 'HOOK_FILE_OUT', key='tree')
print("Success!")        
print("File HOOK_FILE_OUT saved!" )

