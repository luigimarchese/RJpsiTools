## It works only with 1 file at the time! FIXME the output
#double decay
import ROOT
from time import time
from datetime import datetime, timedelta
from hammer.hammerlib import Hammer, IOBuffer, Particle, Process, FourMomentum, WTerm
from hammer import hepmc, pdg
from root_pandas import read_root, to_root
import numpy as np
from glob import glob
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--files_per_job', dest='files_per_job', default=2    , type=int)
parser.add_argument('--jobid'        , dest='jobid'        , default=0    , type=int)
parser.add_argument('--verbose'      , dest='verbose'      , default=0    , type=int)
parser.add_argument('--destination'  , dest='destination'  , default='.'  , type=str)
parser.add_argument('--maxevents'    , dest='maxevents'    , default=-1   , type=int)
args = parser.parse_args()

files_per_job = args.files_per_job
jobid         = args.jobid
verbose       = args.verbose
destination   = args.destination
maxevents     = args.maxevents

ham = Hammer()
fbBuffer = IOBuffer

ham.include_decay(["BcJpsiTauNu"])

ff_input_scheme = dict()
ff_input_scheme["BcJpsi"] = "Kiselev"
ham.set_ff_input_scheme(ff_input_scheme)

ff_scheme  = dict()

ff_scheme['BcJpsi'] = 'BGLVar'
ham.add_ff_scheme('BGL', ff_scheme)

ham.set_units("GeV")
ham.init_run()


files = glob('/pnfs/psi.ch/cms/trivcat/store/user/friti/HOOK_INPUT/*.root')
files.sort()
files = files[2:]
files = files[(jobid)*files_per_job:(jobid+1)*files_per_job]
print("files: ",files)


for fname in files:
    fin = ROOT.TFile.Open(fname)
    tree = fin.Get('tree')
    
    maxevents = -1
    
    tree_df = read_root(fname, 'tree', where='is_jpsi_tau & is3m & ismu3fromtau & bhad_pdgid == 541')
    
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

        if not ev.is_jpsi_tau: continue
        if not ev.is3m: continue
        if not ev.ismu3fromtau: continue
        if not ev.bhad_pdgid == 541: continue
        ham.init_event()
    
        thebc_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.bhad_pt, ev.bhad_eta, ev.bhad_phi, ev.bhad_m)
        thejpsi_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.jpsi_pt, ev.jpsi_eta, ev.jpsi_phi, ev.jpsi_m)
        thetau_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.tau_pt, ev.tau_eta, ev.tau_phi, ev.tau_m)
        thenutau_p4   = thebc_p4 - thetau_p4 - thejpsi_p4
        
        # not used
        themu_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.mu3_pt , ev.mu3_eta , ev.mu3_phi , 0.1056   )
        thenutaubar_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.nutau_pt , ev.nutau_eta , ev.nutau_phi , 0   )
        thenumu_p4   = thetau_p4 - themu_p4 - thenutaubar_p4
        
        thebc   = Particle(FourMomentum(thebc_p4.e()  , thebc_p4.px()  , thebc_p4.py()  , thebc_p4.pz()  ), ev.bhad_pdgid)
        thejpsi = Particle(FourMomentum(thejpsi_p4.e(), thejpsi_p4.px(), thejpsi_p4.py(), thejpsi_p4.pz()), 443          )
        thetau = Particle(FourMomentum(thetau_p4.e(), thetau_p4.px(), thetau_p4.py(), thetau_p4.pz()), -15          )
        thenutau   = Particle(FourMomentum(thenutau_p4.e()  , thenutau_p4.px()  , thenutau_p4.py()  , thenutau_p4.pz())  , 16           )
        
        themu   = Particle(FourMomentum(themu_p4.e()  , themu_p4.px()  , themu_p4.py()  , themu_p4.pz()  ), -13          )
        thenutaubar   = Particle(FourMomentum(thenutaubar_p4.e()  , thenutaubar_p4.px()  , thenutaubar_p4.py()  , thenutaubar_p4.pz())  , -16           )
        thenumu   = Particle(FourMomentum(thenumu_p4.e()  , thenumu_p4.px()  , thenumu_p4.py()  , thenumu_p4.pz())  , 14           )
        
        Bc2JpsiLNu = Process()
        
        # each of these add_particle operations returns an index, needed to define vertices 
        thebc_idx   = Bc2JpsiLNu.add_particle(thebc  )
        thejpsi_idx = Bc2JpsiLNu.add_particle(thejpsi)
        thetau_idx   = Bc2JpsiLNu.add_particle(thetau  )
        thenutau_idx   = Bc2JpsiLNu.add_particle(thenutau  )
        
        themu_idx   = Bc2JpsiLNu.add_particle(themu  )
        thenutaubar_idx   = Bc2JpsiLNu.add_particle(thenutaubar  )
        thenumu_idx   = Bc2JpsiLNu.add_particle(thenumu  )
        
        # define decay vertex
        Bc2JpsiLNu.add_vertex(thebc_idx, [thejpsi_idx, thetau_idx, thenutau_idx])
        
        # save process id to later retrieve the per-event weight
        pid = ham.add_process(Bc2JpsiLNu)
        pids.append(pid)
        ham.process_event()
        weights.append(ham.get_weight('BGL', [pid]))
        
        if i>maxevents: break

    reduced_tree = tree_df[:len(weights)].copy()
    reduced_tree['hammer'] = np.nan_to_num(np.array(weights)) # sone NaNs, check the manual
    to_root(reduced_tree, 'HOOK_FILE_OUT', key='tree')

    print("Success!")        
    print("File HOOK_FILE_OUT saved!" )
