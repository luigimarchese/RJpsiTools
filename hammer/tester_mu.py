import ROOT
from time import time
from datetime import datetime, timedelta
from hammer.hammerlib import Hammer, IOBuffer, Particle, Process, FourMomentum, WTerm
from hammer import hepmc, pdg
from root_pandas import read_root, to_root
import numpy as np

ham = Hammer()
fbBuffer = IOBuffer
# "BcJpsiMuNu", "BcJpsiTauNu", TauEllNuNu
ham.include_decay("BcJpsiMuNu")

# ham.set_options({'ProcessCalc': {'CheckForNaNs': True}})

ff_input_scheme = dict()
ff_input_scheme["BcJpsi"] = "EFG"
#ff_input_scheme["BcJpsi"] = "Kiselev"
ham.set_ff_input_scheme(ff_input_scheme)

ff_scheme  = dict()

#ff_scheme['BcJpsi'] = 'BGLVar'
#ham.add_ff_scheme('BGL', ff_scheme)
# closure test
ff_scheme['BcJpsi'] = 'Kiselev'
ham.add_ff_scheme('Kiselev', ff_scheme)
#ff_scheme['BcJpsi'] = 'EFG'
#ham.add_ff_scheme('EFG', ff_scheme)

#ham.add_histogram("pEllVsQ2:Bc", [6, 5], True, [[0., 2.5], [3., 12.]])
ham.add_total_sum_of_weights() # adds "Total Sum of Weights" histo with auto bin filling
#ham.keep_errors_in_histogram("pEllVsQ2:Bc", True)
#ham.collapse_processes_in_histogram("pEllVsQ2:Bc")
ham.set_units("GeV")
ham.init_run()

#input di ebert(efg), reweighted as Kiselev
#fname = 'flat_tree_ebert20k.root'
#fname = 'flat_tree_ebert_all_newgen.root'
fname = 'inspector/flat_tree_ebert_mu_14Apr21.root'

fin = ROOT.TFile.Open(fname)
tree = fin.Get('tree')

maxevents = -1

tree_df = read_root(fname, 'tree', where='is_jpsi_mu & is3m')

pids = []
weights = []

start = time()
maxevents = maxevents if maxevents>=0 else tree.GetEntries() # total number of events in the files

# for i, ev in enumerate(tree):

for i, ev in enumerate(tree):

    if (i+1)>maxevents:
        break
        
    if i%1000==0:
        percentage = float(i)/maxevents*100.
        speed = float(i)/(time()-start)
        eta = datetime.now() + timedelta(seconds=(maxevents-i) / max(0.1, speed))
        print('\t===> processing %d / %d event \t completed %.1f%s \t %.1f ev/s \t ETA %s s' %(i, maxevents, percentage, '%', speed, eta.strftime('%Y-%m-%d %H:%M:%S')))

    #     if i<9800: continue
    
    if not ev.is_jpsi_mu: continue
    if not ev.is3m: continue

    #     print ("processing event", i)
    
    ham.init_event()
    
    #     print ("chekpoint A", i)

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

    #     print ("chekpoint B", i)

    # define decay vertex
    Bc2JpsiLNu.add_vertex(thebc_idx, [thejpsi_idx, themu_idx, thenu_idx])

    # save process id to later retrieve the per-event weight
    pid = ham.add_process(Bc2JpsiLNu)
    pids.append(pid)

    pl =  ev.mu3_pt
    q2 =  ev.q2
    
    #    ham.fill_event_histogram("pEllVsQ2:Bc", [pl, q2])

    #     print ("chekpoint C", i)

    ham.process_event()

    #     print ("chekpoint D", i, pid)
    #     import pdb ; pdb.set_trace()
    
    
    #print (pid, ham.get_weight('BGL', [pid]))
    #print (pid, ham.get_weight('Kiselev', [pid]))
    #     weights.append(ham.get_weight('BGL', [pid]))
    #weights.append(ham.get_weight('BGL'))
    weights.append(ham.get_weight('Kiselev'))

    #     print ("chekpoint E", i)

    if i>maxevents: break

reduced_tree = tree_df[:len(weights)]
reduced_tree['hammer'] = np.nan_to_num(np.array(weights)) # sone NaNs, check the manual
to_root(reduced_tree, 'reweighed_bc_tree_mu_fromEfgtoKis_14Apr21.root', key='tree')

