import ROOT
from time import time
from datetime import datetime, timedelta
from hammer.hammerlib import Hammer, IOBuffer, Particle, Process, FourMomentum, WTerm
from hammer import hepmc, pdg
from root_pandas import read_root, to_root
import numpy as np
from bgl_variations import variations
from itertools import product

ham = Hammer()
fbBuffer = IOBuffer
# "BcJpsiMuNu", "BcJpsiTauNu", TauEllNuNu
ham.include_decay("BcJpsiMuNu")

# ham.set_options({'ProcessCalc': {'CheckForNaNs': True}})

ff_input_scheme = dict()
# ff_input_scheme["Bc+JpsiMu+Num"] = "Kiselev"
ff_input_scheme["BcJpsi"] = "Kiselev"
ham.set_ff_input_scheme(ff_input_scheme) # "denominator"

ff_schemes  = dict()
# ff_schemes['bgl'    ] = {'BcJpsi':'BGL'    }
ff_schemes['bglvar' ] = {'BcJpsi':'BGLVar' }
# ff_schemes['kiselev'] = {'BcJpsi':'Kiselev'}
# ff_schemes['ebert'  ] = {'BcJpsi':'EFG'    }

for i, j in product(range(11), ['up', 'down']):
    unc = 'e%d%s'%(i,j)
    ff_schemes['bglvar_%s'%unc] = {'BcJpsi':'BGLVar_%s'%unc  }

for k, v in ff_schemes.items():
    ham.add_ff_scheme(k, v)

# ham.set_options({"BctoJpsi*BGLVar": {"avec": [1., 1., 1.], "bvec": [1., 1., 1.], "cvec": [1., 1.], "dvec": [1., 1.]})

# import pdb ; pdb.set_trace()

# Setting Wilson Coefficients
# ham.setWilsonCoefficients("BtoCMuNu", {{"S_qLlL", 1.}, {"T_qLlL",0.25}});
# ham.setWilsonCoefficients("BtoCTauNu", {{"S_qLlL", 1.}, {"T_qLlL",0.25}});


# ham.add_histogram("pEllVsQ2:Bc", [6, 5], True, [[0., 2.5], [3., 12.]])
# ham.add_total_sum_of_weights() # adds "Total Sum of Weights" histo with auto bin filling
# ham.keep_errors_in_histogram("pEllVsQ2:Bc", True)
# ham.collapse_processes_in_histogram("pEllVsQ2:Bc")
ham.set_units("GeV")
ham.init_run()

# this needs to be put AFTER init_run
for i, j in product(range(11), ['up', 'down']):
    unc = 'e%d%s'%(i,j)
    ham.set_ff_eigenvectors('BctoJpsi', 'BGLVar_%s'%unc, variations['e%d'%i][j])

fname = 'flat_tree_bc_chunk0.root'

fin = ROOT.TFile.Open(fname)
tree = fin.Get('tree')

maxevents = -1

tree_df = read_root(fname, 'tree', where='is_jpsi_mu & is3m')

pids = []
weights = dict()
for k in ff_schemes.keys():
    weights[k] = []

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

    thebc_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.bhad_pt, ev.bhad_eta, ev.bhad_phi, ev.bhad_m   )
    themu_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.mu3_pt , ev.mu3_eta , ev.mu3_phi , 0.1056583755)
    thejpsi_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.jpsi_pt, ev.jpsi_eta, ev.jpsi_phi, ev.jpsi_m   )
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

#     pl =  ev.mu3_pt
#     q2 =  ev.q2
    
#     ham.fill_event_histogram("pEllVsQ2:Bc", [pl, q2])

#     print ("chekpoint C", i)

    ham.process_event()

    for k in ff_schemes.keys():
        weights[k].append(ham.get_weight(k))

#     print ("chekpoint D", i, pid)
#     import pdb ; pdb.set_trace()

    
#     print (pid, ham.get_weight('BGL', [pid]))
#     print (pid, ham.get_weight('Kiselev', [pid]))
#     weights.append(ham.get_weight('BGL', [pid]))
#     weights.append(ham.get_weight('BGL'))
#     weights.append(ham.get_weight('Kiselev', [pid]))

#     print ("chekpoint E", i)

    if i>maxevents: break

reduced_tree = tree_df[:len(list(weights.values())[0])]
for k in ff_schemes.keys():
    reduced_tree['weight_%s'%k] = np.nan_to_num(np.array(weights[k])) # sone NaNs, check the manual
to_root(reduced_tree, 'reweighed_bc_tree.root', key='tree')

