#double decay
import ROOT
from time import time
from datetime import datetime, timedelta
from hammer.hammerlib import Hammer, IOBuffer, Particle, Process, FourMomentum, WTerm
from hammer import hepmc, pdg
from root_pandas import read_root, to_root
import numpy as np

ham = Hammer()
fbBuffer = IOBuffer

#ham.include_decay(["BcJpsiTauNu","TauMuNuNuBar"])
ham.include_decay(["BcJpsiTauNu"])

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

ham.set_units("GeV")
ham.init_run()


# input file (BcToJpsiTauNu )
#fname = 'flat_tree_bc_chunk0.root'
fname = 'inspector/flat_tree_tau_ebert_13Apr21.root'

fin = ROOT.TFile.Open(fname)
tree = fin.Get('tree')

maxevents = -1

tree_df = read_root(fname, 'tree', where='is_jpsi_tau & is3m & ismu3fromtau & bhad_pdgid == 541')

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
    
    if not ev.is_jpsi_tau: continue
    if not ev.is3m: continue
    if not ev.ismu3fromtau: continue
    if not ev.bhad_pdgid == 541: continue
    #     print ("processing event", i)
    
    ham.init_event()
    
    #     print ("chekpoint A", i)

    thebc_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.bhad_pt, ev.bhad_eta, ev.bhad_phi, ev.bhad_m)
    thejpsi_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.jpsi_pt, ev.jpsi_eta, ev.jpsi_phi, ev.jpsi_m)
    thetau_p4 = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.tau_pt, ev.tau_eta, ev.tau_phi, ev.tau_m)
    thenutau_p4   = thebc_p4 - thetau_p4 - thejpsi_p4
    
    themu_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.mu3_pt , ev.mu3_eta , ev.mu3_phi , 0.1056   )
    thenutaubar_p4   = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(ev.nutau_pt , ev.nutau_eta , ev.nutau_phi , 0   )
    thenumu_p4   = thetau_p4 - themu_p4 - thenutaubar_p4

    #    print(ev.bhad_pdgid,ev.tau_pdgid,ev.nutau_pdgid)#,ev.jpsi_pdgid,ev.tau_pdgid,ev.mu3_pdgid,ev.nutau_pdgid)
    thebc   = Particle(FourMomentum(thebc_p4.e()  , thebc_p4.px()  , thebc_p4.py()  , thebc_p4.pz()  ), ev.bhad_pdgid)
    thejpsi = Particle(FourMomentum(thejpsi_p4.e(), thejpsi_p4.px(), thejpsi_p4.py(), thejpsi_p4.pz()), 443          )
    thetau = Particle(FourMomentum(thetau_p4.e(), thetau_p4.px(), thetau_p4.py(), thetau_p4.pz()), -15          )
    thenutau   = Particle(FourMomentum(thenutau_p4.e()  , thenutau_p4.px()  , thenutau_p4.py()  , thenutau_p4.pz())  , 16           )

    themu   = Particle(FourMomentum(themu_p4.e()  , themu_p4.px()  , themu_p4.py()  , themu_p4.pz()  ), -13          )
    thenutaubar   = Particle(FourMomentum(thenutaubar_p4.e()  , thenutaubar_p4.px()  , thenutaubar_p4.py()  , thenutaubar_p4.pz())  , -16           )
    thenumu   = Particle(FourMomentum(thenumu_p4.e()  , thenumu_p4.px()  , thenumu_p4.py()  , thenumu_p4.pz())  , 14           )

    #    print(ev.bhad_pdgid)
    #print(thenumu_p4.mass())
    Bc2JpsiLNu = Process()
    
    # each of these add_particle operations returns an index, needed to define vertices 
    thebc_idx   = Bc2JpsiLNu.add_particle(thebc  )
    thejpsi_idx = Bc2JpsiLNu.add_particle(thejpsi)
    thetau_idx   = Bc2JpsiLNu.add_particle(thetau  )
    thenutau_idx   = Bc2JpsiLNu.add_particle(thenutau  )

    themu_idx   = Bc2JpsiLNu.add_particle(themu  )
    thenutaubar_idx   = Bc2JpsiLNu.add_particle(thenutaubar  )
    thenumu_idx   = Bc2JpsiLNu.add_particle(thenumu  )
    #     print ("chekpoint B", i)

    # define decay vertex
    Bc2JpsiLNu.add_vertex(thebc_idx, [thejpsi_idx, thetau_idx, thenutau_idx])
    #Bc2JpsiLNu.add_vertex(thetau_idx, [themu_idx,  thenutaubar_idx, thenumu_idx])

    # save process id to later retrieve the per-event weight
    pid = ham.add_process(Bc2JpsiLNu)
    pids.append(pid)

    #    pl =  ev.mu3_pt
    #    q2 =  ev.q2
    
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
to_root(reduced_tree, 'reweighed_bc_tree_tau_EFTtoKis_14Apr21_1vertx.root', key='tree')

