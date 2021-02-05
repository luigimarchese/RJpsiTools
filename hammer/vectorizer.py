import ROOT
from hammer.hammerlib import Hammer, IOBuffer, Particle, Process, FourMomentum, WTerm
from hammer import hepmc, pdg
from root_pandas import read_root, to_root
import numpy as np
# from numba import float32    # import the types
# from numba.experimental import jitclass
from hammer.hammerlib import Hammer, IOBuffer, Particle, Process, FourMomentum, WTerm
from hammer import hepmc, pdg

# spec = [
#     ('e' , float32),
#     ('px', float32),
#     ('py', float32),
#     ('pz', float32),
# ]
# 
# @jitclass(spec)
# class MyFourMomentum(FourMomentum):
#     def __init__(self, e, px, py, pz):
#         self = FourMomentum(e, px, py, pz)
# 
# @np.vectorize
# class MyFourMomentum(FourMomentum):
#     def __init__(self, e, px, py, pz):
#         self = FourMomentum(e, px, py, pz)


def MyFourMomentum(e, px, py, pz):
    return FourMomentum(e, px, py, pz)

def MyParticle(e, px, py, pz, pdgid):
    p4 = FourMomentum(e, px, py, pz)
    return Particle(p4, np.int32(pdgid))

# vectorize
FourMomentumVec = np.frompyfunc(MyFourMomentum, 4, 1)
ParticleVec = np.frompyfunc(MyParticle, 5, 1)

es   = np.ones(10)*10.
pxs  = np.ones(10)*2.
pys  = np.ones(10)*1.
pzs  = np.ones(10)*3.
pdgids = np.ones(10)*541.

four_momenta = FourMomentumVec(es, pxs, pys, pzs)
particles = ParticleVec(es, pxs, pys, pzs, pdgids)

# fname = 'flat_tree_bc_chunk0.root'
# tree_df = read_root(fname, 'tree', where='is_jpsi_mu & is3m')




# FourMomentum(10., 2., 1., 3.)