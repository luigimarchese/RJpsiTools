from root_pandas import read_root, to_root
import numpy as np
from samples import sample_names
from selections import prepreselection

leading = '((mu1pt)*(mu1pt > mu2pt & mu1pt > kpt) + (mu2pt)*(mu2pt > mu1pt & mu2pt > kpt) + (kpt)*(kpt > mu2pt & kpt > mu1pt))'
trailing = '((mu1pt)*(mu1pt < mu2pt & mu1pt < kpt) + (mu2pt)*(mu2pt < mu1pt & mu2pt < kpt) + (kpt)*(kpt < mu2pt & kpt < mu1pt))'
subleading = '((mu1pt)*(mu1pt != '+leading+' & mu1pt != '+trailing+') + (mu2pt)*(mu2pt != '+leading+' & mu2pt != '+trailing+') + (kpt)*(kpt != '+leading+' & kpt != '+trailing+'))'

for sname in sample_names:
    if sname == 'data':
        df = read_root("/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/"+sname+"_fakerate.root","BTo3Mu", where = prepreselection)
    else:
        df = read_root("/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/"+sname+"_sf_werrors.root","BTo3Mu", where = prepreselection)
        
    df.to_root("/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/"+sname+"_prepresel.root","BTo3Mu")
