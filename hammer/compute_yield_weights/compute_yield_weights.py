# with this script we compute the sum for the hammer weights for mu and tau

from root_pandas import read_root, to_root
import numpy as np

path_storage = "/pnfs/psi.ch/cms/trivcat/store/user/friti/"
#folder_mu =  "Rjpsi_hammer_mu_01Dec21_v1"
#folder_tau = "Rjpsi_hammer_tau_12nov21_v1"
folder_mu =  "Rjpsi_hammer_mu_24jan22_v2"
folder_tau = "Rjpsi_hammer_tau_24jan22_v2"

file_mu = path_storage + folder_mu + "/rjpsi_hammer_mu_merged.root"
file_tau = path_storage + folder_tau + "/rjpsi_hammer_tau_merged.root"

tree_mu = read_root(file_mu, 'tree')
tree_tau = read_root(file_tau, 'tree')

hammer_syst = [
    'bglvar',
    'bglvar_e0',
    'bglvar_e1',
    'bglvar_e2',
    'bglvar_e3',
    'bglvar_e4',
    'bglvar_e5',
    'bglvar_e6',
    'bglvar_e7',
    'bglvar_e8',
    'bglvar_e9',
    'bglvar_e10',
]

for hammer in hammer_syst:
    hammer = 'hammer_'+hammer
    if hammer != 'hammer_bglvar':
        hammer = [hammer+'up',hammer+'down']
    else:
        hammer = [hammer]
    for ham in hammer:
        #print("Number of events with hammer weight = 0 for mu: ",len(tree_mu[ham][(tree_mu[ham] ==0)]))
        #print("Number of events with hammer weight = 0 for tau: ",len(tree_tau[ham][(tree_tau[ham] ==0)]))
        print("Average hammer weight "+ham+" for tau: ",np.mean(tree_tau[ham])," +- ",np.std(tree_tau[ham])/np.sqrt(len(tree_tau[ham])))
        print("Average hammer weight "+ham+" for mu: ",np.mean(tree_mu[ham])," +- ",np.std(tree_mu[ham])/np.sqrt(len(tree_mu[ham])))

