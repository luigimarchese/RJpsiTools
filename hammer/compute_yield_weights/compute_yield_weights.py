# with this script we compute the sum for the hammer weights for mu and tau

from root_pandas import read_root, to_root
import numpy as np

path_storage = "/pnfs/psi.ch/cms/trivcat/store/user/friti/"
folder_mu =  "Rjpsi_hammer_mu_01Dec21_v1"
folder_tau = "Rjpsi_hammer_tau_12nov21_v1"

file_mu = path_storage + folder_mu + "/rjpsi_hammer_mu_merged.root"
file_tau = path_storage + folder_tau + "/rjpsi_hammer_tau_merged.root"

tree_mu = read_root(file_mu, 'tree')
tree_tau = read_root(file_tau, 'tree')

print("Number of events with hammer weight = 0 for mu: ",len(tree_mu.hammer[(tree_mu.hammer ==0)]))
print("Number of events with hammer weight = 0 for tau: ",len(tree_tau.hammer[(tree_tau.hammer ==0)]))
print("Average hammer weight for mu: ",np.mean(tree_mu.hammer)," +- ",np.std(tree_mu.hammer)/np.sqrt(len(tree_mu.hammer)))
print("Average hammer weight for tau: ",np.mean(tree_tau.hammer)," +- ",np.std(tree_tau.hammer)/np.sqrt(len(tree_tau.hammer)))

