'''
Script to compute the scale factors id and reco for the muons from json tables
- It saves scale factor branches in the flat nanos, both central values and errors
- It also saves the value of the global error
'''
from root_pandas import read_root
from root_pandas import to_root
from samples import sample_names
import json
import pandas as pd
import ROOT
import sys
import numpy as np

# This variable is False because all the shape euncertainties are actually normalisations only, so we apply a unique normalisation uncertainty (the max one) to the fit (see plots in `15Jul2021_13h18m04s`)

compute_error = False
compute_error_global = True

# Path for final root files 
path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021'

#Open json files with scale factors info depending on eta and pt
with open('reco_muon.json') as f:
  reco_json = json.load(f)
with open('id_muon.json') as f:
  id_json = json.load(f)

def find_sf(df, i, which_mu, json, initial_folder):
  '''Function to compute the scale factor for each event;
  Takes as input:
  - df -> dataframe of the ntuples
  - i -> number of event
  - which_mu -> string that indicated which of the three muons 
  - json -> which json file (reco or id)
  - initial_folder -> depending on the json file, the initial folder is different
  '''
  flag_pt = 0.
  flag_eta = 0.
  for ieta,eta_key in enumerate(json[initial_folder]['abseta_pt'].keys()):
    low_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[0]
    high_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[1]
    if abs(df[which_mu+'eta'][i])<= float(high_bound_eta) and abs(df[which_mu+'eta'][i])>= float(low_bound_eta):
      flag_eta =1.
      flag_pt = 0.
      for ipt,pt_key in enumerate(json[initial_folder]['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']'].keys()):
        low_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[0]
        high_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[1]
        if abs(df[which_mu+'pt'][i])<= float(high_bound_pt) and abs(df[which_mu+'pt'][i])>= float(low_bound_pt):
          flag_pt = 1.
          value = json[initial_folder]['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['value']
          error = json[initial_folder]['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['error']
          cell =  ipt + 16 * ieta 
  if flag_pt==0. or flag_eta==0.:
    value = 1.
    error = 1.
    cell = None

  return value, error, cell


for sname in sample_names+['jpsi_x']:
    if sname == 'data':
        continue
    print("Computing sample ",sname)
    df = read_root(path+'/'+sname+'_fakerate.root','BTo3Mu',warn_missing_tree=True)
    #df = read_root(path+'/'+sname+'_fakerate_mva.root','BTo3Mu',warn_missing_tree=True)
    #df = read_root(path+'/'+sname+'_sf.root','BTo3Mu',warn_missing_tree=True)
    
    #reorder the indices 
    df.index= [i for i in range(len(df))]

    mu1_reco_features = []
    mu2_reco_features = []
    k_reco_features = []
    mu1_id_features = []
    mu2_id_features = []
    k_id_features = []
    
    for i in range(len(df)):
      mu1_reco_features.append( find_sf(df,i,'mu1', reco_json, 'NUM_TrackerMuons_DEN_genTracks') )
      mu2_reco_features.append( find_sf(df,i,'mu2', reco_json, 'NUM_TrackerMuons_DEN_genTracks') )
      k_reco_features.append( find_sf(df,i,'k', reco_json, 'NUM_TrackerMuons_DEN_genTracks') )

      mu1_id_features.append( find_sf(df,i,'mu1', id_json, 'NUM_MediumID_DEN_TrackerMuons') )
      mu2_id_features.append( find_sf(df,i,'mu2', id_json, 'NUM_MediumID_DEN_TrackerMuons') )
      k_id_features.append( find_sf(df,i,'k', id_json, 'NUM_MediumID_DEN_TrackerMuons') )

    # transpose the 2d array
    mu1_reco_features = np.array(mu1_reco_features).T
    mu2_reco_features = np.array(mu2_reco_features).T
    k_reco_features = np.array(k_reco_features).T
    mu1_id_features = np.array(mu1_id_features).T
    mu2_id_features = np.array(mu2_id_features).T
    k_id_features = np.array(k_id_features).T

    # weights for the central value
    df['sf_reco_total'] = (mu1_reco_features[0] * mu2_reco_features[0] * k_reco_features[0]).astype(float)
    df['sf_id_jpsi'] = (mu1_id_features[0] * mu2_id_features[0] ).astype(float)
    df['sf_id_k'] = (k_id_features[0]).astype(float)
    
    if compute_error:
      # build weights for the shape/ normalisation uncertainties (one for each cell)
      # number of cells in the json files
      for ireco in range(0, 16*4):      
        # transfor the cell number in bool (yes or not)
        # The weights will be equal to 1 if the cell number (2nd element in the array) is the one we want ot work on
        weight_mu1_reco = list(map(lambda item: item == ireco, mu1_reco_features[2])) #cell number
        weight_mu2_reco = list(map(lambda item: item == ireco, mu2_reco_features[2]))
        weight_k_reco = list(map(lambda item: item == ireco, k_reco_features[2]))

        # value + error * true/false
        df['sf_reco_'+str(ireco)+'_up'] = ((mu1_reco_features[0]+mu1_reco_features[1]*weight_mu1_reco)*(mu2_reco_features[0]+mu2_reco_features[1]*weight_mu2_reco)*(k_reco_features[0]+k_reco_features[1]*weight_k_reco)).astype(float)
        df['sf_reco_'+str(ireco)+'_down'] = ((mu1_reco_features[0]-mu1_reco_features[1]*weight_mu1_reco)*(mu2_reco_features[0]-mu2_reco_features[1]*weight_mu2_reco)*(k_reco_features[0]-k_reco_features[1]*weight_k_reco)).astype(float)
      
      
      for iid in range(0, 16*4):
          # transfor the cell number in bool (yes or not)
          weight_mu1_id = list(map(lambda item: item == iid, mu1_id_features[2])) #cell number
          weight_mu2_id = list(map(lambda item: item == iid, mu2_id_features[2]))
          weight_k_id = list(map(lambda item: item == iid, k_id_features[2]))
          
          # value + error * true/false
          # I divide into jpsi and muon because in the fail region the third muon doesn't want the sf_id
          df['sf_id_'+str(iid)+'_jpsi_up'] = ((mu1_id_features[0]+mu1_id_features[1]*weight_mu1_id)*(mu2_id_features[0]+mu2_id_features[1]*weight_mu2_id)).astype(float)
          df['sf_id_'+str(iid)+'_jpsi_down'] = ((mu1_id_features[0]-mu1_id_features[1]*weight_mu1_id)*(mu2_id_features[0]-mu2_id_features[1]*weight_mu2_id)).astype(float)
          df['sf_id_'+str(iid)+'_k_up'] = ((k_id_features[0]+k_id_features[1]*weight_k_id)).astype(float)
          df['sf_id_'+str(iid)+'_k_down'] = ((k_id_features[0]-k_id_features[1]*weight_k_id)).astype(float)

    if compute_error_global:
      # worst case when I apply only the normalisation nuisance to the fit
      df['sf_reco_all_up'] = ((mu1_reco_features[0]+mu1_reco_features[1])*(mu2_reco_features[0]+mu2_reco_features[1])*(k_reco_features[0]+k_reco_features[1])).astype(float)
      df['sf_reco_all_down'] = ((mu1_reco_features[0]-mu1_reco_features[1])*(mu2_reco_features[0]-mu2_reco_features[1])*(k_reco_features[0]-k_reco_features[1])).astype(float)
      # worst case when I compute only the normalisation nuisance to the fit
      df['sf_id_all_jpsi_up'] = ((mu1_id_features[0]+mu1_id_features[1])*(mu2_id_features[0]+mu2_id_features[1])).astype(float)
      df['sf_id_all_jpsi_down'] = ((mu1_id_features[0]-mu1_id_features[1])*(mu2_id_features[0]-mu2_id_features[1])).astype(float)
      df['sf_id_all_k_up'] = ((k_id_features[0]+k_id_features[1])).astype(float)
      df['sf_id_all_k_down'] = ((k_id_features[0]-k_id_features[1])).astype(float)
          
    df.to_root(path+'/'+sname+'_sf_werrors.root', key='BTo3Mu')
