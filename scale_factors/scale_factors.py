from root_pandas import read_root
from root_pandas import to_root
from samples import sample_names
import json
import pandas as pd
import ROOT
import sys
path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021May31_nn'
with open('reco_muon.json') as f:
  reco = json.load(f)
with open('id_muon.json') as f:
  id = json.load(f)

#print(reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt']['abseta:[1.20,2.10]'].keys())
#print(id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt'].keys())

for sname in sample_names:
    if sname == 'data':
        continue
    print("Computing sample ",sname)
    df = read_root(path+'/'+sname+'_fakerate.root','BTo3Mu', warn_missing_tree=True)
    #rename the indices 
    df.index= [i for i in range(len(df))]

    # save the scale factors
    # reco
    mu1_value = []
    mu1_error = []
    mu2_value = []
    mu2_error = []
    k_value = []
    k_error = []
    # id (medium)
    mu1_value_id = []
    mu1_error_id = []
    mu2_value_id = []
    mu2_error_id = []
    k_value_id = []
    k_error_id = []

    print(len(df))
    for i in range(len(df)):
        ##############################################
        ###### RECO Scale Factor #####################
        ##############################################        

        # mu1
        flag_eta = 0.
        for eta_key in reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt'].keys():
            low_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[0]
            high_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[1]
            if abs(df.mu1eta[i])<= float(high_bound_eta) and abs(df.mu1eta[i])>= float(low_bound_eta):
                flag_eta =1.
                flag_pt = 0.
                for pt_key in reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']'].keys():
                    low_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[0]
                    high_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[1]
                    if abs(df.mu1pt[i])<= float(high_bound_pt) and abs(df.mu1pt[i])>= float(low_bound_pt):
                        flag_pt = 1.
                        mu1_value.append(reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['value'])
                        mu1_error.append(reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['error'])
        if flag_pt==0. or flag_eta==0.:
            mu1_value.append(1.)
            mu1_error.append(1.)

        # mu2
        flag_eta = 0.
        for eta_key in reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt'].keys():
            low_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[0]
            high_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[1]
            if abs(df.mu2eta[i])<= float(high_bound_eta) and abs(df.mu2eta[i])>= float(low_bound_eta):
                flag_eta =1.
                flag_pt = 0.
                for pt_key in reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']'].keys():
                    low_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[0]
                    high_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[1]
                    if abs(df.mu2pt[i])<= float(high_bound_pt) and abs(df.mu2pt[i])>= float(low_bound_pt):
                        flag_pt = 1.
                        mu2_value.append(reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['value'])
                        mu2_error.append(reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['error'])
        if flag_pt==0. or flag_eta==0.:
            mu2_value.append(1.)
            mu2_error.append(1.)

        # mu
        flag_eta = 0.
        for eta_key in reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt'].keys():
            low_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[0]
            high_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[1]
            if abs(df.keta[i])< float(high_bound_eta) and abs(df.keta[i])>= float(low_bound_eta):
                flag_eta =1.
                flag_pt = 0.
                for pt_key in reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']'].keys():
                    low_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[0]
                    high_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[1]
                    if abs(df.kpt[i])< float(high_bound_pt) and abs(df.kpt[i])>= float(low_bound_pt):
                        flag_pt = 1.
                        k_value.append(reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['value'])
                        k_error.append(reco['NUM_TrackerMuons_DEN_genTracks']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['error'])
        if flag_pt==0. or flag_eta==0.:
            k_value.append(1.)
            k_error.append(1.)

        if len(k_value)!= i+1:
            print("ERR",len(k_value), i, df.kpt[i],df.keta[i],flag_eta,flag_pt)
            sys.exit()
        ##############################################
        ###### ID Scale Factor #######################
        ##############################################        

        # mu1
        flag_eta = 0.
        for eta_key in id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt'].keys():
            low_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[0]
            high_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[1]
            if abs(df.mu1eta[i])<= float(high_bound_eta) and abs(df.mu1eta[i])>= float(low_bound_eta):
                flag_eta =1.
                flag_pt = 0.
                for pt_key in id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']'].keys():
                    low_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[0]
                    high_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[1]
                    if abs(df.mu1pt[i])<= float(high_bound_pt) and abs(df.mu1pt[i])>= float(low_bound_pt):
                        flag_pt = 1.
                        mu1_value_id.append(id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['value'])
                        mu1_error_id.append(id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['error'])
        if flag_pt==0. or flag_eta==0.:
            mu1_value_id.append(1.)
            mu1_error_id.append(1.)

        # mu2
        flag_eta = 0.
        for eta_key in id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt'].keys():
            low_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[0]
            high_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[1]
            if abs(df.mu2eta[i])<= float(high_bound_eta) and abs(df.mu2eta[i])>= float(low_bound_eta):
                flag_eta =1.
                flag_pt = 0.
                for pt_key in id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']'].keys():
                    low_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[0]
                    high_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[1]
                    if abs(df.mu2pt[i])<= float(high_bound_pt) and abs(df.mu2pt[i])>= float(low_bound_pt):
                        flag_pt = 1.
                        mu2_value_id.append(id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['value'])
                        mu2_error_id.append(id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['error'])
        if flag_pt==0. or flag_eta==0.:
            mu2_value_id.append(1.)
            mu2_error_id.append(1.)

        # mu
        flag_eta = 0.
        for eta_key in id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt'].keys():
            low_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[0]
            high_bound_eta = eta_key.strip('abseta:').strip(']').strip('[').split(',')[1]
            if abs(df.keta[i])< float(high_bound_eta) and abs(df.keta[i])>= float(low_bound_eta):
                flag_eta =1.
                flag_pt = 0.
                for pt_key in id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']'].keys():
                    low_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[0]
                    high_bound_pt = pt_key.strip('pt:').strip(']').strip('[').split(',')[1]
                    if abs(df.kpt[i])< float(high_bound_pt) and abs(df.kpt[i])>= float(low_bound_pt):
                        flag_pt = 1.
                        k_value_id.append(id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['value'])
                        k_error_id.append(id['NUM_MediumID_DEN_TrackerMuons']['abseta_pt']['abseta:['+low_bound_eta+','+high_bound_eta+']']['pt:['+low_bound_pt+','+high_bound_pt+']']['error'])
        if flag_pt==0. or flag_eta==0.:
            k_value_id.append(1.)
            k_error_id.append(1.)


    #print(len(mu1_value),len(mu2_value),len(k_value))
    
    df['sf_reco_value_mu1'] = mu1_value
    df['sf_reco_error_mu1'] = mu1_error
    df['sf_reco_value_mu2'] = mu2_value
    df['sf_reco_error_mu2'] = mu2_error
    df['sf_reco_value_k'] = k_value
    df['sf_reco_error_k'] = k_error

    df['sf_mediumid_value_mu1'] = mu1_value_id
    df['sf_mediumid_error_mu1'] = mu1_error_id
    df['sf_mediumid_value_mu2'] = mu2_value_id
    df['sf_mediumid_error_mu2'] = mu2_error_id
    df['sf_mediumid_value_k'] = k_value_id
    df['sf_mediumid_error_k'] = k_error_id

    df['sf_total'] = df['sf_reco_value_mu1']*df['sf_mediumid_value_mu1']*df['sf_reco_value_mu2']*df['sf_mediumid_value_mu2']*df['sf_reco_value_k']*df['sf_mediumid_value_k']

    df.to_root(path+'/'+sname+'_sf.root', key='BTo3Mu')
    


    
