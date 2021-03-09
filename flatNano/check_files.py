import os
from personal_settings import *

#data, HbToJPsiMuMu, BcToJPsiMuMu

dataset = 'BcToJPsiMuMu'

#dataset = 'HbToJPsiMuMu'
dateFolder = '2021Mar08'

#res = True #already resubmitted?
res = False

if(res == False):
    fin = open("dataframes_"+dateFolder+"/"+dataset+"/"+dataset+"_files_check.txt","r")
else:
    fin = open("dataframes_"+dateFolder+"/"+dataset+"/"+dataset+"_files_check_resubmitted.txt","r")

list_of_expected_files = [s.split(' \n')[0] for s in fin.readlines()]

if (not 'BcToJPsiMuMu' in dataset):
    list_of_created_files = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset).readlines()
    list_of_created_files = [s.split('\n')[0] for s in list_of_created_files]
    for fl in list_of_expected_files:
        if fl not in list_of_created_files:
            print(fl)

if('BcToJPsiMuMu' in dataset):
    list_of_created_jpsi_tau = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_jpsi_tau").readlines()
    list_of_created_jpsi_mu = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_jpsi_mu").readlines()
    list_of_created_jpsi_pi = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_jpsi_pi").readlines()
    list_of_created_jpsi_hc = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_jpsi_hc").readlines()
    list_of_created_jpsi_3pi = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_jpsi_3pi").readlines()
    list_of_created_chic0_mu = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_chic0_mu").readlines()
    list_of_created_chic1_mu = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_chic1_mu").readlines()
    list_of_created_chic2_mu = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_chic2_mu").readlines()
    list_of_created_hc_mu = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_hc_mu").readlines()
    list_of_created_psi2s_mu = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_psi2s_mu").readlines()
    list_of_created_psi2s_tau = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_psi2s_tau").readlines()
    list_of_created_jpsi_tau = [s.split('\n')[0] for s in list_of_created_jpsi_tau]
    list_of_created_jpsi_mu = [s.split('\n')[0] for s in list_of_created_jpsi_mu]
    list_of_created_jpsi_pi = [s.split('\n')[0] for s in list_of_created_jpsi_pi]
    list_of_created_jpsi_hc = [s.split('\n')[0] for s in list_of_created_jpsi_hc]
    list_of_created_jpsi_3pi = [s.split('\n')[0] for s in list_of_created_jpsi_3pi]
    list_of_created_chic0_mu = [s.split('\n')[0] for s in list_of_created_chic0_mu]
    list_of_created_chic1_mu = [s.split('\n')[0] for s in list_of_created_chic1_mu]
    list_of_created_chic2_mu = [s.split('\n')[0] for s in list_of_created_chic2_mu]
    list_of_created_hc_mu = [s.split('\n')[0] for s in list_of_created_hc_mu]
    list_of_created_psi2s_mu = [s.split('\n')[0] for s in list_of_created_psi2s_mu]
    list_of_created_psi2s_tau = [s.split('\n')[0] for s in list_of_created_psi2s_tau]

    for fl in list_of_expected_files:
        if ('is_jpsi_tau' in fl ) and (fl not in list_of_created_jpsi_tau):
            print(fl)
        if ('is_jpsi_mu' in fl ) and (fl not in list_of_created_jpsi_mu):
            print(fl)
        if ('is_jpsi_psi' in fl ) and (fl not in list_of_created_jpsi_psi):
            print(fl)
        if ('is_jpsi_hc' in fl ) and (fl not in list_of_created_jpsi_hc):
            print(fl)
        if ('is_jpsi_3pi' in fl ) and (fl not in list_of_created_jpsi_3pi):
            print(fl)
        if ('is_chic0_mu' in fl ) and (fl not in list_of_created_chic0_mu):
            print(fl)
        if ('is_chic1_mu' in fl ) and (fl not in list_of_created_chic1_mu):
            print(fl)
        if ('is_chic2_mu' in fl ) and (fl not in list_of_created_chic2_mu):
            print(fl)
        if ('is_hc_mu' in fl ) and (fl not in list_of_created_hc_mu):
            print(fl)
        if ('is_psi2s_mu' in fl ) and (fl not in list_of_created_psi2s_mu):
            print(fl)
        if ('is_psi2s_tau' in fl ) and (fl not in list_of_created_psi2s_tau):
            print(fl)

