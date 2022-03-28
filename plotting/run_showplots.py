# Run all the categories you need
# Save the paths
from datetime import datetime
import os
import time

asimov = True

categories_1 = ['ip3d_sig_dcorr<0','ip3d_sig>=0 & ip3d_sig<2','ip3d_sig>=2']
#load_dimuons = ['28Mar2022_10h53m44s', '28Mar2022_10h53m46s', '28Mar2022_10h53m48s']

#categories_1 = ['bdt_v5<0.72','bdt_v5>=0.72 & bdt_v5<0.825','bdt_v5>=0.825 & bdt_v5<0.89','bdt_v5>=0.89']
#categories_1 = ['bdt_v5<0.825','bdt_v5>=0.825 & bdt_v5<0.89','bdt_v5>=0.89']
#categories_1 = ['bdt_tau_fakes_v6<0.82','bdt_tau_fakes_v6>=0.82 & bdt_tau_fakes_v6<0.885','bdt_tau_fakes_v6>=0.885']
#categories_2 = ['bdt_tau_mu_v2<0.8','bdt_tau_mu_v2>=0.8']
#categories_2 = ['bdt_tau_mu_v1<0.7','bdt_tau_mu_v1>=0.7']
#categories_2 = ['bdt_tau_mu_v1<0.75','bdt_tau_mu_v1>=0.75']
#categories_2 = ['bvtx_svprob>=0 & bvtx_svprob<0.2','bvtx_svprob>=0.2 & bvtx_svprob<0.45','bvtx_svprob>=0.45 & bvtx_svprob<1']

labels = []
for cat1 in categories_1:
#for cat1,load_dimuon in zip(categories_1,load_dimuons):
                #for cat2 in categories_2:
                preselection_plus = cat1 #+" & "+cat2
                label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
                command ='python showplots_v15.py --asimov --add_dimuon --compute_dimuon  --preselection_plus "'+preselection_plus+'" --label '+str(label)+' > plots_ul/logs/log_'+str(label)+'.log &' 
                #command ='python showplots_v15.py --add_dimuon --dimuon_load "'+load_dimuon+'"  --preselection_plus "'+preselection_plus+'" --label '+str(label)+' > plots_ul/logs/log_'+str(label)+'.log &' 
                
                print(command)
                os.system(command)
                labels.append(str(label))
                time.sleep(2)

print("LABELS : ",labels)
    
