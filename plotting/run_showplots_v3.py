# Run all the categories you need
# Save the paths
from datetime import datetime
import os
import time

asimov = False

if asimov:
    addition = '--asimov'
else:
    addition = ''

#categories_1 = ['ip3d_sig_dcorr<-2 & Q_sq>5.5','ip3d_sig_dcorr>=-2 & ip3d_sig_dcorr<0 & Q_sq>5.5','ip3d_sig_dcorr>=0 & ip3d_sig_dcorr<2 & Q_sq>5.5','ip3d_sig_dcorr>=2 & Q_sq>5.5 & jpsivtx_log10_lxy_sig<=0.4','ip3d_sig_dcorr>=2 & Q_sq>5.5 & jpsivtx_log10_lxy_sig>0.4','ip3d_sig_dcorr<0 & Q_sq<4.5',' ip3d_sig_dcorr>=0 & Q_sq<4.5']

categories_1 = ['ip3d_sig_dcorr<-2 & Q_sq>5.5','ip3d_sig_dcorr>=-2 & ip3d_sig_dcorr<0 & Q_sq>5.5','ip3d_sig_dcorr>=0 & ip3d_sig_dcorr<2 & Q_sq>5.5','ip3d_sig_dcorr>=2 & Q_sq>5.5','ip3d_sig_dcorr<0 & Q_sq<4.5',' ip3d_sig_dcorr>=0 & Q_sq<4.5']


labels = []
labels_lowq2 = []
for cat1 in categories_1:
#for cat1,load_dimuon in zip(categories_1,load_dimuons):
                #for cat2 in categories_2:
                preselection_plus = cat1 #+" & "+cat2
                label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
                #command ='python showplots_v16.py   --add_dimuon --compute_dimuon   --preselection_plus "'+preselection_plus+'" --label '+str(label)+' > plots_ul/logs/log_'+str(label)+'.log &' 
                if 'Q_sq<4.5' in cat1:
                                command ='python showplots_v21.py  '+addition+'   --low_q2    --preselection_plus "'+preselection_plus+'" --label '+str(label)+' > plots_ul/logs/log_'+str(label)+'.log &' 
                                labels_lowq2.append(str(label))
                else:
                                command ='python showplots_v21.py   '+addition+'    --preselection_plus "'+preselection_plus+'" --label '+str(label)+' > plots_ul/logs/log_'+str(label)+'.log &' 
                                labels.append(str(label))
                #command ='python showplots_v15.py --add_dimuon --dimuon_load "'+load_dimuon+'"  --preselection_plus "'+preselection_plus+'" --label '+str(label)+' > plots_ul/logs/log_'+str(label)+'.log &' 
                
                print(command)
                os.system(command)
                time.sleep(2)

# last plots without any cut to use for the high mass region
label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
command ='python showplots_v21.py   '+addition+'" --label '+str(label)+' > plots_ul/logs/log_'+str(label)+'.log ' 
print(command)
os.system(command)

print("Labels for the fit")
print("LABELS high q2 : ",labels)
print("LABELS low q2: ",labels_lowq2)
print("LABELS high mass: ",label)
    
