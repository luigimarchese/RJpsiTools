def create_datacard_pass(histos,name,label):

    
    f= open('plots_ul/%s/datacards/datacard_pass_%s.txt' %(label, name),"w")
    f.write("imax 1 number of channels \n")
    f.write("jmax * number of backgrounds \n")
    f.write("kmax * number of independent systematical uncertainties \n") 
    f.write("--------------------------------------------------------------------------- \n") 
    f.write("shapes data_obs    signal   param_ws.root wspace:data_obs_SR \n")
    f.write("shapes mu       signal   param_ws.root wspace:mc_mu_SR #wspace:mc_mu_ctauUp_SR wspace:mc_mu_ctauDown_SR \n")
    f.write("shapes tau      signal   param_ws.root wspace:mc_tau_SR #wspace:mc_tau_ctauUp_SR wspace:mc_tau_ctauDown_SR \n")
    f.write("shapes chic0      signal   param_ws.root wspace:chic0_SR \n")
    f.write("shapes chic1      signal   param_ws.root wspace:chic1_SR \n")
    f.write("shapes chic2      signal   param_ws.root wspace:chic2_SR \n")
    f.write("shapes hc_mu      signal   param_ws.root wspace:hc_mu_SR \n")
    f.write("shapes jpsi_hc      signal   param_ws.root wspace:jpsi_hc_SR \n")
    f.write("shapes psi2s_mu      signal   param_ws.root wspace:psi2s_mu_SR \n")
    f.write("#shapes psi2s_tau      signal   param_ws.root wspace:psi2s_tau_SR \n")
    f.write("shapes comb      signal   param_ws.root wspace:mc_comb_SR \n")
    f.write("shapes mis_id      signal   param_ws.root wspace:mis_id_SR \n")
    f.write("--------------------------------------------------------------------------- \n") 
    f.write("bin signal\n") 
    f.write("observation %.2f\n"%histos['data'].Integral())
    f.write("----------------------------------------------------------------------------\n") 
    f.write("bin  \t signal \t signal \t signal \t signal \t signal \t signal \t signal \t signal \t signal \t signal \n") 
    f.write("process \t mu \t tau \t chic0 \t chic1 \t chic2 \t hc_mu \t jpsi_hc \t psi2s_mu  \t comb \t mis_id\n") 
    f.write("process \t 1 \t 0 \t 2 \t 3 \t 4 \t 5 \t 6 \t 7 \t 8 \t 9  \n") 
    f.write("rate  \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f \t  %.2f \t %.2f \t %.2f \t %.2f\n"%(histos['jpsi_mu'].Integral(),histos['jpsi_tau'].Integral(),histos['chic0_mu'].Integral(),histos['chic1_mu' ].Integral(),histos['chic2_mu'].Integral(),histos['hc_mu'].Integral(),histos['jpsi_hc'].Integral(),histos['psi2s_mu'].Integral(),histos['onia'].Integral(),histos['fakes'].Integral()))
    f.write("-------------------------------------------------------------------- --------------------\n")

    f.write("br_tau_over_mu  lnN - 1.15  - - - - - - - -\n")
    f.write("br_pi_over_mu   lnN 1.15 - - - - - - - - -\n")
    f.write("br_chic0_over_mu lnN - - 1.15 - - - - - - -\n")
    f.write("br_chic1_over_mu lnN - - - 1.15 - - - - - -\n")
    f.write("br_chic2_over_mu lnN - - - - 1.15  - - - - -\n")
    f.write("br_hc_over_mu lnN  - - - - - 1.15   - - - -\n")
    f.write("br_psi2s_over_mu lnN - - - - - - - 1.15  - -\n")
    f.write("br_jpsi_hc_over_mu lnN - - - - - - 1.15 - - -\n")
    f.write("fake_rate lnN - - - - - - - - - 1.5\n")
    f.write("muon_id lnN                 1.05        1.05        1.05        1.05        1.05        1.05        1.05        1.05        1.05               -\n")
    f.write("trigger lnN                 1.05        1.05        1.05        1.05        1.05        1.05        1.05        1.05        1.05               -\n")
    f.write("jpsi_plus_x lnN             -           -           -           -           -           -           -           -           1.30        -                  \n")

    f.write("----------------------------------------------------------------------------\n") 
    f.write("mu_norm rateParam signal mu 1\n") 
    f.write("mu_norm rateParam signal tau 1\n") 
    f.write("mu_norm rateParam signal chic0 1\n")
    f.write("mu_norm rateParam signal chic1 1\n") 
    f.write("mu_norm rateParam signal chic2 1\n")
    f.write("mu_norm rateParam signal hc_mu 1\n")
    f.write("mu_norm rateParam signal jpsi_hc 1\n")
    f.write("mu_norm rateParam signal psi2s_mu 1\n")
    f.write("#mu_norm rateParam signal psi2s_tau 1\n")
    f.write("comb_norm rateParam signal comb 1\n")
    f.write("#mis_id_unc lnN - - 1.2 - \n") 
    f.write("#recoMuon lnN 1.10 1.10 - 1.10 \n") 
    f.write("#ctau shape 1 1 - - - - - - - - - \n") 
    f.write("#mc_comb_unc lnN - - - 1.5 \n") 
    f.write("#signal autoMCStats 1\n") 
    f.close()

def create_datacard_fail(histos,name,label):

    f= open('plots_ul/%s/datacards/datacard_fail_%s.txt' %(label, name),"w")
    f.write("imax 1 number of channels \n")
    f.write("jmax * number of backgrounds \n")
    f.write("kmax * number of independent systematical uncertainties \n") 
    f.write("--------------------------------------------------------------------------- \n") 
    f.write("shapes data_obs    control   param_ws.root wspace:data_obs_CR \n")
    f.write("shapes mu       control   param_ws.root wspace:mc_mu_CR #wspace:mc_mu_ctauUp_CR wspace:mc_mu_ctauDown_CR \n")
    f.write("shapes tau      control   param_ws.root wspace:mc_tau_CR #wspace:mc_tau_ctauUp_CR wspace:mc_tau_ctauDown_CR \n")
    f.write("shapes chic0      control   param_ws.root wspace:chic0_CR \n")
    f.write("shapes chic1      control   param_ws.root wspace:chic1_CR \n")
    f.write("shapes chic2      control   param_ws.root wspace:chic2_CR \n")
    f.write("shapes hc_mu      control   param_ws.root wspace:hc_mu_CR \n")
    f.write("shapes jpsi_hc      control   param_ws.root wspace:jpsi_hc_CR \n")
    f.write("shapes psi2s_mu      control   param_ws.root wspace:psi2s_mu_CR \n")
    f.write("#shapes psi2s_tau      control   param_ws.root wspace:psi2s_tau_CR \n")
    f.write("shapes comb      control   param_ws.root wspace:mc_comb_CR \n")
    f.write("shapes mis_id      control   param_ws.root wspace:mis_id_CR \n")
    f.write("--------------------------------------------------------------------------- \n") 

    f.write("bin control\n") 
    f.write("observation %.2f\n"%histos['data'].Integral())
    f.write("----------------------------------------------------------------------------\n") 
    f.write("bin  \t control \t control \t control \t control \t control \t control \t control \t control \t control \t control \n") 
    f.write("process \t mu \t tau \t chic0 \t chic1 \t chic2 \t hc_mu \t jpsi_hc \t psi2s_mu  \t comb \t mis_id\n") 
    f.write("process \t 1 \t 0 \t 2 \t 3 \t 4 \t 5 \t 6 \t 7 \t 8 \t 9 \n") 
    f.write("rate  \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f \t  %.2f \t %.2f \t %.2f \t %.2f\n"%(histos['jpsi_mu'].Integral(),histos['jpsi_tau'].Integral(),histos['chic0_mu'].Integral(),histos['chic1_mu' ].Integral(),histos['chic2_mu'].Integral(),histos['hc_mu'].Integral(),histos['jpsi_hc'].Integral(),histos['psi2s_mu'].Integral(),histos['onia'].Integral(),histos['fakes'].Integral()))

    f.write("-------------------------------------------------------------------- --------------------\n")
    f.write("br_tau_over_mu  lnN - 1.15 - - - - - - - -\n")
    f.write("br_pi_over_mu   lnN 1.15 - - - - - - - - -\n")
    f.write("br_chic0_over_mu lnN - - 1.15 - - - - - - -\n")
    f.write("br_chic1_over_mu lnN - - - 1.15 - - - - - -\n")
    f.write("br_chic2_over_mu lnN - - - - 1.15  - - - - -\n")
    f.write("br_hc_over_mu lnN  - - - - - 1.15   - - - -\n")
    f.write("br_psi2s_over_mu lnN - - - - - - - 1.15  - -\n")
    f.write("br_jpsi_hc_over_mu lnN - - - - - - 1.15 - - -\n")
    #    f.write("fake_rate lnN - - - - - - - - - 1.5\n")
    f.write("muon_id lnN                 0.95        0.95        0.95        0.95        0.95        0.95        0.95        0.95        0.95               -\n")
    f.write("trigger lnN                 1.05        1.05        1.05        1.05        1.05        1.05        1.05        1.05        1.05               -\n")
    f.write("jpsi_plus_x lnN             -           -           -           -           -           -           -           -           1.30        -                  \n")
    f.write("----------------------------------------------------------------------------\n") 
    f.write("mu_norm rateParam control mu 1\n") 
    f.write("mu_norm rateParam control tau 1\n") 
    f.write("mu_norm rateParam control chic0 1\n")
    f.write("mu_norm rateParam control chic1 1\n") 
    f.write("mu_norm rateParam control chic2 1\n")
    f.write("mu_norm rateParam control hc_mu 1\n")
    f.write("mu_norm rateParam control jpsi_hc 1\n")
    f.write("mu_norm rateParam control psi2s_mu 1\n")
    f.write("#mu_norm rateParam control psi2s_tau 1\n")
    f.write("#mis_id_unc lnN - - 1.2 - \n") 
    f.write("#recoMuon lnN 0.9 0.9 - 0.9 \n") 
    f.write("#ctau shape 1 1 - - - - - - - - - - \n") 
    f.write("comb_norm rateParam control comb 1\n")
    f.write("#mc_comb_unc lnN - - - 1.5 \n") 
    f.write("-----------------------------------------------\n") 
    for i in range(1,histos['data'].GetNbinsX()+1):
        f.write("misid_CR_bin"+str(i)+"  flatParam \n")
    '''
    f.write("misid_CR_bin2  flatParam \n")
    f.write("misid_CR_bin3  flatParam \n")
    f.write("misid_CR_bin4  flatParam \n")
    f.write("misid_CR_bin5  flatParam \n")
    f.write("misid_CR_bin6  flatParam \n")
    f.write("misid_CR_bin7  flatParam \n")
    f.write("misid_CR_bin8  flatParam \n")
    f.write("misid_CR_bin9  flatParam \n")    
    f.write("misid_CR_bin10  flatParam \n")
    f.write("misid_CR_bin11  flatParam \n")
    f.write("misid_CR_bin12  flatParam \n")
    f.write("misid_CR_bin13  flatParam \n")
    f.write("misid_CR_bin14  flatParam \n")
    f.write("misid_CR_bin15  flatParam \n")
    '''
    f.close()
  
