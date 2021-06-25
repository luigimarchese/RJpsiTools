def create_datacard_pass(histos,name,label):

    f= open('plots_ul/%s/datacards/datacard_pass_%s.txt' %(label, name),"w")
    f.write("imax 1 number of channels \n")
    f.write("jmax * number of backgrounds \n")
    f.write("kmax * number of independent systematical uncertainties \n") 
    f.write("--------------------------------------------------------------------------- \n") 
    f.write("shapes *    *  param_ws.root wspace:$PROCESS_$CHANNEL wspace:$PROCESS_$SYSTEMATIC_$CHANNEL \n")
    f.write("--------------------------------------------------------------------------- \n") 
    f.write("bin ch1\n") 
    f.write("observation %.2f\n"%histos['data'].Integral())

    f.write("----------------------------------------------------------------------------\n") 
    f.write("bin                   ch1      ch1      ch1      ch1      ch1      ch1      ch1      ch1      ch1      ch1      ch1 \n") 
    f.write("process           jpsi_mu     jpsi_tau    chic0_mu    chic1_mu    chic2_mu    hc_mu       jpsi_hc     psi2s_mu    psi2s_tau   jpsi_x_mu   fakes\n") 
    f.write("process               1          0           2            3           4           5           6           7          8           9           10 \n") 
    #f.write("rate      %.2f        %.2f        %.2f        %.2f         %.2f        %.2f        %.2f        %.2f       %.2f        %.2f        %.2f\n"%(histos['jpsi_mu'].Integral(),histos['jpsi_tau'].Integral(),histos['chic0_mu'].Integral(),histos['chic1_mu' ].Integral(),histos['chic2_mu'].Integral(),histos['hc_mu'].Integral(),histos['jpsi_hc'].Integral(),histos['psi2s_mu'].Integral(),histos['psi2s_tau'].Integral(),histos['jpsi_x_mu'].Integral(),histos['fakes'].Integral()))
    f.write("rate      %.2f        %.2f        %.2f        %.2f         %.2f        %.2f        %.2f        %.2f       %.2f        %.2f        1\n"%(histos['jpsi_mu'].Integral(),histos['jpsi_tau'].Integral(),histos['chic0_mu'].Integral(),histos['chic1_mu' ].Integral(),histos['chic2_mu'].Integral(),histos['hc_mu'].Integral(),histos['jpsi_hc'].Integral(),histos['psi2s_mu'].Integral(),histos['psi2s_tau'].Integral(),histos['jpsi_x_mu'].Integral()))
    f.write("-------------------------------------------------------------------- --------------------\n")

    f.write("ctau               shape      1           1          1            1           1           1           1          1          1           -             -\n")
    f.write("puWeight            shape      1           1          1            1           1           1           1          1          1           1             -\n")
    f.write("bbbfakes  shape      -           -          -            -           -           -           -          -          -           -          1 \n")
    f.write("bglvar_e0  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e1  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e2  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e3  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e4  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e5  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e6  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e7  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e8  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e9  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_mu_bglvar_e10 shape      1           -          -            -           -           -           -          -          -           -          -\n")

    #    f.write("jpsi_tau_bglvar_e0  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e1  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e2  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e3  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e4  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e5  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e6  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e7  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e8  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e9  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e10 shape      -           1          -            -           -           -           -          -          -           -          -\n")
    f.write("#bc               lnN            1.3         1.3        1.3          1.3         1.3         1.3        1.3         1.3        1.3         -          -\n")
    f.write("br_chic0_over_mu lnN            -           -           1.15         -            -         -          -           -          -           -           -\n")
    f.write("br_chic1_over_mu lnN             -          -            -           1.15         -         -          -           -          -           -           -\n")
    f.write("br_chic2_over_mu lnN             -          -            -           -           1.15        -         -           -           -           -           -\n")
    f.write("br_hc_over_mu    lnN             -           -           -           -           -           1.15      -           -           -           -           -\n")
    f.write("br_jpsi_hc_over_mu lnN           -           -           -           -           -           -        1.15         -           -           -           -\n")
    f.write("br_psi2s_over_mu lnN             -           -           -           -           -           -         -           1.15         -          -           -\n")
    f.write("br_psi2s_over_tau lnN            -           -           -           -           -           -         -           -           1.15        -           - \n")
    f.write("jpsi_plus_x       lnN            -           -           -           -           -           -           -           -           -         1.3         -\n")
    f.write("fake_rate         lnN            -           -           -           -           -           -           -           -           -          -           1.5\n")
    f.write("muon_id           lnN        1.05       1.05        1.05        1.05        1.05        1.05        1.05        1.05         1.05       1.05             -\n")
    f.write("trigger           lnN           1.05        1.05        1.05        1.05        1.05        1.05        1.05        1.05         1.05       1.05        -\n")
    for i in range(1,histos['jpsi_x_mu'].GetNbinsX()+1):
        f.write("bbb"+str(i)+"pass            shape      -           -          -            -           -           -           -          -          -           1          -\n")

    f.write("----------------------------------------------------------------------------\n") 
    f.write("bc rateParam ch1 jpsi_mu 1\n") 
    f.write("bc rateParam ch1 jpsi_tau 1\n") 
    f.write("bc rateParam ch1 chic0_mu 1\n")
    f.write("bc rateParam ch1 chic1_mu 1\n") 
    f.write("bc rateParam ch1 chic2_mu 1\n")
    f.write("bc rateParam ch1 hc_mu 1\n")
    f.write("bc rateParam ch1 jpsi_hc 1\n")
    f.write("bc rateParam ch1 psi2s_mu 1\n")
    f.write("bc rateParam ch1 psi2s_tau 1\n")
    f.write("#fakes_rp rateParam ch1 fakes 1\n")

    f.write("#comb_norm rateParam ch1 comb 1\n")
    f.write("#mis_id_unc lnN - - 1.2 - \n") 
    f.write("#recoMuon lnN 1.10 1.10 - 1.10 \n") 
    f.write("#ctau shape 1 1 - - - - - - - - - \n") 
    f.write("#mc_comb_unc lnN - - - 1.5 \n") 
    f.write("#ch1 autoMCStats 1\n") 
    f.close()

def create_datacard_fail(histos,name,label):

    f= open('plots_ul/%s/datacards/datacard_fail_%s.txt' %(label, name),"w")
    f.write("imax 1 number of channels \n")
    f.write("jmax * number of backgrounds \n")
    f.write("kmax * number of independent systematical uncertainties \n") 
    f.write("--------------------------------------------------------------------------- \n") 
    f.write("shapes *    *  param_ws.root wspace:$PROCESS_$CHANNEL wspace:$PROCESS_$SYSTEMATIC_$CHANNEL \n")
    f.write("--------------------------------------------------------------------------- \n") 
    f.write("bin ch2\n")
    f.write("observation %.2f\n"%histos['data'].Integral())
    f.write("----------------------------------------------------------------------------\n") 
    f.write("bin                   ch2      ch2      ch2      ch2      ch2      ch2      ch2      ch2      ch2      ch2      ch2 \n") 
    f.write("process           jpsi_mu     jpsi_tau    chic0_mu    chic1_mu    chic2_mu    hc_mu       jpsi_hc     psi2s_mu    psi2s_tau   jpsi_x_mu   fakes\n") 
    f.write("process               1          0           2            3           4           5           6           7          8           9           10 \n") 
    #f.write("rate      %.2f        %.2f        %.2f        %.2f         %.2f        %.2f        %.2f        %.2f       %.2f        %.2f        %.2f\n"%(histos['jpsi_mu'].Integral(),histos['jpsi_tau'].Integral(),histos['chic0_mu'].Integral(),histos['chic1_mu' ].Integral(),histos['chic2_mu'].Integral(),histos['hc_mu'].Integral(),histos['jpsi_hc'].Integral(),histos['psi2s_mu'].Integral(),histos['psi2s_tau'].Integral(),histos['jpsi_x_mu'].Integral(),histos['fakes'].Integral()))
    f.write("rate      %.2f        %.2f        %.2f        %.2f         %.2f        %.2f        %.2f        %.2f       %.2f        %.2f        1\n"%(histos['jpsi_mu'].Integral(),histos['jpsi_tau'].Integral(),histos['chic0_mu'].Integral(),histos['chic1_mu' ].Integral(),histos['chic2_mu'].Integral(),histos['hc_mu'].Integral(),histos['jpsi_hc'].Integral(),histos['psi2s_mu'].Integral(),histos['psi2s_tau'].Integral(),histos['jpsi_x_mu'].Integral()))
    f.write("-------------------------------------------------------------------- --------------------\n")

    f.write("ctau               shape      1           1          1            1           1           1           1          1          1           -             -\n")
    f.write("puWeight             shape      1           1          1            1           1           1           1          1          1           1             -\n")
    f.write("bglvar_e0  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e1  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e2  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e3  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e4  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e5  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e6  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e7  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e8  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e9  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_mu_bglvar_e10 shape      1           -          -            -           -           -           -          -          -           -          -\n")

    #    f.write("jpsi_tau_bglvar_e0  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e1  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e2  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e3  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e4  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e5  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e6  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e7  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e8  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e9  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e10 shape      -           1          -            -           -           -           -          -          -           -          -\n")
    f.write("#bc               lnN            1.3         1.3        1.3          1.3         1.3         1.3        1.3         1.3        1.3         -          -\n")
    f.write("br_chic0_over_mu lnN            -           -           1.15         -            -         -          -           -          -           -           -\n")
    f.write("br_chic1_over_mu lnN             -          -            -           1.15         -         -          -           -          -           -           -\n")
    f.write("br_chic2_over_mu lnN             -          -            -           -           1.15        -         -           -           -           -           -\n")
    f.write("br_hc_over_mu    lnN             -           -           -           -           -           1.15      -           -           -           -           -\n")
    f.write("br_jpsi_hc_over_mu lnN           -           -           -           -           -           -        1.15         -           -           -           -\n")
    f.write("br_psi2s_over_mu lnN             -           -           -           -           -           -         -           1.15         -          -           -\n")
    f.write("br_psi2s_over_tau lnN            -           -           -           -           -           -         -           -           1.15        -           - \n")
    f.write("jpsi_plus_x       lnN            -           -           -           -           -           -           -           -           -         1.3         -\n")
    f.write("muon_id lnN                 0.95        0.95        0.95        0.95        0.95        0.95        0.95        0.95        0.95      0.95         -\n")
    f.write("trigger lnN                 1.05        1.05        1.05        1.05        1.05        1.05        1.05        1.05        1.05         1.05      -\n")
    for i in range(1,histos['jpsi_x_mu'].GetNbinsX()+1):
        f.write("bbb"+str(i)+"fail            shape      -           -          -            -           -           -           -          -          -           1          -\n")
    #f.write("fake_rate         lnN            -           -           -           -           -           -           -           -           -          -           1.5\n")

    f.write("----------------------------------------------------------------------------\n") 
    f.write("----------------------------------------------------------------------------\n") 
    f.write("bc rateParam ch2 jpsi_mu 1\n") 
    f.write("bc rateParam ch2 jpsi_tau 1\n") 
    f.write("bc rateParam ch2 chic0_mu 1\n")
    f.write("bc rateParam ch2 chic1_mu 1\n") 
    f.write("bc rateParam ch2 chic2_mu 1\n")
    f.write("bc rateParam ch2 hc_mu 1\n")
    f.write("bc rateParam ch2 jpsi_hc 1\n")
    f.write("bc rateParam ch2 psi2s_mu 1\n")
    f.write("bc rateParam ch2 psi2s_tau 1\n")
    f.write("#fakes_rp rateParam ch2 fakes 1\n")

    f.write("#comb_norm rateParam ch2 comb 1\n")
    f.write("#mis_id_unc lnN - - 1.2 - \n") 
    f.write("#recoMuon lnN 1.10 1.10 - 1.10 \n") 
    f.write("#ctau shape 1 1 - - - - - - - - - \n") 
    f.write("#mc_comb_unc lnN - - - 1.5 \n") 
    f.write("#ch2 autoMCStats 1\n") 
    f.write("-----------------------------------------------\n") 
    for i in range(1,histos['data'].GetNbinsX()+1):
        f.write("fakes_cr_bin"+str(i)+"  flatParam \n")
        #f.write("stat_bin"+str(i)+"  flatParam \n")

    f.close()
  
def create_datacard_onlypass(histos,name,label):

    
    f= open('plots_ul/%s/datacards/datacard_pass_%s.txt' %(label, name),"w")
    f.write("imax 1 number of channels \n")
    f.write("jmax * number of backgrounds \n")
    f.write("kmax * number of independent systematical uncertainties \n") 
    f.write("--------------------------------------------------------------------------- \n") 
    f.write("shapes * * datacard_pass_"+name+".root $PROCESS $PROCESS_$SYSTEMATIC \n")
    f.write("--------------------------------------------------------------------------- \n") 
    f.write("bin signal\n") 
    f.write("observation %.2f\n"%histos['data'].Integral())

    f.write("----------------------------------------------------------------------------\n") 
    f.write("bin                   signal      signal      signal      signal      signal      signal      signal      signal      signal      signal      signal \n") 
    f.write("process           jpsi_mu     jpsi_tau    chic0_mu    chic1_mu    chic2_mu    hc_mu       jpsi_hc     psi2s_mu    psi2s_tau   jpsi_x_mu   jpsi_x\n") 
    f.write("process               1          0           2            3           4           5           6           7          8           9           10 \n") 
    f.write("rate      %.2f        %.2f        %.2f        %.2f         %.2f        %.2f        %.2f        %.2f       %.2f        %.2f        %.2f\n"%(histos['jpsi_mu'].Integral(),histos['jpsi_tau'].Integral(),histos['chic0_mu'].Integral(),histos['chic1_mu' ].Integral(),histos['chic2_mu'].Integral(),histos['hc_mu'].Integral(),histos['jpsi_hc'].Integral(),histos['psi2s_mu'].Integral(),histos['psi2s_tau'].Integral(),histos['jpsi_x_mu'].Integral(),histos['jpsi_x'].Integral()))
    f.write("-------------------------------------------------------------------- --------------------\n")

    f.write("#ctau               shape      1           1          1            1           1           1           1          1          1           -             -\n")
    f.write("bglvar_e0  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e1  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e2  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e3  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e4  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e5  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e6  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e7  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e8  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    f.write("bglvar_e9  shape      1           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_mu_bglvar_e10 shape      1           -          -            -           -           -           -          -          -           -          -\n")

    #    f.write("jpsi_tau_bglvar_e0  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e1  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e2  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e3  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e4  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e5  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e6  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e7  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e8  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e9  shape      -           1          -            -           -           -           -          -          -           -          -\n")
    #f.write("jpsi_tau_bglvar_e10 shape      -           1          -            -           -           -           -          -          -           -          -\n")
    f.write("#bc               lnN            1.3         1.3        1.3          1.3         1.3         1.3        1.3         1.3        1.3         -          -\n")
    f.write("br_tau_over_mu   lnN            -           1.15       -            -            -          -          -           -          -           -          -\n")
    f.write("br_chic0_over_mu lnN            -           -           1.15         -            -         -          -           -          -           -           -\n")
    f.write("br_chic1_over_mu lnN             -          -            -           1.15         -         -          -           -          -           -           -\n")
    f.write("br_chic2_over_mu lnN             -          -            -           -           1.15        -         -           -           -           -           -\n")
    f.write("br_hc_over_mu    lnN             -           -           -           -           -           1.15      -           -           -           -           -\n")
    f.write("br_jpsi_hc_over_mu lnN           -           -           -           -           -           -        1.15         -           -           -           -\n")
    f.write("br_psi2s_over_mu lnN             -           -           -           -           -           -         -           1.15         -          -           -\n")
    f.write("br_psi2s_over_tau lnN            -           -           -           -           -           -         -           -           1.15        -           - \n")
    f.write("jpsi_plus_x       lnN            -           -           -           -           -           -           -           -           -         1.3         -\n")
    f.write("#fake_rate         lnN            -           -           -           -           -           -           -           -           -          -           1.5\n")
    f.write("muon_id           lnN        1.5       1.05        1.05        1.05        1.05        1.05        1.05        1.05         1.05       1.05             -\n")
    f.write("trigger           lnN           1.05        1.05        1.05        1.05        1.05        1.05        1.05        1.05         1.05       1.05        -\n")

    f.write("----------------------------------------------------------------------------\n") 
    f.write("bc rateParam signal jpsi_mu 1\n") 
    f.write("bc rateParam signal jpsi_tau 1\n") 
    f.write("bc rateParam signal chic0_mu 1\n")
    f.write("bc rateParam signal chic1_mu 1\n") 
    f.write("bc rateParam signal chic2_mu 1\n")
    f.write("bc rateParam signal hc_mu 1\n")
    f.write("bc rateParam signal jpsi_hc 1\n")
    f.write("bc rateParam signal psi2s_mu 1\n")
    f.write("bc rateParam signal psi2s_tau 1\n")
    f.write("jpsi_x_rp rateParam signal jpsi_x 1\n")

    f.write("#comb_norm rateParam signal comb 1\n")
    f.write("#mis_id_unc lnN - - 1.2 - \n") 
    f.write("#recoMuon lnN 1.10 1.10 - 1.10 \n") 
    f.write("#ctau shape 1 1 - - - - - - - - - \n") 
    f.write("#mc_comb_unc lnN - - - 1.5 \n") 
    f.write("signal autoMCStats 1\n") 
    f.close()
