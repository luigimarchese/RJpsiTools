def first_part(f, channel, histos, shapes_file = 'param_ws.root wspace:$PROCESS_$CHANNEL wspace:$PROCESS_$SYSTEMATIC_$CHANNEL'):
    f.write("imax 1 number of channels \n")
    f.write("jmax * number of backgrounds \n")
    f.write("kmax * number of independent systematical uncertainties \n") 
    f.write("--------------------------------------------------------------------------- \n") 
    f.write("shapes * * %s    \n"%shapes_file)
    f.write("--------------------------------------------------------------------------- \n") 
    f.write("bin %s\n"%channel) 
    f.write("observation %.2f\n"%histos['data'].Integral())

    f.write("----------------------------------------------------------------------------\n") 

def rates(f, channel, histos,  jpsi_split, fakes_1 = True):
    # Define strings for process and rates
    process_numbers_string = ''
    process_string = ''
    bins_string = ''
    rates_string = ''
    for i,histo in enumerate(histos):        
        if histo == 'data':
            continue
        if histo == 'jpsi_x_mu' and jpsi_split:
            continue
        process_numbers_string+=' %s '%i
        process_string+=' %s '%histo
        bins_string+=' %s '%channel
        if histo == 'fakes' and fakes_1:
            rates_string+= ' 1 '
        else:
            rates_string+= ' %.3f '%histos[histo].Integral()
        #print(histo, histos[histo].Integral())
    f.write("bin    %s   \n"%bins_string) 
    f.write("process           %s \n"%process_string) 
    f.write("process               %s \n"%process_numbers_string) 
    f.write("rate  %s   \n"%rates_string)
    f.write("-------------------------------------------------------------------- ------------------- \n")

def norm_nuisances(f, channel, histos, jpsi_split):
    # Define strings for nuisances
    ctau_string = 'ctau shape '
    pu_string   = 'puWeight shape '
    fakes_lm_string = 'fake_rate_lm lnN '
    fakes_hm_string = 'fake_rate_hm lnN '
    trigger_string = 'trigger lnN '
    for i,histo in enumerate(histos):
        if histo == 'data':
            continue
        elif histo == 'fakes':
            ctau_string    += ' - '
            pu_string      += ' - '
            fakes_lm_string   += ' 1.5 '
            fakes_hm_string   += ' 1.5 '
            trigger_string += ' - '
        elif histo == 'jpsi_x_mu' and jpsi_split:
            continue
        elif'jpsi_x_mu' in histo:
            ctau_string    += ' - '
            pu_string      += ' 1 '
            fakes_lm_string   += ' - '
            fakes_hm_string   += ' - '
            trigger_string += ' 1.05 '
        else:
            ctau_string    += ' 1 '
            pu_string      += ' 1 '
            fakes_lm_string   += ' - '
            fakes_hm_string   += ' - '
            trigger_string += ' 1.05 '

    f.write(" %s \n"%ctau_string)
    f.write(" %s  \n"%pu_string)
    if channel == 'ch1' or channel == 'ch3':
        f.write(" %s  \n"%fakes_lm_string)
    #elif  channel == 'ch3':
    #    f.write(" %s  \n"%fakes_hm_string)
    f.write(" %s  \n"%trigger_string)

def ff_nuisances(f, channel, histos, jpsi_split):
    # Form factor nuisances
    bgl_strings = []
    for j in range(10):
        bgl_string   = 'bglvar_e%d shape '%j
        for i,histo in enumerate(histos):
            if histo == 'data':
                continue
            elif histo == 'jpsi_mu' or histo == 'jpsi_tau':
                bgl_string += ' 1 '
            elif histo == 'jpsi_x_mu' and jpsi_split:
                continue
            else:
                bgl_string += ' - '
        bgl_strings.append(bgl_string)

    for bglvar in bgl_strings:
            f.write('%s \n'%bglvar)

def br_nuisances(f, channel, histos, jpsi_split):
    br_nuisances ={
        'chic0_mu': 1.16,
        'chic1_mu': 1.10,
        'chic2_mu': 1.22,
        'hc_mu':    1.15,
        'jpsi_hc':  1.38,
        'psi2s_mu': 1.13,
        'psi2s_tau':1.15}

    # br nuisances
    br_nuisances_strings = []
    for sample in br_nuisances:
        string = 'br_%s_over_mu lnN '%sample
        for histo in histos:
            if histo == 'data':
                continue
            elif histo == 'jpsi_x_mu' and jpsi_split:
                continue
            elif histo != sample:
                string += ' - '
            else:
                string += ' %s '%br_nuisances[sample]
        f.write('%s \n'%string)

def bbb_nuisances(f, channel, histos, jpsi_split, hmlm_split, jpsi_x_mu_samples):
    
    nbins = histos['data'].GetNbinsX()
    if jpsi_split:

        jpsi_x_mu_samples = [string.replace('jpsi_x_mu','') for string in jpsi_x_mu_samples if ('bzero' not in string and 'bplus' not in string and 'other' not in string)]
        jpsimothers = [string.replace('jpsi_x_mu','') for string in jpsi_x_mu_samples]
        hmlm = ['_hm','_lm']
        
        if hmlm_split:
            for mother in jpsimothers:
                for hl in hmlm:
                    for nbin in range(1,nbins+1):
                        string = 'jpsi_x_mu%s%s_bbb%d%s shape '%(mother,hl,nbin,channel)
                        for histo in histos:
                            if histo == 'data':
                                continue
                            elif histo == 'jpsi_x_mu%s%s'%(mother,hl) and hl in histo:
                                string += ' 1 '
                            elif histo == 'jpsi_x_mu':
                                continue
                            else:
                                string += ' - '
                        f.write('%s \n'%string)
        else:
            for mother in jpsimothers:
                for nbin in range(1,nbins+1):
                    string = 'jpsi_x_mu%s_bbb%d%s shape '%(mother,nbin,channel)
                    for histo in histos:
                        if histo == 'data':
                            continue
                        elif histo == 'jpsi_x_mu%s'%(mother):
                            string += ' 1 '
                        elif histo == 'jpsi_x_mu':
                            continue
                        else:
                            string += ' - '
                    f.write('%s \n'%string)
        
            
    else: 
        for nbin in range(1,nbins+1):
            string = 'jpsi_x_mu_bbb%d%s shape '%(nbin,channel)
            for histo in histos:
                if histo == 'data':
                    continue
                elif histo != 'jpsi_x_mu':
                    string += ' - '
            f.write('%s \n'%string)

# only 1 bbb per bin (if jpsi_x_mu splitted)
def bbb_single_nuisances(f, channel, histos, which_sample_bbb_unc):
    #print(channel,"which_sample_bbb_unc",which_sample_bbb_unc)
    nbins = histos['data'].GetNbinsX()

    for i,sample in enumerate(which_sample_bbb_unc):
        string = 'jpsi_x_mu_from_%s_single_bbb%d%s shape '%(sample,i+1,channel)
        for histo in histos:
            if histo == 'data':
                continue
            elif histo == 'jpsi_x_mu_from_%s'%(sample):
                string += ' 1 '
            elif histo == 'jpsi_x_mu':
                continue
            else:
                string += ' - '
        f.write('%s \n'%string)


def sf_nuisances(f, channel, histos, sfReco_nuisances, sfIdjpsi_nuisances, sfIdk_nuisances,jpsi_split):
    string_reco      = 'sfReco lnN '
    string_id_jpsi   = 'sfIdJpsi lnN '
    string_id_k      = 'sfIdk lnN '
    for histo in histos:
        if histo == 'data':
            continue
        elif histo == 'jpsi_x_mu' and jpsi_split:
            continue
        elif 'jpsi_x_mu' in histo:
            string_id_jpsi += ' %s '%sfIdjpsi_nuisances['jpsi_x_mu']
            string_id_k += ' %s '%sfIdk_nuisances['jpsi_x_mu']
            string_reco += ' %s '%sfReco_nuisances['jpsi_x_mu']
        elif histo == 'fakes':
            string_id_jpsi += ' - '
            string_id_k += ' - '
            string_reco += ' - '
        else:
            string_id_jpsi += ' %s '%sfIdjpsi_nuisances[histo]
            string_id_k += ' %s '%sfIdk_nuisances[histo]
            string_reco += ' %s '%sfReco_nuisances[histo]


    f.write('%s \n'%string_reco)
    f.write('%s \n'%string_id_jpsi)
    f.write('%s \n'%string_id_k)

def jpsimother_nuisances(f, channel, histos, jpsi_x_mu_samples):
    # jpsimother nuisances
    jpsimothers = [string.replace('jpsi_x_mu_from_','') for string in jpsi_x_mu_samples]
    #print(jpsimothers)
    jpsimother_nuisances_strings = []
    for mother in jpsimothers: #total of 10 contributes
        string = 'jpsimother_%s lnN '%mother
        for histo in histos:
            if histo == 'jpsi_x_mu':
                continue
            elif histo == 'data':
                continue
            elif histo == 'jpsi_x_mu_from_%s'%mother:
                string += ' 1.1 '
            else:
                string += ' - '
        f.write('%s \n'%string)

def lm_nuisances(f, channel, histos):
    # Define strings for nuisances
    lm_string = 'lm lnN '
    for i,histo in enumerate(histos):
        if histo == 'data':
            continue
        elif '_lm' in histo:
            lm_string   += ' 1.1 '
        else:
            lm_string   += ' - '

    f.write(" %s \n"%lm_string)
    
# date; name of the variable for the fit; array of histos; bool if jpsi_x_mu must be split; bool if we also have the high mass low mass split; jpsi_x_mu_samples

def create_datacard_ch1(label, var_name,  histos, hmlm_split, jpsi_x_mu_samples, which_sample_bbb_unc =[]):
    jpsi_split = len(jpsi_x_mu_samples)>1

    f= open('plots_ul/%s/datacards/datacard_ch1_%s.txt' %(label, var_name),"w")

    first_part(f, 'ch1', histos)
    rates(f, 'ch1', histos, jpsi_split)
    norm_nuisances(f, 'ch1', histos, jpsi_split)
    ff_nuisances(f, 'ch1', histos, jpsi_split)
    br_nuisances(f, 'ch1', histos, jpsi_split)
    if jpsi_split:
        jpsimother_nuisances(f, 'ch1', histos, jpsi_x_mu_samples)
        if hmlm_split:
            lm_nuisances(f, 'ch1', histos)
        
    sfReco_nuisances = {
        'jpsi_mu' :1.031,
        'jpsi_tau':1.030,
        'chic0_mu':1.027,
        'chic1_mu':1.029,
        'chic2_mu':1.030,
        'hc_mu'   :1.041,
        'jpsi_hc' :1.032,
        'psi2s_mu':1.028,
        'psi2s_tau':1.022,
        'jpsi_x_mu':1.029}

    sfIdjpsi_nuisances = {
        'jpsi_mu' :1.027,
        'jpsi_tau':1.027,
        'chic0_mu':1.026,
        'chic1_mu':1.026,
        'chic2_mu':1.027,
        'hc_mu'   :1.041,
        'jpsi_hc' :1.029,
        'psi2s_mu':1.026,
        'psi2s_tau':1.024,
        'jpsi_x_mu':1.028}

    sfIdk_nuisances = {
        'jpsi_mu' :1.013,
        'jpsi_tau':1.013,
        'chic0_mu':1.012,
        'chic1_mu':1.013,
        'chic2_mu':1.014,
        'hc_mu'   :1.013,
        'jpsi_hc' :1.013,
        'psi2s_mu':1.013,
        'psi2s_tau':1.011,
        'jpsi_x_mu':1.016}

    sf_nuisances(f, 'ch1', histos, sfReco_nuisances, sfIdjpsi_nuisances, sfIdk_nuisances, jpsi_split)

    if len(which_sample_bbb_unc)<=1:
        bbb_nuisances(f, 'ch1', histos, jpsi_split, hmlm_split, jpsi_x_mu_samples)
    else:
        bbb_single_nuisances(f, 'ch1', histos, which_sample_bbb_unc)

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
    #f.write("fake_rate rateParam ch1 fakes 1\n")
    
    # bkg rateParam
    for histo in histos:
        if 'jpsi_x_mu' in histo:
            if histo == 'jpsi_x_mu' and jpsi_split:
                continue
            f.write('bkg rateParam ch1 %s 1 \n'%histo)
    f.close()

def create_datacard_ch2(label, var_name,  histos, hmlm_split, jpsi_x_mu_samples, which_sample_bbb_unc =[]):
    jpsi_split = len(jpsi_x_mu_samples)>1

    f= open('plots_ul/%s/datacards/datacard_ch2_%s.txt' %(label, var_name),"w")
    first_part(f, 'ch2', histos)
    rates(f, 'ch2', histos, jpsi_split)
    norm_nuisances(f, 'ch2', histos, jpsi_split)
    ff_nuisances(f, 'ch2', histos, jpsi_split)
    br_nuisances(f, 'ch2', histos, jpsi_split)
    if jpsi_split:
        jpsimother_nuisances(f, 'ch2', histos, jpsi_x_mu_samples)
        if hmlm_split:
            lm_nuisances(f, 'ch2', histos)

    sfReco_nuisances = {
        'jpsi_mu' : 1.026,
        'jpsi_tau':1.026,
        'chic0_mu':1.026,
        'chic1_mu':1.027,
        'chic2_mu':1.026,
        'hc_mu'   :1.029,
        'jpsi_hc' :1.028,
        'psi2s_mu':1.026,
        'psi2s_tau':1.030,
        'jpsi_x_mu':1.028}

    sfIdjpsi_nuisances = {
        'jpsi_mu' :1.026,
        'jpsi_tau':1.026,
        'chic0_mu':1.026,
        'chic1_mu':1.026,
        'chic2_mu':1.025,
        'hc_mu'   :1.029,
        'jpsi_hc' :1.026,
        'psi2s_mu': 1.025,
        'psi2s_tau':1.028,
        'jpsi_x_mu':1.026}

    # anticorrelated to the ones in ch1
    sfIdk_nuisances = {
        'jpsi_mu' :1.013,
        'jpsi_tau':1.013,
        'chic0_mu':1.012,
        'chic1_mu':1.013,
        'chic2_mu':1.013,
        'hc_mu'   :1.013,
        'jpsi_hc' :1.013,
        'psi2s_mu':1.013,
        'psi2s_tau':1.011,
        'jpsi_x_mu':1.015,}

    sf_nuisances(f, 'ch2', histos, sfReco_nuisances, sfIdjpsi_nuisances,sfIdk_nuisances, jpsi_split)

    if len(which_sample_bbb_unc)<=1:
        bbb_nuisances(f, 'ch2', histos, jpsi_split, hmlm_split, jpsi_x_mu_samples)
    else:
        bbb_single_nuisances(f, 'ch2', histos, which_sample_bbb_unc)



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

    # bkg rateParam
    for histo in histos:
        if 'jpsi_x_mu' in histo:
            if histo == 'jpsi_x_mu' and jpsi_split:
                continue
            f.write('bkg rateParam ch2 %s 1 \n'%histo)

    for i in range(1,histos['data'].GetNbinsX()+1):
        f.write("fakes_ch2_bin"+str(i)+"  flatParam \n")

    f.close()

def create_datacard_ch3(label, var_name,  histos, hmlm_split, jpsi_x_mu_samples, which_sample_bbb_unc =[]):
    jpsi_split = len(jpsi_x_mu_samples)>1

    f= open('plots_ul/%s/datacards/datacard_ch3_%s.txt' %(label, var_name),"w")
    first_part(f, 'ch3', histos)
    rates(f, 'ch3', histos, jpsi_split)
    norm_nuisances(f, 'ch3', histos,jpsi_split)
    #ff_nuisances(f, 'ch3', histos)
    #br_nuisances(f, 'ch3', histos)
    if jpsi_split:
        jpsimother_nuisances(f, 'ch3', histos, jpsi_x_mu_samples)
        if hmlm_split:
            lm_nuisances(f, 'ch3', histos)
    
    sfReco_nuisances = {
        'jpsi_x_mu':1.032}

    sfIdjpsi_nuisances = {
        'jpsi_x_mu':1.030}

    sfIdk_nuisances = {
        'jpsi_x_mu':1.015}

    sf_nuisances(f, 'ch3', histos, sfReco_nuisances, sfIdjpsi_nuisances, sfIdk_nuisances,jpsi_split)
    if len(which_sample_bbb_unc)<1:
        bbb_nuisances(f, 'ch3', histos, jpsi_split, hmlm_split, jpsi_x_mu_samples)
    else:
        bbb_single_nuisances(f, 'ch3', histos, which_sample_bbb_unc)


    f.write("----------------------------------------------------------------------------\n") 
   # bkg rateParam
    for histo in histos:
        if 'jpsi_x_mu' in histo:
            if histo == 'jpsi_x_mu' and jpsi_split:
                continue
            f.write('bkg rateParam ch3 %s 1 \n'%histo)

    #f.write("fake_rate rateParam ch3 fakes 1\n")
    f.close()


def create_datacard_ch4(label, var_name,  histos, hmlm_split, jpsi_x_mu_samples, which_sample_bbb_unc =[]):
    jpsi_split = len(jpsi_x_mu_samples)>1

    f= open('plots_ul/%s/datacards/datacard_ch4_%s.txt' %(label, var_name),"w")
    first_part(f, 'ch4', histos)
    rates(f, 'ch4', histos, jpsi_split)
    norm_nuisances(f, 'ch4', histos, jpsi_split)
    if jpsi_split:
        jpsimother_nuisances(f, 'ch4', histos, jpsi_x_mu_samples)
        if hmlm_split:
            lm_nuisances(f, 'ch4', histos)

    sfReco_nuisances = {
        'jpsi_x_mu':1.030}

    sfIdjpsi_nuisances = {
        'jpsi_x_mu':1.027}

    sfIdk_nuisances = {
        'jpsi_x_mu':1.016,}


    sf_nuisances(f, 'ch4', histos, sfReco_nuisances, sfIdjpsi_nuisances, sfIdk_nuisances, jpsi_split)
    if len(which_sample_bbb_unc)<1:
        bbb_nuisances(f, 'ch4', histos, jpsi_split, hmlm_split, jpsi_x_mu_samples)
    else:
        bbb_single_nuisances(f, 'ch4', histos, which_sample_bbb_unc)



    f.write("----------------------------------------------------------------------------\n") 

    f.write("-----------------------------------------------\n") 
   # bkg rateParam
    for histo in histos:
        if 'jpsi_x_mu' in histo:
            if histo == 'jpsi_x_mu' and jpsi_split:
                continue
            f.write('bkg rateParam ch4 %s 1 \n'%histo)

    for i in range(1,histos['data'].GetNbinsX()+1):
        f.write("fakes_ch4_bin"+str(i)+"  flatParam \n")
        #f.write("stat_bin"+str(i)+"  flatParam \n")

    f.close()
  
def create_datacard_ch1_onlypass(label, var_name,  histos, hmlm_split, jpsi_x_mu_samples, which_sample_bbb_unc =[]):
    jpsi_split = len(jpsi_x_mu_samples)>1

    f= open('plots_ul/%s/datacards/datacard_ch1_%s.txt' %(label, var_name),"w")

    shapes_file = 'datacard_ch1_%s.root $PROCESS_$CHANNEL $PROCESS_$SYSTEMATIC_$CHANNEL'%var_name
    first_part(f, 'ch1', histos, shapes_file)
    rates(f, 'ch1', histos,  jpsi_split, False)
    norm_nuisances(f, 'ch1', histos, jpsi_split)
    ff_nuisances(f, 'ch1', histos, jpsi_split)
    br_nuisances(f, 'ch1', histos, jpsi_split)
    if jpsi_split:
        jpsimother_nuisances(f, 'ch1', histos, jpsi_x_mu_samples)
        if hmlm_split:
            lm_nuisances(f, 'ch1', histos)
    
    sfReco_nuisances = {
        'jpsi_mu' :1.031,
        'jpsi_tau':1.030,
        'chic0_mu':1.027,
        'chic1_mu':1.029,
        'chic2_mu':1.030,
        'hc_mu'   :1.041,
        'jpsi_hc' :1.032,
        'psi2s_mu':1.028,
        'psi2s_tau':1.022,
        'jpsi_x_mu':1.029}

    sfIdjpsi_nuisances = {
        'jpsi_mu' :1.027,
        'jpsi_tau':1.027,
        'chic0_mu':1.026,
        'chic1_mu':1.026,
        'chic2_mu':1.027,
        'hc_mu'   :1.041,
        'jpsi_hc' :1.029,
        'psi2s_mu':1.026,
        'psi2s_tau':1.024,
        'jpsi_x_mu':1.028}

    sfIdk_nuisances = {
        'jpsi_mu' :1.013,
        'jpsi_tau':1.013,
        'chic0_mu':1.012,
        'chic1_mu':1.013,
        'chic2_mu':1.014,
        'hc_mu'   :1.013,
        'jpsi_hc' :1.013,
        'psi2s_mu':1.013,
        'psi2s_tau':1.011,
        'jpsi_x_mu':1.016}

    sf_nuisances(f, 'ch1', histos, sfReco_nuisances, sfIdjpsi_nuisances,  sfIdk_nuisances,jpsi_split)
    if len(which_sample_bbb_unc)<=1:
        bbb_nuisances(f, 'ch1', histos, jpsi_split, hmlm_split, jpsi_x_mu_samples)
    else:
        bbb_single_nuisances(f, 'ch1', histos, which_sample_bbb_unc)

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
    
    # bkg rateParam
    for histo in histos:
        if 'jpsi_x_mu' in histo:
            if histo == 'jpsi_x_mu' and jpsi_split:
                continue
            f.write('bkg rateParam ch1 %s 1 \n'%histo)
    f.close()

def create_datacard_ch3_onlypass(label, var_name,  histos, hmlm_split, jpsi_x_mu_samples, which_sample_bbb_unc =[]):
    jpsi_split = len(jpsi_x_mu_samples)>1

    f= open('plots_ul/%s/datacards/datacard_ch3_%s.txt' %(label, var_name),"w")

    shapes_file = 'datacard_ch3_%s.root $PROCESS_$CHANNEL $PROCESS_$SYSTEMATIC_$CHANNEL'%var_name
    first_part(f, 'ch3', histos, shapes_file)
    rates(f, 'ch3', histos, jpsi_split, False)
    norm_nuisances(f, 'ch3', histos, jpsi_split)
    ff_nuisances(f, 'ch3', histos, jpsi_split)
    br_nuisances(f, 'ch3', histos, jpsi_split)
    if jpsi_split:
        jpsimother_nuisances(f, 'ch3', histos, jpsi_x_mu_samples)
        if hmlm_split:
            lm_nuisances(f, 'ch3', histos)
    sfReco_nuisances = {
        'jpsi_x_mu':1.032}

    sfIdjpsi_nuisances = {
        'jpsi_x_mu':1.030}

    sfIdk_nuisances = {
        'jpsi_x_mu':1.015}

    sf_nuisances(f, 'ch3', histos, sfReco_nuisances, sfIdjpsi_nuisances, sfIdk_nuisances, jpsi_split)
    if len(which_sample_bbb_unc)<1:
        bbb_nuisances(f, 'ch3', histos, jpsi_split, hmlm_split, jpsi_x_mu_samples)
    else:
        bbb_single_nuisances(f, 'ch3', histos, which_sample_bbb_unc)


    f.write("----------------------------------------------------------------------------\n") 
    
    # bkg rateParam
    for histo in histos:
        if 'jpsi_x_mu' in histo:
            if jpsi_split and histo == 'jpsi_x_mu':
                continue
            f.write('bkg rateParam ch3 %s 1 \n'%histo)
    f.close()
