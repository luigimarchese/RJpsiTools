basic_samples_names = [
    'jpsi_tau' ,
    'jpsi_mu'  ,
    'chic0_mu' ,
    'chic1_mu' ,
    'chic2_mu' ,
    'jpsi_hc'  ,
    'hc_mu'    ,
    'psi2s_mu' ,
    'psi2s_tau',
     'data'     ,
]

jpsi_x_mu_sample = ['jpsi_x_mu',]
jpsi_x_mu_sample_jpsimother_splitting = [
    'jpsi_x_mu_from_bzero',
    'jpsi_x_mu_from_bplus',
    'jpsi_x_mu_from_bzero_s',
    #'jpsi_x_mu_from_bplus_c',
    'jpsi_x_mu_from_sigmaminus_b',
    'jpsi_x_mu_from_lambdazero_b',
    'jpsi_x_mu_from_ximinus_b',
    'jpsi_x_mu_from_sigmazero_b',
    'jpsi_x_mu_from_xizero_b',
    #'jpsi_x_mu_from_other',
]

jpsi_x_mu_sample_jpsimother_splitting_compressed = [ #sigma and xi are compressed in 1 contribute each
    'jpsi_x_mu_from_bzero',
    'jpsi_x_mu_from_bplus',
    'jpsi_x_mu_from_bzero_s',
    'jpsi_x_mu_from_sigma',
    'jpsi_x_mu_from_lambdazero_b',
    'jpsi_x_mu_from_xi',
]

jpsi_x_mu_sample_hmlm_splitting = [sample + hmlm for sample in jpsi_x_mu_sample for hmlm in ['_hm','_lm']]
jpsi_x_mu_sample_all_splitting = [sample + hmlm for sample in jpsi_x_mu_sample_jpsimother_splitting for hmlm in ['_hm','_lm']]


sample_names = basic_samples_names +jpsi_x_mu_sample
sample_names_explicit_hmlm = basic_samples_names + jpsi_x_mu_sample_hmlm_splitting
sample_names_explicit_jpsimother = basic_samples_names + jpsi_x_mu_sample_jpsimother_splitting
sample_names_explicit_jpsimother_compressed = basic_samples_names + jpsi_x_mu_sample_jpsimother_splitting_compressed
#sample_names_explicit_jpsimother_compressed = ['jpsi_mu','data']
sample_names_explicit_all = basic_samples_names + jpsi_x_mu_sample_all_splitting

# Overall weight to use to renormalize the Bc->jpsi mu and Bc-> jpsi tau samples due to FF reweighting
# Computed in tools/hammer/compute_yield_weights

ff_weights = dict()
ff_weights['jpsi_tau' ] = 1./0.554
ff_weights['jpsi_mu'  ] = 1./0.586
ff_weights['jpsi_tau_bglvar_e0Up' ] = 1./0.524
ff_weights['jpsi_mu_bglvar_e0Up'  ] = 1./0.602
ff_weights['jpsi_tau_bglvar_e0Down' ] = 1./0.588
ff_weights['jpsi_mu_bglvar_e0Down'  ] = 1./0.573
ff_weights['jpsi_tau_bglvar_e1Up' ] = 1./0.617
ff_weights['jpsi_mu_bglvar_e1Up'  ] = 1./0.658
ff_weights['jpsi_tau_bglvar_e1Down' ] = 1./0.497
ff_weights['jpsi_mu_bglvar_e1Down'  ] = 1./0.520
ff_weights['jpsi_tau_bglvar_e2Up' ] = 1./0.610
ff_weights['jpsi_mu_bglvar_e2Up'  ] = 1./0.616
ff_weights['jpsi_tau_bglvar_e2Down' ] = 1./0.502
ff_weights['jpsi_mu_bglvar_e2Down'  ] = 1./0.559
ff_weights['jpsi_tau_bglvar_e3Up' ] = 1./0.480
ff_weights['jpsi_mu_bglvar_e3Up'  ] = 1./0.516
ff_weights['jpsi_tau_bglvar_e3Down' ] = 1./0.635
ff_weights['jpsi_mu_bglvar_e3Down'  ] = 1./0.662
ff_weights['jpsi_tau_bglvar_e4Up' ] = 1./0.481
ff_weights['jpsi_mu_bglvar_e4Up'  ] = 1./0.504
ff_weights['jpsi_tau_bglvar_e4Down' ] = 1./0.633
ff_weights['jpsi_mu_bglvar_e4Down'  ] = 1./0.675
ff_weights['jpsi_tau_bglvar_e5Up' ] = 1./0.512
ff_weights['jpsi_mu_bglvar_e5Up'  ] = 1./0.538
ff_weights['jpsi_tau_bglvar_e5Down' ] = 1./0.599
ff_weights['jpsi_mu_bglvar_e5Down'  ] = 1./0.637
ff_weights['jpsi_tau_bglvar_e6Up' ] = 1./0.566
ff_weights['jpsi_mu_bglvar_e6Up'  ] = 1./0.599
ff_weights['jpsi_tau_bglvar_e6Down' ] = 1./0.543
ff_weights['jpsi_mu_bglvar_e6Down'  ] = 1./0.573
ff_weights['jpsi_tau_bglvar_e7Up' ] = 1./0.625
ff_weights['jpsi_mu_bglvar_e7Up'  ] = 1./0.669
ff_weights['jpsi_tau_bglvar_e7Down' ] = 1./0.488
ff_weights['jpsi_mu_bglvar_e7Down'  ] = 1./0.508
ff_weights['jpsi_tau_bglvar_e8Up' ] = 1./0.542
ff_weights['jpsi_mu_bglvar_e8Up'  ] = 1./0.572
ff_weights['jpsi_tau_bglvar_e8Down' ] = 1./0.568
ff_weights['jpsi_mu_bglvar_e8Down'  ] = 1./0.601
ff_weights['jpsi_tau_bglvar_e9Up' ] = 1./0.587
ff_weights['jpsi_mu_bglvar_e9Up'  ] = 1./0.616
ff_weights['jpsi_tau_bglvar_e9Down' ] = 1./0.523
ff_weights['jpsi_mu_bglvar_e9Down'  ] = 1./0.557
ff_weights['jpsi_tau_bglvar_e10Up' ] = 1./0.554
ff_weights['jpsi_mu_bglvar_e10Up'  ] = 1./0.586
ff_weights['jpsi_tau_bglvar_e10Down' ] = 1./0.554
ff_weights['jpsi_mu_bglvar_e10Down'  ] = 1./0.586




#ff_weights = dict()
#ff_weights['jpsi_tau' ] = 1./0.554
#ff_weights['jpsi_mu'  ] = 1./0.603
#ff_weights['jpsi_tau_bglvar_e0Up' ] = 1./0.523
#ff_weights['jpsi_mu_bglvar_e0Up'  ] = 1./0.632
#ff_weights['jpsi_tau_bglvar_e0Down' ] = 1./0.589
#ff_weights['jpsi_mu_bglvar_e0Down'  ] = 1./0.578
#ff_weights['jpsi_tau_bglvar_e1Up' ] = 1./0.617
#ff_weights['jpsi_mu_bglvar_e1Up'  ] = 1./0.682
#ff_weights['jpsi_tau_bglvar_e1Down' ] = 1./0.498
#ff_weights['jpsi_mu_bglvar_e1Down'  ] = 1./0.532
#ff_weights['jpsi_tau_bglvar_e2Up' ] = 1./0.609
#ff_weights['jpsi_mu_bglvar_e2Up'  ] = 1./0.623
#ff_weights['jpsi_tau_bglvar_e2Down' ] = 1./0.502
#ff_weights['jpsi_mu_bglvar_e2Down'  ] = 1./0.586
#ff_weights['jpsi_tau_bglvar_e3Up' ] = 1./0.480
#ff_weights['jpsi_mu_bglvar_e3Up'  ] = 1./0.529
#ff_weights['jpsi_tau_bglvar_e3Down' ] = 1./0.635
#ff_weights['jpsi_mu_bglvar_e3Down'  ] = 1./0.684
#ff_weights['jpsi_tau_bglvar_e4Up' ] = 1./0.481
#ff_weights['jpsi_mu_bglvar_e4Up'  ] = 1./0.517
#ff_weights['jpsi_tau_bglvar_e4Down' ] = 1./0.633
#ff_weights['jpsi_mu_bglvar_e4Down'  ] = 1./0.698
#ff_weights['jpsi_tau_bglvar_e5Up' ] = 1./0.512
#ff_weights['jpsi_mu_bglvar_e5Up'  ] = 1./0.552
#ff_weights['jpsi_tau_bglvar_e5Down' ] = 1./0.599
#ff_weights['jpsi_mu_bglvar_e5Down'  ] = 1./0.659
#ff_weights['jpsi_tau_bglvar_e6Up' ] = 1./0.566
#ff_weights['jpsi_mu_bglvar_e6Up'  ] = 1./0.618
#ff_weights['jpsi_tau_bglvar_e6Down' ] = 1./0.543
#ff_weights['jpsi_mu_bglvar_e6Down'  ] = 1./0.589
#ff_weights['jpsi_tau_bglvar_e7Up' ] = 1./0.625
#ff_weights['jpsi_mu_bglvar_e7Up'  ] = 1./0.693
#ff_weights['jpsi_tau_bglvar_e7Down' ] = 1./0.488
#ff_weights['jpsi_mu_bglvar_e7Down'  ] = 1./0.521
#ff_weights['jpsi_tau_bglvar_e8Up' ] = 1./0.541
#ff_weights['jpsi_mu_bglvar_e8Up'  ] = 1./0.589
#ff_weights['jpsi_tau_bglvar_e8Down' ] = 1./0.568
#ff_weights['jpsi_mu_bglvar_e8Down'  ] = 1./0.618
#ff_weights['jpsi_tau_bglvar_e9Up' ] = 1./0.586
#ff_weights['jpsi_mu_bglvar_e9Up'  ] = 1./0.633
#ff_weights['jpsi_tau_bglvar_e9Down' ] = 1./0.523
#ff_weights['jpsi_mu_bglvar_e9Down'  ] = 1./0.575
#ff_weights['jpsi_tau_bglvar_e10Up' ] = 1./0.554
#ff_weights['jpsi_mu_bglvar_e10Up'  ] = 1./0.603
#ff_weights['jpsi_tau_bglvar_e10Down' ] = 1./0.554
#ff_weights['jpsi_mu_bglvar_e10Down'  ] = 1./0.603

weights = dict()
bc_weight = 0.09 *1.1 *1.04 * 0.85 * 0.9 *1.4
#bc_weight = 0.15  #for prev ff_weights
#bc_weight = 1.53 #for prev oct21
fr = 0.19
#hbmu_norm = 8.5 *1.5 #without weights
#hbmu_norm = 8.5 *0.5 *0.65 #0.65 from fit, so it will make the fit converge better (old sample)
hbmu_norm = 0.3 * 0.85 *0.7*0.1 * 2.7 *1.6 *0.85 * 1.8 *1.4    # new sample (0.85 from fit)


weights['jpsi_tau' ] = bc_weight * ff_weights['jpsi_tau' ]
weights['jpsi_mu'  ] = bc_weight * ff_weights['jpsi_mu'  ]
weights['psi2s_mu' ] = bc_weight
weights['chic0_mu' ] = bc_weight
weights['chic1_mu' ] = bc_weight
weights['chic2_mu' ] = bc_weight
weights['hc_mu'    ] = bc_weight
weights['psi2s_tau'] = bc_weight
weights['jpsi_hc'  ] = bc_weight
weights['fakes'    ] = 1.  #fr/(1-fr) #2.7 # 2.5 # 2.7
#weights['fakes'    ] = 2 * 1.2 #fr/(1-fr) #2.7 # 2.5 # 2.7
weights['data'     ] = 1.
#weights['dimuon'     ] = 1.
weights['jpsi_x'] = 15
weights['jpsi_x_mu'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_hm'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_lm'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_from_other'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_from_bzero'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_from_bplus'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_from_bzero_s'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_from_bplus_c'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_from_sigmaminus_b'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_from_lambdazero_b'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_from_ximinus_b'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_from_sigmazero_b'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_from_xizero_b'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_from_xi'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_x_mu_from_sigma'] = 1. * hbmu_norm # 10. # 8.5
weights['jpsi_pi'  ] = bc_weight
weights['jpsi_3pi' ] = bc_weight

for name in sample_names_explicit_all:
    if 'jpsi_x_mu' in name:
        weights[name] = 1. * hbmu_norm


titles = dict()
titles['jpsi_tau' ] = 'B_{c}#rightarrowJ/#Psi#tau'
titles['jpsi_mu'  ] = 'B_{c}#rightarrowJ/#Psi#mu'
titles['onia'     ] = 'J/#Psi + #mu'
titles['jpsi_pi'  ] = 'B_{c}#rightarrowJ/#Psi#pi'
titles['psi2s_mu' ] = 'B_{c}#rightarrow#Psi(2S)#mu'
titles['chic0_mu' ] = 'B_{c}#rightarrow#chi_{c0}#mu'
titles['chic1_mu' ] = 'B_{c}#rightarrow#chi_{c1}#mu'
titles['chic2_mu' ] = 'B_{c}#rightarrow#chi_{c2}#mu'
titles['hc_mu'    ] = 'B_{c}#rightarrowh_{c}#mu'
titles['psi2s_tau'] = 'B_{c}#rightarrow#Psi(2S)#tau'
titles['jpsi_3pi' ] = 'B_{c}#rightarrowJ/#Psi3#pi'
titles['jpsi_hc'  ] = 'B_{c}#rightarrowJ/#PsiH_{c}'
titles['data'     ] = 'observed'
titles['fakes'   ] = 'fakes'#'J/#Psi + X'
titles['dimuon'] = 'comb J/#Psi'
titles['jpsi_x'] = 'J/#Psi + X'
titles['jpsi_x_mu'] = 'J/#Psi + #mu'
titles['jpsi_x_mu_hm'] = 'J/#Psi + #mu HM'
titles['jpsi_x_mu_lm'] = 'J/#Psi + #mu LM'
titles['jpsi_x_mu_from_bzero'] = 'B^{0}#rightarrowJ/#Psi#muX'
titles['jpsi_x_mu_from_bplus'] = 'B^{+}#rightarrowJ/#Psi#muX'
titles['jpsi_x_mu_from_bzero_s'] = 'B^{0}_{s}#rightarrowJ/#Psi#muX'
titles['jpsi_x_mu_from_bplus_c'] = 'B_{c}#rightarrowJ/#Psi#muX'
titles['jpsi_x_mu_from_sigmaminus_b'] = '#Sigma^{-}_{b}#rightarrowJ/#Psi#muX'
titles['jpsi_x_mu_from_lambdazero_b'] = '#Lambda^{0}_{b}#rightarrowJ/#Psi#muX'
titles['jpsi_x_mu_from_ximinus_b'] = '#Xi^{-}_{b}#rightarrowJ/#Psi#muX'
titles['jpsi_x_mu_from_sigmazero_b'] = '#Sigma^{0}_{b}#rightarrowJ/#Psi#muX'
titles['jpsi_x_mu_from_xizero_b'] = '#Xi^{0}_{b}#rightarrowJ/#Psi#muX'
titles['jpsi_x_mu_from_other'] = 'other B'

titles['jpsi_x_mu_from_sigma'] = '#Sigma^{0/-}_{b}#rightarrowJ/#Psi#muX'
titles['jpsi_x_mu_from_xi'] = '#Xi^{0/-}_{b}#rightarrowJ/#Psi#muX'


for name in sample_names_explicit_all:
    if 'jpsi_x_mu' in name:
        if 'hm' in name:
            titles[name] = titles[name.replace('_hm','')]+' HM'
        else:
            titles[name] = titles[name.replace('_lm','')]+' LM'


#colors
import ROOT
from bokeh.palettes import viridis, all_palettes
col = list(map(ROOT.TColor.GetColor, all_palettes['Spectral'][11]))
col2 = list(map(ROOT.TColor.GetColor, all_palettes['Pastel1'][9]))
col3 = list(map(ROOT.TColor.GetColor, all_palettes['Set1'][9]))
colours = dict()
colours['jpsi_tau' ] = col[0]
colours['jpsi_mu'  ] = col[1]
colours['chic0_mu' ] = col[2]
colours['chic1_mu' ] = col[3]
colours['chic2_mu' ] = col[4]
colours['jpsi_hc'  ] = col[5]
colours['hc_mu'    ] = col[6]
colours['psi2s_mu' ] = col[7]
colours['psi2s_tau'] = col2[7]
colours['jpsi_x_mu'] = col[9]
colours['jpsi_x'] = col[9]
colours['fakes'   ] = col[10]
colours['data'     ] = ROOT.kBlack
colours['jpsi_x_mu_hm'] = col2[0]
colours['jpsi_x_mu_lm'] = col2[1]
colours['jpsi_x_mu_from_bzero'] = col2[0]
colours['jpsi_x_mu_from_bplus'] = col2[1]
colours['jpsi_x_mu_from_bzero_s'] = col2[2]
#colours['jpsi_x_mu_from_bplus_c'] = col2[3]
colours['jpsi_x_mu_from_sigmaminus_b'] = col2[3]
colours['jpsi_x_mu_from_lambdazero_b'] = col2[4]
colours['jpsi_x_mu_from_ximinus_b'] = col2[5]
colours['jpsi_x_mu_from_sigmazero_b'] = col2[6]
colours['jpsi_x_mu_from_xizero_b'] = col2[7]
colours['jpsi_x_mu_from_other'] = col2[8]

colours['jpsi_x_mu_from_sigma'] = col2[6]
colours['jpsi_x_mu_from_xi'] = col[8]

colours['dimuon'] = col2[8]
colours['jpsi_3pi'] = col2[6]
colours['jpsi_pi'] = col2[7]


for i,name in enumerate(jpsi_x_mu_sample_all_splitting):
    if 'jpsi_x_mu' in name:
        if 'hm' in name:
            colours[name] = col2[int(i/2)]
        else:
            colours[name] = col3[int((i-1)/2)]
'''
mean_nn_data_weights_lm = dict()
mean_nn_mc_weights_lm = dict()

mean_nn_data_weights_lm['jpsi_tau' ] = 0.201
mean_nn_data_weights_lm['jpsi_mu'  ] = 0.202
mean_nn_data_weights_lm['psi2s_mu' ] = 0.202
mean_nn_data_weights_lm['chic0_mu' ] = 0.199
mean_nn_data_weights_lm['chic1_mu' ] = 0.198
mean_nn_data_weights_lm['chic2_mu' ] = 0.201
mean_nn_data_weights_lm['hc_mu'    ] = 0.200
mean_nn_data_weights_lm['psi2s_tau'] = 0.231
mean_nn_data_weights_lm['jpsi_hc'  ] = 0.197
mean_nn_data_weights_lm['jpsi_x_mu_from_bzero'] = 0.113
mean_nn_data_weights_lm['jpsi_x_mu_from_bplus'] = 0.114
mean_nn_data_weights_lm['jpsi_x_mu_from_bzero_s'] = 0.119 
mean_nn_data_weights_lm['jpsi_x_mu_from_lambdazero_b'] = 0.109
mean_nn_data_weights_lm['jpsi_x_mu_from_xi'] = 0.104
mean_nn_data_weights_lm['jpsi_x_mu_from_sigma'] = 0.108

mean_nn_mc_weights_lm['jpsi_tau' ] = 2.422
mean_nn_mc_weights_lm['jpsi_mu'  ] = 2.363
mean_nn_mc_weights_lm['psi2s_mu' ] = 2.373
mean_nn_mc_weights_lm['chic0_mu' ] = 2.384
mean_nn_mc_weights_lm['chic1_mu' ] = 2.366
mean_nn_mc_weights_lm['chic2_mu' ] = 2.391
mean_nn_mc_weights_lm['hc_mu'    ] = 2.356
mean_nn_mc_weights_lm['psi2s_tau'] = 2.412
mean_nn_mc_weights_lm['jpsi_hc'  ] = 2.516
mean_nn_mc_weights_lm['jpsi_x_mu_from_bzero'] = 2.389
mean_nn_mc_weights_lm['jpsi_x_mu_from_bplus'] = 2.380
mean_nn_mc_weights_lm['jpsi_x_mu_from_bzero_s'] = 2.388 
mean_nn_mc_weights_lm['jpsi_x_mu_from_lambdazero_b'] = 2.386
mean_nn_mc_weights_lm['jpsi_x_mu_from_xi'] = 2.371
mean_nn_mc_weights_lm['jpsi_x_mu_from_sigma'] = 2.380

mean_nn_data_weights_hm = dict()
mean_nn_mc_weights_hm = dict()

mean_nn_data_weights_hm['jpsi_x_mu_from_bzero'] = 0.313
mean_nn_data_weights_hm['jpsi_x_mu_from_bplus'] = 0.313
mean_nn_data_weights_hm['jpsi_x_mu_from_bzero_s'] = 0.319
mean_nn_data_weights_hm['jpsi_x_mu_from_lambdazero_b'] = 0.303
mean_nn_data_weights_hm['jpsi_x_mu_from_xi'] = 0.308
mean_nn_data_weights_hm['jpsi_x_mu_from_sigma'] = 0.319

mean_nn_mc_weights_hm['jpsi_x_mu_from_bzero'] = 2.552
mean_nn_mc_weights_hm['jpsi_x_mu_from_bplus'] = 2.551
mean_nn_mc_weights_hm['jpsi_x_mu_from_bzero_s'] = 2.527 
mean_nn_mc_weights_hm['jpsi_x_mu_from_lambdazero_b'] = 2.563
mean_nn_mc_weights_hm['jpsi_x_mu_from_xi'] = 2.564
mean_nn_mc_weights_hm['jpsi_x_mu_from_sigma'] = 2.577

'''
