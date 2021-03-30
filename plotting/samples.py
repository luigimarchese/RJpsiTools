sample_names = [
    'jpsi_tau' ,
    'jpsi_mu'  ,
#     'onia'     ,
#     'jpsi_pi'  ,
    'psi2s_mu' ,
    'chic0_mu' ,
    'chic1_mu' ,
    'chic2_mu' ,
    'hc_mu'    ,
    'psi2s_tau',
#     'jpsi_3pi' ,
    'jpsi_hc'  ,
#     'jpsi_x'   ,
    'jpsi_x_mu',
    'data'     ,
]

bc_common = 0.30
#bc_common = 0.45
#fr = 0.119 without trigger selection
#fr = 0.423 #with trigger selection
fr = 0.80
all_comm = 1.
hb_norm = 6.7
weights = dict()
weights['data'     ] = 1.

weights['jpsi_x'   ] = 6.7
weights['jpsi_x_mu'] = 1. * 8.5 # 10. # 8.5
weights['onia'     ] = 1.
weights['jpsi_pi'  ] = 1.
weights['jpsi_3pi' ] = 1.

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
titles['jpsi_x'   ] = 'J/#Psi + X'
titles['jpsi_x_mu'] = 'J/#Psi + #mu'


# import ROOT
# colours = list(map(ROOT.TColor.GetColor, all_palettes['Spectral'][len(samples)]))
