sample_names = [
    'jpsi_tau' ,
    'jpsi_mu'  ,
    'onia'     ,
#     'jpsi_pi'  ,
    'psi2s_mu' ,
    'chic0_mu' ,
    'chic1_mu' ,
    'chic2_mu' ,
    'hc_mu'    ,
#    'psi2s_tau',
#     'jpsi_3pi' ,
    'jpsi_hc'  ,
    'data'     ,
]

bc_common = 0.35
#bc_common = 0.45
#fr = 0.119 without trigger selection
#fr = 0.423 #with trigger selection
fr = 0.6
all_comm = 1.
hb_norm = 6.7
weights = dict()
#weights['jpsi_tau' ] = 1
weights['jpsi_tau' ] = bc_common * all_comm
weights['jpsi_mu'  ] = bc_common * all_comm# * 0.7 # 1.
weights['jpsi_pi'  ] = bc_common * all_comm# * 1.
weights['psi2s_mu' ] = bc_common * all_comm# * 0.336000000 + 0.177300000 + 0.032800000 + 0.001300000
weights['chic0_mu' ] = bc_common * all_comm# * 0.011600000
weights['chic1_mu' ] = bc_common * all_comm# * 0.344000000
weights['chic2_mu' ] = bc_common* all_comm # * 0.195000000
weights['hc_mu'    ] = bc_common* all_comm # * 0.01
weights['psi2s_tau'] = bc_common* all_comm# * 0.336000000 + 0.177300000 + 0.032800000 + 0.001300000
weights['jpsi_3pi' ] = bc_common* all_comm# * 1.
weights['jpsi_hc'  ] = bc_common* all_comm# * 1.
#weights['fakes'    ] = 1.15 * 0.46/(1-0.46)
weights['fakes'    ] = fr/(1-fr) * all_comm
weights['onia'     ] = hb_norm * all_comm# * 6.7 #28.33
weights['data'     ] = 1.

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
