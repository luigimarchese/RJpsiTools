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
    'psi2s_tau',
#     'jpsi_3pi' ,
    'jpsi_hc'  ,
    'data'     ,
]

weights = dict()
weights['jpsi_tau' ] = 1.15 * 0.79 * 0.7
weights['jpsi_mu'  ] = 1.15 * 0.79 * 0.7 # 1.
weights['onia'     ] = 1.15 * 0.79 * 10. #28.33
weights['jpsi_pi'  ] = 1.15 * 0.79 * 1.
weights['psi2s_mu' ] = 1.15 * 0.79 * 0.336000000 + 0.177300000 + 0.032800000 + 0.001300000
weights['chic0_mu' ] = 1.15 * 0.79 * 0.011600000
weights['chic1_mu' ] = 1.15 * 0.79 * 0.344000000
weights['chic2_mu' ] = 1.15 * 0.79 * 0.195000000
weights['hc_mu'    ] = 1.15 * 0.79 * 0.01
weights['psi2s_tau'] = 1.15 * 0.79 * 0.336000000 + 0.177300000 + 0.032800000 + 0.001300000
weights['jpsi_3pi' ] = 1.15 * 0.79 * 1.
weights['jpsi_hc'  ] = 1.15 * 0.79 * 1.
weights['fakes'    ] = 1.15 * 2.5
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
