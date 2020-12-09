from collections import defaultdict

class Decay():
    def __init__(self, br, final_state, model, comment=''):
        self.br = br
        self.final_state = final_state
        self.model = model
        self.comment = comment
    def __str__(self):
        mystr = []
        mystr.append('%.7f' %self.br)
        mystr.append(' '.join([iparticle.name for iparticle in self.final_state]))
        mystr.append(self.model) 
        return ' '.join(mystr) + '; # ' + self.comment

class Particle():
    def __init__(self, name, decays=[], charge_conjugate=None):
        self.name = name
        self.decays = decays
        self.initial_total_br = self.total_br() if len(decays) else 1.
        self._is_total_br_already_normalised = False
        self._are_forced_decays_already_factored_in = False
        self.charge_conjugate = charge_conjugate
    
    def total_br(self):
        if not len(self.decays): return 1.
        return sum([idecay.br for idecay in self.decays])

    def factor_in_forced_decays(self, force=False):
        if self._are_forced_decays_already_factored_in and force==False:
            print("%s forced decays already factored in, won't redo" %self.name)
#         elif self._is_total_br_already_normalised:
#             print("total BR already normalised to unity, this should happen *after* factoring forced decays in. Won't do")
        else:
            print("if a decay contains a particle that has forced decays, then the decay's BR gets scaled down by the total BR that is forced")
            new_decays = []
            for idecay in self.decays:
                forced_decays_factor = 1.
                for iparticle in idecay.final_state:
#                     if iparticle.initial_total_br != 1.:
                    if iparticle.total_br() != 1.:
                        print('%s'%iparticle.name)
                        iparticle.factor_in_forced_decays()
#                         forced_decays_factor *= iparticle.initial_total_br
                        forced_decays_factor *= iparticle.total_br()
                new_decays.append(Decay(idecay.br*forced_decays_factor, idecay.final_state, idecay.model, idecay.comment))
            self.decays = new_decays
            self._are_forced_decays_already_factored_in = True
            
    def normalise_total_br(self, force=False):
        if self._is_total_br_already_normalised and force==False:
            print("total BR already normalised to unity, won't redo")
        else:
            print("normalising total BR to unity")
            total_br = self.total_br()
            self.decays = [Decay(idecay.br/total_br, idecay.final_state, idecay.model, idecay.comment) for idecay in self.decays]
            self._is_total_br_already_normalised = True
            
    def __str__(self):
        mystr  = ['Decay %s  # original total forced BR = %.7f' %(self.name, self.initial_total_br)]
        mystr += [idecay.__str__() for idecay in self.decays]
        mystr += ['Enddecay']
        if self.charge_conjugate:
             mystr += ['CDecay %s' %self.charge_conjugate]
        return '\n'.join(mystr)        

class particles_dict(defaultdict):
    '''
    Specialised dictionary, if a key is missing it returns a default particle named
    as the missing key.
    This spares the effort of manually defining all particles, even those that are not
    customised, for example by defining forced decays.
    '''
    def __missing__(self, missing_key):
        return Particle(missing_key)

##########################################################################################
##########################################################################################
       
if __name__ == '__main__':

    particles = particles_dict()
    
    # forced decays
    particles['MyJ/psi'] = Particle(
        'MyJ/psi',
        [
            Decay(0.0593, [particles['mu+'], particles['mu-']], 'PHOTOS  VLL', ''),
        ]
    )
    
    particles['Mychi_c0'] = Particle(
        'Mychi_c0',
        [
            Decay(0.0116, [particles['gamma'], particles['MyJ/psi']], 'PHSP', ''),
        ]
    )

    particles['Mychi_c1'] = Particle(
        'Mychi_c1',
        [
            Decay(0.344, [particles['MyJ/psi'], particles['gamma']], 'VVP 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0', ''),
        ]
    )
    particles['Mychi_c2'] = Particle(
        'Mychi_c2',
        [
            Decay(0.195, [particles['gamma'], particles['MyJ/psi']], 'PHSP', ''),
        ]
    )

    particles['Myh_c'] = Particle(
        'Myh_c',
        [
            Decay(0.01, [particles['MyJ/psi'], particles['pi0']], 'PHSP', ''),
        ]
    )

    particles['Mypsi(2S)'] = Particle(
        'Mypsi(2S)',
        [
            Decay(0.3360, [particles['MyJ/psi'], particles['pi+'     ], particles['pi-']], 'VVPIPI'                          ),     
            Decay(0.1773, [particles['MyJ/psi'], particles['pi0'     ], particles['pi0']], 'VVPIPI'                          ),     
            Decay(0.0328, [particles['MyJ/psi'], particles['eta'     ]                  ], 'PARTWAVE 0.0 0.0 1.0 0.0 0.0 0.0'),     
            Decay(0.0013, [particles['MyJ/psi'], particles['pi0'     ]                  ], 'PARTWAVE 0.0 0.0 1.0 0.0 0.0 0.0'),     
            Decay(0.0962, [particles['gamma'  ], particles['Mychi_c0']                  ], 'PHSP'                            ),     
            Decay(0.0920, [particles['gamma'  ], particles['Mychi_c1']                  ], 'PHSP'                            ),     
            Decay(0.0874, [particles['gamma'  ], particles['Mychi_c2']                  ], 'PHSP'                            ),     
            Decay(0.0008, [particles['Myh_c'  ], particles['gamma'   ]                  ], 'PHSP'                            ),     
        ]
    )

    particles['Mypsi(3770)'] = Particle(
        'Mypsi(3770)',
        [
            Decay(0.001930000, [particles['MyJ/psi'], particles['pi+'     ], particles['pi-']], 'PHSP'),     
            Decay(0.000800000, [particles['MyJ/psi'], particles['pi0'     ], particles['pi0']], 'PHSP'),     
            Decay(0.000900000, [particles['MyJ/psi'], particles['eta'     ]                  ], 'PHSP'),     
        ]
    )


    particles['MyB+'] = Particle(
        'MyB+',
        [
            Decay(0.001014000   , [particles['MyJ/psi'    ], particles['K+'      ],                                   ], 'SVS'                                                             ),
            Decay(0.001430000   , [particles['MyJ/psi'    ], particles['K*+'     ],                                   ], 'SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHmin'),
            Decay(0.000049000   , [particles['MyJ/psi'    ], particles['pi+'     ],                                   ], 'SVS'                                                             ),
            Decay(0.000050000   , [particles['MyJ/psi'    ], particles['rho+'    ],                                   ], 'SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHmin'), 
            Decay(0.0002        , [particles['MyJ/psi'    ], particles['K0'      ], particles['pi+']                  ], 'PHSP'                                                            ),
            Decay(0.0001        , [particles['MyJ/psi'    ], particles['K+'      ], particles['pi0']                  ], 'PHSP'                                                            ),
            Decay(0.0001        , [particles['MyJ/psi'    ], particles["K'_1+"   ],                                   ], 'SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0'                              ),
            Decay(0.0005        , [particles['MyJ/psi'    ], particles['K_2*+'   ],                                   ], 'PHSP'                                                            ),
            Decay(0.001800000   , [particles['MyJ/psi'    ], particles['K_1+'    ],                                   ], 'SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0'                              ),
            Decay(0.000052000   , [particles['MyJ/psi'    ], particles['phi'     ], particles['K+' ]                  ], 'PHSP'                                                            ), 
            Decay(0.001070000   , [particles['MyJ/psi'    ], particles['K+'      ], particles['pi+'], particles['pi-']], 'PHSP'                                                            ), 
            Decay(0.000108000   , [particles['MyJ/psi'    ], particles['eta'     ], particles['K+']                   ], 'PHSP'                                                            ), 
            Decay(0.000350000   , [particles['MyJ/psi'    ], particles['omega'   ], particles['K+']                   ], 'PHSP'                                                            ), 
            Decay(0.000011800   , [particles['MyJ/psi'    ], particles['p+'      ], particles['anti-Lambda0']         ], 'PHSP'                                                            ), 
            Decay(0.000646000   , [particles['Mypsi(2S)'  ], particles['K+'      ],                                   ], 'SVS'                                                             ),
            Decay(0.000620000   , [particles['Mypsi(2S)'  ], particles['K*+'     ],                                   ], 'SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHmin'), 
            Decay(0.0004        , [particles['Mypsi(2S)'  ], particles['K0'      ], particles['pi+']                  ], 'PHSP'                                                            ),
            Decay(0.0002        , [particles['Mypsi(2S)'  ], particles['K+'      ], particles['pi0']                  ], 'PHSP'                                                            ),
            Decay(0.001900000   , [particles['Mypsi(2S)'  ], particles['K+'      ], particles['pi-'], particles['pi+']], 'PHSP'                                                            ),
            Decay(0.0001        , [particles['Mypsi(2S)'  ], particles['K+'      ], particles['pi0'], particles['pi0']], 'PHSP'                                                            ),
            Decay(0.0001        , [particles['Mypsi(2S)'  ], particles['K0'      ], particles['pi+'], particles['pi0']], 'PHSP'                                                            ),
            Decay(0.0004        , [particles['Mypsi(2S)'  ], particles['K_1+'    ],                                   ], 'PHSP'                                                            ),
            Decay(0.000025800   , [particles['Mypsi(2S)'  ], particles['pi+'     ],                                   ], 'PHSP'                                                            ),
            Decay(0.000490000   , [particles['Mypsi(3770)'], particles['K+'      ],                                   ], 'SVS'                                                             ),
            Decay(0.0005        , [particles['Mypsi(3770)'], particles['K*+'     ],                                   ], 'PHSP'                                                            ),
            Decay(0.0003        , [particles['Mypsi(3770)'], particles['K0'      ], particles['pi+']                  ], 'PHSP'                                                            ),
            Decay(0.0002        , [particles['Mypsi(3770)'], particles['K+'      ], particles['pi0']                  ], 'PHSP'                                                            ),
            Decay(0.0002        , [particles['Mypsi(3770)'], particles['K+'      ], particles['pi-'], particles['pi+']], 'PHSP'                                                            ),
            Decay(0.0001        , [particles['Mypsi(3770)'], particles['K+'      ], particles['pi0'], particles['pi0']], 'PHSP'                                                            ),
            Decay(0.0001        , [particles['Mypsi(3770)'], particles['K0'      ], particles['pi+'], particles['pi0']], 'PHSP'                                                            ),
            Decay(0.0003        , [particles['Mypsi(3770)'], particles['K_1+'    ],                                   ], 'PHSP'                                                            ),
            Decay(0.000133000   , [particles['Mychi_c0'   ], particles['K+'      ],                                   ], 'PHSP'                                                            ),
            Decay(0.0004        , [particles['K*+'        ], particles['Mychi_c0'],                                   ], 'SVS'                                                             ),
            Decay(0.0002        , [particles['Mychi_c0'   ], particles['K0'      ], particles['pi+']                  ], 'PHSP'                                                            ),
            Decay(0.0001        , [particles['Mychi_c0'   ], particles['K+'      ], particles['pi0']                  ], 'PHSP'                                                            ),
            Decay(0.0002        , [particles['Mychi_c0'   ], particles['K+'      ], particles['pi-'], particles['pi+']], 'PHSP'                                                            ),
            Decay(0.0001        , [particles['Mychi_c0'   ], particles['K+'      ], particles['pi0'], particles['pi0']], 'PHSP'                                                            ),
            Decay(0.0001        , [particles['Mychi_c0'   ], particles['K0'      ], particles['pi+'], particles['pi0']], 'PHSP'                                                            ),
            Decay(0.000460000   , [particles['Mychi_c1'   ], particles['K+'      ],                                   ], 'SVS'                                                             ),
            Decay(0.000300000   , [particles['Mychi_c1'   ], particles['K*+'     ],                                   ], 'SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHmin'), 
            Decay(0.0004        , [particles['Mychi_c1'   ], particles['K0'      ], particles['pi+']                  ], 'PHSP'                                                            ),
            Decay(0.0002        , [particles['Mychi_c1'   ], particles['K+'      ], particles['pi0']                  ], 'PHSP'                                                            ),
            Decay(0.0004        , [particles['Mychi_c1'   ], particles['K+'      ], particles['pi-'], particles['pi+']], 'PHSP'                                                            ),
            Decay(0.0002        , [particles['Mychi_c1'   ], particles['K+'      ], particles['pi0'], particles['pi0']], 'PHSP'                                                            ),
            Decay(0.0002        , [particles['Mychi_c1'   ], particles['K0'      ], particles['pi+'], particles['pi0']], 'PHSP'                                                            ),
            Decay(0.000020000   , [particles['Mychi_c1'   ], particles['pi+'     ],                                   ], 'PHSP'                                                            ),
            Decay(0.00002       , [particles['Mychi_c2'   ], particles['K+'      ],                                   ], 'STS'                                                             ),
            Decay(0.00002       , [particles['Mychi_c2'   ], particles['K*+'     ],                                   ], 'PHSP'                                                            ),
            Decay(0.0002        , [particles['Mychi_c2'   ], particles['K0'      ], particles['pi+']                  ], 'PHSP'                                                            ),
            Decay(0.0001        , [particles['Mychi_c2'   ], particles['K+'      ], particles['pi0']                  ], 'PHSP'                                                            ),
            Decay(0.0002        , [particles['Mychi_c2'   ], particles['K+'      ], particles['pi-'], particles['pi+']], 'PHSP'                                                            ),
            Decay(0.0001        , [particles['Mychi_c2'   ], particles['K+'      ], particles['pi0'], particles['pi0']], 'PHSP'                                                            ),
            Decay(0.0001        , [particles['Mychi_c2'   ], particles['K0'      ], particles['pi+'], particles['pi0']], 'PHSP'                                                            ),
        ],
        charge_conjugate = 'MyB-'
    )

#     mypsi2s = particles['Mypsi(2S)']
#     print('\n\n')  
#     print(mypsi2s)  
#     print('\n\n')  
#     mypsi2s.factor_in_forced_decays()
#     print('\n\n')  
#     print(mypsi2s)
#     mypsi2s.normalise_total_br()
#     print('\n\n')  
#     print(mypsi2s)    

#     mychic0 = particles['Mychi_c0']
#     print('\n\n')  
#     print(mychic0)
# 
#     mybplus = particles['MyB+']
#     print('\n\n')  
#     print(mybplus)
#     print('\n\n')  
#     mybplus.factor_in_forced_decays()
#     print('\n\n')  
#     print(mybplus)
#     mybplus.normalise_total_br()
#     print('\n\n')  
#     print(mybplus)    

    # now let's create the .dec file
    particles_to_print_in_dec_file = []
    
    particles_to_print_in_dec_file.append(particles['MyJ/psi'    ])
    particles_to_print_in_dec_file.append(particles['Mychi_c0'   ])
    particles_to_print_in_dec_file.append(particles['Mychi_c1'   ])
    particles_to_print_in_dec_file.append(particles['Mychi_c2'   ])
    particles_to_print_in_dec_file.append(particles['Mypsi(2S)'  ])
    particles_to_print_in_dec_file.append(particles['Mypsi(3770)'])
    particles_to_print_in_dec_file.append(particles['Myh_c'      ])
    particles_to_print_in_dec_file.append(particles['MyB+'       ])

    for iparticle in particles_to_print_in_dec_file:
        iparticle.factor_in_forced_decays()
        
    # WARNING! the normalisation MUST happen only as last step!
    for iparticle in particles_to_print_in_dec_file:
        iparticle.normalise_total_br()
    
    with open('BToJpsiMuMuInclusive.dec', 'w') as ff:
        # Preamble and Aliases
        
        # Charmonium states
        print('Alias      MyJ/psi      J/psi    ', file=ff)
        print('Alias      Mypsi(2S)    psi(2S)  ', file=ff)
        print('Alias      Mypsi(3770)  psi(3770)', file=ff)
        print('Alias      Mychi_c0     chi_c0   ', file=ff)
        print('Alias      Mychi_c1     chi_c1   ', file=ff)
        print('Alias      Mychi_c2     chi_c2   ', file=ff)
        print('Alias      Myh_c        h_c      ', file=ff)
        
        # B mesons
        print('Alias      MyB+         B+'       , file=ff)
        print('Alias      MyB-         B-'       , file=ff)
        print('ChargeConj MyB-         MyB+'     , file=ff)

        for iparticle in particles_to_print_in_dec_file:
            print('\n', file=ff)  
            print(iparticle, file=ff)    

        print('\nEnd\n', file=ff)












