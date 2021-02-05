from collections import defaultdict

class Decay():
    def __init__(self, br, final_state, model, comment=''):
        self.br = br
        self.final_state = final_state
        self.model = model
        self.comment = comment
    def __str__(self):
        mystr = []
        mystr.append('%.8f' %self.br)
        mystr.append(' '.join([iparticle.name for iparticle in self.final_state]))
        mystr.append(self.model) 
        final_str = ' '.join(mystr) + ';'
        if len(self.comment):
            final_str += ' # ' + self.comment
        return final_str 

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
        mystr  = ['Decay %s  # original total forced BR = %.8f' %(self.name, self.initial_total_br)]
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
            Decay(0.001014000   , [particles['MyJ/psi'    ], particles['K+'      ],                                   ], 'SVS'                                                               ),
            Decay(0.001430000   , [particles['MyJ/psi'    ], particles['K*+'     ],                                   ], 'SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus'),
            Decay(0.000049000   , [particles['MyJ/psi'    ], particles['pi+'     ],                                   ], 'SVS'                                                               ),
            Decay(0.000050000   , [particles['MyJ/psi'    ], particles['rho+'    ],                                   ], 'SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus'), 
            Decay(0.0002        , [particles['MyJ/psi'    ], particles['K0'      ], particles['pi+']                  ], 'PHSP'                                                              ),
            Decay(0.0001        , [particles['MyJ/psi'    ], particles['K+'      ], particles['pi0']                  ], 'PHSP'                                                              ),
            Decay(0.0001        , [particles['MyJ/psi'    ], particles["K'_1+"   ],                                   ], 'SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0'                                ),
            Decay(0.0005        , [particles['MyJ/psi'    ], particles['K_2*+'   ],                                   ], 'PHSP'                                                              ),
            Decay(0.001800000   , [particles['MyJ/psi'    ], particles['K_1+'    ],                                   ], 'SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0'                                ),
            Decay(0.000052000   , [particles['MyJ/psi'    ], particles['phi'     ], particles['K+' ]                  ], 'PHSP'                                                              ), 
            Decay(0.001070000   , [particles['MyJ/psi'    ], particles['K+'      ], particles['pi+'], particles['pi-']], 'PHSP'                                                              ), 
            Decay(0.000108000   , [particles['MyJ/psi'    ], particles['eta'     ], particles['K+']                   ], 'PHSP'                                                              ), 
            Decay(0.000350000   , [particles['MyJ/psi'    ], particles['omega'   ], particles['K+']                   ], 'PHSP'                                                              ), 
            Decay(0.000011800   , [particles['MyJ/psi'    ], particles['p+'      ], particles['anti-Lambda0']         ], 'PHSP'                                                              ), 
            Decay(0.000646000   , [particles['Mypsi(2S)'  ], particles['K+'      ],                                   ], 'SVS'                                                               ),
            Decay(0.000620000   , [particles['Mypsi(2S)'  ], particles['K*+'     ],                                   ], 'SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus'), 
            Decay(0.0004        , [particles['Mypsi(2S)'  ], particles['K0'      ], particles['pi+']                  ], 'PHSP'                                                              ),
            Decay(0.0002        , [particles['Mypsi(2S)'  ], particles['K+'      ], particles['pi0']                  ], 'PHSP'                                                              ),
            Decay(0.001900000   , [particles['Mypsi(2S)'  ], particles['K+'      ], particles['pi-'], particles['pi+']], 'PHSP'                                                              ),
            Decay(0.0001        , [particles['Mypsi(2S)'  ], particles['K+'      ], particles['pi0'], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.0001        , [particles['Mypsi(2S)'  ], particles['K0'      ], particles['pi+'], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.0004        , [particles['Mypsi(2S)'  ], particles['K_1+'    ],                                   ], 'PHSP'                                                              ),
            Decay(0.000025800   , [particles['Mypsi(2S)'  ], particles['pi+'     ],                                   ], 'PHSP'                                                              ),
            Decay(0.000490000   , [particles['Mypsi(3770)'], particles['K+'      ],                                   ], 'SVS'                                                               ),
            Decay(0.0005        , [particles['Mypsi(3770)'], particles['K*+'     ],                                   ], 'PHSP'                                                              ),
            Decay(0.0003        , [particles['Mypsi(3770)'], particles['K0'      ], particles['pi+']                  ], 'PHSP'                                                              ),
            Decay(0.0002        , [particles['Mypsi(3770)'], particles['K+'      ], particles['pi0']                  ], 'PHSP'                                                              ),
            Decay(0.0002        , [particles['Mypsi(3770)'], particles['K+'      ], particles['pi-'], particles['pi+']], 'PHSP'                                                              ),
            Decay(0.0001        , [particles['Mypsi(3770)'], particles['K+'      ], particles['pi0'], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.0001        , [particles['Mypsi(3770)'], particles['K0'      ], particles['pi+'], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.0003        , [particles['Mypsi(3770)'], particles['K_1+'    ],                                   ], 'PHSP'                                                              ),
            Decay(0.000133000   , [particles['Mychi_c0'   ], particles['K+'      ],                                   ], 'PHSP'                                                              ),
            Decay(0.0004        , [particles['K*+'        ], particles['Mychi_c0'],                                   ], 'SVS'                                                               ),
            Decay(0.0002        , [particles['Mychi_c0'   ], particles['K0'      ], particles['pi+']                  ], 'PHSP'                                                              ),
            Decay(0.0001        , [particles['Mychi_c0'   ], particles['K+'      ], particles['pi0']                  ], 'PHSP'                                                              ),
            Decay(0.0002        , [particles['Mychi_c0'   ], particles['K+'      ], particles['pi-'], particles['pi+']], 'PHSP'                                                              ),
            Decay(0.0001        , [particles['Mychi_c0'   ], particles['K+'      ], particles['pi0'], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.0001        , [particles['Mychi_c0'   ], particles['K0'      ], particles['pi+'], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.000460000   , [particles['Mychi_c1'   ], particles['K+'      ],                                   ], 'SVS'                                                               ),
            Decay(0.000300000   , [particles['Mychi_c1'   ], particles['K*+'     ],                                   ], 'SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus'), 
            Decay(0.0004        , [particles['Mychi_c1'   ], particles['K0'      ], particles['pi+']                  ], 'PHSP'                                                              ),
            Decay(0.0002        , [particles['Mychi_c1'   ], particles['K+'      ], particles['pi0']                  ], 'PHSP'                                                              ),
            Decay(0.0004        , [particles['Mychi_c1'   ], particles['K+'      ], particles['pi-'], particles['pi+']], 'PHSP'                                                              ),
            Decay(0.0002        , [particles['Mychi_c1'   ], particles['K+'      ], particles['pi0'], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.0002        , [particles['Mychi_c1'   ], particles['K0'      ], particles['pi+'], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.000020000   , [particles['Mychi_c1'   ], particles['pi+'     ],                                   ], 'PHSP'                                                              ),
            Decay(0.00002       , [particles['Mychi_c2'   ], particles['K+'      ],                                   ], 'STS'                                                               ),
            Decay(0.00002       , [particles['Mychi_c2'   ], particles['K*+'     ],                                   ], 'PHSP'                                                              ),
            Decay(0.0002        , [particles['Mychi_c2'   ], particles['K0'      ], particles['pi+']                  ], 'PHSP'                                                              ),
            Decay(0.0001        , [particles['Mychi_c2'   ], particles['K+'      ], particles['pi0']                  ], 'PHSP'                                                              ),
            Decay(0.0002        , [particles['Mychi_c2'   ], particles['K+'      ], particles['pi-'], particles['pi+']], 'PHSP'                                                              ),
            Decay(0.0001        , [particles['Mychi_c2'   ], particles['K+'      ], particles['pi0'], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.0001        , [particles['Mychi_c2'   ], particles['K0'      ], particles['pi+'], particles['pi0']], 'PHSP'                                                              ),
        ],
        charge_conjugate = 'MyB-'
    )

    particles['Myanti-B0'] = Particle(
        'Myanti-B0',
        [                                                    
            Decay(0.000871000, [particles['MyJ/psi'    ], particles['anti-K0'   ]                                        ], 'PHSP'                                                              ),
            Decay(0.000310000, [particles['MyJ/psi'    ], particles['omega'     ], particles['anti-K0']                  ], 'PHSP'                                                              ),
            Decay(0.000009500, [particles['MyJ/psi'    ], particles['eta'       ]                                        ], 'PHSP'                                                              ),
            Decay(0.000019000, [particles['MyJ/psi'    ], particles['pi-'       ], particles['pi+'    ]                  ], 'PHSP'                                                              ),
            Decay(0.000460000, [particles['MyJ/psi'    ], particles['anti-K0'   ], particles['pi-'    ], particles['pi+']], 'PHSP'                                                              ),
            Decay(0.000540000, [particles['MyJ/psi'    ], particles['anti-K0'   ], particles['rho0'   ]                  ], 'PHSP'                                                              ),
            Decay(0.000800000, [particles['MyJ/psi'    ], particles['K*-'       ], particles['pi+'    ]                  ], 'PHSP'                                                              ),
            Decay(0.000660000, [particles['MyJ/psi'    ], particles['anti-K*0'  ], particles['pi-'    ], particles['pi+']], 'PHSP'                                                              ),
            Decay(0.000435500, [particles['MyJ/psi'    ], particles['K_S0'      ]                                        ], 'SVS'                                                               ),
            Decay(0.000435500, [particles['MyJ/psi'    ], particles['K_L0'      ]                                        ], 'SVS'                                                               ),
            Decay(0.001330000, [particles['MyJ/psi'    ], particles['anti-K*0'  ]                                        ], 'SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus'),
            Decay(0.000017600, [particles['MyJ/psi'    ], particles['pi0'       ]                                        ], 'SVS'                                                               ),
            Decay(0.000027000, [particles['MyJ/psi'    ], particles['rho0'      ]                                        ], 'SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus'),
            Decay(0.000030   , [particles['MyJ/psi'    ], particles['omega'     ]                                        ], 'SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus'),
            Decay(0.000000000, [particles['MyJ/psi'    ], particles['K-'        ], particles['pi+'    ]                  ], 'PHSP'                                                              ),
            Decay(0.0001     , [particles['MyJ/psi'    ], particles['anti-K0'   ], particles['pi0'    ]                  ], 'PHSP'                                                              ),
            Decay(0.001300000, [particles['MyJ/psi'    ], particles['anti-K_10' ]                                        ], 'SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0'                                ),
            Decay(0.0001     , [particles['MyJ/psi'    ], particles["anti-K'_10"]                                        ], 'SVV_HELAMP 0.5 0.0 1.0 0.0 0.5 0.0'                                ),
            Decay(0.0005     , [particles['MyJ/psi'    ], particles['anti-K_2*0']                                        ], 'PHSP'                                                              ),
            Decay(0.000094000, [particles['MyJ/psi'    ], particles['phi'       ], particles['anti-K0']                  ], 'PHSP'                                                              ),
            Decay(0.000620000, [particles['Mypsi(2S)'  ], particles['anti-K0'   ]                                        ], 'PHSP'                                                              ),
            Decay(0.000310000, [particles['Mypsi(2S)'  ], particles['K_S0'      ]                                        ], 'SVS'                                                               ),
            Decay(0.000310000, [particles['Mypsi(2S)'  ], particles['K_L0'      ]                                        ], 'SVS'                                                               ),
            Decay(0.000610000, [particles['Mypsi(2S)'  ], particles['anti-K*0'  ]                                        ], 'SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus'),
            Decay(0.0004     , [particles['Mypsi(2S)'  ], particles['K-'        ], particles['pi+'    ]                  ], 'PHSP'                                                              ),
            Decay(0.0002     , [particles['Mypsi(2S)'  ], particles['anti-K0'   ], particles['pi0'    ]                  ], 'PHSP'                                                              ),
            Decay(0.0002     , [particles['Mypsi(2S)'  ], particles['anti-K0'   ], particles['pi+'    ], particles['pi-']], 'PHSP'                                                              ),
            Decay(0.0001     , [particles['Mypsi(2S)'  ], particles['anti-K0'   ], particles['pi0'    ], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.0001     , [particles['Mypsi(2S)'  ], particles['K-'        ], particles['pi+'    ], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.0004     , [particles['Mypsi(2S)'  ], particles['anti-K_10' ]                                        ], 'PHSP'                                                              ),
            Decay(0.00024    , [particles['Mypsi(3770)'], particles['K_S0'      ]                                        ], 'SVS'                                                               ),
            Decay(0.00024    , [particles['Mypsi(3770)'], particles['K_L0'      ]                                        ], 'SVS'                                                               ),
            Decay(0.00048    , [particles['Mypsi(3770)'], particles['anti-K*0'  ]                                        ], 'SVV_HELAMP PKHplus PKphHplus PKHzero PKphHzero PKHminus PKphHminus'),
            Decay(0.00014    , [particles['Mypsi(3770)'], particles['K-'        ], particles['pi+'    ]                  ], 'PHSP'                                                              ),
            Decay(0.00014    , [particles['Mypsi(3770)'], particles['anti-K0'   ], particles['pi0'    ]                  ], 'PHSP'                                                              ),
            Decay(0.00014    , [particles['Mypsi(3770)'], particles['anti-K0'   ], particles['pi+'    ], particles['pi-']], 'PHSP'                                                              ),
            Decay(0.00007    , [particles['Mypsi(3770)'], particles['anti-K0'   ], particles['pi0'    ], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.00007    , [particles['Mypsi(3770)'], particles['K-'        ], particles['pi+'    ], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.00029    , [particles['Mypsi(3770)'], particles['anti-K_10' ]                                        ], 'PHSP'                                                              ),
            Decay(0.000070000, [particles['Mychi_c0'   ], particles['K_S0'      ]                                        ], 'PHSP'                                                              ),
            Decay(0.000070000, [particles['Mychi_c0'   ], particles['K_L0'      ]                                        ], 'PHSP'                                                              ),
            Decay(0.00030    , [particles['anti-K*0'   ], particles['Mychi_c0'  ]                                        ], 'SVS'                                                               ),
            Decay(0.0002     , [particles['Mychi_c0'   ], particles['K-'        ], particles['pi+'    ]                  ], 'PHSP'                                                              ),
            Decay(0.0001     , [particles['Mychi_c0'   ], particles['anti-K0'   ], particles['pi0'    ]                  ], 'PHSP'                                                              ),
            Decay(0.0002     , [particles['Mychi_c0'   ], particles['anti-K0'   ], particles['pi+'    ], particles['pi-']], 'PHSP'                                                              ),
            Decay(0.0001     , [particles['Mychi_c0'   ], particles['anti-K0'   ], particles['pi0'    ], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.0001     , [particles['Mychi_c0'   ], particles['K-'        ], particles['pi+'    ], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.000140000, [particles['Mychi_c0'   ], particles['anti-K0'   ]                                        ], 'PHSP'                                                              ),
            Decay(0.000195000, [particles['Mychi_c1'   ], particles['K_S0'      ]                                        ], 'SVS'                                                               ),
            Decay(0.000195000, [particles['Mychi_c1'   ], particles['K_L0'      ]                                        ], 'SVS'                                                               ),
            Decay(0.000222000, [particles['Mychi_c1'   ], particles['anti-K*0'  ]                                        ], 'SVV_HELAMP PKHminus PKphHminus PKHzero PKphHzero PKHplus PKphHplus'),
            Decay(0.0004     , [particles['Mychi_c1'   ], particles['K-'        ], particles['pi+'    ]                  ], 'PHSP'                                                              ),
            Decay(0.0002     , [particles['Mychi_c1'   ], particles['anti-K0'   ], particles['pi0'    ]                  ], 'PHSP'                                                              ),
            Decay(0.0004     , [particles['Mychi_c1'   ], particles['anti-K0'   ], particles['pi+'    ], particles['pi-']], 'PHSP'                                                              ),
            Decay(0.0002     , [particles['Mychi_c1'   ], particles['anti-K0'   ], particles['pi0'    ], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.0002     , [particles['Mychi_c1'   ], particles['K-'        ], particles['pi+'    ], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.000011200, [particles['Mychi_c1'   ], particles['pi0'       ]                                        ], 'PHSP'                                                              ),
            Decay(0.000390000, [particles['Mychi_c1'   ], particles['anti-K0'   ]                                        ], 'PHSP'                                                              ),
            Decay(0.000158000, [particles['Mychi_c1'   ], particles['K+'        ], particles['pi-'    ]                  ], 'PHSP'                                                              ),
            Decay(0.00005    , [particles['Mychi_c2'   ], particles['K_S0'      ]                                        ], 'STS'                                                               ),
            Decay(0.00005    , [particles['Mychi_c2'   ], particles['K_L0'      ]                                        ], 'STS'                                                               ),
            Decay(0.00003    , [particles['Mychi_c2'   ], particles['anti-K*0'  ]                                        ], 'PHSP'                                                              ),
            Decay(0.0002     , [particles['Mychi_c2'   ], particles['K-'        ], particles['pi+'    ]                  ], 'PHSP'                                                              ),
            Decay(0.0001     , [particles['Mychi_c2'   ], particles['anti-K0'   ], particles['pi0'    ]                  ], 'PHSP'                                                              ),
            Decay(0.0002     , [particles['Mychi_c2'   ], particles['anti-K0'   ], particles['pi+'    ], particles['pi-']], 'PHSP'                                                              ),
            Decay(0.0001     , [particles['Mychi_c2'   ], particles['anti-K0'   ], particles['pi0'    ], particles['pi0']], 'PHSP'                                                              ),
            Decay(0.0001     , [particles['Mychi_c2'   ], particles['K-'        ], particles['pi+'    ], particles['pi0']], 'PHSP'                                                              ),
        ],
        charge_conjugate = 'MyB0'
    )

    particles['MyBs'] = Particle(
        'MyBs',
        [
            Decay(0.00064    , [particles['MyJ/psi'  ], particles["eta'"    ]		                             ], 'SVS'                                 ),
            Decay(0.00032    , [particles['MyJ/psi'  ], particles["eta"	    ]                                    ], 'SVS'                                 ),
            Decay(0.001300000, [particles['MyJ/psi'  ], particles["phi"     ]                                    ], 'SVV_HELAMP  1.0 0.0 1.0 0.0 1.0 0.0' ),
            Decay(0.00008    , [particles['MyJ/psi'  ], particles["K0"	    ]	                                 ], 'SVS'                                 ),
            Decay(0.00070    , [particles['MyJ/psi'  ], particles["K-"      ], particles["K+" ]                  ], 'PHSP'                                ),
            Decay(0.00070    , [particles['MyJ/psi'  ], particles["anti-K0" ], particles["K0" ]                  ], 'PHSP'                                ),
            Decay(0.00070    , [particles['MyJ/psi'  ], particles["K0"      ], particles["K-" ], particles["pi+"]], 'PHSP'                                ),
            Decay(0.00070    , [particles['MyJ/psi'  ], particles["anti-K0" ], particles["K0" ], particles["pi0"]], 'PHSP'                                ),
            Decay(0.00070    , [particles['MyJ/psi'  ], particles["K-"      ], particles["K+" ], particles["pi0"]], 'PHSP'                                ),
            Decay(0.00039    , [particles['MyJ/psi'  ], particles["phi"     ], particles["pi+"], particles["pi-"]], 'PHSP'                                ),
            Decay(0.00039    , [particles['MyJ/psi'  ], particles["phi"     ], particles["pi0"], particles["pi0"]], 'PHSP'                                ),
            Decay(0.0002     , [particles['MyJ/psi'  ], particles["eta"     ], particles["pi+"], particles["pi-"]], 'PHSP'                                ),
            Decay(0.0002     , [particles['MyJ/psi'  ], particles["eta"     ], particles["pi0"], particles["pi0"]], 'PHSP'                                ),
            Decay(0.0004     , [particles['MyJ/psi'  ], particles["eta'"    ], particles["pi+"], particles["pi-"]], 'PHSP'                                ),
            Decay(0.0004     , [particles['MyJ/psi'  ], particles["eta'"    ], particles["pi0"], particles["pi0"]], 'PHSP'                                ),
            Decay(0.0002     , [particles['MyJ/psi'  ], particles["pi+"     ], particles["pi-"]                  ], 'PHSP'                                ),
            Decay(0.0002     , [particles['MyJ/psi'  ], particles["pi0"     ], particles["pi0"]                  ], 'PHSP'                                ),
            Decay(0.000465   , [particles['Mypsi(2S)'], particles["eta'"    ]	                                 ], 'SVS'                                 ),
            Decay(0.000235   , [particles['Mypsi(2S)'], particles["eta"     ]                                    ], 'SVS'                                 ),
            Decay(0.000680000, [particles['Mypsi(2S)'], particles["phi"     ]                                    ], 'SVV_HELAMP  1.0 0.0 1.0 0.0 1.0 0.0' ),
            Decay(0.0003     , [particles['Mypsi(2S)'], particles["K-"      ], particles["K+" ]                  ], 'PHSP'                                ),
            Decay(0.0003     , [particles['Mypsi(2S)'], particles["anti-K0" ], particles["K0" ]                  ], 'PHSP'                                ),
            Decay(0.0003     , [particles['Mypsi(2S)'], particles["K0"      ], particles["K-" ], particles["pi+"]], 'PHSP'                                ),
            Decay(0.0003     , [particles['Mypsi(2S)'], particles["anti-K0" ], particles["K0" ], particles["pi0"]], 'PHSP'                                ),
            Decay(0.0003     , [particles['Mypsi(2S)'], particles["K-"      ], particles["K+" ], particles["pi0"]], 'PHSP'                                ),
            Decay(0.00034    , [particles['Mypsi(2S)'], particles["phi"     ], particles["pi+"], particles["pi-"]], 'PHSP'                                ),
            Decay(0.00034    , [particles['Mypsi(2S)'], particles["phi"     ], particles["pi0"], particles["pi0"]], 'PHSP'                                ),
            Decay(0.0002     , [particles['Mypsi(2S)'], particles["eta"     ], particles["pi+"], particles["pi-"]], 'PHSP'                                ),
            Decay(0.0002     , [particles['Mypsi(2S)'], particles["eta"     ], particles["pi0"], particles["pi0"]], 'PHSP'                                ),
            Decay(0.0004     , [particles['Mypsi(2S)'], particles["eta'"    ], particles["pi+"], particles["pi-"]], 'PHSP'                                ),
            Decay(0.0004     , [particles['Mypsi(2S)'], particles["eta'"    ], particles["pi0"], particles["pi0"]], 'PHSP'                                ),
            Decay(0.0002     , [particles['Mypsi(2S)'], particles["pi+"     ], particles["pi-"]                  ], 'PHSP'                                ),
            Decay(0.0002     , [particles['Mypsi(2S)'], particles["pi0"     ], particles["pi0"]                  ], 'PHSP'                                ),
            Decay(0.00010    , [particles['Mychi_c0' ], particles["eta'"    ]                                    ], 'PHSP'                                ),
            Decay(0.00005    , [particles['Mychi_c0' ], particles["eta"     ]                                    ], 'PHSP'                                ),
            Decay(0.00020    , [particles['phi'      ], particles["Mychi_c0"]                                    ], 'SVS'                                 ),
            Decay(0.00003    , [particles['Mychi_c0' ], particles["K-"      ], particles["K+" ]                  ], 'PHSP'                                ),
            Decay(0.00003    , [particles['Mychi_c0' ], particles["anti-K0" ], particles["K0" ]                  ], 'PHSP'                                ),
            Decay(0.00003    , [particles['Mychi_c0' ], particles["K0"      ], particles["K-" ], particles["pi+"]], 'PHSP'                                ),
            Decay(0.00003    , [particles['Mychi_c0' ], particles["anti-K0" ], particles["K0" ], particles["pi0"]], 'PHSP'                                ),
            Decay(0.00003    , [particles['Mychi_c0' ], particles["K-"      ], particles["K+" ], particles["pi0"]], 'PHSP'                                ),
            Decay(0.0007     , [particles['Mychi_c1' ], particles["eta'"	]                                    ], 'SVS'                                 ),
            Decay(0.0003     , [particles['Mychi_c1' ], particles["eta"     ]                                    ], 'SVS'                                 ),
            Decay(0.0014     , [particles['Mychi_c1' ], particles["phi"     ]                                    ], 'SVV_HELAMP  1.0 0.0 1.0 0.0 1.0 0.0' ),
            Decay(0.00026    , [particles['Mychi_c1' ], particles["K-"      ], particles["K+" ]                  ], 'PHSP'                                ),
            Decay(0.00026    , [particles['Mychi_c1' ], particles["anti-K0" ], particles["K0" ]                  ], 'PHSP'                                ),
            Decay(0.00026    , [particles['Mychi_c1' ], particles["K0"      ], particles["K-" ], particles["pi+"]], 'PHSP'                                ),
            Decay(0.00026    , [particles['Mychi_c1' ], particles["anti-K0" ], particles["K0" ], particles["pi0"]], 'PHSP'                                ),
            Decay(0.00026    , [particles['Mychi_c1' ], particles["K-"      ], particles["K+" ], particles["pi0"]], 'PHSP'                                ),
            Decay(0.00040    , [particles['Mychi_c1' ], particles["phi"     ], particles["pi+"], particles["pi-"]], 'PHSP'                                ),
            Decay(0.00040    , [particles['Mychi_c1' ], particles["phi"     ], particles["pi0"], particles["pi0"]], 'PHSP'                                ),
            Decay(0.0001     , [particles['Mychi_c1' ], particles["eta"     ], particles["pi+"], particles["pi-"]], 'PHSP'                                ),
            Decay(0.0001     , [particles['Mychi_c1' ], particles["eta"     ], particles["pi0"], particles["pi0"]], 'PHSP'                                ),
            Decay(0.0002     , [particles['Mychi_c1' ], particles["eta'"    ], particles["pi+"], particles["pi-"]], 'PHSP'                                ),
            Decay(0.0002     , [particles['Mychi_c1' ], particles["eta'"    ], particles["pi0"], particles["pi0"]], 'PHSP'                                ),
            Decay(0.000465   , [particles['Mychi_c2' ], particles["eta'"    ]                                    ], 'STS'                                 ),
            Decay(0.000235   , [particles['Mychi_c2' ], particles["eta"     ]                                    ], 'STS'                                 ),
            Decay(0.00016    , [particles['Mychi_c2' ], particles["K-"      ], particles["K+"]                   ], 'PHSP'                                ),
            Decay(0.00016    , [particles['Mychi_c2' ], particles["anti-K0" ], particles["K0"]                   ], 'PHSP'                                ),
            Decay(0.00016    , [particles['Mychi_c2' ], particles["K0"      ], particles["K-"], particles["pi+"] ], 'PHSP'                                ),
            Decay(0.00016    , [particles['Mychi_c2' ], particles["anti-K0" ], particles["K0"], particles["pi0"] ], 'PHSP'                                ),
            Decay(0.00016    , [particles['Mychi_c2' ], particles["K-"      ], particles["K+"], particles["pi0"] ], 'PHSP'                                ),
            Decay(0.000465   , [particles['Myh_c'    ], particles["eta'"	]                                    ], 'SVS'                                 ),
            Decay(0.000235   , [particles['Myh_c'    ], particles["eta"     ]                                    ], 'SVS'                                 ),
            Decay(0.0010     , [particles['Myh_c'    ], particles["phi"     ]                                    ], 'SVV_HELAMP  1.0 0.0 1.0 0.0 1.0 0.0' ),
            Decay(0.00016    , [particles['Myh_c'    ], particles["K-"      ], particles["K+"]	                 ], 'PHSP'                                ),
            Decay(0.00016    , [particles['Myh_c'    ], particles["anti-K0" ], particles["K0"]                   ], 'PHSP'                                ),
            Decay(0.00016    , [particles['Myh_c'    ], particles["K0"      ], particles["K-"], particles["pi+"] ], 'PHSP'                                ),
            Decay(0.00016    , [particles['Myh_c'    ], particles["anti-K0" ], particles["K0"], particles["pi0"] ], 'PHSP'                                ),
            Decay(0.00016    , [particles['Myh_c'    ], particles["K-"      ], particles["K+"], particles["pi0"] ], 'PHSP'                                ),
        ],
        charge_conjugate = 'Myanti-Bs'
    )

    particles['MyB*+'] = Particle(
        'MyB*+',
        [
            Decay(1., [particles["MyB+"], particles["gamma"]], 'VSP_PWAVE'),
        ],
        charge_conjugate = 'MyB*-'
    )

    particles['MyB*0'] = Particle(
        'MyB*0',
        [
            Decay(1., [particles["MyB0"], particles["gamma"]], 'VSP_PWAVE'),
        ],
        charge_conjugate = 'Myanti-B*0'
    )

    particles['MyBs*'] = Particle(
        'MyBs*',
        [
            Decay(1., [particles["MyBs"], particles["gamma"]], 'VSP_PWAVE'),
        ],
        charge_conjugate = 'Myanti-Bs*'
    )

    particles['MyBc+'] = Particle(
        'MyBc+',
        [
            Decay(0.01900     , [particles["MyJ/psi"  ], particles["mu+"    ], particles["nu_mu" ]], "PHOTOS BC_VMN 1"                   ),
            Decay(0.00094     , [particles["Mypsi(2S)"], particles["mu+"    ], particles["nu_mu" ]], "PHOTOS BC_VMN 1"                   ),
            Decay(0.0020577   , [particles["Mychi_c0" ], particles["mu+"    ], particles["nu_mu" ]], "PHOTOS BC_SMN 3"                   ),
            Decay(0.0020577   , [particles["Mychi_c1" ], particles["mu+"    ], particles["nu_mu" ]], "PHOTOS BC_VMN 3"                   ),
            Decay(0.0020577   , [particles["Mychi_c2" ], particles["mu+"    ], particles["nu_mu" ]], "PHOTOS BC_TMN 3"                   ),
            Decay(0.0029260   , [particles["Myh_c"    ], particles["mu+"    ], particles["nu_mu" ]], "PHOTOS PHSP"                       ),
            Decay(0.04030     , [particles["MyBs"     ], particles["mu+"    ], particles["nu_mu" ]], "PHOTOS PHSP"                       ),
            Decay(0.05060     , [particles["MyBs*"    ], particles["mu+"    ], particles["nu_mu" ]], "PHOTOS PHSP"                       ),
            Decay(0.00340     , [particles["MyB0"     ], particles["mu+"    ], particles["nu_mu" ]], "PHOTOS PHSP"                       ),
            Decay(0.00580     , [particles["MyB*0"    ], particles["mu+"    ], particles["nu_mu" ]], "PHOTOS PHSP"                       ),
            Decay(0.01900     , [particles["MyJ/psi"  ], particles["e+"     ], particles["nu_e"  ]], "PHOTOS BC_VMN 1"                   ),
            Decay(0.00094     , [particles["Mypsi(2S)"], particles["e+"     ], particles["nu_e"  ]], "PHOTOS BC_VMN 1"                   ),
            Decay(0.0020577   , [particles["Mychi_c0" ], particles["e+"     ], particles["nu_e"  ]], "PHOTOS BC_SMN 3"                   ),
            Decay(0.0020577   , [particles["Mychi_c1" ], particles["e+"     ], particles["nu_e"  ]], "PHOTOS BC_VMN 3"                   ),
            Decay(0.0020577   , [particles["Mychi_c2" ], particles["e+"     ], particles["nu_e"  ]], "PHOTOS BC_TMN 3"                   ),
            Decay(0.0029260   , [particles["Myh_c"    ], particles["e+"     ], particles["nu_e"  ]], "PHOTOS PHSP"                       ),
            Decay(0.04030     , [particles["MyBs"     ], particles["e+"     ], particles["nu_e"  ]], "PHOTOS PHSP"                       ),
            Decay(0.05060     , [particles["MyBs*"    ], particles["e+"     ], particles["nu_e"  ]], "PHOTOS PHSP"                       ),
            Decay(0.00340     , [particles["MyB0"     ], particles["e+"     ], particles["nu_e"  ]], "PHOTOS PHSP"                       ),
            Decay(0.00580     , [particles["MyB*0"    ], particles["e+"     ], particles["nu_e"  ]], "PHOTOS PHSP"                       ),
            Decay(0.00480     , [particles["MyJ/psi"  ], particles["tau+"   ], particles["nu_tau"]], "PHOTOS BC_VMN 1"                   ),
            Decay(0.00008     , [particles["Mypsi(2S)"], particles["tau+"   ], particles["nu_tau"]], "PHOTOS BC_VMN 1"                   ),
            Decay(0.00130     , [particles["MyJ/psi"  ], particles["pi+"    ]                     ], "SVS"                               ),
            Decay(0.00400     , [particles["MyJ/psi"  ], particles["rho+"   ]                     ], "SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0"),
            Decay(0.00011     , [particles["MyJ/psi"  ], particles["K+"     ]                     ], "SVS"                               ),
            Decay(0.00022     , [particles["MyJ/psi"  ], particles["K*+"    ]                     ], "SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0"),
            Decay(0.00170     , [particles["MyJ/psi"  ], particles["D_s+"   ]                     ], "SVS"                               ),
            Decay(0.00670     , [particles["MyJ/psi"  ], particles["D_s*+"  ]                     ], "SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0"),
            Decay(0.00009     , [particles["MyJ/psi"  ], particles["D+"     ]                     ], "SVS"                               ),
            Decay(0.00028     , [particles["MyJ/psi"  ], particles["D*+"    ]                     ], "SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0"),
            Decay(0.16400     , [particles["MyBs"     ], particles["pi+"    ]                     ], "PHSP"                              ),
            Decay(0.07200     , [particles["rho+"     ], particles["MyBs"   ]                     ], "SVS"                               ),
            Decay(0.01060     , [particles["MyBs"     ], particles["K+"     ]                     ], "PHSP"                              ),
            Decay(0.00000     , [particles["K*+"      ], particles["MyBs"   ]                     ], "SVS"                               ),
            Decay(0.06500     , [particles["MyBs*"    ], particles["pi+"    ]                     ], "SVS"                               ),
            Decay(0.20200     , [particles["MyBs*"    ], particles["rho+"   ]                     ], "SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0"),
            Decay(0.00370     , [particles["MyBs*"    ], particles["K+"     ]                     ], "SVS"                               ),
            Decay(0.00000     , [particles["MyBs*"    ], particles["K*+"    ]                     ], "SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0"),
            Decay(0.01060     , [particles["MyB0"     ], particles["pi+"    ]                     ], "PHSP"                              ),
            Decay(0.00960     , [particles["rho+"     ], particles["MyB0"   ]                     ], "SVS"                               ),
            Decay(0.00070     , [particles["MyB0"     ], particles["K+"     ]                     ], "PHSP"                              ),
            Decay(0.00015     , [particles["K*+"      ], particles["MyB0"   ]                     ], "SVS"                               ),
            Decay(0.00950     , [particles["MyB*0"    ], particles["pi+"    ]                     ], "SVS"                               ),
            Decay(0.02570     , [particles["MyB*0"    ], particles["rho+"   ]                     ], "SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0"),
            Decay(0.00055     , [particles["MyB*0"    ], particles["K+"     ]                     ], "SVS"                               ),
            Decay(0.00058     , [particles["MyB*0"    ], particles["K*+"    ]                     ], "SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0"),
            Decay(0.00037     , [particles["MyB+"     ], particles["pi0"    ]                     ], "PHSP"                              ),
            Decay(0.00034     , [particles["rho0"     ], particles["MyB+"   ]                     ], "SVS"                               ),
            Decay(0.01980     , [particles["MyB+"     ], particles["anti-K0"]                     ], "PHSP"                              ),
            Decay(0.00430     , [particles["K*0"      ], particles["MyB+"   ]                     ], "SVS"                               ),
            Decay(0.00033     , [particles["MyB*+"    ], particles["pi0"    ]                     ], "SVS"                               ),
            Decay(0.00090     , [particles["MyB*+"    ], particles["rho0"   ]                     ], "SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0"),
            Decay(0.01600     , [particles["MyB*+"    ], particles["anti-K0"]                     ], "SVS"                               ),
            Decay(0.01670     , [particles["MyB*+"    ], particles["K*0"    ]                     ], "SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0"),
        ],
        charge_conjugate = 'MyBc-'
    )

    particles['MyLambda_b0'] = Particle(
        'MyLambda_b0',
        [
            Decay(0.00047, [particles["Lambda0"], particles["MyJ/psi"  ]], 'PHSP'),
            Decay(0.00038, [particles["Lambda0"], particles["Mypsi(2S)"]], 'PHSP'),
        ],
    )

    particles['MyXi_b-'] = Particle(
        'MyXi_b-',
        [
            Decay(0.00047, [particles["Xi-"], particles["MyJ/psi"  ]], 'PHSP'),
            Decay(0.00038, [particles["Xi-"], particles["Mypsi(2S)"]], 'PHSP'),
        ],
        charge_conjugate = 'Myanti-Xi_b+'
    )

    particles['MyXi_b0-'] = Particle(
        'MyXi_b0',
        [
            Decay(0.00047, [particles["Xi0"], particles["MyJ/psi"  ]], 'PHSP'),
        ],
        charge_conjugate = 'Myanti-Xi_b0'
    )

    particles['MyOmega_b-'] = Particle(
        'MyOmega_b-',
        [
            Decay(0.00047, [particles["Omega-"], particles["MyJ/psi"  ]], 'PHSP'),
            Decay(0.00038, [particles["Omega-"], particles["Mypsi(2S)"]], 'PHSP'),
        ],
        charge_conjugate = 'Myanti-Omega_b+'
    )
    
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
    particles_to_print_in_dec_file.append(particles['Myanti-B0'  ])
    particles_to_print_in_dec_file.append(particles['MyBs'       ])
    particles_to_print_in_dec_file.append(particles['MyBc+'      ])
    particles_to_print_in_dec_file.append(particles['MyB*+'      ])
    particles_to_print_in_dec_file.append(particles['MyB*0'      ])
    particles_to_print_in_dec_file.append(particles['MyBs*'      ])
    particles_to_print_in_dec_file.append(particles['MyLambda_b0'])
    particles_to_print_in_dec_file.append(particles['MyXi_b-'    ])
    particles_to_print_in_dec_file.append(particles['MyXi_b0-'   ])
    particles_to_print_in_dec_file.append(particles['MyOmega_b-' ])

    for iparticle in particles_to_print_in_dec_file:
        iparticle.factor_in_forced_decays()
        
    # WARNING! the normalisation MUST happen only as last step!
    for iparticle in particles_to_print_in_dec_file:
        iparticle.normalise_total_br()
    
    with open('HbToJpsiMuMuInclusive.dec', 'w') as ff:
        # Preamble and Aliases
                
        # Charmonium states
        print('Alias      MyJ/psi          J/psi'          , file=ff)
        print('Alias      Mypsi(2S)        psi(2S)'        , file=ff)
        print('Alias      Mypsi(3770)      psi(3770)'      , file=ff)
        print('Alias      Mychi_c0         chi_c0'         , file=ff)
        print('Alias      Mychi_c1         chi_c1'         , file=ff)
        print('Alias      Mychi_c2         chi_c2'         , file=ff)
        print('Alias      Myh_c            h_c'            , file=ff)
                
        # B mesons        
        print('Alias      MyB+             B+'             , file=ff)
        print('Alias      MyB-             B-'             , file=ff)
        print('Alias      Myanti-B0        anti-B0'        , file=ff)
        print('Alias      MyB0             B0'             , file=ff)
        print('Alias      Myanti-Bs        anti-B_s0'      , file=ff)
        print('Alias      MyBs             B_s0'           , file=ff)
        print('Alias      MyBc+            B_c+'           , file=ff)
        print('Alias      MyBc-            B_c-'           , file=ff)
        print('Alias      MyB*+            B*+'            , file=ff)
        print('Alias      MyB*-            B*-'            , file=ff)
        print('Alias      MyB*0            B*0'            , file=ff)
        print('Alias      Myanti-B*0       anti-B*0'       , file=ff)
        print('Alias      MyBs*            B_s*0'          , file=ff)
        print('Alias      Myanti-Bs*       anti-B_s*0'     , file=ff)

        print('Alias      MyLambda_b0      Lambda_b0'      , file=ff)
        print('Alias      MyXi_b-          Xi_b-'          , file=ff)
        print('Alias      Myanti-Xi_b+     anti-Xi_b+'     , file=ff)
        print('Alias      MyXi_b0          Xi_b0'          , file=ff)
        print('Alias      Myanti-Xi_b0     anti-Xi_b0'     , file=ff)
        print('Alias      MyOmega_b-       Omega_b-'       , file=ff)
        print('Alias      Myanti-Omega_b+  anti-Omega_b+'  , file=ff)

        print('ChargeConj MyB-             MyB+'           , file=ff)
        print('ChargeConj Myanti-B0        MyB0'           , file=ff)
        print('ChargeConj Myanti-Bs        MyBs'           , file=ff)
        print('ChargeConj MyBc-            MyBc+'          , file=ff)

        print('ChargeConj MyB*-            MyB*+'          , file=ff)
        print('ChargeConj MyB*0            Myanti-B*0'     , file=ff)
        print('ChargeConj MyBs*            Myanti-Bs*'     , file=ff)
    
        print('ChargeConj MyXi_b-          Myanti-Xi_b+'   , file=ff)
        print('ChargeConj MyXi_b0          Myanti-Xi_b0'   , file=ff)
        print('ChargeConj MyOmega_b-       Myanti-Omega_b+', file=ff)

        for iparticle in particles_to_print_in_dec_file:
            print('\n', file=ff)  
            print(iparticle, file=ff)    

        print('\nEnd\n', file=ff)

