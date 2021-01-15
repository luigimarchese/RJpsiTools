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

    # missing quasi-2-body decays, i.e. Ds*
    # https://gitlab.cern.ch/lhcb-datapkg/Gen/DecFiles/-/blob/master/dkfiles/Bc_JpsiCharmQuasi2Body,mm,muX=JpsiLeptonInAcceptance.dec
    # using Kieselev FF
    particles['MyBc-'] = Particle(
        'MyBc-',
        [
            Decay(1.00000 , [particles["MyJ/psi"  ], particles["mu-"     ], particles["anti-nu_mu " ]                   ], "PHOTOS BC_VMN 1"                          , 'from PDG 2020 https://bit.ly/30NDbWm'                                                                                                                          ),
            Decay(0.05000 , [particles["Mypsi(2S)"], particles["mu-"     ], particles["anti-nu_mu " ]                   ], "PHOTOS BC_VMN 1"                          , 'BR(J/Psi mu nu) = 1.95%, BR(Psi(2S) mu nu) = 0.1%, reported from various sources, summarised here https://bit.ly/3jLaZuO and in agreement with EvtGen defaults'),
            Decay(0.10830 , [particles["Mychi_c0" ], particles["mu-"     ], particles["anti-nu_mu " ]                   ], "PHOTOS BC_SMN 3"                          , 'average from https://bit.ly/3jLaZuO table 2, then equally split by 3 for each chi_c species'                                                                   ),
            Decay(0.10830 , [particles["Mychi_c1" ], particles["mu-"     ], particles["anti-nu_mu " ]                   ], "PHOTOS BC_VMN 3"                          , 'average from https://bit.ly/3jLaZuO table 2, then equally split by 3 for each chi_c species'                                                                   ),
            Decay(0.10830 , [particles["Mychi_c2" ], particles["mu-"     ], particles["anti-nu_mu " ]                   ], "PHOTOS BC_TMN 3"                          , 'average from https://bit.ly/3jLaZuO table 2, then equally split by 3 for each chi_c species'                                                                   ),
            Decay(0.15400 , [particles["Myh_c"    ], particles["mu-"     ], particles["anti-nu_mu " ]                   ], "PHOTOS PHSP"                              , 'average from https://bit.ly/3jLaZuO table 2'                                                                                                                   ),
            Decay(1.00000 , [particles["MyJ/psi"  ], particles["tau-"    ], particles["anti-nu_tau" ]                   ], "PHOTOS BC_VMN 1"                          , 'Set to be equal to that of the muon channel. SM 0.25-0.29, LHCb 0.71 https://bit.ly/3deLlfr'                                                                   ),
            Decay(0.00420 , [particles["Mypsi(2S)"], particles["tau-"    ], particles["anti-nu_tau" ]                   ], "PHOTOS BC_VMN 1"                          , 'from EvtGen defaults, assumes SM R(J/Psi)=0.25'                                                                                                                ),
            Decay(0.04690 , [particles["MyJ/psi"  ], particles["pi-"     ]                                              ], "PHOTOS SVS"                               , 'from PDG 2020 https://bit.ly/30NDbWm'                                                                                                                          ),
#             Decay(0.11256 , [particles["MyJ/psi"  ], particles["pi-"     ], particles["pi-"         ], particles["pi+" ]], "PHOTOS BC_VHAD 1"                         , 'from PDG 2020 https://bit.ly/30NDbWm # https://bit.ly/2VDZ2Nn'                                                                                                 ),
#             Decay(0.11256 , [particles["MyJ/psi"  ], particles["pi-"     ], particles["pi-"         ], particles["pi+" ]], "PHOTOS PHSP"                         , 'from PDG 2020 https://bit.ly/30NDbWm # https://bit.ly/2VDZ2Nn'                                                                                                 ),
            Decay(0.11256 , [particles["MyJ/psi"  ], particles["pi-"     ], particles["pi-"         ], particles["pi+" ]], "PHOTOS BC_VNPI 1"                         , 'from PDG 2020 https://bit.ly/30NDbWm # https://bit.ly/2VDZ2Nn'                                                                                                 ),
            Decay(0.003705, [particles["MyJ/psi"  ], particles["K-"      ]                                              ], "PHOTOS SVS"                               , 'PDG https://pdglive.lbl.gov/BranchingRatio.action?desig=14&parCode=S091'                                                                                       ),
            Decay(0.13601 , [particles["MyJ/psi"  ], particles["D_s-"    ]                                              ], "PHOTOS SVS"                               , 'LHCb BR(J/Psi Ds)/BR(J/Psi pi) = 2.90  https://journals.aps.org/prd/abstract/10.1103/PhysRevD.87.112012, https://inspirehep.net/literature/1385102'            ),
            Decay(0.32234 , [particles["MyJ/psi"  ], particles["D_s*-"   ]                                              ], "PHOTOS SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0", 'LHCb BR(J/Psi Ds*)/BR(J/Psi Ds) = 2.37 https://journals.aps.org/prd/abstract/10.1103/PhysRevD.87.112012, https://inspirehep.net/literature/1385102'            ),
            Decay(0.02026 , [particles["MyJ/psi"  ], particles["anti-D0" ], particles["K-"          ]                   ], "PHOTOS PHSP"                              , 'https://journals.aps.org/prd/pdf/10.1103/PhysRevD.95.032005'                                                                                                   ),
            Decay(0.10333 , [particles["MyJ/psi"  ], particles["anti-D*0"], particles["K-"          ]                   ], "PHOTOS PHSP"                              , 'https://journals.aps.org/prd/pdf/10.1103/PhysRevD.95.032005'                                                                                                   ),
            Decay(0.042546, [particles["MyJ/psi"  ], particles["D*-"     ], particles["anti-K*0"    ]                   ], "PHOTOS PHSP"                              , 'https://journals.aps.org/prd/pdf/10.1103/PhysRevD.95.032005'                                                                                                   ),
            Decay(0.012764, [particles["MyJ/psi"  ], particles["D-"      ], particles["anti-K*0"    ]                   ], "PHOTOS PHSP"                              , 'https://journals.aps.org/prd/pdf/10.1103/PhysRevD.95.032005'                                                                                                   ),
            Decay(0.02084 , [particles["MyJ/psi"  ], particles["D- "     ]                                              ], "PHOTOS SVS"                               , 'https://gitlab.cern.ch/lhcb-datapkg/Gen/DecFiles/-/blob/master/dkfiles/Bc_JpsiCharm2Body,mm,muX=JpsiLeptonInAcceptance.dec#L120-121'                           ),
            Decay(0.01581 , [particles["MyJ/psi"  ], particles["D*-"     ]                                              ], "PHOTOS SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0", 'https://gitlab.cern.ch/lhcb-datapkg/Gen/DecFiles/-/blob/master/dkfiles/Bc_JpsiCharm2Body,mm,muX=JpsiLeptonInAcceptance.dec#L120-121'                           ),
            Decay(0.00671 , [particles["MyJ/psi"  ], particles["p+"      ], particles["anti-p-"     ], particles["pi-" ]], "PHOTOS PHSP"                              , 'https://inspirehep.net/literature/1309880'                                                                                                                     ),
            Decay(0.02486 , [particles["MyJ/psi"  ], particles["K+"      ], particles["K-"          ], particles["pi-" ]], "PHOTOS PHSP"                              , 'https://inspirehep.net/literature/1252556'                                                                                                                     ),
# Decay B_c+sig
#   0.1000  MyJ/psi    MyD_s0*+                           PHOTOS SVS
#   0.8153  MyJ/psi    MyD_s1+                            PHOTOS SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0
#   0.0651  MyJ/psi    MyD'_s1+                           PHOTOS SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0
#   0.0197  MyJ/psi    MyD_s2*+                           PHOTOS PHSP
# 
#             Decay(0.00671 , [particles["MyJ/psi"  ], particles["D_s0*+"   ]                                             ], "PHOTOS SVS"                               , 'https://gitlab.cern.ch/lhcb-datapkg/Gen/DecFiles/-/blob/master/dkfiles/Bc_JpsiCharmQuasi2Body,mm,muX=JpsiLeptonInAcceptance.dec#L170-174'                      ),
#             Decay(0.00671 , [particles["MyJ/psi"  ], particles["D_s1+"    ]                                             ], "PHOTOS SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0", 'https://gitlab.cern.ch/lhcb-datapkg/Gen/DecFiles/-/blob/master/dkfiles/Bc_JpsiCharmQuasi2Body,mm,muX=JpsiLeptonInAcceptance.dec#L170-174'                      ),
#             Decay(0.00671 , [particles["MyJ/psi"  ], particles["D'_s1+"   ]                                             ], "PHOTOS SVV_HELAMP 1.0 0.0 1.0 0.0 1.0 0.0", 'https://gitlab.cern.ch/lhcb-datapkg/Gen/DecFiles/-/blob/master/dkfiles/Bc_JpsiCharmQuasi2Body,mm,muX=JpsiLeptonInAcceptance.dec#L170-174'                      ),
#             Decay(0.00671 , [particles["MyJ/psi"  ], particles["D_s2*+"   ]                                             ], "PHOTOS PHSP"                              , 'https://gitlab.cern.ch/lhcb-datapkg/Gen/DecFiles/-/blob/master/dkfiles/Bc_JpsiCharmQuasi2Body,mm,muX=JpsiLeptonInAcceptance.dec#L170-174'                      ),
        ],
        charge_conjugate = 'MyBc+'
    )
    
    # now let's create the .dec file
    particles_to_print_in_dec_file = []
    
    particles_to_print_in_dec_file.append(particles['MyJ/psi'    ])
    particles_to_print_in_dec_file.append(particles['Mychi_c0'   ])
    particles_to_print_in_dec_file.append(particles['Mychi_c1'   ])
    particles_to_print_in_dec_file.append(particles['Mychi_c2'   ])
    particles_to_print_in_dec_file.append(particles['Mypsi(2S)'  ])
    particles_to_print_in_dec_file.append(particles['Myh_c'      ])
    particles_to_print_in_dec_file.append(particles['MyBc-'      ])

    for iparticle in particles_to_print_in_dec_file:
        iparticle.factor_in_forced_decays()
        
    # WARNING! the normalisation MUST happen only as last step!
    for iparticle in particles_to_print_in_dec_file:
        iparticle.normalise_total_br()
    
    with open('BcToJpsiMuMuInclusive.dec', 'w') as ff:
        # Preamble and Aliases
                
        # Charmonium states
        print('Alias      MyJ/psi          J/psi'          , file=ff)
        print('Alias      Mypsi(2S)        psi(2S)'        , file=ff)
        print('Alias      Mychi_c0         chi_c0'         , file=ff)
        print('Alias      Mychi_c1         chi_c1'         , file=ff)
        print('Alias      Mychi_c2         chi_c2'         , file=ff)
        print('Alias      Myh_c            h_c'            , file=ff)
                
        # B mesons        
        print('Alias      MyBc+            B_c+'           , file=ff)
        print('Alias      MyBc-            B_c-'           , file=ff)

        print('ChargeConj MyBc-            MyBc+'          , file=ff)

        for iparticle in particles_to_print_in_dec_file:
            print('\n', file=ff)  
            print(iparticle, file=ff)    

        print('\nEnd\n', file=ff)

