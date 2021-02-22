import re
from particle import Particle
from collections import OrderedDict
# https://stackoverflow.com/questions/1546226/is-there-a-simple-way-to-remove-multiple-spaces-in-a-string

new_lines_latex = []
new_lines_pdg = []

decays = OrderedDict()

aliases = OrderedDict()

with open('dec_file.dec', 'r') as dec_file:
    idx = 0
    for iline in dec_file.readlines():
        iline = iline.rstrip()
        iline = re.sub(' +', ' ', iline)
        bits = iline.split(' ')
        
        if len(bits)<2:
            continue
        
        if bits[0]=='ChargeConj':
            continue
        
        if bits[0]=='Alias':
            aliases[bits[1]] = bits[2]
            continue

        if bits[0]=='Decay':
            pname = aliases[bits[1]] if bits[1] in aliases.keys() else bits[1]
            particle = Particle.from_evtgen_name(pname)
            particle_pdgid = int(particle.pdgid)
            decays[abs(particle_pdgid)] = OrderedDict()
            continue

        if bits[0]=='CDecay':
            continue
                        
        new_bits_latex = []
        new_bits_pdg = []
        for ibit in bits:
            if ibit in aliases.keys():
                ibit = aliases[ibit]
            try:
                new_bit_latex = Particle.from_evtgen_name(ibit).latex_name.replace('\\', '#')
                new_bit_pdg = str(int(Particle.from_evtgen_name(ibit).pdgid))
                new_bits_latex.append(new_bit_latex)
                new_bits_pdg.append(new_bit_pdg)
            except:
                pass
        
        mykey = list(map(abs, map(int, new_bits_pdg)))   
        mykey.sort(key = lambda x: abs(x), reverse = True)     
        decays[abs(particle_pdgid)][tuple(mykey)] = (idx, new_bits_latex)
        idx += 1
        
        new_lines_latex.append(' '.join(new_bits_latex))
        new_lines_pdg.append(' '.join(new_bits_pdg))

with open('pdg_file.txt', 'w') as pdg_file:
    for ii in new_lines_pdg: 
        print(ii, file=pdg_file)

with open('latex_file.txt', 'w') as latex_file:
    for ii in new_lines_latex: 
        print(ii, file=latex_file)

with open('decays_dict.py', 'w') as decays_file:
    print('from collections import OrderedDict', file=decays_file)
    print('decays = ', decays, file=decays_file)
    
