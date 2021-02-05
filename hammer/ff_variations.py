'''
All numbers from https://doi.org/10.1103/PhysRevD.100.094503
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from itertools import combinations

# don't use numpy matrices
# https://numpy.org/doc/stable/reference/generated/numpy.matrix.html

names_bgl_coefficients = [
    'g_a0' ,
    'g_a1' ,
    'g_a2' ,
    'f_a0' ,
    'f_a1' ,
    'f_a2' ,
#     'F1_a0',
    'F1_a1',
    'F1_a2',
    'F2_a0',
    'F2_a1',
    'F2_a2',
]

names_bgl_coefficients_hammer = [
    'a0',
    'a1',
    'a2',
    'b0',
    'b1',
    'b2',
#     'c0',  <== missing in Hammer
    'c1',
    'c2',
    'd0',
    'd1',
    'd2',
]

names_eigenvectors = ['e%d' %i for i in range(len(names_bgl_coefficients_hammer))]

bgl_coefficients = np.array([
     0.004698059596935804 , # g  a0 --> Hammer a0
    -0.022183478236662734 , # g  a1 --> Hammer a1
     0.1502562517362494   , # g  a2 --> Hammer a2
     0.0034238384795902575, # f  a0 --> Hammer b0
    -0.025953732571963286 , # f  a1 --> Hammer b1
     0.38967327083088565  , # f  a2 --> Hammer b2
#      0.000577003829194323 , # F1 a0 --> Hammer c0 <=== MISSING FROM HAMMER
    -0.0031639433493629403, # F1 a1 --> Hammer c1
     0.08731423252244401  , # F1 a2 --> Hammer c2
     0.04011232167856242  , # F2 a0 --> Hammer d0
    -0.21320857365985693  , # F2 a1 --> Hammer d1
     0.00815737883762693  , # F2 a2 --> Hammer d2
]).astype(np.float64)

bgl_coefficient_uncertainties = np.array([
    0.000826571525943808    , # g  a0 --> Hammer a0
    0.02196490016291772     , # g  a1 --> Hammer a1
    0.545977490995423       , # g  a2 --> Hammer a2
    0.0004860943851721208   , # f  a0 --> Hammer b0
    0.019917544712186736    , # f  a1 --> Hammer b1
    0.5358672468274461      , # f  a2 --> Hammer b2
#     0.00009603720028421325  , # F1 a0 --> Hammer c0 <=== MISSING FROM HAMMER
    0.002803896045528811    , # F1 a1 --> Hammer c1
    0.07179225382668045     , # F1 a2 --> Hammer c2
    0.009259405433788202    , # F2 a0 --> Hammer d0
    0.15391959700192073     , # F2 a1 --> Hammer d1
    0.008005154443761296    , # F2 a2 --> Hammer d2
]).astype(np.float64)

# correlation matrix (normalised covariance matrix)
# c0 column/row is removed
corr = np.array([
##### BGL nomenclature
#     g  a0     g  a1     g  a2    f  a0      f  a1     f  a2     F1 a1     F1 a2     F2 a0     F2 a1     F2 a2
##### Hammer nomenclature                                       
#     a0        a1        a2       b0         b1        b2        c1        c2        d0        d1        d2
    [ 1.     , -0.57184, -0.59452, 0.02863 ,  0.01685, -0.05287, -0.05387, -0.17094, -0.1075 , -0.01861,  0.17341], # g  a0 --> Hammer a0
    [-0.57184,  1.     ,  0.05372, 0.10823 ,  0.47257, -0.51642,  0.54132, -0.26322, -0.16775,  0.58525, -0.42259], # g  a1 --> Hammer a1
    [-0.59452,  0.05372,  1.     , -0.12352, -0.64803,  0.54139, -0.51489,  0.61174,  0.42299, -0.64167,  0.23507], # g  a2 --> Hammer a2
    [ 0.02863,  0.10823, -0.12352,  1.     , -0.13439, -0.58584,  0.34573, -0.04210,  0.54837,  0.31989, -0.07857], # f  a0 --> Hammer b0
    [ 0.01685,  0.47257, -0.64803, -0.13439,  1.     , -0.40318,  0.6154 , -0.58483, -0.6239 ,  0.7606 , -0.45403], # f  a1 --> Hammer b1
    [-0.05287, -0.51642,  0.54139, -0.58584, -0.40318,  1.     , -0.70115,  0.50202,  0.06024, -0.78539,  0.34941], # f  a2 --> Hammer b2
    [-0.05387,  0.54132, -0.51489,  0.34573,  0.6154 , -0.70115,  1.     , -0.17813,  0.1071 ,  0.94788, -0.52455], # F1 a1 --> Hammer c1
    [-0.17094, -0.26322,  0.61174, -0.04210, -0.58483,  0.50202, -0.17813,  1.     ,  0.76217, -0.41757,  0.17784], # F1 a2 --> Hammer c2
    [-0.1075 , -0.16775,  0.42299,  0.54837, -0.6239 ,  0.06024,  0.1071 ,  0.76217,  1.     , -0.1488 ,  0.09672], # F2 a0 --> Hammer d0
    [-0.01861,  0.58525, -0.64167,  0.31989,  0.7606 , -0.78539,  0.94788, -0.41757, -0.1488 ,  1.     , -0.53803], # F2 a1 --> Hammer d1
    [ 0.17341, -0.42259,  0.23507, -0.07857, -0.45403,  0.34941, -0.52455,  0.17784,  0.09672, -0.53803,  1.     ], # F2 a2 --> Hammer d2
]).astype(np.float64)

# here we need elementwise multiplication, not matrix multiplication
# need to use atleast_2d to really traspose the vector
cov = np.atleast_2d(bgl_coefficient_uncertainties).T * corr * bgl_coefficient_uncertainties

print('covariance matrix')
print(cov)

# Get Eigenvectors and eigenvalues of the covariance matrix
eVals, eVecs = np.linalg.eig(cov)    

print('eigenvectors V (components wrt x,y)')
for i, iev in enumerate(eVecs):
    print(i, iev)

# eigenvalues matrix
diag = np.identity(len(bgl_coefficients)) * eVals
print('eigenvalues matrix W')
print(diag)

# rotate back to original basis
cov_new = eVecs.dot(diag.dot(eVecs.T))
rtol = 1e-5
print('closure test: rebuild the covariance matrix from the eigenvectors, eigenvalues')
print('\tpassed at {} level?'.format(rtol), np.isclose(cov, cov_new, rtol=1e-5).all())

# sigmas should be squared root of the eigenvalues
eUncs = np.nan_to_num(np.sqrt(eVals))

# principal components
print('principal components')
principal_comp = np.atleast_2d(eUncs*eVecs).T
print(principal_comp)

print('\n\n')

variations = dict()

# create a structure, i.e. a dictionary that allows:
# for each movement along the direction of the j-th eigenvector
# define two possible ways, up and down
for iname in names_eigenvectors:
    variations[iname] = dict()
    variations[iname]['up'  ] = dict()
    variations[iname]['down'] = dict()
    # for each of these, specify ho much distance needs to be travelled in
    # the x, y basis
    for jname in names_bgl_coefficients_hammer:
        variations[iname]['up'  ]['delta_'+jname] = 0.
        variations[iname]['down']['delta_'+jname] = 0.

# now fill this dictionary
for i in range(eUncs.shape[0]):
    for j in range(eVecs.shape[0]):
        variations[names_eigenvectors[j]]['up'  ]['delta_'+names_bgl_coefficients_hammer[i]] =  principal_comp[j, i]
        variations[names_eigenvectors[j]]['down']['delta_'+names_bgl_coefficients_hammer[i]] = -principal_comp[j, i]

# save values of up/down variations
with open('bgl_variations.py', 'w') as fout:
    print('variations =', variations, file=fout)

print ('='*80+'\n\n')
print(variations)
print ('='*80+'\n\n')


# Plot 2D correlations

for i,j in combinations(range(11),2): 
    print('studying %s vs %s' %(names_bgl_coefficients_hammer[i], names_bgl_coefficients_hammer[j]))
    
    cval = np.array([
        bgl_coefficients[i],
        bgl_coefficients[j],
    ])
    
    uncs = np.array([
        bgl_coefficient_uncertainties[i],
        bgl_coefficient_uncertainties[j],
    ])
    
    # extract submatrix from cov matrix
    minicov = cov[np.ix_([i,j],[i,j])]
    
    # extract the relevant principal components
    mini_principal_comp = principal_comp.T[np.ix_([i,j],[i,j])]
    
#     xmin = -2.*uncs[0] + cval[0]
#     xmax =  2.*uncs[0] + cval[0]
#     ymin = -2.*uncs[1] + cval[1]
#     ymax =  2.*uncs[1] + cval[1]

    # need to keep the same scaling for x- and y-axis
    # because arrows, quiver... don't scale.
#     max_unc = max(uncs)
#     xmin = -2.*max_unc + cval[0]
#     xmax =  2.*max_unc + cval[0]
#     ymin = -2.*max_unc + cval[1]
#     ymax =  2.*max_unc + cval[1]

    min_unc = min(uncs)
    xmin = -6.*min_unc + cval[0]
    xmax =  6.*min_unc + cval[0]
    ymin = -6.*min_unc + cval[1]
    ymax =  6.*min_unc + cval[1]

    x = np.linspace(xmin, xmax, 500)
    y = np.linspace(ymin, ymax, 500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; 
    pos[:, :, 1] = Y
    rv = multivariate_normal(mean=cval, cov=minicov)
    
    # 2D plot
    plt.clf()
    fig = plt.figure(figsize=(5,5))
    aspect = abs(xmax-xmin)/abs(ymax-ymin)
    subplt = fig.add_subplot(111, box_aspect=1., aspect=aspect)

    Z = rv.pdf(pos)

    plt.imshow(Z, origin='lower', extent=[xmin, xmax, ymin, ymax])

    levels = [
        np.power(np.e, -9. ) * abs(np.max(Z)-np.min(Z))+np.min(Z),  # 3 sigma 
        np.power(np.e, -2. ) * abs(np.max(Z)-np.min(Z))+np.min(Z),  # 2 sigma 
        np.power(np.e, -0.5) * abs(np.max(Z)-np.min(Z))+np.min(Z),  # 1 sigma    
        abs(np.max(Z))                                           ,  # max
    ] 
    levels_str = [r'3 $\sigma$', r'2 $\sigma$', r'1 $\sigma$', r'0 $\sigma$']
    contours = subplt.contour(X, Y, Z, levels=levels, colors='silver')
    fmt = {}
    for l, s in zip(contours.levels, levels_str):
        fmt[l] = s
    subplt.clabel(contours, contours.levels[:-1], inline=True, fmt=fmt)
    
    origin = np.array([np.ones(2)*cval[0], np.ones(2)*cval[1]])
    subplt.quiver(*origin, mini_principal_comp[:,0], mini_principal_comp[:,1], units='xy', color=['r','b'], angles='xy', scale_units='xy', scale=1.)

    plt.xlabel(names_bgl_coefficients_hammer[i])
    plt.ylabel(names_bgl_coefficients_hammer[j])

#     for ix, iy in mini_principal_comp+origin:
#         plt.text(ix, iy, '(%.2f, %.2f)'%(ix, iy))

    subplt.scatter(*cval)
    plt.text(*cval, '({:.1e}, {:.1e})'.format(*cval), c='silver')
    
    plot_margin = 0.2
    plt.subplots_adjust(left=0+plot_margin, bottom=0+plot_margin)#, right=1, top=1, wspace=0, hspace=0)
    plt.savefig('contour_%s_vs_%s.pdf' %(names_bgl_coefficients_hammer[i], names_bgl_coefficients_hammer[j]))
    
    break

# closure test
# rebuild the covariance matrix from the principal components
cov_from_pc = principal_comp.T.dot(np.identity(len(principal_comp)).dot(principal_comp))
rtol = 1e-5
print('closure test: rebuild the covariance matrix from the principal components')
print('\tpassed at {} level?'.format(rtol), np.isclose(cov, cov_from_pc, rtol=1e-5).all())

# principal components + multivariate gaus mean
print('principal components + multivariate gaus mean')
print(principal_comp+bgl_coefficients)















# corr = np.array([
# ##### BGL nomenclature
# #     g  a0     g  a1     g  a2    f  a0      f  a1     f  a2     F1 a0     F1 a1     F1 a2     F2 a0     F2 a1     F2 a2
# ##### Hammer nomenclature                                         MISSING!
# #     a0        a1        a2       b0         b1        b2        c0        c1        c2        d0        d1        d2
#     [ 1.     , -0.57184, -0.59452, 0.02863 ,  0.01685, -0.05287,  0.0119 , -0.05387, -0.17094, -0.1075 , -0.01861,  0.17341], # g  a0 --> Hammer a0
#     [-0.57184,  1.     ,  0.05372, 0.10823 ,  0.47257, -0.51642, -0.01997,  0.54132, -0.26322, -0.16775,  0.58525, -0.42259], # g  a1 --> Hammer a1
#     [-0.59452,  0.05372,  1.     , -0.12352, -0.64803,  0.54139,  0.04933, -0.51489,  0.61174,  0.42299, -0.64167,  0.23507], # g  a2 --> Hammer a2
#     [ 0.02863,  0.10823, -0.12352,  1.     , -0.13439, -0.58584,  0.92851,  0.34573, -0.04210,  0.54837,  0.31989, -0.07857], # f  a0 --> Hammer b0
#     [ 0.01685,  0.47257, -0.64803, -0.13439,  1.     , -0.40318, -0.40009,  0.6154 , -0.58483, -0.6239 ,  0.7606 , -0.45403], # f  a1 --> Hammer b1
#     [-0.05287, -0.51642,  0.54139, -0.58584, -0.40318,  1.     , -0.41505, -0.70115,  0.50202,  0.06024, -0.78539,  0.34941], # f  a2 --> Hammer b2
#     [ 0.0119 , -0.01997,  0.04933,  0.92851, -0.40009, -0.41505,  1.     ,  0.2554 ,  0.1437 ,  0.73757,  0.14654,  0.0085 ], # F1 a0 --> Hammer c0 <=== MISSING FROM HAMMER
#     [-0.05387,  0.54132, -0.51489,  0.34573,  0.6154 , -0.70115,  0.2554 ,  1.     , -0.17813,  0.1071 ,  0.94788, -0.52455], # F1 a1 --> Hammer c1
#     [-0.17094, -0.26322,  0.61174, -0.04210, -0.58483,  0.50202,  0.1437 , -0.17813,  1.     ,  0.76217, -0.41757,  0.17784], # F1 a2 --> Hammer c2
#     [-0.1075 , -0.16775,  0.42299,  0.54837, -0.6239 ,  0.06024,  0.73757,  0.1071 ,  0.76217,  1.     , -0.1488 ,  0.09672], # F2 a0 --> Hammer d0
#     [-0.01861,  0.58525, -0.64167,  0.31989,  0.7606 , -0.78539,  0.14654,  0.94788, -0.41757, -0.1488 ,  1.     , -0.53803], # F2 a1 --> Hammer d1
#     [ 0.17341, -0.42259,  0.23507, -0.07857, -0.45403,  0.34941,  0.0085 , -0.52455,  0.17784,  0.09672, -0.53803,  1.     ], # F2 a2 --> Hammer d2
# ]).astype(np.float64)
