'''
2D simplistic example.
x, y     correspond to the parameter space g, f, F1, F2
e0, e1   correspond to the eigenvector space a0...d2
'''

import numpy as np

from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

names_param = [
    'param0',
    'param1',
]

names_eigen = [
    'e0',
    'e1',
]

cval = np.array([
    2.,
    1.,
])

uncs = np.array([
    0.2,
    0.5,
])

corr = np.array([
    [ 1.  , 0.5 ],
    [ 0.5 , 1.  ],
#     [ 1.  , 0.  ],
#     [ 0.  , 1.  ],
]).astype(np.float64)

# here we need elementwise multiplication, not matrix multiplication
# need to use atleast_2d to really traspose the vector
cov = np.atleast_2d(uncs).T * corr * uncs
print('covariance matrix')
print(cov)

# cov = np.array([[ 6.83220488e-07, -1.03820760e-05],
#                 [-1.03820760e-05,  4.82456839e-04]])
# 
# cval = np.array([ 0.00469806, -0.02218348])
# uncs = np.array([0.00082657, 0.0219649 ])

# Get Eigenvectors and eigenvalues of the covariance matrix
eVals, eVecs = np.linalg.eig(cov)    

print('eigenvectors V (components wrt x,y)')
for i, iev in enumerate(eVecs):
    print(i, iev)

# eigenvalues matrix
diag = np.identity(len(cval)) * eVals
print('eigenvalues matrix W')
print(diag)

# rotate back to original basis
cov_new = eVecs.dot(diag.dot(eVecs.T))
print('closure test, recreate the cov matrix by cov = V W V_T')
print(cov_new)

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
for iname in names_eigen:
    variations[iname] = dict()
    variations[iname]['up'  ] = dict()
    variations[iname]['down'] = dict()
    # for each of these, specify ho much distance needs to be travelled in
    # the x, y basis
    for jname in names_param:
        variations[iname]['up'  ]['delta_'+jname] = 0.
        variations[iname]['down']['delta_'+jname] = 0.


# now fill this dictionary
for i in range(eUncs.shape[0]):
    for j in range(eVecs.shape[0]):
        variations[names_eigen[j]]['up'  ]['delta_'+names_param[i]] =  principal_comp[j, i]
        variations[names_eigen[j]]['down']['delta_'+names_param[i]] = -principal_comp[j, i]
    print('\n')


print ('='*80+'\n\n')
print(variations)

plt.arrow(cval[0], cval[1], principal_comp[0][0], principal_comp[0][1], length_includes_head=True, width=1e-6, head_width=0., head_length=0., fc='k', ec='k', label='e0')
plt.arrow(cval[0], cval[1], principal_comp[1][0], principal_comp[1][1], length_includes_head=True, width=1e-6, head_width=0., head_length=0., fc='k', ec='k', label='e1')

# max_unc = min(uncs)
max_unc = max(uncs)
xmin = -2.*max_unc + cval[0]
xmax =  2.*max_unc + cval[0]
ymin = -2.*max_unc + cval[1]
ymax =  2.*max_unc + cval[1]
# xmin = -2.*uncs[0] + cval[0]
# xmax =  2.*uncs[0] + cval[0]
# ymin = -2.*uncs[1] + cval[1]
# ymax =  2.*uncs[1] + cval[1]

x = np.linspace(xmin, xmax, 500)
y = np.linspace(ymin, ymax, 500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; 
pos[:, :, 1] = Y
rv = multivariate_normal(mean=cval, cov=cov)

# 2D plot
plt.clf()
fig = plt.figure(figsize=(5,5))
aspect = abs(xmax-xmin)/abs(ymax-ymin)
# aspect = 'equal'
subplt = fig.add_subplot(111, aspect=aspect, box_aspect=1.)

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

plt.xlabel(names_param[0])
plt.ylabel(names_param[1])

for ix, iy in (principal_comp+np.atleast_2d(origin).T):
    plt.text(ix, iy, '(%.3f, %.3f)'%(ix, iy), c='silver')

subplt.scatter(*cval)
plt.text(*cval, '({:.3f}, {:.3f})'.format(*cval), c='silver')

subplt.quiver(*origin, principal_comp[:,0], principal_comp[:,1], units='xy', color=['r','b'], angles='xy', scale_units='xy', scale=1.)
subplt.quiver(*origin, principal_comp[:,0], principal_comp[:,1], units='xy', color=['r','b'], angles='xy', scale_units='xy', scale=1.)

plot_margin = 0.2
plt.subplots_adjust(left=0+plot_margin, bottom=0+plot_margin)#, right=1, top=1, wspace=0, hspace=0)

plt.savefig('contour_pre.pdf')

# principal components + multivariate gaus mean
print('principal components + multivariate gaus mean')
print(principal_comp+np.atleast_2d(origin).T)

# save values of up/down variations
with open('dummy_variations.py', 'w') as fout:
    print('variations =', variations, file=fout)

