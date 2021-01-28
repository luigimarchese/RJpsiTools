'''
2D simplistic example.
x, y     correspond to the parameter space g, f, F1, F2
e0, e1   correspond to the eigenvector space a0...d2
'''

import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

names_param = [
    'param0',
    'param1',
]

names_eigen = [
    'e0',
    'e1',
]

cval = np.array([
    0.,
    0.,
])

uncs = np.array([
    0.2,
    0.5,
])

corr = np.array([
    [ 1.  , 0.5 ],
    [ 0.5 , 1.  ],
]).astype(np.float64)

# here we need elementwise multiplication, not matrix multiplication
# need to use atleast_2d to really traspose the vector
cov = np.atleast_2d(uncs).T * corr * uncs
print('covariance matrix')
print(cov)

# Get Eigenvectors and eigenvalues of the covariance matrix
eVals, eVecs = np.linalg.eig(cov)    

print('eigenvectors V (components wrt x,y)')
for i, iev in enumerate(eVecs):
    print(i, iev)

# eigenvalues matrix
diag = np.identity(2) * eVals
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
#     print('eUncs: {:.3f}'.format(eUncs[i]))
#     print('eVecs: ' + (len(eVecs)*' {:.3f}').format(*eVecs[:, i]))    
#     print('move in the direction of each eigenvector by an amount corresponding to +/- sqrt of the corresponding eigenvalue')
#     a = str(np.column_stack((eUncs[i]*eVecs[:, i], -eUncs[i]*eVecs[:, i])))
#     print(a.replace('\n', ',').replace('[', '{').replace(']', '}'))
    for j in range(eVecs.shape[0]):
        variations[names_eigen[j]]['up'  ]['delta_'+names_param[i]] =  principal_comp[j, i]
        variations[names_eigen[j]]['down']['delta_'+names_param[i]] = -principal_comp[j, i]
    print('\n')


print ('='*80+'\n\n')
print(variations)

xmin = -max(uncs) + cval[0]
xmax =  max(uncs) + cval[0]
ymin = xmin
ymax = xmax

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
# ax0 = fig.add_subplot(111)
# plt.contour(rv.pdf(pos).reshape(500,500))
# plt.contour(rv.pdf(pos))
plt.contourf(X, Y, rv.pdf(pos))
origin = np.array([np.ones(2)*cval[0], np.ones(2)*cval[1]])
plt.quiver(*origin, principal_comp[:,0], principal_comp[:,1], color=['r','b'], scale=abs(xmin-xmax))

plt.xlabel(names_param[0])
plt.ylabel(names_param[1])

for ix, iy in principal_comp+origin:
    plt.text(ix, iy, '(%.2f, %.2f)'%(ix, iy))

plt.scatter(*cval)
plt.savefig('contour.pdf')

# principal components + multivariate gaus mean
print('principal components + multivariate gaus mean')
print(principal_comp+origin)


# save values of up/down variations
with open('dummy_variations.py', 'w') as fout:
    print('variations =', variations, file=fout)

