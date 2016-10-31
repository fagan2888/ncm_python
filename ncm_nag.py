
# coding: utf-8

# # Nearest Correlation Matrix with NAG

# Import required modules and set print options
import numpy as np
import nag4py.g02 as nag_g02
import nag4py.util as nag_util
import matplotlib.pyplot as plt
"Set the print precision."
np.set_printoptions(precision=4)


# ## Start with an incomplete matrix P

"Define a 2-d array and us np.nan to set the NaNs."
P = np.array([
  [59.875, 42.734, 47.938, 60.359, 54.016, 69.625, 61.500, 62.125],
  [53.188, 49.000, 39.500, np.nan, 34.750, np.nan, 83.000, 44.500],
  [55.750, 50.000, 38.938, np.nan, 30.188, np.nan, 70.875, 29.938],
  [65.500, 51.063, 45.563, 69.313, 48.250, 62.375, 85.250, np.nan],
  [69.938, 47.000, 52.313, 71.016, np.nan, 59.359, 61.188, 48.219],
  [61.500, 44.188, 53.438, 57.000, 35.313, 55.813, 51.500, 62.188],
  [59.230, 48.210, 62.190, 61.390, 54.310, 70.170, 61.750, 91.080],
  [61.230, 48.700, 60.300, 68.580, 61.250, 70.340, np.nan, np.nan],
  [52.900, 52.690, 54.230, np.nan, 68.170, 70.600, 57.870, 88.640],
  [57.370, 59.040, 59.870, 62.090, 61.620, 66.470, 65.370, 85.840]])


def cov_bar(P):
    """Returns an approximate sample covarience matrix"""
    "P.shape returns a tuple (m, n) that we unpack to m and n."
    m, n = P.shape
    "Initialise an nxn zero matrix."
    S = np.zeros((n, n))
    "i = 0, ..., 7."
    for i in range(0, n):
        "Take the ith column."
        xi = P[:, i]
        "j = 0, ..., i"
        for j in range(0, i+1):
            "Take the jth column, where j <= i."
            xj = P[:, j]
            """masked_array masks elements marked as True; this is the
            opposite behaviour of the MATLAB script. Therefore, we need
            ~(~np.isnan(xi) & ~np.isnan(xj)), which I have
            simplified below and will label notp."""
            "Set mask such that all nans are True."
            notp = np.isnan(xi) | np.isnan(xj)
            "Apply the mask to xi"
            xim = np.ma.masked_array(xi, mask=notp)
            "Apply the mask to xj"
            xjm = np.ma.masked_array(xj, mask=notp)
            S[i, j] = np.ma.dot(xim - np.mean(xim), xjm - np.mean(xjm))
            """As we used notp as our mask, we must take the sum over ~notp
            to give the same result as the MATLAB script when normalising."""
            S[i, j] = 1.0 / (sum(~notp) - 1) * S[i, j]
            S[j, i] = S[i, j]
    return S


# In[4]:

def cor_bar(P):
    """Returns an approximate sample correlation matrix"""
    S = cov_bar(P)
    D = np.diag(1.0 / np.sqrt(np.diag(S)))
    "This is will only work in Python3"
    return D @ S @ D


# ## Calculate the correlation matrix of P.

# In[5]:

G = cor_bar(P)
print(G)


"""Some of the eigenvalues are negative and therefore the matrix is
not semi-positive definite."""

# In[6]:

print("The sorted eigenvalues of G {}".format(np.sort(np.linalg.eig(G)[0])))


"""Use g02aa to calculate the nearest correlation matrix in the
Frobenius norm"""

# In[7]:

order = nag_util.Nag_RowMajor
Gflat = G.flatten()
n = G.shape[0]
pdg = n
errtol = 0.0
maxits = 0
maxit = 0
Xflat = np.empty_like(Gflat)
pdx = n
itr = np.array([0])
feval = np.array([0])
nrmgrd = np.array([0.0])
fail = nag_util.noisy_fail()
nag_g02.g02aac(order, Gflat, pdg, n, errtol, maxits,
               maxit, Xflat, pdx, itr, feval, nrmgrd, fail)


# In[8]:

"Unflatten X to have the same shape as G for comparison."
X = np.reshape(Xflat, G.shape)
print(X)


# In[9]:

print("The sorted eigenvalues of X [{0}]".format(''.join(
            ['{:.4f} '.format(x) for x in np.sort(np.linalg.eig(X)[0])])))


# In[10]:

fig1, ax1 = plt.subplots(figsize=(14, 7))
cax1 = ax1.imshow(abs(X-G),
                  interpolation='none',
                  cmap=plt.cm.Blues,
                  vmin=0, vmax=0.2)
cbar = fig1.colorbar(cax1, ticks=np.linspace(0.0, 0.2, 11, endpoint=True),
                     boundaries=np.linspace(0.0, 0.2, 11, endpoint=True))
cbar.set_clim([0, 0.2])
ax1.tick_params(axis='both', which='both',
                bottom='off', top='off', left='off', right='off',
                labelbottom='off', labelleft='off')
ax1.set_title(
  r"""g02aa: interations: {0}, $||G-X||_F = {1:.5f}$""".format(
    itr[0], np.linalg.norm(X-G)))
plt.show()


"""Use g02ab to calculate the nearest correlation
matrix with row and column weighting"""

# ### Define the weights as follows

# In[11]:

W = np.array([10, 10, 10, 1, 1, 1, 1, 1], dtype=np.float64)


# In[12]:

order = nag_util.Nag_RowMajor
Gflat = G.flatten()
n = G.shape[0]
opt = nag_util.Nag_Both
alpha = 0.001
pdg = n
errtol = 0.0
maxits = 0
maxit = 0
Xflat = np.empty_like(Gflat)
pdx = n
itr = np.array([0])
feval = np.array([0])
nrmgrd = np.array([0.0])
fail = nag_util.noisy_fail()
nag_g02.g02abc(order, Gflat, pdg, n, opt, alpha, W, errtol, maxits,
               maxit, Xflat, pdx, itr, feval, nrmgrd, fail)


# In[13]:

X = np.reshape(Xflat, G.shape)
print(X)


# In[14]:

print("The sorted eigenvalues of X [{0}]".format(''.join(
            ['{:.4f} '.format(x) for x in np.sort(np.linalg.eig(X)[0])])))


# In[15]:

fig1, ax1 = plt.subplots(figsize=(14, 7))
cax1 = ax1.imshow(abs(X-G),
                  interpolation='none',
                  cmap=plt.cm.Blues,
                  vmin=0, vmax=0.2)
cbar = fig1.colorbar(cax1, ticks=np.linspace(0.0, 0.2, 11, endpoint=True),
                     boundaries=np.linspace(0.0, 0.2, 11, endpoint=True))
cbar.set_clim([0, 0.2])
ax1.tick_params(axis='both', which='both',
                bottom='off', top='off', left='off', right='off',
                labelbottom='off', labelleft='off')
ax1.set_title(r"""g02ab: interations: {0}, $||G-X||_F = {1:.4f}$""".format(
  itr[0], np.linalg.norm(X-G)))
plt.show()


"""Use g02aj to calculate the nearest correlation
matrix with element-wise weighting"""

# In[16]:

H = np.ones([n, n])
H[0:3, 0:3] = 100


# In[17]:

Gflat = G.flatten()
n = G.shape[0]
alpha = 0.001
Hflat = H.flatten()
pdg = n
pdh = n
errtol = 0.0
maxits = 0
maxit = 500
Xflat = np.empty_like(Gflat)
pdx = n
itr = np.array([0])
norm = np.array([0.0])
fail = nag_util.noisy_fail()
nag_g02.g02ajc(Gflat, pdg, n, alpha, Hflat, pdh, errtol,
               maxit, Xflat, pdx, itr, norm, fail)


# In[18]:

X = np.reshape(Xflat, G.shape)
print(X)


# In[19]:

print("The sorted eigenvalues of X [{0}]".format(''.join(
            ['{:.4f} '.format(x) for x in np.sort(np.linalg.eig(X)[0])])))


# In[20]:

fig1, ax1 = plt.subplots(figsize=(14, 7))
cax1 = ax1.imshow(abs(X-G),
                  interpolation='none',
                  cmap=plt.cm.Blues,
                  vmin=0, vmax=0.2)
cbar = fig1.colorbar(cax1, ticks=np.linspace(0.0, 0.2, 11, endpoint=True),
                     boundaries=np.linspace(0.0, 0.2, 11, endpoint=True))
cbar.set_clim([0, 0.2])
ax1.tick_params(axis='both', which='both',
                bottom='off', top='off', left='off', right='off',
                labelbottom='off', labelleft='off')
ax1.set_title(r"""g02aj: interations: {0}, $||G-X||_F = {1:.4f}$""".format(
  itr[0], np.linalg.norm(X-G)))
plt.show()


"""Use g02an to calculate the nearest correlation
matrix with fixed leading block"""

# In[21]:

Gflat = G.flatten()
n = G.shape[0]
pdg = n
k = 3
errtol = 0.0
eigtol = 0.0
maxits = 0
maxit = 500
Xflat = np.empty_like(Gflat)
pdx = n
alpha = np.array([0.0])
itr = np.array([0])
eigmin = np.array([0.0])
norm = np.array([0.0])
fail = nag_util.noisy_fail()
nag_g02.g02anc(Gflat, pdg, n, k, errtol, eigtol, Xflat, pdx,
               alpha, itr, eigmin, norm, fail)


# In[22]:

X = np.reshape(Xflat, G.shape)
print(X)


# In[23]:

print("The sorted eigenvalues of X [{0}]".format(''.join(
            ['{:.4f} '.format(x) for x in np.sort(np.linalg.eig(X)[0])])))


# In[24]:

fig1, ax1 = plt.subplots(figsize=(14, 7))
cax1 = ax1.imshow(abs(X-G),
                  interpolation='none',
                  cmap=plt.cm.Blues,
                  vmin=0, vmax=0.2)
cbar = fig1.colorbar(cax1, ticks=np.linspace(0.0, 0.2, 11, endpoint=True),
                     boundaries=np.linspace(0.0, 0.2, 11, endpoint=True))
cbar.set_clim([0, 0.2])
ax1.tick_params(axis='both', which='both',
                bottom='off', top='off', left='off', right='off',
                labelbottom='off', labelleft='off')
ax1.set_title(r"""g02an: interations: {0}, $||G-X||_F = {1:.4f}$""".format(
  itr[0], np.linalg.norm(X-G)))
plt.show()
