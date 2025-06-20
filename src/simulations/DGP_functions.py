import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
import doubleml as dml
from doubleml.datasets import make_ssm_data
from scipy.stats import t

# Default DGP
def generate_dgp_1_MAR(seed, n_obs=2000, theta=1.0, dim_x=100):

    np.random.seed(seed)

    sigma = np.array([[1, 0], [0, 1]])
    gamma = 0

    e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n_obs).T

    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    X = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    beta = [0.4 / (k**2) for k in range(1, dim_x + 1)]

    d = np.where(np.dot(X, beta) + np.random.randn(n_obs) > 0, 1, 0)
    z = np.random.randn(n_obs)
    s = np.where(np.dot(X, beta) + d + gamma * z + e[0] > 0, 1, 0)

    y = np.dot(X, beta) + theta * d + e[1]

    y_real = y.copy()

    y[s == 0] = 0

    return pd.DataFrame(np.column_stack([X, y, y_real, d, s]), columns=[f'X{i+1}' for i in range(dim_x)] + ['y', 'y_real', 'd', 's'])

def generate_dgp_1_MNAR(seed, n_obs=2000, theta=1.0, dim_x=100):
    
    np.random.seed(seed)

    sigma = np.array([[1, 0.8], [0.8, 1]])
    gamma = 1

    e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n_obs).T

    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])
    X = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    beta = [0.4 / (k**2) for k in range(1, dim_x + 1)]

    d = np.where(np.dot(X, beta) + np.random.randn(n_obs) > 0, 1, 0)
    z = np.random.randn(n_obs)
    s = np.where(np.dot(X, beta) + d + gamma * z + e[0] > 0, 1, 0)

    y = np.dot(X, beta) + theta * d + e[1]

    y_real = y.copy()

    y[s == 0] = 0

    return pd.DataFrame(np.column_stack([X, y, y_real, d, s, z]), columns=[f'X{i+1}' for i in range(dim_x)] + ['y', 'y_real' ,'d', 's', 'z'])

# Only added non-linearity in the outcome (y)
def generate_dgp_2_MAR(seed, n_obs=2000, theta=1.0, dim_x=100, mar=True):

    np.random.seed(seed)

    # Define error term covariance and gamma based on MAR condition
    if mar:
        sigma = np.array([[1, 0], [0, 1]])
        gamma = 0
    else:
        sigma = np.array([[1, 0.8], [0.8, 1]])
        gamma = 1

    e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n_obs).T

    # Covariates (X) - Correlated multivariate normal
    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])  # AR(1) structure
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    # Coefficients for linear effects
    beta = np.array([0.4 / (k**2) for k in range(1, dim_x + 1)])

    # Treatment assignment remains linear
    d = np.where(np.dot(x, beta) + np.random.randn(n_obs) > 0, 1, 0)
    
    # Instrument (Z)
    z = np.random.randn(n_obs)
    
    # Selection process remains linear
    s = np.where(np.dot(x, beta) + d + gamma * z + e[0] > 0, 1, 0)

    # Outcome with additional nonlinear transformations
    y = np.dot(x, beta) + theta * d + 0.5 * x[:, 2]**2 + 0.3 * np.abs(x[:, 4]) + 0.2 * x[:, 5]**3 + e[1]

    y_real = y.copy()
    
    # Setting y to 0 for unselected observations
    y[s == 0] = 0

    return pd.DataFrame(np.column_stack([x, y, y_real, d, s]), columns=[f'X{i+1}' for i in range(dim_x)] + ['y','y_real', 'd', 's'])

# Only added non-linearity in the outcome (y)
def generate_dgp_2_MNAR(seed, n_obs=2000, theta=1.0, dim_x=100, mar=False):

    np.random.seed(seed)

    # Define error term covariance and gamma based on MAR condition
    if mar:
        sigma = np.array([[1, 0], [0, 1]])
        gamma = 0
    else:
        sigma = np.array([[1, 0.8], [0.8, 1]])
        gamma = 1

    e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n_obs).T

    # Covariates (X) - Correlated multivariate normal
    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])  # AR(1) structure
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    # Coefficients for linear effects
    beta = np.array([0.4 / (k**2) for k in range(1, dim_x + 1)])

    # Treatment assignment remains linear
    d = np.where(np.dot(x, beta) + np.random.randn(n_obs) > 0, 1, 0)
    
    # Instrument (Z)
    z = np.random.randn(n_obs)
    
    # Selection process remains linear
    s = np.where(np.dot(x, beta) + d + gamma * z + e[0] > 0, 1, 0)

    # Outcome with additional nonlinear transformations
    y = (
    np.dot(x, beta)
    + theta * d
    + 0.5 * x[:, 2]**2
    + 0.3 * np.abs(x[:, 4])
    + 0.2 * x[:, 5]**3
    + e[1]
    )

    y_real = y.copy()

    # Setting y to 0 for unselected observations
    y[s == 0] = 0

    return pd.DataFrame(np.column_stack([x, y, y_real, d, s, z]), columns=[f'X{i+1}' for i in range(dim_x)] + ['y', 'y_real', 'd', 's', 'z'])

# Added non-linearity in Y - X and D - X relationships
def generate_dgp_3_MAR(seed, n_obs=2000, theta=1.0, dim_x=100, mar=True):

    np.random.seed(seed)

    # Define error term covariance and gamma based on MAR condition
    if mar:
        sigma = np.array([[1, 0], [0, 1]])
        gamma = 0
    else:
        sigma = np.array([[1, 0.8], [0.8, 1]])
        gamma = 1

    e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n_obs).T

    # Covariates (X) - Correlated multivariate normal
    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])  # AR(1) structure
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    # Coefficients for linear effects
    beta = np.array([0.4 / (k**2) for k in range(1, dim_x + 1)])

    # Treatment assignment with nonlinear relationship to X     d = np.where(np.dot(x, beta) + 0.4 * x[:, 1]**2 + 0.3 * np.abs(x[:, 3]) + np.random.randn(n_obs) > 0, 1, 0)
    d = np.where(
    np.dot(x, beta)
    + 0.5 * x[:, 1]**2
    + 0.3 * np.abs(x[:, 3])
    + 0.2 * x[:, 5] * x[:, 7]
    + 0.1 * np.log(np.abs(x[:, 6]) + 1)
    + np.random.randn(n_obs) > 0,
    1, 0
    ) 
    # Instrument (Z)
    z = np.random.randn(n_obs)
    
    # Selection process remains linear
    s = np.where(np.dot(x, beta) + d + gamma * z + e[0] > 0, 1, 0)

    # Outcome with additional nonlinear transformations  y = np.dot(x, beta) + theta * d + 0.5 * x[:, 2]**2 + 0.3 * np.abs(x[:, 4]) + 0.2 * x[:, 5]**3 + e[1]
    y = (
    np.dot(x, beta)
    + theta * d
    + 0.5 * x[:, 2]**2
    + 0.3 * np.abs(x[:, 4])
    + 0.2 * x[:, 5]**3
    + e[1]
    )

    y_real = y.copy()

    # Setting y to 0 for unselected observations
    y[s == 0] = 0

    return pd.DataFrame(np.column_stack([x, y, y_real, d, s]), columns=[f'X{i+1}' for i in range(dim_x)] + ['y', 'y_real', 'd', 's'])



# Added non-linearity in Y - X and D - X relationships, log, sine and polynomial terms
def generate_dgp_3_MNAR(seed, n_obs=2000, theta=1.0, dim_x=100):

    np.random.seed(seed)

    sigma = np.array([[1, 0.8], [0.8, 1]])
    gamma = 1

    e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n_obs).T

    # Covariates (X) - Correlated multivariate normal
    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])  # AR(1) structure
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    # Coefficients for linear effects
    beta = np.array([0.4 / (k**2) for k in range(1, dim_x + 1)])

    # Treatment assignment with nonlinear relationship to X  d = np.where(np.dot(x, beta) + 0.4 * x[:, 1]**2 + 0.3 * np.abs(x[:, 3]) + np.random.randn(n_obs) > 0, 1, 0)
    d = np.where(np.dot(x, beta) 
                 + 0.2 * np.sin(x[:, 1]) 
                 + 0.2 * x[:, 2] * x[:, 5] 
                 + np.random.randn(n_obs) > 0, 1, 0)
    
    # Instrument (Z)
    z = np.random.randn(n_obs)
    
    # Selection process remains linear
    s = np.where(np.dot(x, beta) + d + gamma * z + e[0] > 0, 1, 0)

    # Outcome with additional nonlinear transformations    y = np.dot(x, beta) + theta * d + 0.5 * x[:, 2]**2 + 0.3 * np.abs(x[:, 4]) + 0.2 * x[:, 5]**3 + e[1]
    y = (np.dot(x, beta) 
         + theta * d 
         + 0.3 * np.log(np.abs(x[:, 3]) + 1) 
         + 0.3 * np.sin(x[:, 4]) 
         + 0.2 * x[:, 7] * x[:, 9] 
         + e[1])
   
    y_real = y.copy()
    

    # Setting y to 0 for unselected observations
    y[s == 0] = 0

    return pd.DataFrame(np.column_stack([x, y, y_real, d, s, z]), columns=[f'X{i+1}' for i in range(dim_x)] + ['y', 'y_real', 'd', 's', 'z'])


# Added non-linearity in Y - X and D - X relationships, sin, exp, and polynomial terms
def generate_dgp_4_MNAR(seed, n_obs = 2000, theta=1.0, dim_x=100, mar=False):

    np.random.seed(seed)

    # Define error term covariance and gamma based on MAR condition
    if mar:
        sigma = np.array([[1, 0], [0, 1]])
        gamma = 0
    else:
        sigma = np.array([[1, 0.8], [0.8, 1]])
        gamma = 1

    e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n_obs).T

    # Covariates (X) - Correlated multivariate normal
    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])  # AR(1) structure
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    # Coefficients for linear effects
    beta = np.array([0.4 / (k**2) for k in range(1, dim_x + 1)])

    # Treatment assignment with nonlinear relationship to X  d = np.where(np.dot(x, beta) + 0.4 * x[:, 1]**2 + 0.3 * np.abs(x[:, 3]) + np.random.randn(n_obs) > 0, 1, 0)
    d = np.where(
        np.dot(x, beta)
        + 0.2 * np.sin(x[:, 2] * x[:, 3])
        + 0.2 * np.exp(-np.abs(x[:, 7])) * x[:, 5]
        + 0.2 * np.sign(x[:, 6]) * x[:, 1]
        + np.random.randn(n_obs) > 0,
        1, 0
    )

    # Instrument (Z)
    z = np.random.randn(n_obs)
    
    # Selection process remains linear
    s = np.where(np.dot(x, beta) + d + gamma * z + e[0] > 0, 1, 0)

    # Outcome with additional nonlinear transformations    y = np.dot(x, beta) + theta * d + 0.5 * x[:, 2]**2 + 0.3 * np.abs(x[:, 4]) + 0.2 * x[:, 5]**3 + e[1]
    y = (
    np.dot(x, beta)
    + theta * d
    + 0.5 * x[:, 2]**2
    + 0.3 * np.abs(x[:, 4])
    + 0.2 * x[:, 5]**3
    + e[1]
    )

    y_real = y.copy()

    # Setting y to 0 for unselected observations
    y[s == 0] = 0

    return pd.DataFrame(np.column_stack([x, y, y_real, d, s, z]), columns=[f'X{i+1}' for i in range(dim_x)] + ['y', 'y_real', 'd', 's', 'z'])


# only beta, non-linearity in Y - X and D - X relationships, standard normal errors, only specific terms in D and Y
def generate_dgp_5_MNAR(seed, n_obs = 2000, theta=1.0, dim_x=100):

    np.random.seed(seed)

    # Define error term covariance and gamma based on MAR condition
    sigma = np.array([[1, 0.8], [0.8, 1]])
    gamma = 1

    e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n_obs).T

    # Covariates (X) - Correlated multivariate normal
    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])  # AR(1) structure
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    # Coefficients for linear effects
    beta = np.array([0.4 / (k**2) for k in range(1, dim_x + 1)])

    # Treatment assignment with nonlinear relationship to X 
    d = np.where(
        beta[ 1] * x[:,  1]                       # α2  * X2
        + beta[ 4] * x[:,  4]                       # α5  * X5
        + beta[11] * x[:, 11]                       # α12 * X12
        + beta[66] * np.sqrt(np.abs(x[:, 66]))      # α67 * √|X67|
        + beta[88] * (x[:, 88] > 19).astype(float)              # 1[X89 > 19] 
        + beta[94] * (x[:, 94] > 5).astype(float) * (x[:, 94] - 3)  # 1[X95 > 5] * (X95 - 3)
        + np.random.randn(n_obs) > 0,
        1, 0)

    # Instrument (Z)
    z = np.random.randn(n_obs)
    
    # Selection process remains linear
    s = np.where(np.dot(x, beta) + d + gamma * z + e[0] > 0, 1, 0)

    # Outcome with additional nonlinear transformations    y = np.dot(x, beta) + theta * d + 0.5 * x[:, 2]**2 + 0.3 * np.abs(x[:, 4]) + 0.2 * x[:, 5]**3 + e[1]
    y = (
        theta * d
        + beta[1]  * x[:,  1]                                # β2 * X2
        + beta[4]  * x[:,  4]                                # β5 * X5
        + beta[1]  * x[:,  1] * x[:,  4]                     # β2 * X2 * X5
        + beta[11] * x[:, 11]                                # β12 * X12
        + beta[22] * x[:, 22]                                # β23 * X23
        + beta[22] * x[:, 22]**2                             # β23 * X23^2
        + beta[11] * x[:, 11] * x[:, 22]**2                  # β12 * X12 * X23^2
        + beta[39] * x[:, 39]                                # β40 * X40      ← ekleme
        + beta[66] * np.sqrt(np.abs(x[:, 66]))               # β67 * √|X67|
        + beta[76] * x[:, 76]                                # β77 * X77      ← ekleme
        + beta[88] * (x[:, 88] > 19).astype(float)                      # 1[X89 > 19]
        + beta[94] * (x[:, 94] > 5).astype(float) * (x[:, 94] - 3)      # 1[X95 > 5] * (X95 - 3)
        + e[1]
    )
    y_real = y.copy()

    # Setting y to 0 for unselected observations
    y[s == 0] = 0

    return pd.DataFrame(np.column_stack([x, y, y_real, d, s, z]), columns=[f'X{i+1}' for i in range(dim_x)] + ['y', 'y_real', 'd', 's', 'z'])

from scipy.stats import t, norm

# DGP 6, alpha and beta separated, nonlinear effects in both D and Y, and standard normal errors
def generate_dgp_6_MNAR(seed, n_obs = 2000, theta=1.0, dim_x=100):

    np.random.seed(seed)

    # 1) α ve β dizileri
    alpha = np.array([0.3 / (k**2) for k in range(1, dim_x+1)])
    beta  = np.array([0.4 / (k**2) for k in range(1, dim_x+1)])
    gamma = 1
    sigma = np.array([[1, 0.8], [0.8, 1]])

    e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n_obs).T

    # Covariates (X) - Correlated multivariate normal
    cov_mat = toeplitz([np.power(0.5, k) for k in range(dim_x)])  # AR(1) structure
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    # Coefficients for linear effects
    beta = np.array([0.4 / (k**2) for k in range(1, dim_x + 1)])

    d = d = np.where(
        alpha[22] * x[:, 22]                       # α23  * X23
        + alpha[22] * (x[:, 22]**2)                  # α23  * X23^2
        + alpha[19] * np.sqrt(np.abs(x[:, 19]))      # α20  * √|X20|
        + alpha[24] * x[:, 24]                       # α25  * X25
        + alpha[29] * (x[:, 29] > 19).astype(float)  # α30  * 1[X30 > 19]
        + alpha[34] * (x[:, 34] > 5).astype(float) * (x[:, 34] - 3)  
                                                    # α35  * 1[X35>5] * (X35–3)
        + alpha[39] * np.exp(x[:, 39])               # α40  * exp(X40)
        + alpha[44] * x[:, 44]                       # α45  * X45
        + alpha[49] * x[:, 49]                       # α50  * X50
        + alpha[54] * x[:, 54]                       # α55  * X55
        + alpha[59] * x[:, 59]                       # α60  * X60
        + alpha[64] * (x[:, 64] > 1).astype(float)   # α65  * 1[X65 > 1]
        + np.random.randn(n_obs) > 0,
        1, 0)

    # Instrument (Z)
    z = np.random.randn(n_obs)
    
    # Selection process remains linear
    s = np.where(np.dot(x, beta) + d + gamma * z + e[0] > 0, 1, 0)

    # Outcome with additional nonlinear transformations    y = np.dot(x, beta) + theta * d + 0.5 * x[:, 2]**2 + 0.3 * np.abs(x[:, 4]) + 0.2 * x[:, 5]**3 + e[1]
    y = (
        theta * d
        + beta[0] * np.sqrt(np.abs(x[:,  0]))             # β1  * √|X1|
        + beta[ 5] * x[:,  5]                              # β6  * X6
        + beta[10] * (x[:, 10] ** 2)                       # β11 * X11^2
        + beta[15] * (x[:, 15] > 2).astype(float)           # β16 * 1[X16 > 2]
        + beta[20] * x[:, 20] * x[:, 25]                   # β21 * X21 * X26
        + beta[30] * np.log(np.abs(x[:, 30]) + 1)          # β31 * log(|X31|+1)
        + beta[35] * np.sin(x[:, 35])                      # β36 * sin(X36)
        + beta[40] * x[:, 40]                              # β41 * X41
        + beta[45] * (x[:, 45] > -1).astype(float) * x[:, 50]  
                                                        # β46 * 1[X46 > -1] * X51
        + beta[55] * np.sqrt(np.abs(x[:, 55]))             # β56 * √|X56|
        + beta[60] * x[:, 60]                              # β61 * X61
        + e[1] 
    )

    y_real = y.copy()

    # Setting y to 0 for unselected observations
    y[s == 0] = 0

    return pd.DataFrame(np.column_stack([x, y, y_real, d, s, z]), columns=[f'X{i+1}' for i in range(dim_x)] + ['y', 'y_real', 'd', 's', 'z'])


# DGP 7, alpha and beta separated, nonlinear effects in both D and Y, and t-distributed errors
def generate_dgp_7_MNAR(seed, n_obs = 2000, theta=1.0, dim_x=100):

    np.random.seed(seed)

    # 1) α ve β dizileri
    alpha = np.array([0.3 / (k**2) for k in range(1, dim_x+1)])
    beta  = np.array([0.4 / (k**2) for k in range(1, dim_x+1)])
    gamma = 1

    # 2) Correlated t(5) hataları
    def multivariate_t_rvs(mean, cov, df, n):
        d = cov.shape[0]
        z = np.random.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)
        w = np.random.chisquare(df, size=n)
        return z * np.sqrt(df / w)[:, None]

    df_t = 5
    sigma = np.array([[1.0, 0.8],
                    [0.8, 1.0]])
    e_mat = multivariate_t_rvs(mean=[0,0], cov=sigma, df=df_t, n=n_obs)
    #   → e_mat[:,0]  seçim hatası için
    #   → e_mat[:,1]  outcome hatası için

    # 3) x matrisi (AR(1) covariance, dim_x=100)
    cov_mat = toeplitz([0.5**k for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=n_obs)

    d = np.where(
        alpha[22] * x[:, 22]                       # α23  * X23
        + alpha[22] * (x[:, 22]**2)                  # α23  * X23^2
        + alpha[19] * np.sqrt(np.abs(x[:, 19]))      # α20  * √|X20|
        + alpha[24] * x[:, 24]                       # α25  * X25
        + alpha[29] * (x[:, 29] > 19).astype(float)  # α30  * 1[X30 > 19]
        + alpha[34] * (x[:, 34] > 5).astype(float) * (x[:, 34] - 3)  
                                                    # α35  * 1[X35>5] * (X35–3)
        + alpha[39] * np.exp(x[:, 39])               # α40  * exp(X40)
        + alpha[44] * x[:, 44]                       # α45  * X45
        + alpha[49] * x[:, 49]                       # α50  * X50
        + alpha[54] * x[:, 54]                       # α55  * X55
        + alpha[59] * x[:, 59]                       # α60  * X60
        + alpha[64] * (x[:, 64] > 1).astype(float)   # α65  * 1[X65 > 1]
        + np.random.randn(n_obs) > 0,
        1, 0)
    
    # Instrument (Z)
    z = np.random.randn(n_obs)
    
    # Selection process remains linear
    s = np.where(np.dot(x, beta) + d + gamma * z + e_mat[:, 0] > 0, 1, 0)

    # Outcome with additional nonlinear transformations    y = np.dot(x, beta) + theta * d + 0.5 * x[:, 2]**2 + 0.3 * np.abs(x[:, 4]) + 0.2 * x[:, 5]**3 + e[1]
    y = (
        theta * d
        + beta[0] * np.sqrt(np.abs(x[:,  0]))             # β1  * √|X1|
        + beta[ 5] * x[:,  5]                              # β6  * X6
        + beta[10] * (x[:, 10] ** 2)                       # β11 * X11^2
        + beta[15] * (x[:, 15] > 2).astype(float)           # β16 * 1[X16 > 2]
        + beta[20] * x[:, 20] * x[:, 25]                   # β21 * X21 * X26
        + beta[30] * np.log(np.abs(x[:, 30]) + 1)          # β31 * log(|X31|+1)
        + beta[35] * np.sin(x[:, 35])                      # β36 * sin(X36)
        + beta[40] * x[:, 40]                              # β41 * X41
        + beta[45] * (x[:, 45] > -1).astype(float) * x[:, 50]  
                                                        # β46 * 1[X46 > -1] * X51
        + beta[55] * np.sqrt(np.abs(x[:, 55]))             # β56 * √|X56|
        + beta[60] * x[:, 60]                              # β61 * X61
        + e_mat[:, 1] 
    )

    y_real = y.copy()

    # Setting y to 0 for unselected observations
    y[s == 0] = 0

    return pd.DataFrame(np.column_stack([x, y, y_real, d, s, z]), columns=[f'X{i+1}' for i in range(dim_x)] + ['y', 'y_real', 'd', 's', 'z'])


