
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np
from scipy.special import expit
from scipy.stats import norm

# p(Pi = pi | X = x)
def p_pi_given_x(pi_value, x, beta, gamma):
    p_d1_given_x = norm.cdf(np.dot(x, beta))
    phi_inv_pi = norm.ppf(pi_value)
    
    # z^* değerlerini hesaplama
    z_d0 = (phi_inv_pi - np.dot(x, beta)) / gamma
    z_d1 = (phi_inv_pi - (np.dot(x, beta) + 1)) / gamma

    # p(z) için PDF hesaplama
    pdf_d0 = norm.pdf(z_d0)
    pdf_d1 = norm.pdf(z_d1)

    # Nihai hesaplama
    result = (1 - p_d1_given_x) * pdf_d0 / (gamma * norm.pdf(phi_inv_pi)) \
        + p_d1_given_x * pdf_d1 / (gamma * norm.pdf(phi_inv_pi))

    return result

# p(Pi = pi | X = x, d = 1)
def p_pi_given_x_d1(pi_value, x, beta, gamma):
    phi_inv_pi = norm.ppf(pi_value)
    z_d1 = (phi_inv_pi - (np.dot(x, beta) + 1)) / gamma

    # Nihai hesaplama (tek terim)
    result = norm.pdf(z_d1) / (gamma * norm.pdf(phi_inv_pi))
    return result


def prepare_oracle_dgp1_MAR(X, d):
    
    dim_x = len(X.columns)
    gamma = 0
    beta = np.array([0.4 / (k**2) for k in range(1, dim_x + 1)])
    theta = 1
    z = np.random.randn(len(X))

    # Compute oracle nuisance parameters
    g_d1_oracle = np.dot(X, beta) + theta * 1  # For the treatment group g(X, d=1)
    g_d0_oracle = np.dot(X, beta) + theta * 0  # For the control group g(X, d=0)

    m_oracle = norm.cdf(np.dot(X, beta))  # True p_d(X)
    pi_oracle = norm.cdf(np.dot(X, beta) + d + gamma * z)  # True pi(D, X, Z)

    return g_d0_oracle, g_d1_oracle, m_oracle, pi_oracle


def prepare_oracle_dgp1_MNAR(X, d, z, s, theta=1.0):

    # 1) Parameters
    dim      = X.shape[1]
    beta     = np.array([0.4/(k**2) for k in range(1, dim+1)])
    gamma    = 1.0
    sigma    = np.array([[1,0.8],[0.8,1]])
    cov_e1e2 = sigma[0,1]
    sigma_e1 = np.sqrt(sigma[0,0])

    # 2) π-oracle = P(S=1|X,d,z)  (linear)
    xβ        = X.values.dot(beta)
    lin_sel   = xβ + d + gamma*z
    pi_oracle = norm.cdf(lin_sel)

    # 3) conditional bias = E[e2|e1 > −lin_sel]
    c = -lin_sel
    cdf_term = norm.cdf(c / sigma_e1)
    denom = np.clip(1 - cdf_term, 1e-8, 1.0)  # 0’a yaklaşmayı engelle
    bias = (cov_e1e2 / sigma_e1) * (norm.pdf(c / sigma_e1) / denom)

    # 4) g-oracle’lar = E[Y|X,d,S=1]  (lineer çıktı)
    g_d1_oracle = xβ + theta*1 + bias * (s == 1)
    g_d0_oracle = xβ + theta*0 + bias * (s == 1)

    # 5) p(D=1|X) — lineer atama kuralı
    p_d1_x = norm.cdf(xβ)

    # 6) f(π|X,D=d) yoğunlukları
    icdf    = norm.ppf(pi_oracle)
    f_pi_d1 = (1/gamma) * norm.pdf((icdf - (xβ+1+gamma*z))/gamma) / norm.pdf(icdf)
    f_pi_d0 = (1/gamma) * norm.pdf((icdf - (xβ+0+gamma*z))/gamma) / norm.pdf(icdf)

    # 7) Bayes adımı: p(D=1|X,π)
    f_pi      = p_d1_x*f_pi_d1 + (1-p_d1_x)*f_pi_d0
    p_d1_x_pi = (f_pi_d1 * p_d1_x) / f_pi

    return g_d0_oracle, g_d1_oracle, p_d1_x, pi_oracle

# Only added non-linearity in the outcome (y)
def prepare_oracle_dgp2_MAR(X, d, theta=1.0):
    dim   = X.shape[1]
    beta  = np.array([0.4/(k**2) for k in range(1, dim+1)])
    theta_val = theta

    # 1) xβ
    xβ = X.values.dot(beta)

    # 2) g-oracle = E[Y | X, d, S=1]  
    nl_term = (
        0.5 * (X.values[:, 2] ** 2)
      + 0.3 * np.abs(X.values[:, 4])
      + 0.2 * (X.values[:, 5] ** 3)
    )
    g_d1_oracle = xβ + theta_val*1 + nl_term
    g_d0_oracle = xβ + theta_val*0 + nl_term

    # 3) p(D=1 | X) — lineer atama (MAR’da aynı şekilde)
    p_d_oracle = norm.cdf(xβ)

    # 4) π-oracle = P(S=1 | X, d)  (MAR’da γ=0, z etkisiz)
    pi_oracle = norm.cdf(xβ + d)

    return g_d0_oracle, g_d1_oracle, p_d_oracle, pi_oracle


# Only added non-linearity in the outcome (y)
def prepare_oracle_dgp2_MNAR(X, d, z, s, theta=1.0, mar=False):

    # 1) Parametreler
    dim = X.shape[1]
    beta = np.array([0.4/(k**2) for k in range(1, dim+1)])
    if mar:
        sigma = np.eye(2);    gamma = 0.0
    else:
        sigma = np.array([[1,0.8],[0.8,1]]);    gamma = 1.0

    cov_e1e2 = sigma[0,1]
    sigma_e1  = np.sqrt(sigma[0,0])

    # 2) π-oracle = P(S=1|X,d,z)
    lin_sel   = X.values.dot(beta) + d + gamma*z
    pi_oracle = norm.cdf(lin_sel)

    # 3) Seleksiyon bias’ı
    c    = -lin_sel
    bias = (cov_e1e2/sigma_e1) * (norm.pdf(c/sigma_e1)/(1 - norm.cdf(c/sigma_e1)))

    # 4) g-oracle’lar = E[Y|X,d,S=1]
    xβ      = X.values.dot(beta)
    nl_term = (
        0.5*(X.values[:,2]**2)
      + 0.3*(np.abs(X.values[:,4]))
      + 0.2*(X.values[:,5]**3)
    )
    g_d1_oracle = xβ + theta*1 + nl_term + bias * (s == 1)
    g_d0_oracle = xβ + theta*0 + nl_term + bias * (s == 1)

    # 5) p(D=1|X) — linear atama kuralı
    p_d1_x = norm.cdf(xβ)

    # 6) f(π|X,D=d) yoğunlukları
    icdf    = norm.ppf(pi_oracle)
    f_pi_d1 = (1/gamma) * norm.pdf((icdf - (xβ+1 + gamma*z))/gamma) / norm.pdf(icdf)
    f_pi_d0 = (1/gamma) * norm.pdf((icdf - (xβ+0 + gamma*z))/gamma) / norm.pdf(icdf)

    # 7) Bayes adımı: p(D=1|X,π)
    f_pi        = p_d1_x*f_pi_d1 + (1-p_d1_x)*f_pi_d0
    p_d1_x_pi   = (f_pi_d1 * p_d1_x) / f_pi

    return g_d0_oracle, g_d1_oracle, p_d1_x_pi, pi_oracle

# Added non-linearity in Y - X and D - X relationships

def prepare_oracle_dgp3_MAR(X, d, theta=1.0):
    """
    DGP-3 için MAR (gamma=0, hata bileşenleri bağımsız) senaryosu.
    X: pd.DataFrame (X1..Xdim)
    d: 1-d array (tedavi göstergesi)
    theta: tedavi etki katsayısı
    Döndürülenler:
      g_d0_oracle, g_d1_oracle, p_d_oracle, pi_oracle
    """
    # 1) Parametreler
    dim   = X.shape[1]
    beta  = np.array([0.4/(k**2) for k in range(1, dim+1)])
    theta_val = theta

    # 2) xβ
    xβ = X.values.dot(beta)

    # 3) g-oracle’lar = E[Y | X, d, S=1]  (Seçim bias’ı yok – çünkü MAR)
    nl_term = (
        0.5 * X.values[:, 2]**2 +
        0.3 * np.abs(X.values[:, 4]) +
        0.2 * X.values[:, 5]**3
    )
    g_d1_oracle = xβ + theta_val*1 + nl_term
    g_d0_oracle = xβ + theta_val*0 + nl_term

    # 4) p(D=1 | X) — non-lineer atama kuralı (DGP-3’de)
    h = (
        xβ +
        0.5 * X.values[:, 1]**2 +
        0.3 * np.abs(X.values[:, 3]) +
        0.2 * X.values[:, 5] * X.values[:, 7] +
        0.1 * np.log(np.abs(X.values[:, 6]) + 1)
    )
    p_d_oracle = norm.cdf(h)

    # 5) π-oracle = P(S=1 | X, d)  (MAR’da γ=0 => z etkisiz)
    pi_oracle = norm.cdf(xβ + d)

    return g_d0_oracle, g_d1_oracle, p_d_oracle, pi_oracle

def prepare_oracle_dgp3_MNAR(X, d, z, s, theta=1.0):
    # 1) sabitler
    dim = X.shape[1]
    beta = np.array([0.4/(k**2) for k in range(1, dim+1)])
    gamma = 1.0
    sigma = np.array([[1,0.8],[0.8,1]])
    cov_e1e2 = sigma[0,1]
    sigma_e1  = np.sqrt(sigma[0,0])

    # 2) pi-oracle
    lin_sel = X.values.dot(beta) + d + gamma*z
    pi_oracle = norm.cdf(lin_sel)

    # 3) conditional bias
    c = -lin_sel
    bias = (cov_e1e2/sigma_e1) * (norm.pdf(c/sigma_e1)/(1 - norm.cdf(c/sigma_e1)))

    # 4) g_oracle’lar (D=0,1 için nonlinear y + bias)
    #    X.values[:,i] Python’da 0-based => X2 = [:,1], X4 = [:,3] vs. sana göre
    xβ = X.values.dot(beta)
    nl_term = (
        0.3*np.log(np.abs(X.values[:,3])+1)
      + 0.3*np.sin(X.values[:,4])
      + 0.2*(X.values[:,7]*X.values[:,9])
    )
    g_d1_oracle = xβ + theta*1 + nl_term + bias * (s == 1)
    g_d0_oracle = xβ + theta*0 + nl_term + bias * (s == 1)

    # 5) p(D=1|X) — non-linear h(x)
    h = (
        X.values.dot(beta)
      + 0.2*np.sin(X.values[:,1])
      + 0.2*(X.values[:,2]*X.values[:,5])
    )
    p_d1_x = norm.cdf(h)

    # 6) f(pi|X,D=d) yoğunlukları
    icdf = norm.ppf(pi_oracle)
    f_pi_d1 = (1/gamma) * norm.pdf((icdf - (xβ+1+gamma*z))/gamma) / norm.pdf(icdf)
    f_pi_d0 = (1/gamma) * norm.pdf((icdf - (xβ+0+gamma*z))/gamma) / norm.pdf(icdf)

    # 7) f(pi|X) ve Bayes ile p(D=1|X,pi)
    f_pi = p_d1_x*f_pi_d1 + (1-p_d1_x)*f_pi_d0
    p_d1_x_pi = (f_pi_d1 * p_d1_x) / f_pi

    return g_d0_oracle, g_d1_oracle, p_d1_x_pi, pi_oracle

def prepare_oracle_dgp4_MNAR(X, d, z, s, theta=1.0):
    # 1) sabitler
    dim = X.shape[1]
    beta = np.array([0.4/(k**2) for k in range(1, dim+1)])
    gamma = 1.0
    sigma = np.array([[1,0.8],[0.8,1]])
    cov_e1e2 = sigma[0,1]
    sigma_e1  = np.sqrt(sigma[0,0])

    # 2) pi-oracle
    lin_sel = X.values.dot(beta) + d + gamma*z
    pi_oracle = norm.cdf(lin_sel)

    # 3) conditional bias
    c = -lin_sel
    bias = (cov_e1e2/sigma_e1) * (norm.pdf(c/sigma_e1)/(1 - norm.cdf(c/sigma_e1)))

    # 4) g_oracle’lar (D=0,1 için nonlinear y + bias)
    #    X.values[:,i] Python’da 0-based => X2 = [:,1], X4 = [:,3] vs. sana göre
    xβ = X.values.dot(beta)
    nl_term = (
        0.5 * X.values[:, 2]**2 +
        0.3 * np.abs(X.values[:, 4]) +
        0.2 * X.values[:, 5]**3
    )
    g_d1_oracle = xβ + theta*1 + nl_term + bias * (s == 1)
    g_d0_oracle = xβ + theta*0 + nl_term + bias * (s == 1)

    # 5) p(D=1|X) — non-linear h(x)
    h = (
        xβ
        + 0.2 * np.sin(X.values[:, 2] * X.values[:, 3])
        + 0.2 * np.exp(-np.abs(X.values[:, 7])) * X.values[:, 5]
        + 0.2 * np.sign(X.values[:, 6]) * X.values[:, 1]
    )

    p_d1_x = norm.cdf(h)

    # 6) f(pi|X,D=d) yoğunlukları
    icdf = norm.ppf(pi_oracle)
    f_pi_d1 = (1/gamma) * norm.pdf((icdf - (xβ+1+gamma*z))/gamma) / norm.pdf(icdf)
    f_pi_d0 = (1/gamma) * norm.pdf((icdf - (xβ+0+gamma*z))/gamma) / norm.pdf(icdf)

    # 7) f(pi|X) ve Bayes ile p(D=1|X,pi)
    f_pi = p_d1_x*f_pi_d1 + (1-p_d1_x)*f_pi_d0
    p_d1_x_pi = (f_pi_d1 * p_d1_x) / f_pi

    return g_d0_oracle, g_d1_oracle, p_d1_x_pi, pi_oracle

def prepare_oracle_dgp5_MNAR(X, d, z, s, theta=1.0):
    # 1) sabitler
    dim = X.shape[1]
    beta = np.array([0.4/(k**2) for k in range(1, dim+1)])
    gamma = 1.0
    sigma = np.array([[1,0.8],[0.8,1]])
    cov_e1e2 = sigma[0,1]
    sigma_e1  = np.sqrt(sigma[0,0])

    # 2) pi-oracle
    lin_sel = X.values.dot(beta) + d + gamma*z
    pi_oracle = norm.cdf(lin_sel)

    # 3) conditional bias
    c = -lin_sel
    bias = (cov_e1e2/sigma_e1) * (norm.pdf(c/sigma_e1)/(1 - norm.cdf(c/sigma_e1)))

    # 4) g_oracle’lar (D=0,1 için nonlinear y + bias)
    #    X.values[:,i] Python’da 0-based => X2 = [:,1], X4 = [:,3] vs. sana göre
    xβ = X.values.dot(beta)
    x_val = X.values

    nl_term = (
        beta[1]   * x_val[:,  1]                                +  # β2 * X2
        beta[4]   * x_val[:,  4]                                +  # β5 * X5
        beta[1]   * x_val[:,  1] * x_val[:,  4]                 +  # β2 * X2 * X5
        beta[11]  * x_val[:, 11]                                +  # β12 * X12
        beta[22]  * x_val[:, 22]                                +  # β23 * X23
        beta[22]  * x_val[:, 22]**2                             +  # β23 * X23^2
        beta[11]  * x_val[:, 11] * x_val[:, 22]**2              +  # β12 * X12 * X23^2
        beta[39]  * x_val[:, 39]                                +  # β40 * X40
        beta[66]  * np.sqrt(np.abs(x_val[:, 66]))               +  # β67 * √|X67|
        beta[76]  * x_val[:, 76]                                +  # β77 * X77
        beta[88] * (x_val[:, 88] > 19).astype(float)                        +  # 1[X89 > 19]
        beta[94] * (x_val[:, 94] > 5).astype(float) * (x_val[:, 94] - 3)      # 1[X95 > 5]*(X95 - 3)
    )

    g_d1_oracle = theta*1 + nl_term + bias * (s == 1)
    g_d0_oracle = theta*0 + nl_term + bias * (s == 1)

    # 5) p(D=1|X) — non-linear h(x)

    # Treatment assignment
    h = (
        beta[ 1] * x_val[:,  1]                       # α2  * x_val2
        + beta[ 4] * x_val[:,  4]                       # α5  * x_val5
        + beta[11] * x_val[:, 11]                       # α12 * x_val12
        + beta[66] * np.sqrt(np.abs(x_val[:, 66]))      # α67 * √|x_val67|
        + beta[88] * (x_val[:, 88] > 19).astype(float)              # 1[x_val89 > 19] 
        + beta[94] * (x_val[:, 94] > 5).astype(float) * (x_val[:, 94] - 3) 
    )

    p_d1_x = norm.cdf(h)

    # 6) f(pi|X,D=d) yoğunlukları
    icdf = norm.ppf(pi_oracle)
    f_pi_d1 = (1/gamma) * norm.pdf((icdf - (xβ+1+gamma*z))/gamma) / norm.pdf(icdf)
    f_pi_d0 = (1/gamma) * norm.pdf((icdf - (xβ+0+gamma*z))/gamma) / norm.pdf(icdf)

    # 7) f(pi|X) ve Bayes ile p(D=1|X,pi)
    f_pi = p_d1_x*f_pi_d1 + (1-p_d1_x)*f_pi_d0
    p_d1_x_pi = (f_pi_d1 * p_d1_x) / f_pi

    return g_d0_oracle, g_d1_oracle, p_d1_x_pi, pi_oracle

from scipy.stats import t, norm

def prepare_oracle_dgp6_MNAR(X, d, z, s, theta=1.0):
    # 1) sabitler
    dim = X.shape[1]
    beta = np.array([0.4/(k**2) for k in range(1, dim+1)])
    alpha = np.array([0.3 / (k**2) for k in range(1, dim+1)])
    gamma = 1.0
    sigma = np.array([[1,0.8],[0.8,1]])
    cov_e1e2 = sigma[0,1]
    sigma_e1  = np.sqrt(sigma[0,0])

    # 2) pi-oracle
    lin_sel = X.values.dot(beta) + d + gamma*z
    pi_oracle = norm.cdf(lin_sel)

    # 3) conditional bias
    c = -lin_sel
    bias = (cov_e1e2/sigma_e1) * (norm.pdf(c/sigma_e1)/(1 - norm.cdf(c/sigma_e1)))


    # 4) g_oracle’lar (D=0,1 için nonlinear y + bias)
    #    X.values[:,i] Python’da 0-based => X2 = [:,1], X4 = [:,3] vs. sana göre
    xβ = X.values.dot(beta)
    x_val = X.values

    nl_term = (
        + beta[0] * np.sqrt(np.abs(x_val[:,  0]))             # β1  * √|x_val1|
        + beta[ 5] * x_val[:,  5]                              # β6  * x_val6
        + beta[10] * (x_val[:, 10] ** 2)                       # β11 * x_val11^2
        + beta[15] * (x_val[:, 15] > 2).astype(float)           # β16 * 1[x_val16 > 2]
        + beta[20] * x_val[:, 20] * x_val[:, 25]                   # β21 * x_val21 * x_val26
        + beta[30] * np.log(np.abs(x_val[:, 30]) + 1)          # β31 * log(|x_val31|+1)
        + beta[35] * np.sin(x_val[:, 35])                      # β36 * sin(x_val36)
        + beta[40] * x_val[:, 40]                              # β41 * x_val41
        + beta[45] * (x_val[:, 45] > -1).astype(float) * x_val[:, 50]  
                                                        # β46 * 1[x_val46 > -1] * x_val51
        + beta[55] * np.sqrt(np.abs(x_val[:, 55]))             # β56 * √|x_val56|
        + beta[60] * x_val[:, 60]                              # β61 * X61
    )

    g_d1_oracle = theta*1 + nl_term + bias * (s == 1)
    g_d0_oracle = theta*0 + nl_term + bias * (s == 1)

    # 5) p(D=1|X) — non-linear h(x)

    # Treatment assignment
    h = (
        alpha[22] * x_val[:, 22]                       # α23  * x_val23
        + alpha[22] * (x_val[:, 22]**2)                  # α23  * x_val23^2
        + alpha[19] * np.sqrt(np.abs(x_val[:, 19]))      # α20  * √|x_val20|
        + alpha[24] * x_val[:, 24]                       # α25  * x_val25
        + alpha[29] * (x_val[:, 29] > 19).astype(float)  # α30  * 1[x_val30 > 19]
        + alpha[34] * (x_val[:, 34] > 5).astype(float) * (x_val[:, 34] - 3)  
                                                    # α35  * 1[x_val35>5] * (x_val35–3)
        + alpha[39] * np.exp(x_val[:, 39])               # α40  * ex_valp(x_val40)
        + alpha[44] * x_val[:, 44]                       # α45  * x_val45
        + alpha[49] * x_val[:, 49]                       # α50  * x_val50
        + alpha[54] * x_val[:, 54]                       # α55  * x_val55
        + alpha[59] * x_val[:, 59]                       # α60  * x_val60
        + alpha[64] * (x_val[:, 64] > 1).astype(float)   # α65  * 1[X65 > 1]
    )

    p_d1_x = norm.cdf(h)

    # 6) f(pi|X,D=d) yoğunlukları
    icdf = norm.ppf(pi_oracle)
    f_pi_d1 = (1/gamma) * norm.pdf((icdf - (xβ+1+gamma*z))/gamma) / norm.pdf(icdf)
    f_pi_d0 = (1/gamma) * norm.pdf((icdf - (xβ+0+gamma*z))/gamma) / norm.pdf(icdf)

    # 7) f(pi|X) ve Bayes ile p(D=1|X,pi)
    f_pi = p_d1_x*f_pi_d1 + (1-p_d1_x)*f_pi_d0
    p_d1_x_pi = (f_pi_d1 * p_d1_x) / f_pi

    return g_d0_oracle, g_d1_oracle, p_d1_x_pi, pi_oracle



# errors t-distributed with 5 degrees of freedom
def prepare_oracle_dgp7_MNAR(X, d, z, theta=1.0):
    # 1) sabitler
    df_t = 5

    dim = X.shape[1]
    beta = np.array([0.4/(k**2) for k in range(1, dim+1)])
    alpha = np.array([0.3 / (k**2) for k in range(1, dim+1)])
    gamma = 1.0
    sigma = np.array([[1,0.8],[0.8,1]])
    cov_e1e2 = sigma[0,1]
    sigma_e1  = np.sqrt(sigma[0,0])

    # 2) pi-oracle
    lin_sel = X.values.dot(beta) + d + gamma*z
    pi_oracle = t.cdf(lin_sel / np.sqrt(sigma[0,0]), df=df_t)

    # 3) conditional bias
    t_inv_mills = t.pdf(-lin_sel / sigma_e1, df=df_t) / t.sf(-lin_sel / sigma_e1, df=df_t)
    bias_t = (cov_e1e2 / sigma_e1) * t_inv_mills

    # 4) g_oracle’lar (D=0,1 için nonlinear y + bias)
    #    X.values[:,i] Python’da 0-based => X2 = [:,1], X4 = [:,3] vs. sana göre
    xβ = X.values.dot(beta)
    x_val = X.values

    nl_term = (
        + beta[0] * np.sqrt(np.abs(x_val[:,  0]))             # β1  * √|x_val1|
        + beta[ 5] * x_val[:,  5]                              # β6  * x_val6
        + beta[10] * (x_val[:, 10] ** 2)                       # β11 * x_val11^2
        + beta[15] * (x_val[:, 15] > 2).astype(float)           # β16 * 1[x_val16 > 2]
        + beta[20] * x_val[:, 20] * x_val[:, 25]                   # β21 * x_val21 * x_val26
        + beta[30] * np.log(np.abs(x_val[:, 30]) + 1)          # β31 * log(|x_val31|+1)
        + beta[35] * np.sin(x_val[:, 35])                      # β36 * sin(x_val36)
        + beta[40] * x_val[:, 40]                              # β41 * x_val41
        + beta[45] * (x_val[:, 45] > -1).astype(float) * x_val[:, 50]  
                                                        # β46 * 1[x_val46 > -1] * x_val51
        + beta[55] * np.sqrt(np.abs(x_val[:, 55]))             # β56 * √|x_val56|
        + beta[60] * x_val[:, 60]                              # β61 * X61
    )

    g_d1_oracle = theta*1 + nl_term + bias_t
    g_d0_oracle = theta*0 + nl_term + bias_t

    # 5) p(D=1|X) — non-linear h(x)

    # Treatment assignment
    h = (
        alpha[22] * x_val[:, 22]                       # α23  * x_val23
        + alpha[22] * (x_val[:, 22]**2)                  # α23  * x_val23^2
        + alpha[19] * np.sqrt(np.abs(x_val[:, 19]))      # α20  * √|x_val20|
        + alpha[24] * x_val[:, 24]                       # α25  * x_val25
        + alpha[29] * (x_val[:, 29] > 19).astype(float)  # α30  * 1[x_val30 > 19]
        + alpha[34] * (x_val[:, 34] > 5).astype(float) * (x_val[:, 34] - 3)  
                                                    # α35  * 1[x_val35>5] * (x_val35–3)
        + alpha[39] * np.exp(x_val[:, 39])               # α40  * ex_valp(x_val40)
        + alpha[44] * x_val[:, 44]                       # α45  * x_val45
        + alpha[49] * x_val[:, 49]                       # α50  * x_val50
        + alpha[54] * x_val[:, 54]                       # α55  * x_val55
        + alpha[59] * x_val[:, 59]                       # α60  * x_val60
        + alpha[64] * (x_val[:, 64] > 1).astype(float)   # α65  * 1[X65 > 1]
    )

    p_d1_x = norm.cdf(h)

    icdf = t.ppf(pi_oracle, df=df_t)
    f_pi_d1 = (1/gamma) * t.pdf((icdf - (xβ + 1 + gamma*z)) / gamma, df=df_t) / t.pdf(icdf, df=df_t)
    f_pi_d0 = (1/gamma) * t.pdf((icdf - (xβ + 0 + gamma*z)) / gamma, df=df_t) / t.pdf(icdf, df=df_t)

    f_pi = p_d1_x * f_pi_d1 + (1 - p_d1_x) * f_pi_d0
    p_d1_x_pi = (f_pi_d1 * p_d1_x) / f_pi

    return g_d0_oracle, g_d1_oracle, p_d1_x_pi, pi_oracle

