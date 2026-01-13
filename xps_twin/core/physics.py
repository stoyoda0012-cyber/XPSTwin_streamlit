import numpy as np
from scipy.special import erf

def fermi_dirac(energy, temp, ef=0):
    kb = 8.617333262e-5
    if temp < 0.1: return np.where(energy <= ef, 1.0, 0.0)
    val = np.clip((energy - ef) / (kb * temp), -100, 100)
    return 1.0 / (np.exp(val) + 1.0)

def skew_gaussian(x, sigma, gamma):
    """非対称ガウス分布"""
    phi = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    cdf = 0.5 * (1 + erf(gamma * x / (sigma * np.sqrt(2))))
    return 2 * phi * cdf

def elliptical_gaussian_2d(X, Y, sigma_x, sigma_y, gamma_x=0, gamma_y=0, rotation=0):
    """
    2D楕円ガウシアン分布（非対称版）

    Parameters:
    -----------
    X, Y : 2D meshgrid
    sigma_x, sigma_y : X/Y方向の標準偏差
    gamma_x, gamma_y : X/Y方向の非対称性パラメータ
    rotation : 回転角度（度）
    """
    # 回転行列を適用
    theta = np.radians(rotation)
    X_rot = X * np.cos(theta) - Y * np.sin(theta)
    Y_rot = X * np.sin(theta) + Y * np.cos(theta)

    # X方向の非対称ガウシアン
    phi_x = np.exp(-X_rot**2 / (2 * sigma_x**2))
    cdf_x = 0.5 * (1 + erf(gamma_x * X_rot / (sigma_x * np.sqrt(2))))
    dist_x = 2 * phi_x * cdf_x

    # Y方向の非対称ガウシアン
    phi_y = np.exp(-Y_rot**2 / (2 * sigma_y**2))
    cdf_y = 0.5 * (1 + erf(gamma_y * Y_rot / (sigma_y * np.sqrt(2))))
    dist_y = 2 * phi_y * cdf_y

    # 2D分布 = X方向 × Y方向
    dist_2d = dist_x * dist_y

    # 正規化
    return dist_2d / (np.sum(dist_2d) + 1e-12)

def apply_smart_padding_conv(data, kernel):
    """
    フェルミエッジ等の特性に合わせ、占有側(左)は端の値を維持し、
    非占有側(右/高BE)は0へ落ち着くようにパディングして畳み込む
    """
    pad_size = len(kernel) // 2
    # 左側(Low BE)はデータの端の値(通常1)、右側(High BE)は0でパディング
    padded = np.pad(data, (pad_size, pad_size), mode='constant', 
                    constant_values=(data[0], 0))
    
    conv_result = np.convolve(padded, kernel, mode='valid')
    return conv_result[:len(data)]