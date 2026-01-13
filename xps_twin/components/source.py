import numpy as np
from xps_twin.core.physics import skew_gaussian, elliptical_gaussian_2d

class XraySource:
    def __init__(self, sigma_x=0.01, sigma_y=1.0, alpha=0.0, gamma_x=0.0, gamma_y=0.0, rotation=0.0):
        self.sigma_x = sigma_x      # X方向のスポットサイズ (eV単位)
        self.sigma_y = sigma_y      # Y方向のスポットサイズ
        self.alpha = alpha          # エネルギー勾配 (dE/dy)
        self.gamma_x = gamma_x      # X方向の非対称性
        self.gamma_y = gamma_y      # Y方向の非対称性
        self.rotation = rotation    # 回転角度（度）

    def get_spatial_distribution(self, y_axis):
        """Y軸方向の空間分布を取得（1Dスライス用）"""
        return skew_gaussian(y_axis, self.sigma_y, self.gamma_y)

    def get_2d_spot_profile(self, grid):
        """2Dスポット形状を取得"""
        return elliptical_gaussian_2d(grid.E, grid.Y, self.sigma_x, self.sigma_y,
                                      self.gamma_x, self.gamma_y, self.rotation)

    def generate_2d_emission(self, grid, true_1d_spectrum):
        # Y方向の空間分布を取得（スポットのY軸強度分布）
        y_distribution = skew_gaussian(grid.y_axis, self.sigma_y, self.gamma_y)

        img = np.zeros(grid.E.shape)
        for i, y_val in enumerate(grid.y_axis):
            shift = self.alpha * y_val
            # left=... で左端の値を維持、right=0 で高エネルギー側を0にする
            shifted_spec = np.interp(grid.e_axis - shift, grid.e_axis, true_1d_spectrum,
                                    left=true_1d_spectrum[0], right=0)
            # Y方向の強度分布のみを掛ける（X方向はフェルミ分布を維持）
            img[i, :] = shifted_spec * y_distribution[i]

        return img