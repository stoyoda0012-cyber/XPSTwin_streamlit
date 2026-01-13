import numpy as np
from scipy.interpolate import RegularGridInterpolator

class Detector2D:
    def __init__(self, kappa=0.0, theta=0.0, sigma_res=0.001):
        self.kappa = kappa        # 曲率
        self.theta = theta        # 傾き
        self.sigma_res = sigma_res # 固有分解能

    def project_to_1d(self, grid, ideal_2d_img):
        theta_rad = np.radians(self.theta)
        y_norm = grid.Y / np.max(np.abs(grid.y_axis))

        # 座標変換 (歪みと回転)
        E_src = grid.E * np.cos(theta_rad) + grid.Y * np.sin(theta_rad)
        Y_src = -grid.E * np.sin(theta_rad) + grid.Y * np.cos(theta_rad)
        E_src_curved = E_src - self.kappa * (y_norm**2)

        # 補間
        interp = RegularGridInterpolator((grid.y_axis, grid.e_axis), ideal_2d_img,
                                            bounds_error=False, fill_value=None)

        pts = np.stack([Y_src.ravel(), E_src_curved.ravel()], axis=-1)
        img_distorted = interp(pts).reshape(grid.E.shape)

        # Y軸方向に積分
        spec_1d = np.sum(img_distorted, axis=0)

        # 装置固有分解能による畳み込み
        if self.sigma_res > 0:
            de = grid.e_axis[1] - grid.e_axis[0]
            gauss_width = int(5 * self.sigma_res / de)  # 5σ幅
            if gauss_width > 0:
                x_gauss = np.arange(-gauss_width, gauss_width + 1) * de
                gauss_kernel = np.exp(-x_gauss**2 / (2 * self.sigma_res**2))
                gauss_kernel /= np.sum(gauss_kernel)

                # エッジ効果を防ぐためパディングを追加
                pad_width = len(gauss_kernel) // 2
                spec_padded = np.pad(spec_1d, pad_width, mode='edge')
                spec_conv = np.convolve(spec_padded, gauss_kernel, mode='same')
                spec_1d = spec_conv[pad_width:-pad_width]

        return spec_1d