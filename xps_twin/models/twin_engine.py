from xps_twin.core.grid import CalculationGrid
from xps_twin.core.physics import fermi_dirac
from xps_twin.components.source import XraySource
from xps_twin.components.analyzer_2d import Detector2D
import numpy as np

class DigitalTwinEngine:
    def __init__(self, e_range=(-0.05, 0.05), e_steps=500):
        self.grid = CalculationGrid(e_range[0], e_range[1], e_steps)
        self.source = XraySource()
        self.detector = Detector2D()

    def simulate(self, temp=5.0):
        # 1. 真の物性 (フェルミ分布) を作成
        true_fd = fermi_dirac(self.grid.e_axis, temp)

        # 2. X線光源による2Dエミッション生成
        img_2d = self.source.generate_2d_emission(self.grid, true_fd)

        # 3. 検出器による歪みと1D投影
        # sigma_source: X線源のエネルギー方向分解能 (meV -> eV)
        sigma_source = self.source.sigma_x / 1000.0
        spec_1d = self.detector.project_to_1d(self.grid, img_2d, sigma_source=sigma_source)

        return self.grid.e_axis, spec_1d

    def get_resolution_components(self):
        """
        分解能の各成分を計算

        Returns:
        --------
        dict : 分解能成分（すべてmeV単位）
            - sigma_source: ソース分解能（X線源のエネルギー方向広がり）
            - sigma_detector: 検出器分解能
            - sigma_combined: 合成分解能（二乗和の平方根）
        """
        sigma_source_mev = self.source.sigma_x  # 既にmeV
        sigma_detector_mev = self.detector.sigma_res * 1000.0  # eV -> meV

        # 合成分解能: sqrt(sigma_source^2 + sigma_detector^2)
        sigma_combined_mev = np.sqrt(sigma_source_mev**2 + sigma_detector_mev**2)

        return {
            'sigma_source': sigma_source_mev,
            'sigma_detector': sigma_detector_mev,
            'sigma_combined': sigma_combined_mev
        }