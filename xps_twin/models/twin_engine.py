from xps_twin.core.grid import CalculationGrid
from xps_twin.core.physics import fermi_dirac
from xps_twin.components.source import XraySource
from xps_twin.components.analyzer_2d import Detector2D

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
        spec_1d = self.detector.project_to_1d(self.grid, img_2d)
        
        return self.grid.e_axis, spec_1d