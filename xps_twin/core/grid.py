import numpy as np

class CalculationGrid:
    def __init__(self, e_start, e_end, e_steps, y_start=-10, y_end=10, y_steps=200):
        self.e_axis = np.linspace(e_start, e_end, e_steps)
        self.y_axis = np.linspace(y_start, y_end, y_steps)
        self.E, self.Y = np.meshgrid(self.e_axis, self.y_axis)
        self.de = self.e_axis[1] - self.e_axis[0]