import numpy as np
from scipy.optimize import least_squares
# 外部からengineをもらう設計ならそのままでOKですが、
# もし内部で参照があれば xps_twin. を頭につけます

class XPSOptimizer:
    def __init__(self, engine):
        self.engine = engine

    def objective_function(self, params, e_data, y_data, temp):
        """
        params: [kappa, theta, sigma_res, ef_shift, alpha]
        の順番で最適化アルゴリズムから渡される
        """
        kappa, theta, sigma_res, ef_shift, alpha = params
        
        # エンジンのパラメータを更新
        self.engine.detector.kappa = kappa
        self.engine.detector.theta = theta
        self.engine.detector.sigma_res = sigma_res
        self.engine.source.alpha = alpha
        
        # シミュレーション実行 (ef_shiftを考慮)
        # 本来のEf=0から、ef_shift分だけずらして計算
        kb = 8.617e-5
        x, y_sim = self.engine.simulate(temp=temp)
        
        # ef_shiftを反映させるため、y_simを再補完
        y_fit = np.interp(e_data, x - ef_shift, y_sim)
        
        # 実測データとの差分を返す
        return y_fit - y_data

    def fit(self, e_data, y_data, temp=5.0, initial_guess=None):
        if initial_guess is None:
            # 初期値: [kappa, theta, sigma, ef_shift, alpha]
            initial_guess = [0.001, 0.05, 0.002, 0.0, 0.01]
        
        # 探索範囲の制限 (物理的にあり得ない値を防ぐ)
        # bounds = ([下限], [上限])
        bounds = (
            [0, -0.2, 0.0001, -0.01, 0],   # 下限
            [0.05, 0.2, 0.01, 0.01, 0.1]    # 上限
        )
        
        res = least_squares(
            self.objective_function, 
            initial_guess, 
            args=(e_data, y_data, temp),
            bounds=bounds,
            verbose=2
        )
        return res