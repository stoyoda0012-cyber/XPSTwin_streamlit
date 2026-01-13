import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import convolve


class XPSDeconvolver:
    def __init__(self, engine):
        self.engine = engine

    def fermi_dirac_convolved(self, energy, ef_shift, temp, sigma_total):
        """
        フェルミディラック関数とガウシアン装置関数のコンボリューション

        Parameters:
        -----------
        energy : array
            エネルギー軸
        ef_shift : float
            フェルミエネルギーのシフト (eV)
        temp : float
            温度 (K)
        sigma_total : float
            合計分解能（ガウシアンの標準偏差, eV）

        Returns:
        --------
        convolved : array
            コンボリューション後のスペクトル
        """
        kb = 8.617e-5  # eV/K

        # エネルギー軸のステップサイズ（平均値を使用）
        de = np.mean(np.diff(energy))

        # パディング数を計算（最小10ポイント、最大1000ポイント）
        n_pad = max(10, min(1000, int(10 * sigma_total / de)))

        # パディングを追加
        e_pad_left = energy[0] - np.arange(n_pad, 0, -1) * de
        e_pad_right = energy[-1] + np.arange(1, n_pad + 1) * de
        energy_padded = np.concatenate([e_pad_left, energy, e_pad_right])

        # パディングされたエネルギー軸でフェルミディラック関数を計算
        fd_padded = 1.0 / (1.0 + np.exp((energy_padded - ef_shift) / (kb * temp)))

        # ガウシアン装置関数（±5σ）
        n_gauss = max(5, min(1000, int(10 * sigma_total / de)))
        if n_gauss % 2 == 0:
            n_gauss += 1  # 奇数にする

        x_gauss = np.linspace(-5*sigma_total, 5*sigma_total, n_gauss)
        gaussian = np.exp(-x_gauss**2 / (2 * sigma_total**2))
        gaussian = gaussian / np.sum(gaussian)  # 正規化

        # コンボリューション（full modeで計算してから中心部分を抽出）
        convolved_full = convolve(fd_padded, gaussian, mode='same')

        # 元のエネルギー範囲に対応する部分を抽出
        convolved = convolved_full[n_pad:n_pad+len(energy)]

        return convolved

    def fit_fermi_edge(self, observed_spectrum, temp, initial_guess=None, fit_temp=True, use_global_opt=True):
        """
        観測スペクトルをフェルミディラック+ガウシアンでフィッティング

        Parameters:
        -----------
        observed_spectrum : array
            観測スペクトル（規格化済み）
        temp : float
            測定温度の初期推定値 (K)
        initial_guess : dict or None
            初期推定値 {'ef_shift': float, 'sigma_total': float, 'temp': float}
        fit_temp : bool
            温度もフィッティングパラメータに含めるか（デフォルト: True）
        use_global_opt : bool
            大域最適化を使用するか（デフォルト: True）

        Returns:
        --------
        result : dict
            フィッティング結果
        """
        energy = self.engine.grid.e_axis

        # 初期推定値
        if initial_guess is None:
            ef_shift_init = 0.0
            sigma_total_init = 0.005  # 5 meV
            temp_init = temp
        else:
            ef_shift_init = initial_guess.get('ef_shift', 0.0)
            sigma_total_init = initial_guess.get('sigma_total', 0.005)
            temp_init = initial_guess.get('temp', temp)

        if fit_temp:
            # 温度もフィッティングパラメータに含める
            def fitting_func(e, ef_shift, sigma_total, temp_fit, amplitude, offset):
                conv = self.fermi_dirac_convolved(e, ef_shift, temp_fit, sigma_total)
                return amplitude * conv + offset

            # パラメータ境界
            bounds = (
                [-0.05, 0.0001, 0.1, 0.5, -0.5],  # 下限
                [0.05, 0.05, 300.0, 2.0, 0.5]     # 上限
            )
            p0 = [ef_shift_init, sigma_total_init, temp_init, 1.0, 0.0]
        else:
            # 温度は固定
            def fitting_func(e, ef_shift, sigma_total, amplitude, offset):
                conv = self.fermi_dirac_convolved(e, ef_shift, temp, sigma_total)
                return amplitude * conv + offset

            bounds = (
                [-0.05, 0.0001, 0.5, -0.5],  # 下限
                [0.05, 0.05, 2.0, 0.5]        # 上限
            )
            p0 = [ef_shift_init, sigma_total_init, 1.0, 0.0]

        try:
            if use_global_opt:
                # ステップ1: Differential Evolutionで大域的探索
                def objective(params):
                    predicted = fitting_func(energy, *params)
                    return np.sum((observed_spectrum - predicted)**2)

                result_de = differential_evolution(
                    objective,
                    bounds=list(zip(bounds[0], bounds[1])),
                    seed=42,
                    maxiter=100,
                    popsize=15,
                    atol=1e-8,
                    tol=1e-8,
                    workers=1
                )
                p0_refined = result_de.x

                # ステップ2: curve_fitで局所最適化（精密化）
                popt, pcov = curve_fit(
                    fitting_func,
                    energy,
                    observed_spectrum,
                    p0=p0_refined,
                    bounds=bounds,
                    maxfev=10000
                )
            else:
                # 局所最適化のみ
                popt, pcov = curve_fit(
                    fitting_func,
                    energy,
                    observed_spectrum,
                    p0=p0,
                    bounds=bounds,
                    maxfev=10000
                )

            if fit_temp:
                ef_shift_fit, sigma_total_fit, temp_fit, amplitude_fit, offset_fit = popt
                perr = np.sqrt(np.diag(pcov))
                temp_error = perr[2]
            else:
                ef_shift_fit, sigma_total_fit, amplitude_fit, offset_fit = popt
                temp_fit = temp
                perr = np.sqrt(np.diag(pcov))
                temp_error = 0.0

            # フィッティング結果の生成
            fitted_spectrum = fitting_func(energy, *popt)

            # フィッティングの品質評価
            residuals = observed_spectrum - fitted_spectrum
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((observed_spectrum - np.mean(observed_spectrum))**2)
            r_squared = 1 - (ss_res / ss_tot)

            return {
                'success': True,
                'ef_shift': ef_shift_fit,
                'ef_shift_error': perr[0],
                'sigma_total': sigma_total_fit,
                'sigma_total_error': perr[1],
                'temp_fit': temp_fit,
                'temp_error': temp_error,
                'amplitude': amplitude_fit,
                'offset': offset_fit,
                'fitted_spectrum': fitted_spectrum,
                'r_squared': r_squared,
                'residuals': residuals,
                'covariance': pcov
            }

        except Exception as e:
            return {
                'success': False,
                'error_message': str(e)
            }

    def calculate_theoretical_resolution(self):
        """
        シミュレータの理論分解能を計算

        Returns:
        --------
        resolution_dict : dict
            各要素の寄与と合計分解能
        """
        # 各要素からの寄与を取得
        # Detector側
        kappa = self.engine.detector.kappa
        theta = self.engine.detector.theta
        sigma_res = self.engine.detector.sigma_res

        # Source側
        alpha = self.engine.source.alpha
        sigma_x = self.engine.source.sigma_x / 1000.0  # meV -> eV
        sigma_y = self.engine.source.sigma_y
        gamma_x = self.engine.source.gamma_x
        gamma_y = self.engine.source.gamma_y
        rotation = self.engine.source.rotation

        # 各要素の分解能への寄与を推定
        # （簡易的な計算: より正確には2Dシミュレーションが必要）

        # Detector intrinsic resolution
        contrib_detector_res = sigma_res

        # Smile curvature contribution (非線形効果の近似)
        # kappaが大きいほどエネルギー分散が増加
        contrib_smile = kappa * 0.01  # 経験的な係数

        # Detector tilt contribution
        contrib_tilt = abs(theta) * 0.001  # 経験的な係数

        # Source size contribution (X方向)
        contrib_source_x = sigma_x

        # Energy gradient contribution
        contrib_gradient = abs(alpha) * sigma_y * 0.1  # 空間とエネルギーのカップリング

        # 非対称性の寄与（RMS的に合算）
        contrib_asymmetry = np.sqrt((gamma_x * 0.0001)**2 + (gamma_y * 0.0001)**2)

        # 合計分解能（各寄与を二乗和平方根で結合）
        total_resolution = np.sqrt(
            contrib_detector_res**2 +
            contrib_smile**2 +
            contrib_tilt**2 +
            contrib_source_x**2 +
            contrib_gradient**2 +
            contrib_asymmetry**2
        )

        return {
            'total_resolution': total_resolution,
            'detector_intrinsic': contrib_detector_res,
            'smile_curvature': contrib_smile,
            'detector_tilt': contrib_tilt,
            'source_size_x': contrib_source_x,
            'energy_gradient': contrib_gradient,
            'asymmetry': contrib_asymmetry
        }

    def estimate_irf_parameters(self, observed_spectrum, temp,
                                 bounds=None,
                                 maxiter=50,
                                 progress_callback=None):
        """
        観測スペクトルからIRFの幾何学的パラメータを推定

        Parameters:
        -----------
        observed_spectrum : array
            ノイズを含む観測スペクトル（規格化済み）
        temp : float
            測定温度 (K)
        bounds : dict or None
            各パラメータの探索範囲を指定する辞書
        maxiter : int
            最適化の最大反復回数
        progress_callback : callable or None
            進捗を通知するコールバック関数 (iteration, loss) を受け取る

        Returns:
        --------
        result : dict
            推定されたパラメータと評価指標を含む辞書
        """

        # デフォルトの探索範囲
        if bounds is None:
            bounds = {
                'kappa': (0.0, 0.1),           # Smile curvature
                'theta': (-0.5, 0.5),          # Detector tilt (deg)
                'sigma_res': (0.0001, 0.010),  # Intrinsic resolution (eV)
                'alpha': (-0.01, 0.01),        # Energy gradient
                'sigma_x': (0.01, 2.0),        # Spot size X (meV)
                'sigma_y': (0.01, 2.0),        # Spot size Y (mm)
                'gamma_x': (-5.0, 5.0),        # Spot skew X
                'gamma_y': (-10.0, 10.0),      # Spot skew Y
                'rotation': (-45.0, 45.0),     # Spot rotation (deg)
            }

        # パラメータ名リストと境界を準備
        param_names = list(bounds.keys())
        bounds_list = [bounds[name] for name in param_names]

        # 目的関数の定義
        def objective(params):
            # パラメータをエンジンに設定
            kappa, theta, sigma_res, alpha, sigma_x, sigma_y, gamma_x, gamma_y, rotation = params

            self.engine.detector.kappa = kappa
            self.engine.detector.theta = theta
            self.engine.detector.sigma_res = sigma_res
            self.engine.source.alpha = alpha
            self.engine.source.sigma_x = sigma_x
            self.engine.source.sigma_y = sigma_y
            self.engine.source.gamma_x = gamma_x
            self.engine.source.gamma_y = gamma_y
            self.engine.source.rotation = rotation

            # シミュレーション実行
            _, y_sim = self.engine.simulate(temp=temp)

            # 規格化
            y_sim = y_sim / (np.max(y_sim) + 1e-12)

            # 損失関数: 平均二乗誤差
            mse = np.mean((observed_spectrum - y_sim)**2)

            return mse

        # Differential Evolution による最適化
        iteration_counter = [0]

        def callback(xk, convergence):
            iteration_counter[0] += 1
            if progress_callback is not None:
                loss = objective(xk)
                progress_callback(iteration_counter[0], loss)

        result = differential_evolution(
            objective,
            bounds_list,
            maxiter=maxiter,
            seed=42,
            callback=callback,
            workers=1,
            updating='deferred',
            polish=True
        )

        # 最適パラメータを取得
        optimal_params = result.x
        param_dict = {name: val for name, val in zip(param_names, optimal_params)}

        # 最適パラメータでシミュレーション
        kappa, theta, sigma_res, alpha, sigma_x, sigma_y, gamma_x, gamma_y, rotation = optimal_params
        self.engine.detector.kappa = kappa
        self.engine.detector.theta = theta
        self.engine.detector.sigma_res = sigma_res
        self.engine.source.alpha = alpha
        self.engine.source.sigma_x = sigma_x
        self.engine.source.sigma_y = sigma_y
        self.engine.source.gamma_x = gamma_x
        self.engine.source.gamma_y = gamma_y
        self.engine.source.rotation = rotation

        _, fitted_spectrum = self.engine.simulate(temp=temp)
        fitted_spectrum = fitted_spectrum / (np.max(fitted_spectrum) + 1e-12)

        # 推定されたIRFを取得
        _, y_step = self.engine.simulate(temp=0.01)
        estimated_irf = -np.gradient(y_step, self.engine.grid.e_axis)
        estimated_irf = estimated_irf / (np.max(np.abs(estimated_irf)) + 1e-12)

        return {
            'parameters': param_dict,
            'fitted_spectrum': fitted_spectrum,
            'estimated_irf': estimated_irf,
            'final_loss': result.fun,
            'success': result.success,
            'message': result.message,
            'nit': result.nit,
            'nfev': result.nfev
        }
