import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# パス設定: 自作パッケージを読み込めるようにする
sys.path.append(os.path.dirname(__file__))

from xps_twin.models.twin_engine import DigitalTwinEngine
from xps_twin.analysis.deconvolution import XPSDeconvolver
from xps_twin.core.physics import fermi_dirac

st.set_page_config(page_title="XPS Digital Twin Lab", layout="wide")

st.title("🔬 XPS Digital Twin & Deconvolution Lab")
st.markdown("装置の幾何学的歪みをパラメタライズし、真の電子状態を復元します。")

# --- サイドバー: パラメータ設定 ---
st.sidebar.header("Instrument Parameters")

# 光源の設定
st.sidebar.subheader("X-ray Source")

# スポットサイズ
sigma_x = st.sidebar.slider(
    "Spot Size X (meV)", 0.01, 2.0, 0.5, format="%.2f",
    help="X線スポットのX方向（エネルギー方向）のサイズ"
)

sigma_y = st.sidebar.slider(
    "Spot Size Y (mm)", 0.01, 2.0, 0.5, format="%.2f",
    help="X線スポットのY方向（空間方向）のサイズ"
)

# 非対称性
gamma_x = st.sidebar.slider(
    "Spot Skew X (gamma_x)", -5.0, 5.0, 0.0, format="%.1f",
    help="X線スポットのX方向の非対称性"
)

gamma_y = st.sidebar.slider(
    "Spot Skew Y (gamma_y)", -10.0, 10.0, 0.0, format="%.1f",
    help="X線スポットのY方向の非対称性"
)

# 回転角度
rotation = st.sidebar.slider(
    "Spot Rotation (deg)", -45.0, 45.0, 0.0, format="%.1f",
    help="X線スポットの回転角度"
)

# エネルギー勾配（最後に配置）
alpha = st.sidebar.slider(
    "Energy Gradient (alpha)", -0.01, 0.01, 0.002, format="%.4f",
    help="試料上のX線スポット内でのエネルギー勾配。スリットを通る位置によるBEのズレを表します。",
    step=0.0001
)

# 検出器の設定
st.sidebar.subheader("2D Detector")
kappa = st.sidebar.slider(
    "Smile Curvature (kappa)", 0.0, 0.2, 0.01, format="%.3f",
    help="アナライザーの収差による『スマイル歪み』。積分後に高BE側へ伸びる非対称な裾野を作ります。",
    step=0.001
)
theta = st.sidebar.slider(
    "Detector Tilt (theta deg)", -1.0, 1.0, 0.08, format="%.2f",
    help="検出器（カメラ）の取り付け角度の微細なズレ。エッジ全体を対称にブロードにします。"
)
sigma_res_mev = st.sidebar.slider(
    "Intrinsic Res (sigma meV)", 0.1, 10.0, 1.5, format="%.1f",
    help="電子光学系や検出器ピクセルサイズに起因する、装置固有の対称な分解能。"
) / 1000.0

# ノイズ設定
st.sidebar.subheader("Detector Noise")
# ログスケールスライダー: 10^(-5) から 10^3 まで (0.00001 から 1000)
poisson_log = st.sidebar.slider(
    "Poisson Noise Level (log₁₀)", -5.0, 3.0, 0.3, format="%.2f", step=0.01,
    help="光子計数統計ノイズ（対数スケール）。値が大きいほどノイズが大きい。信号強度に依存するショットノイズ。-5.0以下でノイズゼロ。"
)
poisson_noise = 10 ** poisson_log

# Poissonの実際の値を表示
if poisson_noise < 0.01:
    st.sidebar.caption(f"実際の値: {poisson_noise:.2e}")
else:
    st.sidebar.caption(f"実際の値: {poisson_noise:.4f}")

gaussian_noise = st.sidebar.slider(
    "Gaussian Readout Noise (%)", 0.0, 10.0, 1.0, format="%.1f",
    help="検出器の読み出しノイズ。信号強度に依存しない一定のノイズ。"
)

# ノイズゼロの判定を表示（両方がゼロの場合）
if poisson_noise <= 1e-5 and gaussian_noise == 0.0:
    st.sidebar.info("ノイズゼロ")

# 測定条件
st.sidebar.subheader("Measurement")
temp = st.sidebar.slider("Temperature (K)", 0.1, 300.0, 5.0)

# --- メイン計算エンジン ---
engine = DigitalTwinEngine(e_range=(-0.1, 0.1), e_steps=500)
engine.source.sigma_x = sigma_x
engine.source.sigma_y = sigma_y
engine.source.alpha = alpha
engine.source.gamma_x = gamma_x
engine.source.gamma_y = gamma_y
engine.source.rotation = rotation
engine.detector.kappa = kappa
engine.detector.theta = theta
engine.detector.sigma_res = sigma_res_mev  # meV -> eV

# --- メイン計算エンジン ---
# (前略: engineのパラメータセットまで)

# 1Dシミュレーション実行
x, y_obs = engine.simulate(temp=temp)

# 【追加】最大値を1に規格化 (分母が0にならないよう微小値を加算)
y_obs = y_obs / (np.max(y_obs) + 1e-12)

# 【追加】IRF（装置関数）の計算
# 温度をほぼ0（0.01K等）にしてシミュレートしたエッジを微分する
_, y_step = engine.simulate(temp=0.01)
y_irf = np.gradient(y_step, x)
y_irf = -y_irf / (np.max(np.abs(y_irf)) + 1e-12) # 微分して規格化（符号反転はBE方向のため）

# --- 表示エリアの分割 ---
# app.py (描画・レイアウト部分)

# --- 表示エリアの分割 (左1.5 : 右1 の比率) ---
col_main, col_sub = st.columns([1.5, 1])

with col_main:
    # --- 1段目: 1D Spectrum ---
    st.subheader("1D Spectrum Simulation")
    # シミュレーションと規格化
    x, y_obs_raw = engine.simulate(temp=temp)
    y_obs_clean = y_obs_raw / (np.max(y_obs_raw) + 1e-12)

    # ノイズを追加（Poisson + Gaussian）
    # 閾値（10^(-5)）以下では完全にノイズゼロ
    if poisson_noise > 1e-5:
        # Poissonノイズ: 信号強度に依存（sqrt(signal)に比例）
        # poisson_noiseが大きいほどノイズが大きくなるよう、逆数を使用
        scale_factor = 1000.0 / poisson_noise  # 値が大きいほどscale_factorは小さくなる
        poisson_component = np.random.poisson(y_obs_clean * scale_factor) / scale_factor
    else:
        # Poissonノイズが閾値以下の場合は元の信号をそのまま使用（ノイズゼロ）
        poisson_component = y_obs_clean.copy()

    # Gaussianノイズ: 信号強度に依存しない
    gaussian_std = gaussian_noise / 100.0
    gaussian_component = np.random.normal(0, gaussian_std, len(y_obs_clean))

    # 両方のノイズを組み合わせ
    y_obs = poisson_component + gaussian_component
    # 負の値を0にクリップ
    y_obs = np.clip(y_obs, 0, None)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    true_fd = fermi_dirac(x, temp)
    ax1.plot(x*1000, true_fd, 'k--', alpha=0.4, label="Ideal Fermi-Dirac")
    ax1.plot(x*1000, y_obs_clean, 'b-', alpha=0.3, linewidth=1.5, label="Clean Observed")
    ax1.plot(x*1000, y_obs, 'r-', linewidth=2, label="Observed (with noise)")
    ax1.set_xlabel("Energy (meV)")
    ax1.set_ylabel("Normalized Intensity")
    ax1.legend()
    st.pyplot(fig1)

    # --- 2段目: Instrumental Function (IRF) ---
    st.subheader("Instrumental Function (IRF)")
    # IRFの抽出と規格化
    _, y_step = engine.simulate(temp=0.01)
    y_irf = -np.gradient(y_step, x) # BE方向への微分
    y_irf = y_irf / (np.max(np.abs(y_irf)) + 1e-12)

    fig_irf, ax_irf = plt.subplots(figsize=(8, 3))
    ax_irf.fill_between(x*1000, y_irf, color='blue', alpha=0.2)
    ax_irf.plot(x*1000, y_irf, 'b-', label="Pure IRF")
    ax_irf.set_xlabel("Energy (meV)")
    ax_irf.set_ylabel("Intensity")
    ax_irf.legend()
    st.pyplot(fig_irf)

    # --- 3段目: Deconvolution Section ---
    st.write("") # スペース空け
    st.subheader("Deconvolution Analysis")

    # タブで2つの機能を分ける
    tab1, tab2 = st.tabs(["Fermi Edge Fitting", "IRF Parameter Estimation"])

    with tab1:
        st.markdown("**フェルミエッジフィッティング**: 観測スペクトルをFermi-Dirac + Gaussianでフィッティングし、Ef shiftと合計分解能を求めます")

        run_fermi_fit = st.button("📊 Run Fermi Edge Fit", use_container_width=True, key="fermi_fit")

    with tab2:
        st.markdown("**IRF逆推定**: 観測スペクトルからIRFの幾何学的パラメータを逆推定します（計算時間がかかります）")
        col_a, col_b = st.columns(2)
        with col_a:
            maxiter = st.number_input("最適化反復回数", min_value=10, max_value=200, value=30, step=10)
        with col_b:
            st.markdown("<br>", unsafe_allow_html=True)  # スペース調整
        run_param_estimation = st.button("🔍 Estimate IRF Parameters", use_container_width=True, key="param_est")

with col_sub:
    # --- 右列上部: 2D Spot Profile (純粋なスポット形状) ---
    st.subheader("2D Spot Profile")
    spot_profile = engine.source.get_2d_spot_profile(engine.grid)

    # Y軸を物理空間（mm）に変換
    y_mm_min = engine.grid.y_axis[0] * 0.5
    y_mm_max = engine.grid.y_axis[-1] * 0.5

    fig_spot, ax_spot = plt.subplots(figsize=(5, 5))
    im_spot = ax_spot.imshow(spot_profile, aspect='auto',
                             extent=[x[0]*1000, x[-1]*1000, y_mm_min, y_mm_max],
                             cmap='hot')
    plt.colorbar(im_spot, ax=ax_spot, orientation='horizontal', pad=0.08)
    ax_spot.set_xlabel("Energy (meV)")
    ax_spot.set_ylabel("Y Position (mm)")
    ax_spot.set_title(f"Spot (σx={sigma_x:.2f}meV, σy={sigma_y:.2f}mm, θ={rotation:.0f}°)")
    st.pyplot(fig_spot)

    # --- 右列下部: 2D Detector Image ---
    st.subheader("2D Detector Image")
    img_2d_source = engine.source.generate_2d_emission(engine.grid, fermi_dirac(x, temp))

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    im = ax2.imshow(img_2d_source, aspect='auto', extent=[x[0]*1000, x[-1]*1000, y_mm_min, y_mm_max], cmap='viridis')
    plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.08)
    ax2.set_xlabel("Energy (meV)")
    ax2.set_ylabel("Y Position (mm)")
    ax2.set_title("After Detector")
    st.pyplot(fig2)

# --- 最下段: フェルミエッジフィッティング結果の表示 ---
if run_fermi_fit:
    st.divider()
    st.subheader("Fermi Edge Fitting Result")

    # シミュレートされたデータを使用
    deconvolver = XPSDeconvolver(engine)
    observed_for_fit = y_obs.copy()
    x_for_plot = x
    st.info("🔬 シミュレートされたデータを使用してフィッティングします")

    with st.spinner("フィッティング実行中..."):
        # フェルミエッジフィッティング
        fit_result = deconvolver.fit_fermi_edge(observed_for_fit, temp=temp)

    if fit_result['success']:
        st.success("フィッティング成功！")

        # フィッティング結果の表示
        col_fit1, col_fit2, col_fit3, col_fit4 = st.columns(4)

        with col_fit1:
            st.metric(
                "Fermi Energy Shift",
                f"{fit_result['ef_shift']*1000:.3f} meV",
                help=f"誤差: ±{fit_result['ef_shift_error']*1000:.3f} meV"
            )

        with col_fit2:
            st.metric(
                "Total Resolution (FWHM)",
                f"{fit_result['sigma_total']*2.355*1000:.2f} meV",
                help=f"Gaussian σ: {fit_result['sigma_total']*1000:.3f} meV (誤差: ±{fit_result['sigma_total_error']*1000:.3f} meV)"
            )

        with col_fit3:
            st.metric(
                "Temperature (Fitted)",
                f"{fit_result['temp_fit']:.2f} K",
                delta=f"{fit_result['temp_fit']-temp:.2f} K",
                help=f"初期値: {temp:.2f} K | 誤差: ±{fit_result['temp_error']:.3f} K"
            )

        with col_fit4:
            st.metric(
                "Fit Quality (R²)",
                f"{fit_result['r_squared']:.6f}",
                help="決定係数（1に近いほど良好）"
            )

        # 理論分解能との比較
        st.subheader("理論分解能との比較")
        theory_res = deconvolver.calculate_theoretical_resolution()

        col_theory1, col_theory2 = st.columns(2)

        with col_theory1:
            st.markdown("**シミュレータの理論分解能**")
            st.metric("Total (FWHM)", f"{theory_res['total_resolution']*2.355*1000:.2f} meV")
            st.caption(f"σ = {theory_res['total_resolution']*1000:.3f} meV")

            # 各成分の寄与を表示
            with st.expander("各成分の寄与を見る"):
                st.write(f"- Detector Intrinsic: {theory_res['detector_intrinsic']*1000:.3f} meV")
                st.write(f"- Smile Curvature: {theory_res['smile_curvature']*1000:.3f} meV")
                st.write(f"- Detector Tilt: {theory_res['detector_tilt']*1000:.3f} meV")
                st.write(f"- Source Size (X): {theory_res['source_size_x']*1000:.3f} meV")
                st.write(f"- Energy Gradient: {theory_res['energy_gradient']*1000:.3f} meV")
                st.write(f"- Asymmetry: {theory_res['asymmetry']*1000:.3f} meV")

        with col_theory2:
            st.markdown("**フィッティングから求めた分解能**")
            st.metric("Total (FWHM)", f"{fit_result['sigma_total']*2.355*1000:.2f} meV")
            st.caption(f"σ = {fit_result['sigma_total']*1000:.3f} meV")

            # 差異を計算
            diff = abs(fit_result['sigma_total'] - theory_res['total_resolution'])
            rel_diff = (diff / theory_res['total_resolution']) * 100
            st.metric("差異", f"{diff*1000:.3f} meV", delta=f"{rel_diff:.1f}%")

        # スペクトルのフィッティング結果プロット
        st.subheader("フィッティング結果")
        fig_fit, ax_fit = plt.subplots(figsize=(12, 5))

        ax_fit.plot(x_for_plot*1000, y_obs_clean, 'b-', alpha=0.3, linewidth=1.5, label="Clean Observed")
        ax_fit.plot(x_for_plot*1000, observed_for_fit, color='gray', alpha=0.6, linewidth=1.5, label="Observed (with noise)")
        ax_fit.plot(x_for_plot*1000, fit_result['fitted_spectrum'], 'r-', linewidth=2, label=f"Fitted (σ={fit_result['sigma_total']*1000:.2f} meV)")
        ax_fit.plot(x_for_plot*1000, true_fd, 'k--', alpha=0.4, linewidth=2, label="True Fermi-Dirac")

        ax_fit.set_xlabel("Energy (meV)")
        ax_fit.set_ylabel("Normalized Intensity")
        ax_fit.legend()
        ax_fit.grid(alpha=0.3)
        st.pyplot(fig_fit)

        # 残差プロット
        st.subheader("フィッティング残差")
        fig_res, ax_res = plt.subplots(figsize=(12, 3))
        ax_res.plot(x_for_plot*1000, fit_result['residuals'], 'g-', alpha=0.7, linewidth=1)
        ax_res.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax_res.set_xlabel("Energy (meV)")
        ax_res.set_ylabel("Residuals")
        ax_res.grid(alpha=0.3)
        st.pyplot(fig_res)

    else:
        st.error(f"フィッティング失敗: {fit_result.get('error_message', 'Unknown error')}")

# --- IRFパラメータ推定の実行 ---
if run_param_estimation:
    st.divider()
    st.subheader("IRF Parameter Estimation Result")

    # 新しいエンジンインスタンスを作成（推定用）
    estimation_engine = DigitalTwinEngine(e_range=(-0.1, 0.1), e_steps=500)

    # 進捗表示
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_text = st.empty()

    # 進捗コールバック
    def progress_callback(iteration, loss):
        progress = min(iteration / maxiter, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Iteration: {iteration}/{maxiter}")
        loss_text.text(f"Current Loss (MSE): {loss:.6e}")

    # デコンボルバーを作成
    deconvolver = XPSDeconvolver(estimation_engine)

    # 観測スペクトル（ノイズ付き）を使用
    observed_for_estimation = y_obs.copy()

    with st.spinner("最適化実行中... しばらくお待ちください"):
        # パラメータ推定を実行
        result = deconvolver.estimate_irf_parameters(
            observed_for_estimation,
            temp=temp,
            maxiter=int(maxiter),
            progress_callback=progress_callback
        )

    progress_bar.progress(1.0)
    status_text.text("最適化完了！")

    # 結果の表示
    st.success(f"最適化成功: {result['success']}")
    st.info(f"最終損失: {result['final_loss']:.6e} | 反復回数: {result['nit']} | 関数評価回数: {result['nfev']}")

    # 推定されたパラメータを表で表示
    st.subheader("推定されたIRFパラメータ")
    params = result['parameters']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Detector Parameters**")
        st.metric("Kappa (Smile)", f"{params['kappa']:.5f}")
        st.metric("Theta (Tilt, deg)", f"{params['theta']:.3f}")
        st.metric("Sigma_res (eV)", f"{params['sigma_res']:.6f}")

    with col2:
        st.markdown("**Source Parameters**")
        st.metric("Alpha (Energy Gradient)", f"{params['alpha']:.6f}")
        st.metric("Sigma_X (meV)", f"{params['sigma_x']:.3f}")
        st.metric("Sigma_Y (mm)", f"{params['sigma_y']:.3f}")

    with col3:
        st.markdown("**Asymmetry Parameters**")
        st.metric("Gamma_X", f"{params['gamma_x']:.2f}")
        st.metric("Gamma_Y", f"{params['gamma_y']:.2f}")
        st.metric("Rotation (deg)", f"{params['rotation']:.2f}")

    # スペクトルのフィッティング結果
    st.subheader("フィッティング結果")
    fig_fit, ax_fit = plt.subplots(figsize=(12, 5))
    ax_fit.plot(x*1000, y_obs_clean, 'b-', alpha=0.3, linewidth=1.5, label="Clean Observed")
    ax_fit.plot(x*1000, observed_for_estimation, color='gray', alpha=0.6, linewidth=1.5, label="Observed (with noise)")
    ax_fit.plot(x*1000, result['fitted_spectrum'], 'r-', linewidth=2, label="Fitted (Estimated IRF)")
    ax_fit.plot(x*1000, true_fd, 'k--', alpha=0.4, linewidth=2, label="True Fermi-Dirac")
    ax_fit.set_xlabel("Energy (meV)")
    ax_fit.set_ylabel("Normalized Intensity")
    ax_fit.legend()
    ax_fit.grid(alpha=0.3)
    st.pyplot(fig_fit)

    # 推定されたIRFと真のIRFの比較
    st.subheader("推定されたIRF vs 真のIRF")
    fig_irf_comp, ax_irf_comp = plt.subplots(figsize=(12, 4))
    ax_irf_comp.fill_between(x*1000, y_irf, color='blue', alpha=0.2, label="True IRF")
    ax_irf_comp.plot(x*1000, y_irf, 'b-', linewidth=2, label="True IRF")
    ax_irf_comp.plot(x*1000, result['estimated_irf'], 'r--', linewidth=2, label="Estimated IRF")
    ax_irf_comp.set_xlabel("Energy (meV)")
    ax_irf_comp.set_ylabel("Intensity")
    ax_irf_comp.legend()
    ax_irf_comp.grid(alpha=0.3)
    st.pyplot(fig_irf_comp)

    # パラメータ比較表（真の値 vs 推定値）
    st.subheader("パラメータ比較（真の値 vs 推定値）")
    true_params = {
        'kappa': kappa,
        'theta': theta,
        'sigma_res': sigma_res_mev / 1000.0,
        'alpha': alpha,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'gamma_x': gamma_x,
        'gamma_y': gamma_y,
        'rotation': rotation
    }

    comparison_data = []
    for param_name in params.keys():
        true_val = true_params[param_name]
        est_val = params[param_name]
        error = abs(est_val - true_val)
        rel_error = (error / (abs(true_val) + 1e-12)) * 100
        comparison_data.append({
            "Parameter": param_name,
            "True Value": f"{true_val:.6f}",
            "Estimated Value": f"{est_val:.6f}",
            "Absolute Error": f"{error:.6f}",
            "Relative Error (%)": f"{rel_error:.2f}"
        })

    import pandas as pd
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)