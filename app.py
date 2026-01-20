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

st.set_page_config(page_title="XPS_IRF_Simulator", layout="wide")

# --- 言語切り替え / Language Toggle ---
lang = st.sidebar.radio("🌐 Language / 言語", ["English", "日本語"], horizontal=True)
is_en = (lang == "English")

# --- Translation Dictionary ---
T = {
    # Main title and description
    "title": "🔬 XPS IRF Simulator & Resolution Explorer",
    "subtitle": {
        True: "Parametrize geometric distortions of instruments and visualize IRF asymmetry.",
        False: "装置の幾何学的歪みをパラメタライズし、IRFの非対称性を可視化します。"
    },

    # Sidebar: Philosophy section
    "philosophy_title": {
        True: "📖 Philosophy of This Simulator",
        False: "📖 このシミュレータの思想"
    },
    "philosophy_content": {
        True: """
### Why We Built This Simulator

Modern synchrotron facilities and analyzers have achieved extremely high energy resolution.
As a result, **approximating the instrumental response function (IRF) with a Gaussian to evaluate resolution** has become "standard practice."

However, this approach has **limitations**:

#### Problems with Gaussian Approximation

1. **Real IRFs are not symmetric**
   - X-ray spot asymmetry
   - Detector smile distortion (parabolic aberration)
   - Slight misalignment of detector tilt angle

2. **Alignment errors become invisible**
   - Multiple aberrations overlap, appearing as a "broadened Gaussian"
   - Individual aberration contributions cannot be separated
   - Optimization direction becomes unclear

#### Purpose of This Simulator

**Develop an intuitive feel for how instrument parameters affect spectra**

- Adjust parameters and **experience spectral shape changes in real-time**
- Visualize **which parameters cause deviations from Gaussian** (asymmetry)
- **Quantitatively compare** "ideal" vs "reality"

#### Educational Value

Through this tool:
- **Move beyond the simplistic understanding** that resolution = Gaussian FWHM
- **Intuitively understand** how geometric arrangement affects spectroscopic performance
- Enable **more precise experimental planning** considering IRF asymmetry components
""",
        False: """
### なぜこのシミュレータを作ったのか

現代の放射光施設やアナライザーは、エネルギー分解能が極めて高くなりました。
その結果、**装置関数（IRF）をガウシアンで近似して分解能を評価する**ことが
「当たり前」になってしまいました。

しかし、これには**限界**があります：

#### ガウシアン近似の問題点

1. **実際のIRFは対称ではない**
   - X線スポットの非対称性
   - 検出器のSmile歪み（放物線状の収差）
   - 検出器の取り付け角度（Tilt）のわずかなズレ

2. **アライメントミスの影響が見えなくなる**
   - 複数の収差が重なると、見かけ上「太ったガウシアン」に見える
   - 個々の収差成分の寄与が分離できない
   - 最適化の方向性が分からなくなる

#### このシミュレータの目的

**「肌感覚」で装置パラメータの影響を理解する**

- 各パラメータを動かして、スペクトル形状の変化を**リアルタイムで体感**
- ガウシアンからのズレ（非対称性）が**どのパラメータに起因するか**を可視化
- 「理想」と「現実」の差を**定量的に比較**

#### 教育的価値

このツールを通じて：
- 分解能＝ガウシアンFWHMという**安易な理解からの脱却**
- 装置の幾何学的配置が分光性能に与える影響の**直感的理解**
- IRFの非対称成分を意識した**より精密な実験計画**

が可能になります。
"""
    },

    # Sidebar: X-ray Source
    "xray_source_model_title": {
        True: "💡 X-ray Source Physics Model",
        False: "💡 X線源の物理モデル"
    },
    "xray_source_model_content": {
        True: """
**2D Profile of X-ray Spot**

At synchrotron beamlines, the X-ray spot is not perfectly circular,
but has an elliptical or asymmetric shape.

```
I(x,y) = A × exp(-x²/2σx² - y²/2σy²) × [1 + erf(γx·x)] × [1 + erf(γy·y)]
```

- **σx**: Energy-direction spread → Direct resolution degradation
- **σy**: Spatial-direction spread → Affects resolution when combined with α
- **γx, γy**: Skewness → Spot asymmetry

---

#### What is Energy Gradient (α)?

In the **energy dispersion direction** of the spectrometer, the X-ray energy
varies slightly depending on position on the sample.

```
E(y) = E₀ + α × y
```

**Physical Origins**:
- Energy spread due to finite size of monochromator crystal
- Optical path difference at entrance slit position
- Slight curvature of crystal surface

**Why is it important?**

Even if σx is small, resolution degrades when **σy × α** is large.
This is a **hidden resolution degradation factor invisible to Gaussian FWHM**,
where spatial spread is converted to energy dispersion.

👉 **Example**: σy=1mm, α=0.005 → Effective 5meV additional broadening
""",
        False: """
**X線スポットの2次元プロファイル**

放射光ビームラインでは、X線スポットは完全な円形ではなく、
楕円形や非対称な形状を持ちます。

```
I(x,y) = A × exp(-x²/2σx² - y²/2σy²) × [1 + erf(γx·x)] × [1 + erf(γy·y)]
```

- **σx**: エネルギー方向の広がり → 直接的な分解能劣化
- **σy**: 空間方向の広がり → αと結合して分解能に影響
- **γx, γy**: 歪度（skewness）→ スポットの非対称性

---

#### Energy Gradient (α) とは？

分光器の**エネルギー分散方向**において、試料上の位置によって
X線のエネルギーがわずかに異なる現象です。

```
E(y) = E₀ + α × y
```

**物理的起源**:
- 分光結晶の有限サイズによるエネルギー広がり
- 入射スリット位置による光路差
- 結晶面の微小な湾曲

**なぜ重要か？**

σxが小さくても、**σy × α** の積が大きいと分解能が劣化します。
空間方向の広がりがエネルギー分散に変換される効果で、
**ガウシアンFWHMだけでは見えない隠れた分解能劣化要因**です。

👉 **例**: σy=1mm, α=0.005 の場合、実効的に 5meV の追加ブロードニング
"""
    },

    # Slider help texts
    "help_sigma_x": {
        True: "X-ray spot size in X direction (energy direction)",
        False: "X線スポットのX方向（エネルギー方向）のサイズ"
    },
    "help_sigma_y": {
        True: "X-ray spot size in Y direction (spatial direction)",
        False: "X線スポットのY方向（空間方向）のサイズ"
    },
    "help_gamma_x": {
        True: "X-ray spot asymmetry in X direction",
        False: "X線スポットのX方向の非対称性"
    },
    "help_gamma_y": {
        True: "X-ray spot asymmetry in Y direction",
        False: "X線スポットのY方向の非対称性"
    },
    "help_alpha": {
        True: "Energy gradient within X-ray spot on sample. Represents BE shift depending on slit position.",
        False: "試料上のX線スポット内でのエネルギー勾配。スリットを通る位置によるBEのズレを表します。"
    },

    # Sidebar: 2D Detector
    "detector_model_title": {
        True: "💡 2D Detector Physics Model",
        False: "💡 2D検出器の物理モデル"
    },
    "detector_model_content": {
        True: """
**Geometric Aberrations of Electron Analyzer**

The combination of hemispherical analyzer and 2D detector
introduces deviations from ideal point-to-point mapping.

#### Smile Distortion (κ: kappa)

On the detector, iso-energy lines become **parabolically curved**:

```
ΔE(y) = κ × y²
```

- No distortion at detector center (y=0)
- Energy shift increases toward edges
- After Y-integration, **asymmetric tail toward higher BE** appears

👉 **This is a typical example that Gaussian cannot evaluate!**

#### Tilt (θ: theta)

Slight misalignment of detector mounting angle:

```
ΔE(y) = tan(θ) × y
```

- Iso-energy lines tilt (linear shift)
- After Y-integration, **symmetric broadening** occurs
- Unlike κ, symmetric so difficult to distinguish

#### Intrinsic Resolution (σres)

**Pure Gaussian component** from electron optics and pixel size.
This is the only component that behaves as an "ideal Gaussian."
""",
        False: """
**電子アナライザーの幾何学的収差**

半球型アナライザーと2D検出器の組み合わせでは、
理想的な点対点写像からのズレが生じます。

#### Smile歪み（κ: kappa）

検出器上で、等エネルギー線が**放物線状に曲がる**現象：

```
ΔE(y) = κ × y²
```

- 検出器中心（y=0）では歪みなし
- 端に行くほどエネルギーシフトが増大
- Y方向に積分すると**高BE側に非対称な裾**が出現

👉 **これが「ガウシアンで評価できない」典型例！**

#### Tilt（θ: theta）

検出器の取り付け角度のわずかなズレ：

```
ΔE(y) = tan(θ) × y
```

- 等エネルギー線が傾く（線形シフト）
- Y方向に積分すると**対称的なブロードニング**
- κと異なり、対称なので見分けにくい

#### 固有分解能（σres）

電子光学系・ピクセルサイズに起因する**純粋なガウシアン成分**。
これだけが「理想的なガウシアン」として振る舞います。
"""
    },
    "help_kappa": {
        True: "Smile distortion from analyzer aberration. Creates asymmetric tail toward higher BE after integration.",
        False: "アナライザーの収差による『スマイル歪み』。積分後に高BE側へ伸びる非対称な裾野を作ります。"
    },
    "help_theta": {
        True: "Slight misalignment of detector (camera) mounting angle. Symmetrically broadens the entire edge.",
        False: "検出器（カメラ）の取り付け角度の微細なズレ。エッジ全体を対称にブロードにします。"
    },
    "help_sigma_res": {
        True: "Symmetric resolution intrinsic to instrument from electron optics and detector pixel size.",
        False: "電子光学系や検出器ピクセルサイズに起因する、装置固有の対称な分解能。"
    },

    # Sidebar: Noise
    "noise_model_title": {
        True: "💡 Physical Origins of Noise",
        False: "💡 ノイズの物理的起源"
    },
    "noise_model_content": {
        True: """
**Detector noise has two origins**

---

#### Poisson Noise (Shot Noise)

Statistical noise from **discrete arrival** of photons and electrons.

```
σ_poisson ∝ √N (N = photon count)
```

**Characteristics**:
- Signal-dependent (brighter areas have more noise)
- Fundamentally unavoidable (quantum mechanical limit)
- S/N ratio improves with integration time (∝ √t)

**Physical Origins**:
- X-ray photon emission is a random process
- Photoelectron generation probability is also statistical

---

#### Gaussian Noise (Readout Noise)

Noise from detector **electronic circuits**.

```
σ_gaussian = constant (independent of signal intensity)
```

**Characteristics**:
- Signal-independent constant noise
- Depends on detector quality and temperature
- Can be reduced by cooling or circuit design

**Physical Origins**:
- Thermal noise in CCD/CMOS readout circuits
- Quantization error during A/D conversion
- Amplifier noise

---

👉 **Gaussian noise dominates at low signal, Poisson noise dominates at high signal**
""",
        False: """
**検出器ノイズには2つの起源があります**

---

#### Poisson ノイズ（ショットノイズ）

光子や電子の**離散的な到着**に起因する統計ノイズ。

```
σ_poisson ∝ √N （Nは光子数）
```

**特徴**:
- 信号強度に依存（明るい部分ほどノイズも大きい）
- 根本的に除去不可能（量子力学的限界）
- 積算時間を増やすとS/N比が改善（√t に比例）

**物理的起源**:
- X線光子の放出はランダム過程
- 光電子の発生確率も確率的

---

#### Gaussian ノイズ（読み出しノイズ）

検出器の**電子回路**に起因するノイズ。

```
σ_gaussian = 一定（信号強度に依存しない）
```

**特徴**:
- 信号強度に依存しない一定のノイズ
- 検出器の品質・温度に依存
- 冷却や回路設計で低減可能

**物理的起源**:
- CCD/CMOSの読み出し回路の熱雑音
- A/D変換時の量子化誤差
- アンプの増幅ノイズ

---

👉 **低信号領域ではGaussianノイズが支配的、高信号領域ではPoissonノイズが支配的**
"""
    },
    "help_poisson": {
        True: "Photon counting statistical noise (log scale). Larger values mean more noise. Shot noise dependent on signal intensity. Noise-free below -5.0.",
        False: "光子計数統計ノイズ（対数スケール）。値が大きいほどノイズが大きい。信号強度に依存するショットノイズ。-5.0以下でノイズゼロ。"
    },
    "help_gaussian": {
        True: "Detector readout noise. Constant noise independent of signal intensity.",
        False: "検出器の読み出しノイズ。信号強度に依存しない一定のノイズ。"
    },
    "actual_value": {
        True: "Actual value",
        False: "実際の値"
    },
    "noise_zero": {
        True: "Noise Zero",
        False: "ノイズゼロ"
    },

    # Sidebar: Asymmetry section
    "asymmetry_title": {
        True: "⚠️ Asymmetry and Alignment Errors",
        False: "⚠️ 非対称性とアライメントミス"
    },
    "asymmetry_content": {
        True: """
### Why Asymmetry Matters

In high-resolution XPS, slight alignment errors produce
**asymmetric IRFs that Gaussian approximation cannot explain**.

#### Main Causes of Asymmetry

| Parameter | Effect | Impact on IRF |
|-----------|--------|---------------|
| κ (Smile) | Parabolic distortion | Tail toward higher BE |
| γx, γy | Spot skewness | Asymmetric peak |
| α × σy | Gradient × spatial spread | Symmetric broadening |

#### Symmetric Components (Gaussian-evaluable)

| Parameter | Effect |
|-----------|--------|
| σx | Energy-direction spread |
| θ (Tilt) | Linear broadening |
| σres | Intrinsic resolution |

#### How to Verify in Experiments

1. **Fermi edge measurement**: Lower temperature to remove thermal broadening
2. **Residual analysis**: Check asymmetric patterns in fit residuals
3. **This simulator**: Reproduce shape by changing parameters

👉 **Look at the IRF "shape" to understand where the problem lies!**
""",
        False: """
### なぜ非対称性が問題なのか

高分解能XPSでは、わずかなアライメントミスが
**ガウシアン近似では説明できない非対称なIRF**を生み出します。

#### 主な非対称性の原因

| パラメータ | 効果 | IRFへの影響 |
|-----------|------|-------------|
| κ (Smile) | 放物線歪み | 高BE側に裾 |
| γx, γy | スポット歪度 | 非対称ピーク |
| α × σy | 勾配×空間広がり | 対称ブロードニング |

#### 対称成分（ガウシアンで評価可能）

| パラメータ | 効果 |
|-----------|------|
| σx | エネルギー方向広がり |
| θ (Tilt) | 線形ブロードニング |
| σres | 固有分解能 |

#### 実験での確認方法

1. **フェルミエッジ測定**: 温度を下げて熱ブロードニングを除去
2. **残差解析**: フィット残差の非対称パターンを確認
3. **このシミュレータ**: パラメータを変えて形状を再現

👉 **IRFの「形」を見れば、どこに問題があるか分かります！**
"""
    },

    # Main content
    "fermi_edge_fitting_desc": {
        True: "**Fermi Edge Fitting**: Fit observed spectrum with Fermi-Dirac + Gaussian to obtain Ef shift and total resolution",
        False: "**フェルミエッジフィッティング**: 観測スペクトルをFermi-Dirac + Gaussianでフィッティングし、Ef shiftと合計分解能を求めます"
    },
    "irf_estimation_desc": {
        True: "**IRF Inverse Estimation**: Estimate geometric IRF parameters from observed spectrum (computation intensive)",
        False: "**IRF逆推定**: 観測スペクトルからIRFの幾何学的パラメータを逆推定します（計算時間がかかります）"
    },
    "optimization_iterations": {
        True: "Optimization Iterations",
        False: "最適化反復回数"
    },

    # Fitting results
    "using_simulated_data": {
        True: "🔬 Using simulated data for fitting",
        False: "🔬 シミュレートされたデータを使用してフィッティングします"
    },
    "fitting_in_progress": {
        True: "Fitting in progress...",
        False: "フィッティング実行中..."
    },
    "fitting_success": {
        True: "Fitting successful!",
        False: "フィッティング成功！"
    },
    "fitting_failed": {
        True: "Fitting failed",
        False: "フィッティング失敗"
    },
    "help_ef_shift_error": {
        True: "Error",
        False: "誤差"
    },
    "help_sigma_error": {
        True: "Gaussian σ",
        False: "Gaussian σ"
    },
    "help_temp_error": {
        True: "Initial value",
        False: "初期値"
    },
    "help_r_squared": {
        True: "Coefficient of determination (closer to 1 is better)",
        False: "決定係数（1に近いほど良好）"
    },
    "theoretical_resolution_comparison": {
        True: "Comparison with Theoretical Resolution",
        False: "理論分解能との比較"
    },
    "simulator_theoretical_resolution": {
        True: "**Simulator Theoretical Resolution**",
        False: "**シミュレータの理論分解能**"
    },
    "fitted_resolution": {
        True: "**Resolution from Fitting**",
        False: "**フィッティングから求めた分解能**"
    },
    "view_component_contributions": {
        True: "View component contributions",
        False: "各成分の寄与を見る"
    },
    "difference": {
        True: "Difference",
        False: "差異"
    },
    "fitting_result": {
        True: "Fitting Result",
        False: "フィッティング結果"
    },
    "fitting_residuals": {
        True: "Fitting Residuals",
        False: "フィッティング残差"
    },

    # IRF Parameter Estimation
    "optimization_in_progress": {
        True: "Optimization in progress... Please wait",
        False: "最適化実行中... しばらくお待ちください"
    },
    "optimization_complete": {
        True: "Optimization complete!",
        False: "最適化完了！"
    },
    "optimization_success": {
        True: "Optimization success",
        False: "最適化成功"
    },
    "final_loss": {
        True: "Final loss",
        False: "最終損失"
    },
    "iterations": {
        True: "Iterations",
        False: "反復回数"
    },
    "function_evaluations": {
        True: "Function evaluations",
        False: "関数評価回数"
    },
    "estimated_irf_parameters": {
        True: "Estimated IRF Parameters",
        False: "推定されたIRFパラメータ"
    },
    "estimated_vs_true_irf": {
        True: "Estimated IRF vs True IRF",
        False: "推定されたIRF vs 真のIRF"
    },
    "parameter_comparison": {
        True: "Parameter Comparison (True vs Estimated)",
        False: "パラメータ比較（真の値 vs 推定値）"
    },

    # Resolution Summary
    "resolution_summary_title": {
        True: "📊 Resolution Summary",
        False: "📊 分解能サマリー"
    },
    "source_resolution_label": {
        True: "Source Resolution",
        False: "ソース分解能"
    },
    "detector_resolution_label": {
        True: "Detector Resolution",
        False: "検出器分解能"
    },
    "combined_resolution_label": {
        True: "Combined Resolution",
        False: "合成分解能"
    },
    "resolution_formula_note": {
        True: "σ = √(σ_source² + σ_detector²)",
        False: "σ = √(σ_source² + σ_detector²)"
    },
}

# Helper function to get translated text
def t(key):
    """Get translated text for the given key."""
    if key in T:
        val = T[key]
        if isinstance(val, dict):
            return val[is_en]
        return val
    return key

st.title(t("title"))
st.markdown(t("subtitle"))
st.markdown("📖 [Mathematical Foundation](https://stoyoda0012-cyber.github.io/XPSTwin_streamlit/XPS_IRF_Simulator_Mathematical_Foundation.html)")

# --- サイドバー: 思想・背景 ---
with st.sidebar.expander(t("philosophy_title"), expanded=False):
    st.markdown(t("philosophy_content"))

st.sidebar.divider()

# --- サイドバー: パラメータ設定 ---
st.sidebar.header("Instrument Parameters")

# 光源の設定
st.sidebar.subheader("X-ray Source")

with st.sidebar.expander(t("xray_source_model_title"), expanded=False):
    st.markdown(t("xray_source_model_content"))

# スポットサイズ
sigma_x = st.sidebar.slider(
    "Spot Size X (meV)", 0.01, 2.0, 0.5, format="%.2f",
    help=t("help_sigma_x")
)

sigma_y = st.sidebar.slider(
    "Spot Size Y (mm)", 0.01, 2.0, 0.5, format="%.2f",
    help=t("help_sigma_y")
)

# 非対称性
gamma_x = st.sidebar.slider(
    "Spot Skew X (gamma_x)", -5.0, 5.0, 0.0, format="%.1f",
    help=t("help_gamma_x")
)

gamma_y = st.sidebar.slider(
    "Spot Skew Y (gamma_y)", -10.0, 10.0, 0.0, format="%.1f",
    help=t("help_gamma_y")
)

# エネルギー勾配（最後に配置）
alpha = st.sidebar.slider(
    "Energy Gradient (alpha)", -0.01, 0.01, 0.002, format="%.4f",
    help=t("help_alpha"),
    step=0.0001
)

# 検出器の設定
st.sidebar.subheader("2D Detector")

with st.sidebar.expander(t("detector_model_title"), expanded=False):
    st.markdown(t("detector_model_content"))

kappa = st.sidebar.slider(
    "Smile Curvature (kappa)", 0.0, 0.2, 0.01, format="%.3f",
    help=t("help_kappa"),
    step=0.001
)
theta = st.sidebar.slider(
    "Detector Tilt (theta deg)", -1.0, 1.0, 0.08, format="%.2f",
    help=t("help_theta")
)
sigma_res_mev = st.sidebar.slider(
    "Intrinsic Res (sigma meV)", 0.1, 10.0, 1.5, format="%.1f",
    help=t("help_sigma_res")
) / 1000.0

# ノイズ設定
st.sidebar.subheader("Detector Noise")

with st.sidebar.expander(t("noise_model_title"), expanded=False):
    st.markdown(t("noise_model_content"))

# ログスケールスライダー: 10^(-5) から 10^3 まで (0.00001 から 1000)
poisson_log = st.sidebar.slider(
    "Poisson Noise Level (log₁₀)", -5.0, 3.0, 0.3, format="%.2f", step=0.01,
    help=t("help_poisson")
)
poisson_noise = 10 ** poisson_log

# Poissonの実際の値を表示
if poisson_noise < 0.01:
    st.sidebar.caption(f"{t('actual_value')}: {poisson_noise:.2e}")
else:
    st.sidebar.caption(f"{t('actual_value')}: {poisson_noise:.4f}")

gaussian_noise = st.sidebar.slider(
    "Gaussian Readout Noise (%)", 0.0, 10.0, 1.0, format="%.1f",
    help=t("help_gaussian")
)

# ノイズゼロの判定を表示（両方がゼロの場合）
if poisson_noise <= 1e-5 and gaussian_noise == 0.0:
    st.sidebar.info(t("noise_zero"))

# 測定条件
st.sidebar.subheader("Measurement")
temp = st.sidebar.slider("Temperature (K)", 0.1, 300.0, 5.0)

st.sidebar.divider()

# --- Resolution Summary ---
st.sidebar.subheader(t("resolution_summary_title"))

# 分解能の計算（meV単位）
sigma_source_mev = sigma_x  # 既にmeV
sigma_detector_mev = sigma_res_mev * 1000.0  # eV -> meV
sigma_combined_mev = np.sqrt(sigma_source_mev**2 + sigma_detector_mev**2)

# 分解能の表示
col_res1, col_res2 = st.sidebar.columns(2)
with col_res1:
    st.metric(t("source_resolution_label"), f"{sigma_source_mev:.2f} meV")
with col_res2:
    st.metric(t("detector_resolution_label"), f"{sigma_detector_mev:.2f} meV")

st.sidebar.metric(t("combined_resolution_label"), f"{sigma_combined_mev:.2f} meV")
st.sidebar.caption(t("resolution_formula_note"))

st.sidebar.divider()

with st.sidebar.expander(t("asymmetry_title"), expanded=False):
    st.markdown(t("asymmetry_content"))

# --- メイン計算エンジン ---
engine = DigitalTwinEngine(e_range=(-0.1, 0.1), e_steps=500)
engine.source.sigma_x = sigma_x
engine.source.sigma_y = sigma_y
engine.source.alpha = alpha
engine.source.gamma_x = gamma_x
engine.source.gamma_y = gamma_y
engine.source.rotation = 0.0  # 回転は無効化
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
        st.markdown(t("fermi_edge_fitting_desc"))

        run_fermi_fit = st.button("📊 Run Fermi Edge Fit", use_container_width=True, key="fermi_fit")

    with tab2:
        st.markdown(t("irf_estimation_desc"))
        col_a, col_b = st.columns(2)
        with col_a:
            maxiter = st.number_input(t("optimization_iterations"), min_value=10, max_value=200, value=30, step=10)
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
    ax_spot.set_title(f"Spot (σx={sigma_x:.2f}meV, σy={sigma_y:.2f}mm)")
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
    st.info(t("using_simulated_data"))

    with st.spinner(t("fitting_in_progress")):
        # フェルミエッジフィッティング
        fit_result = deconvolver.fit_fermi_edge(observed_for_fit, temp=temp)

    if fit_result['success']:
        st.success(t("fitting_success"))

        # フィッティング結果の表示
        col_fit1, col_fit2, col_fit3, col_fit4 = st.columns(4)

        with col_fit1:
            st.metric(
                "Fermi Energy Shift",
                f"{fit_result['ef_shift']*1000:.3f} meV",
                help=f"{t('help_ef_shift_error')}: ±{fit_result['ef_shift_error']*1000:.3f} meV"
            )

        with col_fit2:
            st.metric(
                "Total Resolution (FWHM)",
                f"{fit_result['sigma_total']*2.355*1000:.2f} meV",
                help=f"{t('help_sigma_error')}: {fit_result['sigma_total']*1000:.3f} meV ({t('help_ef_shift_error')}: ±{fit_result['sigma_total_error']*1000:.3f} meV)"
            )

        with col_fit3:
            st.metric(
                "Temperature (Fitted)",
                f"{fit_result['temp_fit']:.2f} K",
                delta=f"{fit_result['temp_fit']-temp:.2f} K",
                help=f"{t('help_temp_error')}: {temp:.2f} K | {t('help_ef_shift_error')}: ±{fit_result['temp_error']:.3f} K"
            )

        with col_fit4:
            st.metric(
                "Fit Quality (R²)",
                f"{fit_result['r_squared']:.6f}",
                help=t("help_r_squared")
            )

        # 理論分解能との比較
        st.subheader(t("theoretical_resolution_comparison"))
        theory_res = deconvolver.calculate_theoretical_resolution()

        col_theory1, col_theory2 = st.columns(2)

        with col_theory1:
            st.markdown(t("simulator_theoretical_resolution"))
            st.metric("Total (FWHM)", f"{theory_res['total_resolution']*2.355*1000:.2f} meV")
            st.caption(f"σ = {theory_res['total_resolution']*1000:.3f} meV")

            # 各成分の寄与を表示
            with st.expander(t("view_component_contributions")):
                st.write(f"- Detector Intrinsic: {theory_res['detector_intrinsic']*1000:.3f} meV")
                st.write(f"- Smile Curvature: {theory_res['smile_curvature']*1000:.3f} meV")
                st.write(f"- Detector Tilt: {theory_res['detector_tilt']*1000:.3f} meV")
                st.write(f"- Source Size (X): {theory_res['source_size_x']*1000:.3f} meV")
                st.write(f"- Energy Gradient: {theory_res['energy_gradient']*1000:.3f} meV")
                st.write(f"- Asymmetry: {theory_res['asymmetry']*1000:.3f} meV")

        with col_theory2:
            st.markdown(t("fitted_resolution"))
            st.metric("Total (FWHM)", f"{fit_result['sigma_total']*2.355*1000:.2f} meV")
            st.caption(f"σ = {fit_result['sigma_total']*1000:.3f} meV")

            # 差異を計算
            diff = abs(fit_result['sigma_total'] - theory_res['total_resolution'])
            rel_diff = (diff / theory_res['total_resolution']) * 100
            st.metric(t("difference"), f"{diff*1000:.3f} meV", delta=f"{rel_diff:.1f}%")

        # スペクトルのフィッティング結果プロット
        st.subheader(t("fitting_result"))
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
        st.subheader(t("fitting_residuals"))
        fig_res, ax_res = plt.subplots(figsize=(12, 3))
        ax_res.plot(x_for_plot*1000, fit_result['residuals'], 'g-', alpha=0.7, linewidth=1)
        ax_res.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax_res.set_xlabel("Energy (meV)")
        ax_res.set_ylabel("Residuals")
        ax_res.grid(alpha=0.3)
        st.pyplot(fig_res)

    else:
        st.error(f"{t('fitting_failed')}: {fit_result.get('error_message', 'Unknown error')}")

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

    with st.spinner(t("optimization_in_progress")):
        # パラメータ推定を実行
        result = deconvolver.estimate_irf_parameters(
            observed_for_estimation,
            temp=temp,
            maxiter=int(maxiter),
            progress_callback=progress_callback
        )

    progress_bar.progress(1.0)
    status_text.text(t("optimization_complete"))

    # 結果の表示
    st.success(f"{t('optimization_success')}: {result['success']}")
    st.info(f"{t('final_loss')}: {result['final_loss']:.6e} | {t('iterations')}: {result['nit']} | {t('function_evaluations')}: {result['nfev']}")

    # 推定されたパラメータを表で表示
    st.subheader(t("estimated_irf_parameters"))
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

    # スペクトルのフィッティング結果
    st.subheader(t("fitting_result"))
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
    st.subheader(t("estimated_vs_true_irf"))
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
    st.subheader(t("parameter_comparison"))
    true_params = {
        'kappa': kappa,
        'theta': theta,
        'sigma_res': sigma_res_mev / 1000.0,
        'alpha': alpha,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'gamma_x': gamma_x,
        'gamma_y': gamma_y,
        'rotation': 0.0  # 回転は無効化
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
