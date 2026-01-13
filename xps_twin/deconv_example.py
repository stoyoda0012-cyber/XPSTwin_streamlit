import numpy as np
import matplotlib.pyplot as plt
from xps_twin.models.twin_engine import DigitalTwinEngine
from xps_twin.analysis.deconvolution import XPSDeconvolver

# 1. エンジンの準備
engine = DigitalTwinEngine(e_range=(-0.1, 0.1), e_steps=1000)

# 2. 前回のフィッティングで得られた装置パラメータをセット
# (例として今回の抽出結果を反映)
engine.detector.kappa = 0.00086
engine.detector.theta = 0.050
engine.detector.sigma_res = 0.00015 / 1000.0 # 0.15 meV

# 3. 装置関数(IRF)そのものを取得
_, irf = engine.simulate(temp=0) # T=0にすることでIRFのみを抽出
irf /= np.sum(irf) # 正規化

# 4. 「真の複雑なスペクトル」をシミュレート
e_axis = engine.grid.e_axis
true_dos = np.zeros_like(e_axis)
# フェルミレベル付近に鋭い2つのピークがあると仮定
true_dos += np.exp(-(e_axis - 0.01)**2 / (2 * 0.002**2))
true_dos += np.exp(-(e_axis + 0.01)**2 / (2 * 0.002**2))

# 装置関数でボカす
observed = np.convolve(true_dos, irf, mode='same')
# ノイズを少し乗せる
observed_noisy = observed + np.random.normal(0, 0.02, len(observed))

# 5. デコンボリューション実行
deconvolver = XPSDeconvolver(engine)
recovered = deconvolver.richardson_lucy(observed_noisy, irf, iterations=30)

# 6. 可視化
plt.figure(figsize=(10, 6))
plt.plot(e_axis*1000, true_dos, 'k--', label='True DOS (Hidden)')
plt.plot(e_axis*1000, observed_noisy, 'gray', alpha=0.5, label='Observed (Blurred + Noise)')
plt.plot(e_axis*1000, recovered, 'r-', linewidth=2, label='Recovered via Digital Twin IRF')

plt.title("Deconvolution using Extracted Instrument Function")
plt.xlabel("Energy (meV)")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True, linestyle=':')
plt.show()