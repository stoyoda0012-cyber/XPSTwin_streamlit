import numpy as np
import matplotlib.pyplot as plt
from models.twin_engine import DigitalTwinEngine
from analysis.optimizer import XPSOptimizer

# 1. エンジンの初期化
engine = DigitalTwinEngine(e_range=(-0.03, 0.03), e_steps=300)

# 2. テスト用データの作成 (本来はここに実測CSVを読み込む)
# 正解: kappa=0.003, theta=0.08, sigma=0.0015
true_params = [0.003, 0.08, 0.0015, 0.0005, 0.01]
engine.detector.kappa = true_params[0]
engine.detector.theta = true_params[1]
engine.detector.sigma_res = true_params[2]
engine.source.alpha = true_params[4]
x_true, y_clean = engine.simulate(temp=5.0)
# ノイズを乗せる
y_obs = y_clean + np.random.normal(0, 0.01, len(y_clean))

# 3. フィッティング実行
optimizer = XPSOptimizer(engine)
print("--- フィッティング開始 ---")
result = optimizer.fit(x_true, y_obs, temp=5.0)

# 4. 結果表示
p = result.x
print(f"\n抽出された装置パラメータ:")
print(f"  曲率 (Kappa): {p[0]:.5f}")
print(f"  傾き (Theta): {p[1]:.3f} deg")
print(f"  固有分解能 (Sigma): {p[2]*1000:.2f} meV")
print(f"  Efシフト: {p[3]*1000:.3f} meV")

# 5. 可視化
plt.scatter(x_true*1000, y_obs, s=10, color='gray', label='Experimental Data', alpha=0.5)
plt.plot(x_true*1000, y_clean, 'k--', label='True Physics')
# 最適化後のパラメータで再シミュレーション
engine.detector.kappa, engine.detector.theta, engine.detector.sigma_res = p[0], p[1], p[2]
engine.source.alpha = p[4]
_, y_fit = engine.simulate(temp=5.0)
plt.plot(x_true*1000, y_fit, 'r-', label='Digital Twin Fit')

plt.xlabel("Energy (meV)")
plt.ylabel("Intensity")
plt.legend()
plt.title("XPS Digital Twin: Parameter Extraction")
plt.show()