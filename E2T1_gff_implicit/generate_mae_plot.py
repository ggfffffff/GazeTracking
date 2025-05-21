import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)

# 设置图表风格
plt.style.use('seaborn')

# 读取MAE值
with open('mae_values.json', 'r') as f:
    data = json.load(f)
    with_implicit = data['with_implicit']
    without_implicit = data['without_implicit']

# 创建轮次列表
rounds = list(range(1, len(with_implicit) + 1))

# 创建图表
plt.figure(figsize=(12, 7))
fig, ax = plt.subplots(figsize=(12, 7))

# 设置背景色
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# 绘制数据线
line1 = ax.plot(rounds, with_implicit, 'o-', color='#1f77b4', linewidth=2.5, 
                markersize=8, label='使用隐式校准', markerfacecolor='white', 
                markeredgewidth=2, markeredgecolor='#1f77b4')
line2 = ax.plot(rounds, without_implicit, 'o-', color='#d62728', linewidth=2.5, 
                markersize=8, label='不使用隐式校准', markerfacecolor='white', 
                markeredgewidth=2, markeredgecolor='#d62728')

# 设置图表样式
ax.set_title('MAE 趋势', fontproperties=font, fontsize=16, pad=20, fontweight='bold')
ax.set_xlabel('轮次', fontproperties=font, fontsize=12, labelpad=10)
ax.set_ylabel('MAE (像素)', fontproperties=font, fontsize=12, labelpad=10)

# 设置网格
ax.grid(True, linestyle='--', alpha=0.3)

# 设置图例
ax.legend(prop=font, fontsize=11, loc='upper right', frameon=True, 
          facecolor='white', edgecolor='none', shadow=True)

# 设置x轴刻度为整数
ax.set_xticks(rounds)

# 设置y轴范围
ax.set_ylim(100, 300)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('mae_trend.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close() 