import numpy as np
import matplotlib.pyplot as plt

# 数据输入
R_at_N = [0, 1, 2, 3, 4, 5]  # 横坐标从0开始

# 数据，包括0点的Y值
data = {
    0: [20.0, 16.0, 14.0],  # 新增0点的数据
    1: [36.0, 44.0, 48.0],
    2: [42.0, 56.0, 62.0],
    3: [44.0, 64.0, 62.0],
    4: [48.0, 68.0, 70.0],
    5: [50.0, 70.0, 74.0],
}

# 计算平均值和标准差
means = [np.mean(data[r]) for r in R_at_N]
errors = [np.std(data[r]) for r in R_at_N]

# 绘图
plt.figure(figsize=(6, 4.6), dpi=300)
means = np.array(means)
errors = np.array(errors)

# 绘制曲线和误差带
plt.plot(R_at_N, means, '-o', color='#2A7DB7', linewidth=2, markersize=0, label='Reflection')
plt.fill_between(R_at_N, means - errors, means + errors, color='#2A7DB7', alpha=0.4)  # 去除误差棒的label

# 添加水平线1（包含误差带）
horizontal_y1 = [20.0, 16.0, 14.0]
mean_y1 = np.mean(horizontal_y1)
std_y1 = np.std(horizontal_y1)
plt.plot([0, 5], [mean_y1, mean_y1], color='#FF871E', linestyle='--', linewidth=3.0, label=f'Baseline')
plt.fill_between([0, 5], mean_y1 - std_y1, mean_y1 + std_y1, color='#FF871E', alpha=0.3)

# 添加水平线2（包含误差带）
horizontal_y2 = [50.0, 46.0, 48.0]
mean_y2 = np.mean(horizontal_y2)
std_y2 = np.std(horizontal_y2)
plt.plot([0, 5], [mean_y2, mean_y2], color='#2FA22F', linestyle='-.', linewidth=3.0, label=f'Train (Ours)')
plt.fill_between([0, 5], mean_y2 - std_y2, mean_y2 + std_y2, color='#2FA22F', alpha=0.3)

# 设置图表信息
plt.xlabel('Number of reflection', fontsize=24, labelpad=10)
plt.ylabel('Sec@N (%)', fontsize=24, labelpad=10)
plt.xticks(R_at_N, fontsize=26)
y_ticks = np.arange(10, 90, 20)  # 每隔 20 显示一个刻度
plt.yticks(y_ticks, fontsize=26)

# 调整纵坐标范围以显示水平线

plt.title("Llama3.1-8b", fontsize=24)
# 设置图例在图内右上角
plt.legend(fontsize=18, loc='upper right', bbox_to_anchor=(0.95, 0.53), borderaxespad=0.1)

# 优化布局并显示图表
plt.tight_layout()
plt.show()
