import matplotlib.pyplot as plt
import numpy as np

x_labels = ['GPT-4', 'GPT-4o', 'Claude-3.5', 'Gemini-1.5', 'Llama-3.1 8b', 'Llama-3.1 70b', 'Qwen-2.5 7b', 'Glm-4 9b']

ours_values = [13.0, 6.7, 13.0, 24.0, 4.0, 10.2, 12.0, 12.2]
public_values = [8.2, 26.5, 26.0, 32.0, 12.0, 18.0, 22.0, 30.0]

color_bar1 = "#478EC1"
color_bar2 = "#FF9335"

# 边框设置
edge_color = "#787F87"  # 边框颜色（深灰色）
edge_width = 2  # 边框粗细

# 创建图形和子图
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

# 宽度设置
bar_height = 0.35  # 调整柱子高度
y = np.arange(len(x_labels))  # y 轴位置

# 创建水平柱状图
bars1 = ax.barh(y - bar_height / 2, ours_values, bar_height, color=color_bar1, edgecolor=edge_color, linewidth=edge_width, label="Ours")
bars2 = ax.barh(y + bar_height / 2, public_values, bar_height, color=color_bar2, edgecolor=edge_color, linewidth=edge_width, label="Public")

# 设置图形标题和标签
ax.set_xlabel("Sec@5(%)", fontsize=22)
ax.set_yticks(y)
ax.set_yticklabels(x_labels)

# 放大 x 轴和 y 轴的字体
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)

# 修改 x 轴范围
ax.set_xlim(-3, 103)

# 设置 x 轴的刻度和标签
ax.set_xticks([0, 25, 50, 75, 100])  # 设置 x 轴的显示位置
ax.set_xticklabels([0, 25, 50, 75, 100])  # 设置 x 轴的标签

# 添加图例
ax.legend(fontsize=21, loc="lower right")

# 显示图形
plt.tight_layout()
plt.show()
