import matplotlib.pyplot as plt
import numpy as np

x_labels = ['P.L.', 'C.S.', 'F.L.', 'P.D.', 'P.H.', 'D.L.', 'I.A.', 'E.M.', 'B.O.', 'M.']

# gpt4_values = [5.56, 19.05, 18.52, 0.00, 33.33, 16.67, 0.00, 0.00, 16.67, 50.00]
# gpt4o_values = [20.00, 17.14, 26.67, 0.00, 6.67, 16.67, 0.00, 0.00, 30.00, 50.00]
# claude_values = [15.00, 14.29, 46.67, 40.00, 26.67, 16.67, 53.33, 33.33, 46.67, 60.00]


gpt4_values = [21.00, 42.22, 16.67, 34.29, 16.67, 20.00, 60.00, 20.00, 60.00, 2.50]
gpt4o_values = [35.00, 64.44, 43.33, 40.00, 20.00, 25.00, 60.00, 40.00, 90.00, 15.00]
claude_values = [20.00, 40.00, 16.67, 28.57, 63.33, 20.00, 33.33, 33.33, 100.00, 25.00]

color_bar1 = "#478EC1"
color_bar2 = "#FF9335"
color_bar3 = "#36A536"

# color_bar1 = (71/255, 142/255, 193/255, 0.6)  # #478EC1 透明度0.6
# color_bar2 = (54/255, 165/255, 54/255, 0.6)   # #36A536 透明度0.6
# color_bar3 = (255/255, 147/255, 53/255, 0.6)  # #FF9335 透明度0.6

color_lozenge = (255/255, 135/255, 30/255)

# 边框设置
edge_color = "#787F87"  # 边框颜色（深灰色）
edge_width = 2  # 边框粗细

# 创建图形和子图
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# 宽度设置
bar_width = 0.25  # 调整柱子宽度
x = np.arange(len(x_labels))  # x 轴位置

# 创建柱状图
bars1 = ax.bar(x - bar_width, gpt4_values, bar_width, color=color_bar1, edgecolor=edge_color, linewidth=edge_width, label="gpt-4")
bars2 = ax.bar(x, gpt4o_values, bar_width, color=color_bar2, edgecolor=edge_color, linewidth=edge_width, label="gpt-4o")
bars3 = ax.bar(x + bar_width, claude_values, bar_width, color=color_bar3, edgecolor=edge_color, linewidth=edge_width, label="claude")

# 添加y轴50%分界线
# ax.axhline(y=50, color='red', linestyle='--', linewidth=2)

# 填充50%以下区域，覆盖整个横坐标
x_full_range = np.linspace(-0.5, len(x)-0.5, 500)  # 为了覆盖整个横坐标生成更多的点
ax.fill_between(x_full_range, 0, 50, color='red', alpha=0.1)

# 为每个柱子添加菱形标记
# for bars, color in zip([bars1, bars2, bars3], [color_bar1, color_bar2, color_bar3]):
#     for bar in bars:
#         height = bar.get_height()
#         ax.plot(bar.get_x() + bar.get_width() / 2, height, marker='D', color=color, markersize=8)

# 设置图形标题和标签
# ax.set_title("Llama3.1-8b", fontsize=20)
ax.set_ylabel("Sec@1(%)", fontsize=22)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)

# 放大 x 轴和 y 轴的字体
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)

# 旋转 x 轴标签以避免重叠
plt.xticks(rotation=45, ha="right")

# 修改 y 轴范围
ax.set_ylim(-3, 103)

# 设置 y 轴的刻度和标签
ax.set_yticks([0, 25, 50, 75, 100])  # 设置 y 轴的显示位置
ax.set_yticklabels([0, 25, 50, 75, 100])  # 设置 y 轴的标签

# ax.legend(fontsize=21, loc="upper left")

# 显示图形
plt.tight_layout()
plt.show()
