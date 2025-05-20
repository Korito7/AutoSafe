import matplotlib.pyplot as plt
import numpy as np

# 数据
# x_labels = [
#     "P.L.", "C.S.", "F.L.",
#     "P.D.", "P.H.", "D.L.",
#     "I.A.", "E.M.",
#     "B.O.", "M."
# ]

# x_labels = ["Ent.", "Bus.", "Hea.", "Fin."]

x_labels = ["Nor.-Nor.", "Nor.-Adv.", "Un.S.-Nor.", "Un.S.-Adv.", "Adv.-Nor.", "Adv.-Adv."]

# x_labels = ["Nor.-Nor.", "Nor.-Adv.", "Un.S.-Nor.", "Un.S.-Adv."]

# y_values_Claude_ours = [15.00, 14.29, 46.67, 40.00, 26.67, 16.67, 53.33, 33.33, 46.67, 60.00]
# y_values_Claude (Naive)_ours = [25.00, 31.43, 35.56, 46.67, 33.33, 16.67, 46.67, 60.00, 43.33, 60.00]
# y_values_GPT-4_ours = [5.56, 19.05, 18.52, 0.00, 33.33, 16.67, 0.00, 0.00, 16.67, 50.00]
# y_values_GPT-4 (Naive)_ours = [47.22, 9.52, 25.93, 22.22, 33.33, 16.67, 22.22, 0.00, 22.22, 100.00]
# y_values_GPT-4o_ours = [20.00, 17.14, 26.67, 0.00, 6.67, 16.67, 0.00, 0.00, 30.00, 50.00]
# y_values_GPT-4o (Naive)_ours = [33.33, 45.71, 33.33, 33.33, 66.67, 16.67, 46.67, 20.00, 63.33, 90.00]
# y_values_Llama3.1-8b (Naive)_ours = [23.33, 48.57, 35.56, 33.33, 13.33, 16.67, 40.00, 6.67, 40.00, 30.00]

# Llama3.1-8b_ours
# y_values1 = [16.67, 14.29, 17.78, 20.00, 0.00, 6.67, 20.00, 0.00, 16.67, 0.00]
# y_values2 = [35.00, 45.71, 77.78, 40.00, 40.00, 33.33, 66.67, 0.00, 46.67, 50.00]

# y_values_Llama3.1-70b (Naive)_ours = [18.33, 37.14, 24.44, 20.00, 6.67, 16.67, 20.00, 6.67, 30.00, 10.00]

# Llama3.1-70b_ours
# y_values1 = [16.67, 25.71, 28.89, 6.67, 13.33, 16.67, 6.67, 0.00, 23.33, 0.00]
# y_values2 = [46.67, 45.71, 55.56, 86.67, 100.00, 46.67, 86.67, 100.00, 60.00, 50.00]

# y_values_Claude_pub = [20.00, 40.00, 16.67, 28.57, 63.33, 20.00, 33.33, 33.33, 100.00, 25.00]
# y_values_Claude (Naive)_pub = [32.00, 60.00, 16.67, 28.57, 56.67, 25.00, 33.33, 33.33, 100.00, 7.50]
# y_values_GPT-4_pub = [21.00, 42.22, 16.67, 34.29, 16.67, 20.00, 60.00, 20.00, 60.00, 2.50]
# y_values_GPT-4 (Naive)_pub = [21.00, 31.11, 13.33, 28.57, 16.67, 20.00, 46.67, 6.67, 60.00, 5.00]
# y_values_GPT-4o_pub = [35.00, 64.44, 43.33, 40.00, 20.00, 25.00, 60.00, 40.00, 90.00, 15.00]
# y_values_GPT-4o (Naive)_pub = [40.00, 66.67, 40.00, 28.57, 20.00, 25.00, 66.67, 33.33, 90.00, 5.00]
# y_values_Llama3.1-8b (Naive)_pub = [24.00, 37.78, 16.67, 25.71, 20.00, 5.00, 20.00, 0.00, 50.00, 15.00]

# Llama3.1-8b_public
# y_values1 = [32.00, 37.78, 16.67, 37.14, 16.67, 20.00, 0.00, 0.00, 0.00, 7.50]
# y_values2 = [57.00, 46.67, 50.00, 42.86, 33.33, 50.00, 53.33, 26.67, 50.00, 17.50]

# y_values_Llama3.1-70b (Naive)_pub = [32.00, 48.89, 33.33, 31.43, 16.67, 25.00, 66.67, 0.00, 40.00, 7.50]

# Llama3.1-70b_public
# y_values1 = [22.00, 37.78, 20.00, 28.57, 16.67, 25.00, 40.00, 0.00, 50.00, 2.50]
# y_values2 = [42.00, 60.00, 66.67, 57.14, 83.33, 55.00, 100.00, 66.67, 60.00, 50.00]

# y_values2 = [30.00, 50.00, 55.00, 45.00, 70.00, 40.00, 90.00, 50.00, 45.00, 40.00]  # 模拟对比数据




# Llama3.1-8b_ours
# y_values1 = [20.00, 8.33, 0.00, 16.36]
# y_values2 = [55.56, 31.67, 44.00, 63.64]

# Llama3.1-70b_ours
# y_values1 = [15.65, 19.17, 8.00, 20.00]
# y_values2 = [66.09, 60.00, 100.0, 54.55]

# Llama3.1-8b_public
# y_values1 = [15.38, 28.13, 23.00, 0.00]
# y_values2 = [43.08, 51.25, 40.00, 25.00]

# Llama3.1-70b_public
# y_values1 = [9.23, 23.75, 16.92, 5.00]
# y_values2 = [40.00, 52.50, 69.23, 50.00]

# Llama3.1-8b_ours
y_values1 = [5.00, 16.67, 11.11, 15.00, 16.36, 26.67]
y_values2 = [45.00, 50.00, 46.67, 45.00, 41.82, 66.67]

# Llama3.1-70b_ours
# y_values1 = [12.50, 30.00, 24.44, 10.00, 14.55, 0.00]
# y_values2 = [35.00, 56.67, 67.78, 45.00, 63.64, 66.67]


# Llama3.1-8b_public
# y_values1 = [18.00, 31.43, 21.33, 0.00, 0.00, 0.00]
# y_values2 = [64.00, 40.00, 42.67, 40.00, 0.00, 0.00]


# Llama3.1-70_public
# y_values1 = [22.00, 45.71, 14.00, 10.00, 0.00, 0.00]
# y_values2 = [62.00, 60.00, 51.33, 50.00, 0.00, 0.00]


# y_values = []


# 颜色（柱体，菱形）
color_bar1 = (177/255, 200/255, 231/255)
color_bar2 = (255/255, 135/255, 30/255)

color_lozenge = (255/255, 135/255, 30/255)

# 边框设置
edge_color = "#787F87"  # 边框颜色（深灰色）
edge_width = 2  # 边框粗细

# 创建图形和子图
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# 宽度设置
bar_width = 0.35  # 单个柱体宽度
x = np.arange(len(x_labels))  # x 轴位置

# 创建柱状图
bars1 = ax.bar(x - bar_width / 2, y_values1, bar_width, color=color_bar1, edgecolor=edge_color, linewidth=edge_width, label="Baseline")
bars2 = ax.bar(x + bar_width / 2, y_values2, bar_width, color=color_bar2, edgecolor=edge_color, linewidth=edge_width, label="Train")

# 为每个柱子添加菱形标记
for bars, color in zip([bars1, bars2], [color_bar1, color_bar2]):
    for bar in bars:
        height = bar.get_height()
        ax.plot(bar.get_x() + bar.get_width() / 2, height)

# 设置图形标题和标签
ax.set_title("Llama3.1-8b", fontsize=20)
ax.set_ylabel("Sec@1(%)", fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)

# 放大 x 轴和 y 轴的字体
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

# 旋转 x 轴标签以避免重叠
plt.xticks(rotation=45, ha="right")

# 修改 y 轴范围
ax.set_ylim(-3, 105)

# 设置 y 轴的刻度和标签
ax.set_yticks([0, 25, 50, 75, 100])  # 设置 y 轴的显示位置
ax.set_yticklabels([0, 25, 50, 75, 100])  # 设置 y 轴的标签

ax.legend(fontsize=20, loc="upper left")

# 显示图形
plt.tight_layout()
plt.show()
