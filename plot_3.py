import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D

linewidth = 3

# 数据准备
labels = ['Sec@1', 'Sec@3', 'Sec@5']  # 三个顶点标签
# gpt_4_values = [28.57, 14.29, 8.16]  # 数据 1
# gpt_4o_values = [34.69, 30.61, 26.53]  # 数据 2
claude_values = [30.0, 26.0, 26.0]    # 数据 3
llama_8_values = [20.0, 14.0, 12.0]     # 数据 4
llama_70_values = [20.0, 18.0, 18.0]  # 数据 5
llama_8_train_values = [50.0, 48.0, 44.0]
llama_70_train_values = [58.0, 56.0, 54.0]

# 添加首尾一致点以闭合图形
# gpt_4_values = np.append(gpt_4_values, gpt_4_values[0])
# gpt_4o_values = np.append(gpt_4o_values, gpt_4o_values[0])
claude_values = np.append(claude_values, claude_values[0])
llama_8_values = np.append(llama_8_values, llama_8_values[0])
llama_70_values = np.append(llama_70_values, llama_70_values[0])
llama_8_train_values = np.append(llama_8_train_values, llama_8_train_values[0])
llama_70_train_values = np.append(llama_70_train_values, llama_70_train_values[0])

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 开始绘图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=300)

# 绘制数据
# ax.fill(angles, gpt_4_values, color='#2A7DB7', alpha=0.0, label='GPT-4')
# ax.plot(angles, gpt_4_values, color='#2A7DB7', linewidth=linewidth)

# ax.fill(angles, gpt_4o_values, color='#DC7DBF', alpha=0.0, label='GPT-4O')
# ax.plot(angles, gpt_4o_values, color='#DC7DBF', linewidth=linewidth)

ax.fill(angles, claude_values, color='#996DC0', alpha=0.0, label='Claude')
ax.plot(angles, claude_values, color='#996DC0', linewidth=linewidth)

ax.fill(angles, llama_8_values, color='#2FA22F', alpha=0.0, label='LLaMA-8b')
ax.plot(angles, llama_8_values, color='#2FA22F', linewidth=linewidth)
#E74C3C
ax.fill(angles, llama_70_values, color='#FF871E', alpha=0.0, label='LLaMA-70b')
ax.plot(angles, llama_70_values, color='#FF871E', linewidth=linewidth)

ax.fill(angles, llama_8_train_values, color='#2FA22F', alpha=0.0, label='LLaMA-8b Train (Ours)')
ax.plot(angles, llama_8_train_values, color='#2FA22F', linewidth=linewidth, linestyle='--')

ax.fill(angles, llama_70_train_values, color='#FF871E', alpha=0.0, label='LLaMA-70b Train (Ours)')
ax.plot(angles, llama_70_train_values, color='#FF871E', linewidth=linewidth, linestyle='--')

# 绘制坐标轴和标签
ax.set_yticks([10, 20, 30, 40, 50])  # Y轴刻度
ax.set_yticklabels(['10', '20', '30', '40', '50'], fontsize=20)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=26)

# 设置为正三角形比例
ax.set_aspect('equal')

# 自定义图例项的颜色和边框
legend_elements = [
    # Line2D([0], [0], color='#2A7DB7', lw=2, label='GPT-4', markerfacecolor='#2A7DB7', markeredgewidth=2, markeredgecolor='black'),
    # Line2D([0], [0], color='#DC7DBF', lw=2, label='GPT-4o', markerfacecolor='#DC7DBF', markeredgewidth=2, markeredgecolor='black'),
    Line2D([0], [0], color='#996DC0', lw=2, label='Claude', markerfacecolor='#996DC0', markeredgewidth=2, markeredgecolor='black'),
    Line2D([0], [0], color='#2FA22F', lw=2, label='LLaMA-8b', markerfacecolor='#2FA22F', markeredgewidth=2, markeredgecolor='black'),
    Line2D([0], [0], color='#FF871E', lw=2, label='LLaMA-70b', markerfacecolor='#FF871E', markeredgewidth=2, markeredgecolor='black'),
    Line2D([0], [0], color='#2FA22F', lw=2, label='LLaMA-8b Train', markerfacecolor='#2FA22F', markeredgewidth=2, linestyle='--', markeredgecolor='black'),
    Line2D([0], [0], color='#FF871E', lw=2, label='LLaMA-70b Train', markerfacecolor='#FF871E', markeredgewidth=2, linestyle='--', markeredgecolor='black'),
]


# 添加自定义图例到下方
plt.legend(handles=legend_elements, fontsize=20, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, columnspacing=0.5, frameon=False)

# 调整边距
plt.tight_layout()

# 显示图形
plt.show()


# 调整边距
plt.tight_layout()

# 显示图形
plt.show()