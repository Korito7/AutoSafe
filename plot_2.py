import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D

linewidth = 3

# 数据准备
# labels = ['Sec@1', 'Sec@3', 'Sec@5']  # 三个顶点标签
labels = ['P.L.', 'C.S.', 'F.L.', 'P.D.', 'P.H.', 'D.L.', 'I.A.', 'E.M.', 'B.O.', 'M.']

# gpt_4_values = [17.39, 13.04, 13.04]  # 数据 1
# gpt_4o_values = [17.78, 8.89, 6.67]  # 数据 2
# claude_values = [26.67, 20.0, 13.0]    # 数据 3
# llama_8_values = [12.0, 6.0, 4.0]     # 数据 4
# llama_70_values = [16.32, 12.24, 10.20]  # 数据 5
# llama_8_train_values = [50.0, 46.0, 44.0]
# llama_70_train_values = [64.0, 60.0, 58.0]

# claude_values = [15.00, 14.29, 46.67, 40.00, 26.67, 16.67, 53.33, 33.33, 46.67, 60.00]

# llama_8_values = [16.67, 14.29, 17.78, 20.00, 0.00, 6.67, 20.00, 0.00, 16.67, 0.00]
# llama_70_values = [16.67, 25.71, 28.89, 6.67, 13.33, 16.67, 6.67, 0.00, 23.33, 0.00]
# llama_8_train_values = [35.00, 45.71, 77.78, 40.00, 40.00, 33.33, 66.67, 0.00, 46.67, 50.00]
# llama_70_train_values = [46.67, 45.71, 55.56, 86.67, 100.00, 46.67, 86.67, 100.00, 60.00, 50.00]
llama_8_values = [26.25, 27.50, 17.33, 32.00, 11.11, 12.00, 10.00, 0.00, 12.50, 6.00]
llama_8_naive_values = [26.88, 43.75, 28.00, 28.00, 13.33, 20.00, 43.33, 3.33, 32.50, 8.00]
llama_8_train_values = [56.88, 65.00, 74.67, 54.00, 68.89, 44.00, 66.67, 56.67, 55.00, 32.00]

# 添加首尾一致点以闭合图形
# gpt_4_values = np.append(gpt_4_values, gpt_4_values[0])
# gpt_4o_values = np.append(gpt_4o_values, gpt_4o_values[0])
# claude_values = np.append(claude_values, claude_values[0])

# llama_70_values = np.append(llama_70_values, llama_70_values[0])

# llama_70_train_values = np.append(llama_70_train_values, llama_70_train_values[0])


llama_8_values = np.append(llama_8_values, llama_8_values[0])
llama_8_naive_values = np.append(llama_8_naive_values, llama_8_naive_values[0])
llama_8_train_values = np.append(llama_8_train_values, llama_8_train_values[0])

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 开始绘图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=300)

# 绘制数据
# ax.fill(angles, gpt_4_values, color='#2A7DB7', alpha=0.0, label='GPT-4')
# ax.plot(angles, gpt_4_values, color='#2A7DB7', linewidth=linewidth)

# ax.fill(angles, gpt_4o_values, color='#DC7DBF', alpha=0.0, label='GPT-4O')
# ax.plot(angles, gpt_4o_values, color='#DC7DBF', linewidth=linewidth)

# ax.fill(angles, claude_values, color='#996DC0', alpha=0.0, label='Claude')
# ax.plot(angles, claude_values, color='#996DC0', linewidth=linewidth)


#E74C3C
# ax.fill(angles, llama_70_values, color='#FF871E', alpha=0.0, label='LLaMA-70b')
# ax.plot(angles, llama_70_values, color='#FF871E', linewidth=linewidth)



# ax.fill(angles, llama_70_train_values, color='#FF871E', alpha=0.0, label='LLaMA-70b Train (Ours)')
# ax.plot(angles, llama_70_train_values, color='#FF871E', linewidth=linewidth, linestyle='--')

ax.fill(angles, llama_8_values, color='#2FA22F', alpha=0.0, label='LLaMA-8b')
ax.plot(angles, llama_8_values, color='#2FA22F', linewidth=linewidth)

ax.fill(angles, llama_8_naive_values, color='#FF871E', linewidth=linewidth, label='LLaMA-8b-Naive')
ax.plot(angles, llama_8_naive_values, color='#FF871E', linewidth=linewidth)

ax.fill(angles, llama_8_train_values, color='#2FA22F', alpha=0.0, label='LLaMA-8b Train (Ours)')
ax.plot(angles, llama_8_train_values, color='#2FA22F', linewidth=linewidth, linestyle='--')


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
    # Line2D([0], [0], color='#996DC0', lw=2, label='Claude', markerfacecolor='#996DC0', markeredgewidth=2, markeredgecolor='black'),
    # Line2D([0], [0], color='#FF871E', lw=2, label='LLaMA-70b', markerfacecolor='#FF871E', markeredgewidth=2, markeredgecolor='black'),
    # Line2D([0], [0], color='#FF871E', lw=2, label='LLaMA-70b Train', markerfacecolor='#FF871E', markeredgewidth=2, linestyle='--', markeredgecolor='black'),
    Line2D([0], [0], color='#2FA22F', lw=2, label='LLaMA-8b', markerfacecolor='#2FA22F', markeredgewidth=2,
           markeredgecolor='black'),
    Line2D([0], [0], color='#2FA22F', lw=2, label='LLaMA-8b Naive', markerfacecolor='#2FA22F', markeredgewidth=2,
           markeredgecolor='black'),
    Line2D([0], [0], color='#2FA22F', lw=2, label='LLaMA-8b Train', markerfacecolor='#2FA22F', markeredgewidth=2, linestyle='--', markeredgecolor='black'),

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
