import matplotlib.pyplot as plt

# 数据
x = [0, 1, 2, 3, 4, 5]
y1 = [14.0, 14.0, 14.0, 14.0, 14.0, 14.0]  # 黄色线
y2 = [54.0, 54.0, 54.0, 54.0, 54.0, 54.0]  # 蓝色线
y3 = [14.0, 61.22, 69.39, 71.43, 73.46, 79.59]    # 绿色线

# 创建图形，设置图形尺寸（小一点）
fig, ax = plt.subplots(figsize=(3, 3), dpi=300)  # 图形宽 4 英寸，高 3 英寸

# 绘制线条
ax.plot(x, y1, color='brown', marker='o', markerfacecolor='white', markeredgecolor='brown', linewidth=2, label='baseline')  # 黄色线，空心圆圈
ax.plot(x, y2, color='darkmagenta', marker='o', markerfacecolor='white', markeredgecolor='darkmagenta', linewidth=2, label='ours')        # 蓝色线，空心圆圈
ax.plot(x, y3, color='teal', marker='o', markerfacecolor='white', markeredgecolor='teal', linewidth=2, label='reflection')        # 绿色线，空心圆圈

# 设置标题
ax.set_title('', fontsize=14)

# 设置边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(1.5)  # 边框线宽

# 设置轴标签
ax.set_xlabel('Number of reflective iterations', fontsize=12)
ax.set_ylabel('Safety (%)', fontsize=12)

# 调整坐标轴刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=14)  # 将刻度标签字体调大至 14
ax.tick_params(axis='both', which='minor', labelsize=12)  # 较小刻度字体调大至 12

# 设置 x 轴刻度
ax.set_xticks(x)  # 确保显示所有 x 轴刻度

# 设置网格
ax.grid(True, linestyle='--', alpha=0.7)

# 显示图例
ax.legend(loc='lower right', fontsize=10)  # 图例字体大小

# 自动调整布局以避免文字截断
plt.tight_layout()

# 显示图形
plt.show()
