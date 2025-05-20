import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

# 数据
data = np.array([
    [29.1, 32.7, 19.8],
    [7.8, 7.6, 2.6]
])

# 标签
columns = ['Nor.', 'Under Spec.', "Adv."]
rows = ['Nor.', 'Adv.']

# 创建热图
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)  # 调整图像尺寸和分辨率

# 创建一个带间隔的网格（通过调整网格间隙）
cmap = plt.cm.YlGnBu  # 使用YlGnBu配色
norm = colors.Normalize(vmin=np.min(data), vmax=np.max(data))  # 归一化
cax = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')  # 绑定到颜色条

# 添加网格间距
ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)  # 小刻度在格子边界间
ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=5)  # 用白色网格间隙隔开
ax.tick_params(which="minor", size=0)

# 添加颜色条
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)  # 调整颜色条的大小和位置
cbar.ax.tick_params(labelsize=40)

# 设置刻度
ax.set_xticks(np.arange(len(columns)))
ax.set_yticks(np.arange(len(rows)))

# 设置刻度标签字体大小
ax.set_xticklabels(columns, fontsize=40, rotation=0)
ax.set_yticklabels(rows, fontsize=40, rotation=90)

# 设置刻度线宽和长度
ax.tick_params(axis='both', which='major', width=2, length=10)  # 增加刻度线的长度为10

# 设置轴标签
ax.set_ylabel('Human', fontsize=40, labelpad=20, verticalalignment='center', horizontalalignment='center')

# 在顶部添加标题
fig.text(0.5, 0.95, 'Vote Evaluator', ha='center', va='center', fontsize=40)  # 标题位置在图形上方

# 去除黑边
for spine in ax.spines.values():
    spine.set_visible(False)

# 添加文本注释并根据背景颜色调整字体颜色
for i in range(len(rows)):
    for j in range(len(columns)):
        # 获取背景颜色的RGB值
        rgba = cmap(norm(data[i, j]))
        # 将RGB转换为灰度值 (亮度)
        brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        # 根据亮度设置字体颜色：亮度低用白色，亮度高用黑色
        text_color = 'white' if brightness < 0.5 else 'black'

        # 将数值转为百分比并格式化
        percentage = data[i, j]  # 转为百分比
        ax.text(j, i, f'{percentage:.1f}%', va='center', ha='center', color=text_color, fontsize=40)

# 显示图像
plt.show()
