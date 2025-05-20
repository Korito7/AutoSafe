import matplotlib.pyplot as plt

# 数据
categories = ['Social Media', 'Finance', 'E-commerce', 'Others', 'Entertainment',
              'Office Work', 'Computer', 'Health', 'IoT']

# ours
# percentages = [25.00, 12.78, 11.67, 3.33, 2.789, 15.00, 15.56, 5.00, 8.89]

#public
percentages = [20.49, 9.02, 6.56, 12.30, 2.46, 12.30, 14.75, 9.02, 13.11]


# 颜色选择 (加入透明度)
colors = ['#42A942', '#FF8A23', '#277CB7', '#986EC0', '#878787', '#936056', '#E47AC3', '#C59610', '#DA4041', '#71C3B3']
colors_with_alpha = [color + '99' for color in colors]  # 设置透明度为 0.6

# 按照百分比从大到小排序
sorted_indices = sorted(range(len(percentages)), key=lambda i: percentages[i], reverse=True)
sorted_categories = [categories[i] for i in sorted_indices]
sorted_percentages = [percentages[i] for i in sorted_indices]
sorted_colors = [colors_with_alpha[i] for i in sorted_indices]

# 计算最小扇区的角度，使其位于水平线
startangle = 90 - (sorted_percentages.index(min(sorted_percentages)) * 360 / sum(sorted_percentages))

# 绘制饼状图
plt.figure(figsize=(10, 5.9), dpi=300)  # 增大图形的整体高度
patches, _, autotexts = plt.pie(
    sorted_percentages,
    labels=None,  # 不显示默认标签
    autopct='%1.2f%%',  # 百分比格式
    colors=sorted_colors,
    wedgeprops={'linewidth': 4, 'edgecolor': 'white'},
    startangle=startangle-25,  # 设置旋转角度
    pctdistance=0.7,  # 调整百分比文本位置
    labeldistance=0.5  # 调整标签位置，避免重叠
)

# 设置百分比字体大小
for autotext in autotexts:
    autotext.set_fontsize(22)

# 添加图例到图右边，居中显示
plt.legend(
    patches,
    sorted_categories,
    loc='center left',  # 图例左对齐
    bbox_to_anchor=(0.9, 0.5),  # 将图例放置在右侧，居中
    fontsize=18,  # 图例字体大小
    ncol=1,  # 图例为一列
    title_fontsize=18,  # 图例标题字体大小
    frameon=False      # 禁用图例边框
)

# 调整布局以减少左边空白并确保图例显示完整
plt.subplots_adjust(left=0.001, right=0.7, top=0.9999, bottom=0.0001)  # 减少左边距并增加右边距

plt.show()
