import matplotlib.pyplot as plt

# 数据
# categories = ['Privacy Leakage', 'Computer Security', 'Financial Loss', 'Property Damage', 'Physical Health',
#               'Data Loss', 'Illegal Activities', 'Ethics & Morality', 'Bias & Offensiveness', 'Miscellaneous']

categories = ['P.L.', 'C.S.', 'F.L.', 'P.D.', 'P.H.',
              'D.L.', 'I.A.', 'E.M.', 'B.O.', 'M.']
percentages = [18.43, 9.44, 16.18, 8.31, 10.79, 13.26, 5.62, 4.94, 8.09, 4.94]

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
    autotext.set_fontsize(24)

# 添加图例到图右边，居中显示
plt.legend(
    patches,
    sorted_categories,
    loc='center left',  # 图例左对齐
    bbox_to_anchor=(0.9, 0.5),  # 将图例放置在右侧，居中
    fontsize=22,  # 图例字体大小
    ncol=1,  # 图例为一列
    title_fontsize=22,  # 图例标题字体大小
    frameon=False      # 禁用图例边框
)

# 调整布局以减少左边空白并确保图例显示完整
plt.subplots_adjust(left=0.001, right=0.7, top=0.9999, bottom=0.0001)  # 减少左边距并增加右边距

plt.show()
