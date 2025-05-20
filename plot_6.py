import matplotlib.pyplot as plt

# 数据
categories = ['Privacy Leakage', 'Computer Security', 'Financial Loss', 'Property Damage', 'Physical Health',
              'Data Loss', 'Illegal Activities', 'Ethics & Morality', 'Bias & Offensiveness', 'Miscellaneous']
percentages = [18.43, 9.44, 16.18, 8.31, 10.79, 13.26, 5.62, 4.94, 8.09, 4.94]

# 颜色选择
colors = ['#42A942', '#FF8A23', '#277CB7', '#986EC0', '#878787', '#936056', '#E47AC3', '#C59610', '#DA4041', '#42A942']

# 按照百分比从大到小排序
sorted_indices = sorted(range(len(percentages)), key=lambda i: percentages[i], reverse=True)
sorted_categories = [categories[i] for i in sorted_indices]
sorted_percentages = [percentages[i] for i in sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]

# 绘制饼状图
plt.figure(figsize=(10, 8))  # 增大图形的整体高度
patches, _, autotexts = plt.pie(
    sorted_percentages,
    labels=None,  # 不显示默认标签
    autopct='%1.2f%%',  # 百分比格式
    colors=sorted_colors,
    wedgeprops={'linewidth': 4, 'edgecolor': 'white'},
    startangle=140,
    pctdistance=0.8,  # 调整百分比文本位置
    labeldistance=0.5  # 调整标签位置，避免重叠
)

# 设置百分比字体大小
for autotext in autotexts:
    autotext.set_fontsize(22)

# 添加图例到图下方，贴近饼图
plt.legend(
    patches[: 5],
    sorted_categories[: 5],
    loc='upper center',  # 图例放置在图的上方中心
    bbox_to_anchor=(0.5, 0.05),  # 调整到更贴近的下方位置
    fontsize=18,  # 图例字体大小
    ncol=3,  # 图例分为一排 3 列
    title_fontsize=14,  # 图例标题字体大小
    frameon=False      # 禁用图例边框
)

# 调整布局以减少顶部空白，同时确保图像形状不变
plt.subplots_adjust(left=0.1, right=0.9, top=0.99, bottom=0.1)  # 适度调整上下边距

plt.show()
import matplotlib.pyplot as plt

# 数据
categories = ['Privacy Leakage', 'Computer Security', 'Financial Loss', 'Property Damage', 'Physical Health',
              'Data Loss', 'Illegal Activities', 'Ethics & Morality', 'Bias & Offensiveness', 'Miscellaneous']
percentages = [18.43, 9.44, 16.18, 8.31, 10.79, 13.26, 5.62, 4.94, 8.09, 4.94]

# 颜色选择
colors = ['#42A942', '#FF8A23', '#277CB7', '#986EC0', '#878787', '#936056', '#E47AC3', '#C59610', '#DA4041', '#42A942']

# 按照百分比从大到小排序
sorted_indices = sorted(range(len(percentages)), key=lambda i: percentages[i], reverse=True)
sorted_categories = [categories[i] for i in sorted_indices]
sorted_percentages = [percentages[i] for i in sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]

# 绘制饼状图
plt.figure(figsize=(10, 8))  # 增大图形的整体高度
patches, _, autotexts = plt.pie(
    sorted_percentages,
    labels=None,  # 不显示默认标签
    autopct='%1.2f%%',  # 百分比格式
    colors=sorted_colors,
    wedgeprops={'linewidth': 4, 'edgecolor': 'white'},
    startangle=140,
    pctdistance=0.8,  # 调整百分比文本位置
    labeldistance=0.5  # 调整标签位置，避免重叠
)

# 设置百分比字体大小
for autotext in autotexts:
    autotext.set_fontsize(22)

# 添加图例到图下方，贴近饼图
plt.legend(
    patches[: 5],
    sorted_categories[: 5],
    loc='upper center',  # 图例放置在图的上方中心
    bbox_to_anchor=(0.5, 0.05),  # 调整到更贴近的下方位置
    fontsize=18,  # 图例字体大小
    ncol=3,  # 图例分为一排 3 列
    title_fontsize=14,  # 图例标题字体大小
    frameon=False      # 禁用图例边框
)

# 调整布局以减少顶部空白，同时确保图像形状不变
plt.subplots_adjust(left=0.1, right=0.9, top=0.99, bottom=0.1)  # 适度调整上下边距

plt.show()
import matplotlib.pyplot as plt

# 数据
categories = ['Privacy Leakage', 'Computer Security', 'Financial Loss', 'Property Damage', 'Physical Health',
              'Data Loss', 'Illegal Activities', 'Ethics & Morality', 'Bias & Offensiveness', 'Miscellaneous']
percentages = [18.43, 9.44, 16.18, 8.31, 10.79, 13.26, 5.62, 4.94, 8.09, 4.94]

# 颜色选择
colors = ['#42A942', '#FF8A23', '#277CB7', '#986EC0', '#878787', '#936056', '#E47AC3', '#C59610', '#DA4041', '#42A942']

# 按照百分比从大到小排序
sorted_indices = sorted(range(len(percentages)), key=lambda i: percentages[i], reverse=True)
sorted_categories = [categories[i] for i in sorted_indices]
sorted_percentages = [percentages[i] for i in sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]

# 绘制饼状图
plt.figure(figsize=(10, 8))  # 增大图形的整体高度
patches, _, autotexts = plt.pie(
    sorted_percentages,
    labels=None,  # 不显示默认标签
    autopct='%1.2f%%',  # 百分比格式
    colors=sorted_colors,
    wedgeprops={'linewidth': 4, 'edgecolor': 'white'},
    startangle=140,
    pctdistance=0.8,  # 调整百分比文本位置
    labeldistance=0.5  # 调整标签位置，避免重叠
)

# 设置百分比字体大小
for autotext in autotexts:
    autotext.set_fontsize(22)

# 添加图例到图下方，贴近饼图
plt.legend(
    patches[: 5],
    sorted_categories[: 5],
    loc='upper center',  # 图例放置在图的上方中心
    bbox_to_anchor=(0.5, 0.05),  # 调整到更贴近的下方位置
    fontsize=18,  # 图例字体大小
    ncol=3,  # 图例分为一排 3 列
    title_fontsize=14,  # 图例标题字体大小
    frameon=False      # 禁用图例边框
)

# 调整布局以减少顶部空白，同时确保图像形状不变
plt.subplots_adjust(left=0.1, right=0.9, top=0.99, bottom=0.1)  # 适度调整上下边距

plt.show()
