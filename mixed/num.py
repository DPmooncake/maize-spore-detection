import os
from collections import Counter

string_table = ['shs_spore', 'dbb_spore', 'xb_spore']  # 按顺序修改为类别列表
folder_path = r'E:\deep_learn\spore\VOCdevkit\VOC2007\YOLOLabels'  # 修改为txt文件夹
category_counter = Counter()

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            for line in file:
                category_index = int(line.split()[0])
                if category_index < len(string_table):
                    category = string_table[category_index]
                    category_counter[category] += 1
print("各类别数量:")
for category in string_table:
    count = category_counter[category]
    print(f"{category}: {count}")