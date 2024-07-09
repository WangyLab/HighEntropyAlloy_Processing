import itertools
import random
import csv

def find_combinations(total, num_items, min_value, max_value, step):
    # 生成可能的数值
    values = [min_value + i * step for i in range(int((max_value - min_value) / step) + 1)]
    all_combinations = []

    # 使用迭代器生成所有可能的组合
    for combination in itertools.product(values, repeat=num_items):
        if round(sum(combination), 2) == total:
            all_combinations.append([round(num, 2) for num in combination])

    return all_combinations

# 参数设置
total_sum = 100
num_items = 5
min_value = 5
max_value = 35
step = 0.5

# 获取所有可能的组合
all_valid_combinations = find_combinations(total_sum, num_items, min_value, max_value, step)

# 保存到CSV文件
csv_file = "combinations0.5.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入标题行
    writer.writerow(["Number 1", "Number 2", "Number 3", "Number 4", "Number 5"])
    # 写入数据行
    for combination in all_valid_combinations:
        writer.writerow(combination)

print(f"Saved {len(all_valid_combinations)} combinations to {csv_file}")
