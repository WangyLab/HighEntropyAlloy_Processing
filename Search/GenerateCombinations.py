import csv
from itertools import product

def generate_combinations(target_sum, min_val, max_val, num_elements):
    for values in product(range(min_val, max_val + 1), repeat=num_elements):
        if sum(values) == target_sum:
            yield values

# 设置参数
target_sum = 60
min_val, max_val = 3, 21
num_elements = 5

# 输出文件路径
output_file = 'combinations.csv'

# 使用生成器产生组合并存储到CSV文件中
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['A', 'B', 'C', 'D', 'E'])  # 写入表头

    for combination in generate_combinations(target_sum, min_val, max_val, num_elements):
        writer.writerow(combination)
