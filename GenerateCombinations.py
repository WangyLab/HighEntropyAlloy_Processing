import pandas as pd
import numpy as np

num_rows = 100000
num_columns = 5
target_sum = 60

def generate_row_no_zero(target_sum, num_columns):
    while True:
        # Generate random integers between 1 and target_sum-1 (to avoid zeros)
        row = np.random.randint(1, target_sum, size=num_columns - 1)
        last_value = target_sum - sum(row)
        # Check if the last value is positive and non-zero
        if last_value > 0:
            row = np.append(row, last_value)
            np.random.shuffle(row)
            return row

rows_no_zero = [generate_row_no_zero(target_sum, num_columns) for _ in range(num_rows)]
df_sum_60_no_zero = pd.DataFrame(rows_no_zero, columns=[f'Column_{i+1}' for i in range(num_columns)])

df_sum_60_no_zero.to_csv('sum_60.csv')