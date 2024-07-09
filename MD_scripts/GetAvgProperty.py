import csv
import numpy as np
import pandas as pd

df = pd.read_csv('out100.csv', header=None)
struct_num = set([num for num in df[0]])

all_dict = []
for num in range(100):
    filtered_df = df[df[0] == num]
    final_dict = dict(zip(filtered_df[1], filtered_df[2]))
    all_dict.append(final_dict)

def generate_square_symmetric_variants(combo):
    """
    Generate all possible rotations and reflections (flips) of a 2x2 square represented by 'combo'.
    The combo is assumed to be in the order of a 2x2 matrix: [A, B, C, D]
    """
    if len(combo) != 8:  # Each element is assumed to be 2 characters long
        return set()

    # Splitting the combo into individual elements
    A, B, C, D = combo[0:2], combo[2:4], combo[4:6], combo[6:8]  # Assuming combo is like 'ABCD'

    # Generating rotations and flips
    rotations_and_flips = {
        A + B + C + D,  # Original
        B + C + D + A,  # Rotated 90 degrees clockwise
        D + C + B + A,  # Rotated 180 degrees
        C + B + A + D,  # Rotated 270 degrees clockwise
        C + D + A + B,  # Horizontally flipped
        A + D + C + B,  # Vertically flipped
        D + A + B + C,  # Diagonal flip (top-left to bottom-right)
        B + A + D + C   # Diagonal flip (top-right to bottom-left)
    }

    return list(rotations_and_flips)

df_properties = pd.read_csv('properties.csv')

sum_100 = []
for d in all_dict:
    total_count = sum(d.values())
    for combo in d:
        d[combo] /= total_count

    keys = list(d.keys())
    values = list(d.values())
    infos = []

    for key, value in zip(keys, values):
        sames = generate_square_symmetric_variants(key)
        found_match = False
        for index, row in df_properties.iterrows():
            if row[0] in sames:
                info = value*row[1:]
                infos.append(info)
                found_match = True
                break
        if not found_match:
            print(key)
    summary = sum(infos)
    sum_100.append(summary)

sum_all = sum(sum_100)/100
sum_all.to_csv('out_property.csv')

