import os
import re
import numpy as np
import csv
import pandas as pd

def get_atom_dict(data):  # 获得每个元素的索引字典 | {'1': 'Mn', '2': 'Fe', '3': 'Co', '4': 'Ni', '5': 'Cu', '6': 'N', '7': 'C', '8': 'Fe'}
    with open(data) as file:
        lines = file.read()
    atom = np.array(
        lines[re.search('Masses', lines).span()[1]:re.search('Bond Coeffs', lines).span()[0]].split()).reshape(-1, 4)
    num = atom[:, 0]
    atom = [re.findall('[a-zA-Z]*', i[-1])[0] for i in atom]
    atom = dict(zip(num, atom))
    return atom

def get_distance(a, b, size):
    distance = np.abs(a - b)  # 计算a金属与b中每一个金属的距离
    distance = np.min([distance, size - distance], axis=0)  # 对比真实距离和周期性距离，取最小值
    distance = (distance ** 2).sum(axis=-1) ** 0.5  # 最后求得两个金属之间的欧氏距离
    distance = distance.tolist()
    return distance

metal_num = []
group_num = []
Metal = ['Co', 'Cu', 'Fe', 'Mn', 'Ni']
group = []

def get_unique_structures(compounds_xyz, name):
    unique_structures = {}
    for i in range(len(compounds_xyz)):
        for j in range(i + 1, len(compounds_xyz)):
            if len(compounds_xyz[i]) == len(compounds_xyz[j]):
                if all(any(np.array_equal(i_elem, k_elem) for k_elem in compounds_xyz[j]) for i_elem in
                       compounds_xyz[i]) \
                        and all(
                    any(np.array_equal(k_elem, i_elem) for i_elem in compounds_xyz[i]) for k_elem in compounds_xyz[j]):
                    structure_key = tuple(map(tuple, compounds_xyz[j]))
                    if structure_key not in unique_structures:
                        unique_structures[structure_key] = name[j]
    unordered_xyz = [list(map(np.array, key)) for key in unique_structures]
    unordered_name = list(unique_structures.values())
    return unordered_xyz, unordered_name

def get_structs(direct):
    print(direct)
    if not os.path.exists(f'{direct}/movie.data'):
        metal_num.append([])
        group_num.append([])
        return
    atom_dict = get_atom_dict(f'{direct}/data.box')  # 获取data.box中的元素种类字典
    with open(f'{direct}/movie.data') as file:
        lines = file.readlines()

    num =  450 # 体系的原子数量
    elements = []
    final_info_100 = []
    metals_100 = []
    elements_100 = []

    for i, line in enumerate(lines):
        if line == 'ITEM: ATOMS id type x y z \n':
            # time = int(lines[i - 7])
            # if time < 1000000:  # 判断是否达到了总步数
            #     continue

            # 盒子边界
            size = lines[i - 3].split()
            size = float(size[1]) - float(size[0]) # 晶格的边长, 立方盒子

            atoms = lines[i + 1:i + num + 1]  # 定位最后一步的原子信息
            metals, elements = [], []  # metals包含了金属的坐标信息，elements是金属的字符串

            # 识别金属与获取坐标
            for atom in atoms:
                atom = atom.split()  # index: 0为原子编号, 1为原子类型, 2-4对应x,y,z
                element = atom_dict[atom[1]]  # 得到atom的原子类型 | 元素字符串
                if element in Metal:
                    elements.append(element)
                    metals.append(atom[-3:])

            metals = np.array(metals).astype(float)  # shape: (90, 3), 每行对应一个金属原子坐标信息
            elements = np.array(elements)  # shape: (90,), 含有所有金属元素

            name=[]  # 最终的组合name信息(字符串)
            compounds_xyz = []  # 最终组合的xyz信息


            for index,metal in enumerate(metals):  # metal是每个金属的坐标 | array
                distances=get_distance(metal, metals, size)  # 计算该金属与所有金属的欧氏距离
                for id,dis in enumerate(distances):
                    distances[id]=[dis,id] #距离，金属的index | distance: [[0.0, 0], [4.142241729426229, 1], [3.768254835689859, 2]]
                distances=np.array(distances)
                neighbor = distances[np.argsort(distances[:, 0])] # 按照距离的大小排序，[距离，序号], array | shape: (90, 2)

                # 中心金属原子+最近的4个金属原子名字, example: 'FeNiCoCuCu'
                near4_center = [elements[int(neighbor[0][1])] + elements[int(neighbor[1][1])] + elements[int(neighbor[2][1])] + elements[int(neighbor[3][1])]+elements[int(neighbor[4][1])]]

                # 计算ABCD中的邻位与对位，这是四个原子的坐标
                near1_xyz = metals[int(neighbor[1][1])]
                near2_xyz = metals[int(neighbor[2][1])]
                near3_xyz = metals[int(neighbor[3][1])]
                near4_xyz = metals[int(neighbor[4][1])]

                ## near1与near3是对位，与near2, near4是邻位, near1是array,包含了坐标信息
                dis_ABCD = [get_distance(near1_xyz, near2_xyz, size),get_distance(near1_xyz, near3_xyz, size),get_distance(near1_xyz, near4_xyz, size)]
                index_duiwei = dis_ABCD.index(max(dis_ABCD)) # 对位原子的index
                aa=[near2_xyz, near3_xyz, near4_xyz]
                near1 = near1_xyz
                near3 = aa[index_duiwei]
                del aa[index_duiwei]
                near2, near4 = aa[0], aa[1]  # 1与3是对位

                ## 获取near对应的name
                near1_name = elements[int(neighbor[1][1])]
                bb = [elements[int(neighbor[2][1])],elements[int(neighbor[3][1])],elements[int(neighbor[4][1])]]
                near3_name = bb[index_duiwei]
                del bb[index_duiwei]
                near2_name, near4_name = bb[0], bb[1]

                # M+2个第一配位原子的组合，另一个原子利用距离判断
                # compounds包含坐标信息，compounds_name包含元素种类(字符串)信息
                # 顺序：第一个是M，第二、三个是对位的元素   |  在菱形体系内，而不是X体系内
                compounds = [(metal, near1, near2), (metal, near1, near4),
                             (metal, near2, near3), (metal, near3, near4)]
                compounds_name = [(elements[int(neighbor[0][1])], near1_name, near2_name), (elements[int(neighbor[0][1])], near1_name, near4_name),
                             (elements[int(neighbor[0][1])], near2_name, near3_name), (elements[int(neighbor[0][1])], near3_name, near4_name)]

                for i, compound in enumerate(compounds):
                    # 搜索到这三个元素最近的金属信息
                    distances4=[]
                    for mm in range(len(get_distance(compound[1],metals,size))):
                        sum = get_distance(compound[1],metals,size)[mm] + get_distance(compound[2],metals,size)[mm] + get_distance(compound[0],metals,size)[mm]
                        distances4.append(sum)

                    for id, dis in enumerate(distances4):
                        distances4[id] = [dis, id]  # 距离，金属的序号
                    distances4 = np.array(distances4)
                    neighbor4 = distances4[np.argsort(distances4[:, 0])]

                    near_other = metals[int(neighbor4[3][1])] # 找到的原子坐标, 四个原子组成：compound和near_other
                    all_xyz = compound+(near_other,)  # 四个原子的坐标信息
                    compounds_xyz.append(all_xyz)

                    # 顺序：第一个是金属，第二个是compound[1]，第三个是compound[2]，第四个是搜索出的金属
                    all_name = compounds_name[i][0]+compounds_name[i][1]+compounds_name[i][2]+elements[int(neighbor4[3][1])]
                    name.append(all_name)

            # uniq_xyz, uniq_name = get_unique_structures(compounds_xyz, name)
            final_info_100.append((name, compounds_xyz))
            metals_100.append(metals)
            elements_100.append(elements)

    return final_info_100, metals_100, elements_100  # 返回的最好是group,有组合名字和出现次数


def find_name(array_xyz, metals, elements):
    index = None
    for i, row in enumerate(metals):
        if np.array_equal(row,array_xyz):
            index=i
            break
    name = elements[index]
    return name

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

    return rotations_and_flips


def merge_symmetric_combinations_with_count(combinations):
    """
    Merge combinations that are the same due to the symmetry of a 2x2 square (rotations and flips),
    and count the occurrences of each unique combination including its symmetric variants.
    """
    combination_counts = {}

    for combo in combinations:
        # Check if the combo length is correct
        if len(combo) != 8:
            continue

        # Generate all symmetric variants of the current combination
        variants = generate_square_symmetric_variants(combo)

        # Find if any variant is already in the combination_counts
        found_variant = None
        for variant in variants:
            if variant in combination_counts:
                found_variant = variant
                break

        if found_variant:
            # If a variant is found in the counts, increment its count
            combination_counts[found_variant] += 1
        else:
            # Otherwise, add the current combination as a new entry with count 1
            combination_counts[combo] = 1

    return combination_counts

for i in range(1, 2):
    final_info_100, metals_100, elements_100 = get_structs(f'{i}')
    # 100组结构的最终信息 | 多个dict
    summary_dicts_all = []
    for idx, info in enumerate(final_info_100):
        names = info[0]
        compounds_xyz = info[1]
        unique_structures = {}

        for i in range(len(compounds_xyz)):
            for j in range(i + 1, len(compounds_xyz)):
                if len(compounds_xyz[i]) == len(compounds_xyz[j]):
                    if all(any(np.array_equal(i_elem, k_elem) for k_elem in compounds_xyz[j]) for i_elem in
                           compounds_xyz[i]) \
                            and all(any(np.array_equal(k_elem, i_elem) for i_elem in compounds_xyz[i]) for k_elem in
                                    compounds_xyz[j]):
                        structure_key = tuple(map(tuple, compounds_xyz[j]))
                        if structure_key not in unique_structures:
                            unique_structures[structure_key] = names[j]
        unordered_xyz = [list(map(np.array, key)) for key in unique_structures]
        # unordered_name = list(unique_structures.values())

        # 重新设计元素组合顺序
        all_name = []
        for compound in unordered_xyz:
            atom1 = compound[0]
            atom2 = compound[1]
            atom3 = compound[3]
            atom4 = compound[2]
            name = find_name(atom1, metals_100[idx], elements_100[idx]) + find_name(atom2, metals_100[idx],
                                                                                    elements_100[idx]) + find_name(
                atom3, metals_100[idx], elements_100[idx]) + find_name(atom4, metals_100[idx], elements_100[idx])
            all_name.append(name)

        summary_dict = merge_symmetric_combinations_with_count(all_name)  # 当下结构的info
        summary_dicts_all.append(summary_dict)  # 添加到总list中

    file_path = "out100.csv"
    # 写入CSV
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # 写入每个字典的键值对
        for i, d in enumerate(summary_dicts_all):
            for key, value in d.items():
                writer.writerow([i, key, value])

    print("CSV文件已保存")
