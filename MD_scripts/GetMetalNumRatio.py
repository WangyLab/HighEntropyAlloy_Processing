import os
import re
import numpy as np
import csv
import pandas as pd

# Get the index dictionary of each element | {'1': 'Mn', '2': 'Fe', '3': 'Co', '4': 'Ni', '5': 'Cu', '6': 'N', '7': 'C', '8': 'Fe'}
def get_atom_dict(data):
    with open(data) as file:
        lines = file.read()
    atom = np.array(
        lines[re.search('Masses', lines).span()[1]:re.search('Bond Coeffs', lines).span()[0]].split()).reshape(-1, 4)
    num = atom[:, 0]
    atom = [re.findall('[a-zA-Z]*', i[-1])[0] for i in atom]
    atom = dict(zip(num, atom))
    return atom

def get_distance(a, b, size):
    distance = np.abs(a - b)  # Get the distance between metal 'a' and each metal in [b]
    distance = np.min([distance, size - distance], axis=0)  # Compare the actual distance and the periodic distance and take the minimum value
    distance = (distance ** 2).sum(axis=-1) ** 0.5  # Get the Euclidean distance between the two metals
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
    atom_dict = get_atom_dict(f'{direct}/data.box')  # Get the element type dictionary in {data.box}
    with open(f'{direct}/movie.data') as file:
        lines = file.readlines()

    num =  450 # The number of atoms in the system
    elements = []
    final_info_100 = []
    metals_100 = []
    elements_100 = []

    for i, line in enumerate(lines):
        if line == 'ITEM: ATOMS id type x y z \n':
            # time = int(lines[i - 7])
            # if time < 1000000:  # Determine whether reached the total steps
            #     continue

            # 盒子边界
            size = lines[i - 3].split()
            size = float(size[1]) - float(size[0]) # Side length of the lattice, cubic box

            atoms = lines[i + 1:i + num + 1]  # Locating the atomic information of the last step
            metals, elements = [], []  # [metals] contains the coordinate information of the metal, and [elements] is the string of the metal

            # 识别金属与获取坐标
            for atom in atoms:
                atom = atom.split()  # index: 0 is the atom number, 1 is the atom type, 2-4 corresponds to x, y, z
                element = atom_dict[atom[1]]  # Get the atomic type of atom | element string
                if element in Metal:
                    elements.append(element)
                    metals.append(atom[-3:])

            metals = np.array(metals).astype(float)  # shape: (90, 3), each row corresponds to the coordinate information of a metal atom
            elements = np.array(elements)  # shape: (90,), contains all metal elements

            name=[]  # Final combination's name
            compounds_xyz = []  # Final combination's xyz


            for index,metal in enumerate(metals):  # (metal) is the coordinates of each metal | array | array
                distances=get_distance(metal, metals, size)  # Get the Euclidean distance between the metal and all metals
                for id,dis in enumerate(distances):
                    distances[id]=[dis,id] # distance，metal's index | distance: [[0.0, 0], [4.142241729426229, 1], [3.768254835689859, 2]]
                distances=np.array(distances)
                neighbor = distances[np.argsort(distances[:, 0])] # Sort by distance，[distance，index], array | shape: (90, 2)

                # Central metal atom + the names of the 4 nearest metal atoms, example: 'FeNiCoCuCu'
                near4_center = [elements[int(neighbor[0][1])] + elements[int(neighbor[1][1])] + elements[int(neighbor[2][1])] + elements[int(neighbor[3][1])]+elements[int(neighbor[4][1])]]

                # Calculate the ortho and para positions in ABCD, which are the coordinates of the four atoms
                near1_xyz = metals[int(neighbor[1][1])]
                near2_xyz = metals[int(neighbor[2][1])]
                near3_xyz = metals[int(neighbor[3][1])]
                near4_xyz = metals[int(neighbor[4][1])]

                ## near1 and near3 are aligned, and near2 and near4 are adjacent. near1 is an array, which contains coordinate information.
                dis_ABCD = [get_distance(near1_xyz, near2_xyz, size),get_distance(near1_xyz, near3_xyz, size),get_distance(near1_xyz, near4_xyz, size)]
                index_duiwei = dis_ABCD.index(max(dis_ABCD)) # Index of the para atom
                aa=[near2_xyz, near3_xyz, near4_xyz]
                near1 = near1_xyz
                near3 = aa[index_duiwei]
                del aa[index_duiwei]
                near2, near4 = aa[0], aa[1]  # 1 and 3 are opposite

                ## Get the name corresponding to near
                near1_name = elements[int(neighbor[1][1])]
                bb = [elements[int(neighbor[2][1])],elements[int(neighbor[3][1])],elements[int(neighbor[4][1])]]
                near3_name = bb[index_duiwei]
                del bb[index_duiwei]
                near2_name, near4_name = bb[0], bb[1]

                # The combination of M+2 first coordination atoms, and the other atom is determined by distance
                # Compounds contains coordinate information, compounds_name contains element type (string) information
                # Order: The first is M, the second and third are the opposite elements | In the diamond system, not the X system
                compounds = [(metal, near1, near2), (metal, near1, near4),
                             (metal, near2, near3), (metal, near3, near4)]
                compounds_name = [(elements[int(neighbor[0][1])], near1_name, near2_name), (elements[int(neighbor[0][1])], near1_name, near4_name),
                             (elements[int(neighbor[0][1])], near2_name, near3_name), (elements[int(neighbor[0][1])], near3_name, near4_name)]

                for i, compound in enumerate(compounds):
                    # Search for the nearest metal information of these three elements
                    distances4=[]
                    for mm in range(len(get_distance(compound[1],metals,size))):
                        sum = get_distance(compound[1],metals,size)[mm] + get_distance(compound[2],metals,size)[mm] + get_distance(compound[0],metals,size)[mm]
                        distances4.append(sum)

                    for id, dis in enumerate(distances4):
                        distances4[id] = [dis, id]  # Distance, metal serial number
                    distances4 = np.array(distances4)
                    neighbor4 = distances4[np.argsort(distances4[:, 0])]

                    near_other = metals[int(neighbor4[3][1])] # The atomic coordinates found, four atoms: compound and near_other
                    all_xyz = compound+(near_other,)  # Coordinate information of four atoms
                    compounds_xyz.append(all_xyz)

                    # Order: The first one is metal, the second one is compound[1], the third one is compound[2], and the fourth one is the searched metal
                    all_name = compounds_name[i][0]+compounds_name[i][1]+compounds_name[i][2]+elements[int(neighbor4[3][1])]
                    name.append(all_name)

            # uniq_xyz, uniq_name = get_unique_structures(compounds_xyz, name)
            final_info_100.append((name, compounds_xyz))
            metals_100.append(metals)
            elements_100.append(elements)

    return final_info_100, metals_100, elements_100  # The best thing to return is group, with the combination name and number of occurrences


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
    # Final information of 100 structures | Multiple dicts
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

        # Redesign the order of element combination
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

        summary_dict = merge_symmetric_combinations_with_count(all_name)  # Current structure info
        summary_dicts_all.append(summary_dict)

    file_path = "out100.csv"
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for i, d in enumerate(summary_dicts_all):
            for key, value in d.items():
                writer.writerow([i, key, value])
    print("CSV file saved")