from collections import OrderedDict
import itertools

total_metals = ['Mn', 'Fe', 'Co', 'Ni', 'Cu']

def consolidate_elements(file_path):
    """
    Consolidate element information in 1 POSCAR file, reorder the elements and counts,
    and rearrange the atomic coordinates accordingly.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract elements and their counts
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]

    # Consolidate elements and update counts
    consolidated_elements = OrderedDict()
    for element, count in zip(elements, counts):
        if element in consolidated_elements:
            consolidated_elements[element] += count
        else:
            consolidated_elements[element] = count

    # Update lines 5 and 6
    lines[5] = ' '.join(consolidated_elements.keys()) + '\n'
    lines[6] = ' '.join(str(count) for count in consolidated_elements.values()) + '\n'

    # Rearrange atomic coordinates
    atom_index = 0
    sorted_atoms = {element: [] for element in consolidated_elements}
    for element, count in zip(elements, counts):
        for _ in range(count):
            sorted_atoms[element].append(lines[9 + atom_index])
            atom_index += 1

    # Update atomic coordinates in the lines
    lines[9:] = [atom for element in consolidated_elements for atom in sorted_atoms[element]]

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)


with open('POSCAR.txt', 'r') as file:
    lines = file.readlines()
    elements = lines[5].strip().split()
    atoms_num = lines[6].strip().split()
    elements_fir = elements[:4]
    atoms_fir = atoms_num[:4]

for time in range(1, 6):
    # 1. 四种原子为同种元素
    if time == 1:
        for metal in total_metals:
            elements_new = elements_fir+[metal]*4
            atoms_new = atoms_fir+['1']*4
            lines[5] = ' '.join(elements_new) + ' \n'
            lines[6] = ' '.join(atoms_new) + ' \n'
            new_file_name = f'4{metal}.txt'
            with open('data\\1\\'+new_file_name, 'w')as file:
                file.writelines(lines)
            consolidate_elements('data\\1\\'+new_file_name)
    if time == 2:
        for metal1 in total_metals:
            for metal2 in list(set(total_metals)-set([metal1])):
                elements_new = elements_fir+[metal1]*3+[metal2]
                atoms_new = atoms_fir+['1']*4
                lines[5] = ' '.join(elements_new) + ' \n'
                lines[6] = ' '.join(atoms_new) + ' \n'
                new_file_name = f'3{metal1}_{metal2}.txt'
                with open('data\\2\\'+new_file_name, 'w')as file:
                    file.writelines(lines)
                consolidate_elements('data\\2\\'+new_file_name)
    if time == 3:
        combinations=[]
        for q in range(len(total_metals)):
            for k in range(q+1, len(total_metals)):
                combination = (total_metals[q], total_metals[k])
                combinations.append(combination)
        for _, (metal1, metal2) in enumerate(combinations):
            for i in ['diagonal', 'adjacent']:
                if i == 'diagonal':
                    elements_new = elements_fir+[metal1]+[metal1]+[metal2]+[metal2]
                    atoms_new = atoms_fir+['1']*4
                    lines[5] = ' '.join(elements_new) + ' \n'
                    lines[6] = ' '.join(atoms_new) + ' \n'
                    new_file_name = f'2{metal1}_2{metal2}_{i}.txt'
                    with open('data\\3\\' + new_file_name, 'w') as file:
                        file.writelines(lines)
                    consolidate_elements('data\\3\\' + new_file_name)
                if i == 'adjacent':
                    elements_new = elements_fir + [metal1] + [metal2] + [metal1] + [metal2]
                    atoms_new = atoms_fir + ['1'] * 4
                    lines[5] = ' '.join(elements_new) + ' \n'
                    lines[6] = ' '.join(atoms_new) + ' \n'
                    new_file_name = f'2{metal1}_2{metal2}_{i}.txt'
                    with open('data\\3\\' + new_file_name, 'w') as file:
                        file.writelines(lines)
                    consolidate_elements('data\\3\\' + new_file_name)
    if time == 4:
        for metal1 in total_metals:
            combinations=[]
            for q in range(len(list(set(total_metals)-set([metal1])))):
                for k in range(q+1, len(list(set(total_metals)-set([metal1])))):
                    combination = (list(set(total_metals)-set([metal1]))[q],
                                   list(set(total_metals)-set([metal1]))[k])
                    combinations.append(combination)
            for _, (metal2, metal3) in enumerate(combinations):
                for i in ['diagonal', 'adjacent']:
                    if i == 'diagonal':
                        elements_new = elements_fir+[metal1]*2+[metal2]+[metal3]
                        atoms_new = atoms_fir+['1']*4
                        lines[5] = ' '.join(elements_new) + ' \n'
                        lines[6] = ' '.join(atoms_new) + ' \n'
                        new_file_name = f'2{metal1}_{metal2}_{metal3}_{i}.txt'
                        with open('data\\4\\' + new_file_name, 'w') as file:
                            file.writelines(lines)
                        consolidate_elements('data\\4\\' + new_file_name)
                    if i == 'adjacent':
                        elements_new = elements_fir + [metal1] + [metal2] + [metal1] + [metal3]
                        atoms_new = atoms_fir + ['1'] * 4
                        lines[5] = ' '.join(elements_new) + ' \n'
                        lines[6] = ' '.join(atoms_new) + ' \n'
                        new_file_name = f'2{metal1}_{metal2}_{metal3}_{i}.txt'
                        with open('data\\4\\' + new_file_name, 'w') as file:
                            file.writelines(lines)
                        consolidate_elements('data\\4\\' + new_file_name)
    if time == 5:
        combinations1 = list(itertools.combinations(total_metals, 4))
        for _, (metal1, metal2, metal3, metal4) in enumerate(combinations1):
            elements_new1 = elements_fir + [metal1] + [metal2] + [metal3] + [metal4]
            elements_new2 = elements_fir + [metal1] + [metal3] + [metal2] + [metal4]
            elements_new3 = elements_fir + [metal1] + [metal4] + [metal2] + [metal3]
            atoms_new = atoms_fir + ['1'] * 4
            for i in range(1, 4):
                if i == 1:
                    lines[5] = ' '.join(elements_new1) + ' \n'
                    lines[6] = ' '.join(atoms_new) + ' \n'
                    new_file_name = f'{metal1}_{metal2}_{metal3}_{metal4}_{i}.txt'
                    with open('data\\5\\' + new_file_name, 'w') as file:
                        file.writelines(lines)
                    consolidate_elements('data\\5\\' + new_file_name)
                if i == 2:
                    lines[5] = ' '.join(elements_new2) + ' \n'
                    lines[6] = ' '.join(atoms_new) + ' \n'
                    new_file_name = f'{metal1}_{metal2}_{metal3}_{metal4}_{i}.txt'
                    with open('data\\5\\' + new_file_name, 'w') as file:
                        file.writelines(lines)
                    consolidate_elements('data\\5\\' + new_file_name)
                if i == 3:
                    lines[5] = ' '.join(elements_new3) + ' \n'
                    lines[6] = ' '.join(atoms_new) + ' \n'
                    new_file_name = f'{metal1}_{metal2}_{metal3}_{metal4}_{i}.txt'
                    with open('data\\5\\' + new_file_name, 'w') as file:
                        file.writelines(lines)
                    consolidate_elements('data\\5\\' + new_file_name)

