import json
import sys
import os

dataset = 'davis'
with open('data/'+dataset+'/ligands_clean.txt', 'r') as file:
    lines = file.readlines()
data_dict = {}
print('len(ligands_clean)',len(lines))
for line in lines:
    columns = line.strip().split(' ')
    if len(columns) >= 2:
        key = columns[0]
        value = columns[1]
        if key not in data_dict:
            data_dict[key] = value
        else:
            print('key',key,data_dict[key])
print('len(data_dict)',len(data_dict))
with open('data/'+dataset+'/ligands_can.txt', 'w') as output_file:
    json.dump(data_dict, output_file, ensure_ascii=False, indent=4)

print("have saved ligands_can.txt")