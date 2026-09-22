'''
Author: Caoyh
Date: 2024-12-13 18:00:14
LastEditors: BellaCaoyh caoyh_cyh@163.com
LastEditTime: 2025-07-18 16:46:06
'''
from typing import Optional, Tuple, Any, Dict, NamedTuple, List
import json
import pickle

def dict2json(mydict: Dict | List, save_file:str):
    with open(save_file, 'w', encoding='utf-8') as f:
        # indent=4 会让JSON文件格式化，变得非常易读
        json.dump(mydict, f, indent=4)

def json2data(file:str)->Dict:
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def data2pkl(data, file_path: str):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def pkl2data(file:str):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def file2list(file:str)->List:
    with open(file, 'r') as f:
        lines = f.readlines()
    res = [line.strip() for line in lines]
    return res

def list2file(mylist:List, file:str):
    with open(file, 'w') as f:
        for l in mylist:
            f.write(f"{l}\n")

def save_txt(data, path):
    with open(path, 'w') as f:
        for each in data:
            f.write(each + '\n')


def read_txt(path):
    data = []
    with open(path, 'r') as f:
        for each in f.readlines():
            data.append(each.strip('\n'))
        return data

