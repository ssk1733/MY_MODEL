'''
Author: Caoyh
Date: 2025-07-18 16:44:33
LastEditors: BellaCaoyh caoyh_cyh@163.com
LastEditTime: 2025-07-19 09:22:36
'''
import os
import sys
import random
import re
# import pynvml
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
import torch
from typing import List, Dict, Tuple, Optional

from utils.chem import similarity_calculator, get_delta_sascore_for_feasibility
from config.config import SingleStepResult
from enzretro.model.product2order import Product2orderEC, BioCyc
from enzretro.extractor.reaction_actions import templates_to_reactants

_MODEL_CACHE = {}


def _get_single_step_model(single_fn_type: str, ec_level: int, device):
    cache_key = (single_fn_type, ec_level, str(device))
    if cache_key not in _MODEL_CACHE:
        if single_fn_type == 'biocyc':
            _MODEL_CACHE[cache_key] = BioCyc(ec_num=ec_level, device=device)
        else:
            _MODEL_CACHE[cache_key] = Product2orderEC(ec_num=ec_level, device=device)
    return _MODEL_CACHE[cache_key]


def prod2react(prod_smi:str, sequence:str):
    # print(sequence)
    # pat_ec = re.compile(r'\[EC\](\d+[\-\.\dnM]*?)\[')
    # pat_ord = re.compile('(\[\w+\]\[\w+\].*?\:\d+\.\d+|\[\w+\]\[\w+\].*?\:-?\d+)', re.I)
    # pat_ord = re.compile('(\[\w+\]\[\w+\].*?\:(?:(?:\d+\.\d+)|(?:\d+))?)', re.I)
    pat_ord = re.compile(r'\[\w+\]\[\w+\][^:]+:[+-]?\d+(?:\.\d+)?')
    ord_string = re.findall(pat_ord, sequence)
    
    ec_number = sequence.split("[EC]")[-1]
    
    ords = []
    atoms = []
    for s in ord_string:
        res = re.match(re.compile('\[(\w+)\]\[(\w+)\]([^\:].*?)\:(.*)', re.I), s)
        ords.append((' '.join([res.group(1), res.group(2)]), res.group(3)))
        if '.' in res.group(4):
            atoms.append([int(atom) for atom in res.group(4).split('.')])
        else: 
            atoms.append(int(res.group(4)))
    
    ords.append('Terminate')
    
    react_smi = None
    try:
        react_smi = Chem.MolToSmiles(templates_to_reactants(prod_smi, ords, atoms))
    except:
        pass
    return react_smi, ec_number

def split_cofactor(input_smiles: str, building_blocks: List[str]):
    """split reactants and cofactors based on similarity"""
    smis = input_smiles.split(".")
    
    if len(smis)==1:
        return input_smiles, None, max(similarity_calculator(smis[0], building_blocks), default=0.0)
    
    else:
        similarities = [max(similarity_calculator(rsmi, building_blocks), default=0) for rsmi in smis]
        max_similarity = max(similarities)
        max_similarity_idx = similarities.index(max(similarities))
        
        r_smi = smis[max_similarity_idx]
        smis.pop(max_similarity_idx)
        cofactors = '.'.join(smis)
        return r_smi, cofactors, max_similarity



# def single_fn_enzretro(target_smile, building_blocks, device='cpu', ec_level=4):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     p2oe = Product2orderEC(ec_num=ec_level, device=device)
#     result = p2oe(target_smile)
#     scores = result[0][1]
    
#     reacts = []
#     ecs = []
#     similarties = []
#     for s in result[1]:
#         react_smi, ec_number = prod2react(target_smile, s)
#         r_smi, cofactor,similarity = split_cofactor(react_smi, building_blocks)
#         reacts.append(r_smi)
#         ecs.append(ec_number)
#         similarties.append(similarity)
            
#     df = pd.DataFrame(columns=['reactants', 'scores', 'templates'])
#     df['reactants'] = reacts
#     df['scores'] = scores
#     df['templates'] = result[1]
#     df['ec'] = ecs
#     df['similarity'] = similarties
#     df.dropna(subset=['reactants'], axis='index', inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     return df.to_dict(orient='list')


def single_fn(target_smile, building_blocks, ec_level=3, single_fn_type: str='biocyc'):
    # ``model`` means neural-model inference. Historically it was grouped with
    # XLSX lookup aliases, which silently bypassed PyTorch and CUDA.
    if single_fn_type == 'model':
        single_fn_type = 'biocyc'
    if single_fn_type in {'model_xlsx', 'xlsx'}:
        from utils.model_xlsx_retro import single_fn_model_xlsx
        return single_fn_model_xlsx(target_smile, building_blocks)
    if single_fn_type not in {'biocyc', 'ecreact'}:
        raise ValueError(
            "single_fn_type must be 'biocyc'/'ecreact' for neural inference "
            "or 'model_xlsx'/'xlsx' for spreadsheet lookup"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    p2oe = _get_single_step_model(single_fn_type=single_fn_type, ec_level=ec_level, device=device)

    result = p2oe(target_smile)
    scores = result[0][1]
    
    reacts = []
    full_reactants = []
    cofactor_values = []
    ecs = []
    similarties = []
    for s in result[1]:
        react_smi, ec_number = prod2react(target_smile, s)
        r_smi, cofactor,similarity = None, None, 0.0
        if react_smi:
            r_smi, cofactor,similarity = split_cofactor(react_smi, building_blocks)
        reacts.append(r_smi)
        full_reactants.append(react_smi)
        cofactor_values.append(cofactor)
        ecs.append(ec_number)
        similarties.append(similarity)
            
    df = pd.DataFrame(columns=['reactants', 'scores', 'templates'])
    df['reactants'] = reacts
    df['all_reactants'] = full_reactants
    df['cofactors'] = cofactor_values
    df['scores'] = scores
    df['templates'] = result[1]
    df['ec'] = ecs
    df['similarity'] = similarties
    df['feasibility'] = [get_delta_sascore_for_feasibility(target_smile, rsmi) for rsmi in reacts ]
    df['top_k'] = [f"Top-{i+1}" for i in range(len(reacts))]
    df.dropna(subset=['reactants'], axis='index', inplace=True)
    df.drop_duplicates(subset=['all_reactants'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)

    res = []
    for _, row in df.iterrows():
        res.append(SingleStepResult(
            reactants=[row['reactants']],
            scores=round(row['scores'],3),
            templates=row['templates'],
            ec=row['ec'],
            similarity=round(row['similarity'],3), # 当前 reactant 和 commerical molecule 的最大相似度
            feasibility=round(row['feasibility'], 3),
            # No calibrated yield or toxicity model is loaded here. Returning
            # random values makes identical searches rank differently.
            yield_estimate=None,
            toxicity=None,
            all_reactants=[
                item for item in str(row['all_reactants'] or '').split('.') if item
            ],
            cofactors=[
                item for item in str(row['cofactors'] or '').split('.')
                if item and item != 'nan'
            ],
            top_k=row['top_k']
        ))

    return res





