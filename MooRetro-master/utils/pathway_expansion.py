import os
from utils.single_retro_fn import single_fn
from utils.utils import dict2json
from src.nodes import Node, MoleculeNode, ReactionNode
from src.synthesis_tree import SynthesisTree

def pathway_expansion(target_smile, building_blocks, ec_level=3, single_fn_type: str='biocyc', save_dir: str='./'):
    tree = SynthesisTree(target_smile, building_blocks)

    res = single_fn(target_smile, building_blocks, ec_level=ec_level, single_fn_type=single_fn_type)

    for r in res:
        tree.add_reaction(tree.root, r)
    
    molecule_info, reaction_info, connection_info = tree.collect_tree_info()

    result = {
        "molecules": molecule_info,
        "reactions": reaction_info,
        "connection": connection_info
    }
    save_file = os.path.join(save_dir, 'pathwayExpansion.json')
    dict2json(result, save_file)
    return save_file
    
    