from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdChemReactions import ReactionFromSmarts

# 你的带原子映射 RSMI
rsmi = "Cl[C:12]([c:11]1[cH:10][cH:9][c:8]([O:7][CH2:6][CH2:5][CH2:4][CH2:3][CH2:2][CH3:1])[cH:37][cH:36]1)=[O:13].[NH2:14][c:15]1[c:16]([O:17][c:18]2[cH:19][cH:20]([O:21][CH:22]([CH3:23])[c:24]3[n:25][n:26][n:27][nH:28]3)[cH:29][cH:30][cH:31]2)[cH:32][cH:33][cH:34][cH:35]1>CCN(CC)CC.Cl.ClCCl.c1ccccc1>[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][O:7][c:8]1[cH:9][cH:10][c:11]([C:12](=[O:13])[NH:14][c:15]2[c:16]([O:17][c:18]3[cH:19][cH:20]([O:21][CH:22]([CH3:23])[c:24]4[n:25][n:26][n:27][nH:28]4)[cH:29][cH:30][cH:31]3)[cH:32][cH:33][cH:34][cH:35]2)[cH:36][cH:37]1"

# RDKit ReactionFromSmarts 读取反应SMILES
rxn = ReactionFromSmarts(rsmi)
# 清理绘图设置
rxn.Initialize()

# 绘制反应图，调整图片尺寸
img = Draw.ReactionToImage(rxn, subImgSize=(500, 350), useSVG=False)

# 保存图片
img.save("reaction.png")
print("Reaction image saved as reaction.png")

# 如果在jupyter里，可以直接显示
# display(img)
