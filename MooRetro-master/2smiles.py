import pandas as pd
import requests
import time

# =========================
# 1. 读取 Excel
# =========================
file_path = "/root/autodl-tmp/MooRetro-master/model_output.xlsx"
df = pd.read_excel(file_path, sheet_name="Metabolites")

names = df["Name"].dropna().tolist()

# =========================
# 2. PubChem 查询函数
# =========================
def get_smiles_from_pubchem(name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/TXT"
        r = requests.get(url, timeout=10)

        if r.status_code == 200:
            return r.text.strip()
        else:
            return ""
    except:
        return ""

# =========================
# 3. 批量映射
# =========================
smiles_list = []

for i, name in enumerate(names):
    print(f"[{i+1}/{len(names)}] Processing: {name}")

    smiles = get_smiles_from_pubchem(name)

    if smiles:
        smiles_list.append(smiles)
    else:
        smiles_list.append("")

    time.sleep(0.2)  # 防止API限速

# =========================
# 4. 写入 txt（每行一个 SMILES）
# =========================
output_file = "metabolites_smiles.txt"

with open(output_file, "w") as f:
    for smi in smiles_list:
        f.write(smi + "\n")

print("Done! Saved to:", output_file)