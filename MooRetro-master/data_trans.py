import pandas as pd
import requests
import time
import urllib.parse

# ==============================
# 1. PubChem 查询函数
# ==============================
def get_smiles_from_pubchem(name):
    try:
        # URL encode
        encoded_name = urllib.parse.quote(name)

        # Step 1: name -> CID
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/cids/JSON"
        r = requests.get(cid_url, timeout=10)

        if r.status_code != 200:
            return None

        data = r.json()
        cid = data["IdentifierList"]["CID"][0]

        # Step 2: CID -> SMILES
        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
        r2 = requests.get(smiles_url, timeout=10)

        if r2.status_code != 200:
            return None

        smiles = r2.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]

        # ==============================
        # 3. 清洗 SMILES（按你的要求）
        # ==============================
        if smiles:
            smiles = smiles.replace("/", "").replace("\\", "")

        return smiles

    except Exception as e:
        return None


# ==============================
# 2. 读取 Excel
# ==============================
file_path = "/root/1/MooRetro-master/model_output.xlsx"
df = pd.read_excel(file_path, sheet_name="Metabolites")

# ==============================
# 3. 遍历 name 列查询 SMILES
# ==============================
smiles_list = []

for i, name in enumerate(df["Name"]):
    print(f"Processing {i+1}/{len(df)}: {name}")

    if pd.isna(name):
        smiles_list.append(None)
        continue

    smiles = get_smiles_from_pubchem(str(name))
    smiles_list.append(smiles)

    # 防止请求过快
    time.sleep(0.2)


# ==============================
# 4. 写入新列
# ==============================
df["SMILES"] = smiles_list

# ==============================
# 5. 保存结果
# ==============================
output_file = "model_output_with_smiles.xlsx"
df.to_excel(output_file, index=False)

print("Done! Saved to:", output_file)