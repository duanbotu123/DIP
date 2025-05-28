import json
import pandas as pd
import os
import glob

motion_path = '/home/hlp/data/motion_text/3'
json_files = glob.glob(f'{motion_path}/**/*.json', recursive=True)

data = []
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        row = json.load(f)
        data.append(row)

df = pd.DataFrame(data)

df.to_excel(os.path.join(motion_path, 'motion_data.xlsx'), index=False)