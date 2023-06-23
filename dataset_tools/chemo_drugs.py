# From https://raw.githubusercontent.com/mcoplan11/service_denial/master/generate_fine_tuning_data.py

from typing import Tuple, Hashable

import pandas as pd
import json
import numpy as np
from pandas import Series

'''
Fine tuning data sourced from:
https://seer.cancer.gov/seertools/seerrx/
https://drugcomb.fimm.fi/
Hand crafted letters 
'''

HCPCS_mapping = {
'J9035': {'Generic Name': 'Bevacizumab', 'Major Drug Class': 'Monoclonal Antibody'},
'J9271': {'Generic Name': 'Pembrolizumab', 'Major Drug Class': 'Checkpoint Inhibitor'},
'J9305': {'Generic Name': 'Pemetrexed', 'Major Drug Class': 'Antimetabolite'},
'J9299': {'Generic Name': 'Nivolumab', 'Major Drug Class': 'Checkpoint Inhibitor'},
'J9310': {'Generic Name': 'Rituximab', 'Major Drug Class': 'Monoclonal Antibody'},
'J9264': {'Generic Name': 'Paclitaxel', 'Major Drug Class': 'Antimitotic Agent'},
'J9312': {'Generic Name': 'Rituximab', 'Major Drug Class': 'Monoclonal Antibody'},
'J9228': {'Generic Name': 'Ipilimumab', 'Major Drug Class': 'Checkpoint Inhibitor'},
'J9355': {'Generic Name': 'Trastuzumab', 'Major Drug Class': 'Monoclonal Antibody'},
'J9041': {'Generic Name': 'Bortezomib', 'Major Drug Class': 'Proteasome Inhibitor'},
'J9999': {'Generic Name': 'Chemotherapy - non specific', 'Major Drug Class': 'non specific'},
'Q2050': {'Generic Name': 'Doxorubicin', 'Major Drug Class': 'Antitumor Antibiotic'},
'C9492': {'Generic Name': 'Durvalumab', 'Major Drug Class': 'Checkpoint Inhibitor'},
}


file = 'data_sources/drugs.csv'
chemo_drugs = pd.read_csv(file)
chemo_drugs = chemo_drugs.dropna(subset=['Primary Site'])



cases = []
for drug in chemo_drugs.iterrows():
    drug = drug[1].to_dict()
    PROMPT = f"Can you tell me the 'Alternate Name', 'Primary Sites', 'Category', 'Sub-category', and any significant 'Remarks' for the cancer drug named: {drug.get('Name')}?"
    case = {}
    case['drug'] = drug.get('Name')
    case['question'] = PROMPT
    answer = f"'Alternate Name': {drug.get('Name')}, 'Primary Sites': {drug.get('Primary Site')}, 'Category': {drug.get('Category')}, 'Remarks': {drug.get('Remarks')}"
    case['answer'] = answer
    cases.append(case)

df = pd.DataFrame(cases)
print(df)
df.to_csv('data_sources/parsed_chemo_drugs.csv')
