import pandas as pd
import os
INPUT_FILE ='./../sequence_data/baseline_data/combined_toxins_sequence.csv'
df = pd.read_csv(INPUT_FILE,index_col=None)
try:
    del df['description_src']
except:
    pass


pos_df = pd.DataFrame(df.loc[df['label']==1])
neg_df = pd.DataFrame(df.loc[df['label']==0])

# need 2 columns 'headers', 'sequences'
def set_header(row):
    return row['db_id']+ '__' + row['seq_id']

pos_df['headers'] = pos_df.apply(set_header,axis=1)
neg_df['headers'] = neg_df.apply(set_header,axis=1)

pos_df = pos_df.rename(columns= {
'seq' : 'sequences'
})
neg_df = neg_df.rename(columns= {
'seq' : 'sequences'
})

try:
    del pos_df['label']
    del neg_df['label']
    del pos_df['db_id']
    del pos_df['seq_id']
    del neg_df['db_id']
    del neg_df['seq_id']
except:
    pass

print(pos_df.columns)
print(neg_df.columns)
print(len(pos_df))
print(len(neg_df))

# write to files
loc = './../sequence_data/baseline_data'
pos_df.to_csv(
    os.path.join(loc,'pre.venom.csv') , index=False
)
neg_df.to_csv(
    os.path.join(loc,'pre.NOT.venom.csv'),index=False
)