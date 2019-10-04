import pandas as pd
import os

DATA_LOC = './../sequence_data/mlcb_data'

INPUT_FILE_list = {
    'train': 'train_data.csv',
    'test': 'test_data.csv',
}
# Requisite columns : sequences,headers

for key, file in INPUT_FILE_list.items():
    f_path = os.path.join(
        DATA_LOC, file
    )
    df = pd.read_csv(f_path, index_col=None)
    df = df.rename(columns={
        'Sequence' :  'sequences' ,
        'Toxin' : 'label'
    })

    df['label'] = df['label'].astype(float)
    pos_df = pd.DataFrame(df.loc[df['label'] == 1], copy=True)
    neg_df = pd.DataFrame(df.loc[df['label'] == 0], copy=True)

    del pos_df['label']
    del neg_df['label']

    def set_header(_inp, prefix):
        return  "seq_" + prefix  + "_" + str(_inp)

    prefix = key + '_' + 'pos'
    pos_df['headers'] = list(range(1,len(pos_df)+1))
    pos_df['headers'] = pos_df['headers'].apply(set_header,args=(prefix,))

    prefix = key + '_' + 'neg'
    neg_df['headers'] = list(range(1, len(neg_df) + 1))
    neg_df['headers'] = neg_df['headers'].apply(set_header, args=(prefix,))


    op_path_pos = os.path.join(
        DATA_LOC,
        'mlcb_1_pos_'+ key + '.csv'
    )
    pos_df.to_csv(
        op_path_pos,
        index=False
    )
    op_path_neg = os.path.join(
        DATA_LOC,
        'mlcb_1_neg_' + key + '.csv'
    )
    neg_df.to_csv(
        op_path_neg,
        index=False
    )

