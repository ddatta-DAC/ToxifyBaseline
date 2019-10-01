'''
Code to generate csv files from the 2 data sources in the Toxify codebase
'''

import pandas as pd
import os
from Bio import SeqIO

def process(inp_f_path, op_f_path):
    # Paper mentions max len of 500 : to be enforced in model code
    df_columns = [
        'headers' ,'sequence'
    ]
    df = pd.DataFrame(columns=df_columns)

    fasta_sequences = SeqIO.parse(open(inp_f_path),'fasta')
    count = 0
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        _dict = {
            'headers' : name,
            'sequences': str(sequence),
        }
        df = df.append(
            _dict,
            ignore_index=True)
        count+=1
    df.to_csv(op_f_path,index=False)

    print('File  :',inp_f_path, '|| count = ',count)
    return df

def main():
    inp_loc = './../sequence_data/training_data'
    input_file_names = [
        'pre.NOT.venom.fasta',
        'pre.venom.fasta'
    ]

    op_file_names = [
        'pre.NOT.venom.csv',
        'pre.venom.csv'
    ]

    for inp_f_name, op_f_name in zip(input_file_names,op_file_names):
        inp_f_path = os.path.join(
            inp_loc , inp_f_name
        )
        op_f_path = os.path.join(
            inp_loc,
            op_f_name
        )
        process(inp_f_path,op_f_path)


    return


main()
