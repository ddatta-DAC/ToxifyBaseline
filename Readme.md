Goto: toxify/toxify

cd toxify/toxify
python3 executor_main.py --config CONFIG.yaml

You can redefine the config or pass in a new config file(cone original) - preferred

Data is in : toxify/sequence_data
Currently using : baseline_data  (toxify/sequence_data/baseline_data ) [ = Data_Directory ]
Data_Directory must have 2 files - 1 positive and one nehgative
These must be specified in the config file

Results  in directory: toxify/sequence_data/\<Data_Directory\>/results/\<model_signature\>/results.csv

Reported metrics : Precision, Recall, F1  (mean and stddev for each )


