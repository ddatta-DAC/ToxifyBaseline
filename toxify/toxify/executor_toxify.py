import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
import sys
from itertools import groupby
import argparse
from random import shuffle
import random
from sklearn.metrics import precision_score
import argparse
import yaml
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
# ------------------------------------------------------------ #
try:
    from . import fifteenmer
except:
    import fifteenmer
try:
    from . import protfactor
except:
    import protfactor
try:
    from . import seq2window as sw
except:
    import seq2window as sw


# ------------------------------------------------------------ #
try: os.mkdir('models')
except: pass
try: os.mkdir(os.path.join('models','baseline'))
except: pass

class ToxifyModel:
    def __init__(self):
        print('Defining model')
        return

    def set_hyperparams(
            self,
            input_dimension,
            output_dimension,
            N_units,
            seq_len,
            model_dir,
            epochs=50,
            batch_size=512,
            lr=0.01
    ):
        self.model_dir = model_dir
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.N_units = N_units
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.save_dir = self.model_dir + "/saved_model"
        return

    def build(self):

        self.inputs = tf.placeholder(tf.float32, [None, None, self.input_dimension])
        self.target = tf.placeholder(tf.float32, [None, self.output_dimension])

        rnn_units = tf.nn.rnn_cell.GRUCell(
            self.N_units
        )
        rnn_output, _ = tf.nn.dynamic_rnn(
            rnn_units,
            self.inputs,
            dtype=tf.float32
        )

        # Ignore all but the last timesteps
        last = tf.gather(
            rnn_output,
            self.seq_len - 1,
            axis=1
        )
        self.logits = tf.layers.dense(
            last,
            self.output_dimension,
            activation=None)
        self.prediction = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.target,
                logits=self.logits
            )
        )
        # 0-1 loss; compute most likely class and compare with target
        self.accuracy = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.target, 1)
        )
        # Average 0-1 loss
        self.accuracy = tf.reduce_mean(
            tf.cast(self.accuracy, tf.float32)
        )
        # Optimizer
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)

        self.merged_summary_op = tf.summary.merge_all()
        return

    def train(
            self,
            train_X,
            train_Y
    ):

        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.summary_writer = tf.summary.FileWriter(
            self.model_dir,
            graph=tf.get_default_graph()
        )
        self.saver = tf.train.Saver()

        batch_size = self.batch_size
        num_batches = train_X.shape[0] // batch_size + 1
        prev_epoch_loss = 0

        print('Number of batches ', num_batches)
        summary = None
        for epoch in range(self.epochs):
            epoch_loss = []
            print('<----')
            t1 = time.time()
            print(':: Epoch ::', epoch + 1)
            for _b in range(num_batches):

                if _b == num_batches - 1:
                    _train_x = train_X[_b * batch_size:]
                    _train_y = train_Y[_b * batch_size:]
                else:
                    _train_x = train_X[_b * batch_size: (_b + 1) * batch_size]
                    _train_y = train_Y[_b * batch_size: (_b + 1) * batch_size]

                _, loss, summary = self.sess.run(
                    [self.train_step, self.loss, self.merged_summary_op],
                    feed_dict={
                        self.inputs: _train_x,
                        self.target: _train_y
                    }
                )
                epoch_loss.append(loss)
                # if epoch % 5 and _b % 100 == 0:
                #     print('batch >', _b, ' Loss: ', loss)
            if summary is not None:
                self.summary_writer.add_summary(summary, epoch)

            t2 = time.time()
            print(' Time Elapsed ::', (t2 - t1) / 60, ' minutes')
            cur_epoch_loss = np.mean(epoch_loss)
            print('Epoch loss : ', cur_epoch_loss)

            # =====
            # Early breaking
            # =====
            # if abs( cur_epoch_loss - prev_epoch_loss)  <= 0.0000001 :
            #     print('Early stopping ..no loss reduction')


            prev_epoch_loss = cur_epoch_loss
            print('---->')

        if os.path.exists(self.save_dir):
            import shutil
            shutil.rmtree(self.save_dir)

        tf.saved_model.simple_save(
            self.sess,
            self.save_dir,
            inputs={
                "inputs": self.inputs,
                "target": self.target},
            outputs={"predictions": self.prediction}
        )
        self.sess.graph.finalize()
        print('End of training model')
        return

    def predict(
            self,
            test_X
    ):
        print(' ---> Start test phase <----')
        results = []
        batch_size = self.batch_size
        num_batches = test_X.shape[0] // batch_size

        print('Number of batches ', num_batches)
        print('Shape of text_X ', test_X.shape)
        for _b in range(num_batches + 1):
            if _b == num_batches:
                _test_x = test_X[_b * batch_size:]
            else:
                _test_x = test_X[_b * batch_size: (_b + 1) * batch_size]
            # ------
            # Do not close the session !!
            # ------
            output = self.sess.run(
                [self.prediction],
                feed_dict={
                    self.inputs: _test_x
                }
            )
            results.extend(output)
        results = np.vstack(results)
        return results

    def model_close(self):
        tf.reset_default_graph()
        self.sess.close()
        return


def train_and_test_model(
        train_X,
        train_Y,
        test_X,
        test_Y,
        N_units,
        model_dir,
        epochs,
        lr
):
    output_dimension = train_Y.shape[1]  # Output dimension
    input_dimension = train_X.shape[2]  # Input dimension
    seq_len = train_X.shape[1]  # Sequence length

    print('Input dimension ::', input_dimension)
    print('Output dimension ::', output_dimension)

    model_obj = ToxifyModel()
    model_obj.set_hyperparams(
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        N_units=N_units,
        seq_len=seq_len,
        model_dir=model_dir,
        epochs=epochs,
        lr=lr
    )
    model_obj.build()
    model_obj.train(
        train_X,
        train_Y
    )
    results = model_obj.predict(test_X)
    model_obj.model_close()
    return results


def only_predict(model_dir,  test_X):
    from tensorflow.contrib import predictor
    batch_size =1000
    num_batches = test_X.shape[0] // batch_size

    print('Number of batches ', num_batches)
    print('Shape of text_X ', test_X.shape)
    res =[]
    predict_fn = predictor.from_saved_model(model_dir)
    for _b in range(num_batches + 1):
        if _b == num_batches:
            _test_x = test_X[_b * batch_size:]
        else:
            _test_x = test_X[_b * batch_size: (_b + 1) * batch_size]

        predictions = predict_fn(
            {"inputs": _test_x}
        )
        r = predictions['predictions']

        res.append(r)
    res = np.vstack(res)
    return res



def get_data(
        pos_file_train,
        neg_file_train,
        pos_file_test,
        neg_file_test,
        window_size,
        max_seq_len
):

    (train_seqs, test_seqs) = sw.seqs2train(
        pos_file_train,
        neg_file_train,
        pos_file_test,
        neg_file_test,
        window_size,
        max_seq_len
    )

    test_seqs_pd = pd.DataFrame(test_seqs)

    if window_size:
        test_seqs_pd.columns = ['header', 'kmer', 'sequence', 'label']
    else:
        test_seqs_pd.columns = ['header', 'sequence', 'label']

    test_seqs_pd['label'] = test_seqs_pd['label'].astype(float)
    test_mat = []
    test_label_mat = []

    '''
    Label format
        Toxin, non-Toxin
    '''
    for row in test_seqs:
        seq = row[-2]
        label = float(row[-1])

        if label:
            test_label_mat.append([1, 0])
        else:
            test_label_mat.append([0, 1])
        test_mat.append(
            sw.seq2atchley(
                seq,
                window_size,
                max_seq_len
            )
        )
    test_label_np = np.array(test_label_mat)
    test_np = np.array(test_mat)

    train_mat = []
    train_label_mat = []

    for row in train_seqs:
        seq = row[-2]
        train_mat.append(sw.seq2atchley(
            seq,
            window_size,
            max_seq_len
        )
        )
        label = float(row[-1])

        if label:
            train_label_mat.append([1, 0])
        else:
            train_label_mat.append([0, 1])

    train_label_np = np.array(train_label_mat)
    train_np = np.array(train_mat)
    train_X = train_np
    train_Y = train_label_np
    test_X = test_np
    test_Y = test_label_np
    # list_train_X.append(train_X)
    # list_train_Y.append(train_Y)
    # list_test_X.append(test_X)
    # list_test_Y.append(test_Y)
    # list_test_seqs_pd.append(test_seqs_pd)
    print("train_X.shape:", train_X.shape)
    print("train_Y.shape:", train_Y.shape)

    return train_X, train_Y, test_X, test_Y, test_seqs_pd


'''
Since the output is logit,
Take mean of all the 'k-mers' for each class 
'''


def evaluate(
        df_test_seqs,
        arr_results
):

    print(df_test_seqs.head(10))
    df_test_seqs['predicted'] = 0.0
    # Create a copy
    new_df = pd.DataFrame(df_test_seqs, copy=True)

    new_df['predicted_1'] = arr_results[:, 0]
    new_df['predicted_0'] = arr_results[:, 1]

    # ----- #

    # Do a groupby with max
    res_df = new_df.groupby(
        by=['header']
    ).agg(
        {'label': 'mean',
         'predicted_0': 'mean',
         'predicted_1': 'mean'}
    ).reset_index()

    res_df = new_df
    # Set predicted label
    def set_res(row):
        if row['predicted_1'] >= row['predicted_0']:
            return 1.0
        else:
            return 0.0

    res_df['predicted'] = res_df.apply(
        set_res,
        axis=1
    )

    res_df['label'] = res_df['label'].astype(float)
    res_df['predicted'] = res_df['predicted'].astype(float)

    res_df.to_csv('all_output.csv', index=False)
    true_labels = list(res_df['label'])
    pred_labels = list(res_df['predicted'])
    print(set(true_labels))
    print(set(pred_labels))

    report = classification_report(true_labels, pred_labels)
    print('REPORT :: ', report)
    P = precision_score(true_labels, pred_labels)
    R = recall_score(true_labels, pred_labels)
    F1 = f1_score(true_labels, pred_labels)

    print(' Precison :: ', P)
    print(' Recall :: ', R)
    print(' F1 :: ', F1)
    return P, R, F1


# --------------------------------------- #

def main( CONFIG):
    print(' >>> Starting main ')
    maxLen= CONFIG['maxLen']
    window= CONFIG['window']
    N_units=CONFIG['N_units']
    DATA_LOC = CONFIG['data_dir']

    pos_file_train = os.path.join(
        DATA_LOC,
        CONFIG['pos_file_train']
    )
    neg_file_train = os.path.join(
        DATA_LOC,
        CONFIG['neg_file_train']
    )
    pos_file_test = os.path.join(
        DATA_LOC,
        CONFIG['pos_file_test']
    )
    neg_file_test = os.path.join(
        DATA_LOC,
        CONFIG['neg_file_test']
    )

    # here we are given a list of positive fasta files and a list of negative fasta files
    # Paper uses 50 epochs
    if 'epochs' in CONFIG.keys():
        epochs = CONFIG['epochs']
    else:
        epochs = 50

    if 'LR' in CONFIG.keys():
        lr = CONFIG['LR']
    else:
        lr = 0.01

    hyperparams = [ maxLen, window, N_units, epochs ]
    str_hyperparams = [str(_) for _ in hyperparams]
    model_signature = 'toxify_' + '_'.join(str_hyperparams)
    model_dir = os.path.join('./models', model_signature)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)


    result_dir = os.path.join(
        DATA_LOC,'results', model_signature
    )
    if not os.path.exists(os.path.join( DATA_LOC,'results')):
        os.mkdir( os.path.join(DATA_LOC ,'results'))

    if not os.path.exists(os.path.join(DATA_LOC,'results', model_signature)):
        os.mkdir( os.path.join(DATA_LOC,'results', model_signature) )

    print('-----------------------')
    print(' Model signature ::', model_signature)
    print('Data Dir : ', DATA_LOC)
    print ('Results directory :: ', result_dir)

    print('-----------------------')



    max_seq_len = maxLen
    window_size = window

    # -----
    # For now no CV
    # -----
    train_X, train_Y, test_X, test_Y, test_seqs_pd = get_data(
        pos_file_train,
        neg_file_train,
        pos_file_test,
        neg_file_test,
        window_size,
        max_seq_len
    )



    '''
    ----------------- TF Model ---------------------
    '''

    if int(CONFIG['only_predict']) == 1 :
        model_dir_save = model_dir + '/saved_model'
        results = only_predict(model_dir_save, test_X)

        print(' >> ' , results.shape)
    else:
        results = train_and_test_model(
            train_X,
            train_Y,
            test_X,
            test_Y,
            N_units,
            model_dir,
            epochs,
            lr
        )

    P, R, F1 = evaluate(
        test_seqs_pd,
        results
    )


    result_file = 'results.csv'

    if not os.path.exists('./../baseline_results'):
        os.mkdir('./../baseline_results')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    result_file_path = os.path.join(result_dir, result_file)
    if os.path.exists(result_file_path):
        results_df = pd.read_csv(result_file_path, index_col=None)
    else:
        results_df = pd.DataFrame(
            columns=[
                'Precision',
                'Recall',
                'F1',
            ])

    _dict = {
        'Precision': P,
        'Recall': R,
        'F1': F1,
    }

    results_df = results_df.append(
        _dict,
        ignore_index=True
    )

    results_df.to_csv(result_file_path, index=False)
    return


# main(maxLen=150, window=15, N_units=150, DATA=2)
# main(maxLen=150, window=0, N_units=270,  DATA=2)
# main(maxLen=500, window=50, N_units=270, DATA=2)
# main(maxLen=500, window=100, N_units=270, DATA=2)
# main(maxLen=500, window=200, N_units=270,  DATA=2)

# -------------------- #



parser = argparse.ArgumentParser(description='Running toxify ')
parser.add_argument(
    '--config',
    type=str,
    nargs='?',
    default='CONFIG.yaml',
    help='config file'
)


args = parser.parse_args()
print(args.config)
if args.config is not None:
    # load config
    conf_file = args.config
    with open(conf_file,'r') as handle:
        CONFIG = yaml.safe_load(handle)
    print(CONFIG)
    main(CONFIG)



