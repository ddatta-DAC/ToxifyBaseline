import tensorflow as tf
import numpy as np
import pandas as pd
from random import shuffle
import random
import os
import sys
from itertools import groupby
import argparse
import time

import fifteenmer
import protfactor as pf
import seq2window as sw

    # def train(self):
    #     parser = argparse.ArgumentParser(
    #         description='Record changes to the repository'
    #     )
    #     # prefixing the argument with -- means it's optional
    #     parser.add_argument('-pos',action='append',nargs='*')
    #     parser.add_argument('-neg',action='append',nargs='*')
    #     parser.add_argument('-window',type = int,default = 15)
    #     parser.add_argument('-maxLen',type = int,default = 100)
    #     parser.add_argument('-units',type = int,default = 150)
    #     parser.add_argument('-epochs',type = int,default = 150)
    #     parser.add_argument('-lr',type = float,default = 0.01)
    #     # now that we're inside a subcommand, ignore the first
    #     # TWO argvs, ie the command (git) and the subcommand (commit)
    #
    #     print(parser)
    #
    #     args = parser.parse_args(sys.argv[2:])
    #     print('Running toxify train\n positive data:' , args.pos,'\n negative data:' , args.neg)
    #     self.args = args
    #
    #     return(self.args)

    # def predict(self):
    #     parser = argparse.ArgumentParser(
    #         description='Predicts venom probabilities'
    #     )
    #     # NOT prefixing the argument with -- means it's not optional
    #     parser.add_argument('sequences')
    #
    #     parser.add_argument(
    #         '-model',
    #         type = str,
    #         default = os.path.abspath(toxify.__file__).replace("__init__.py",
    #                                                            "models/max_len_500/window_0/units_270/lr_0.01/epochs_50/models/saved_model"))
    #     args = parser.parse_args(sys.argv[2:])
    #     print(' >> ', args)
    #     print('Running toxify predict\n input data:' , args.sequences)
    #     self.args = args
    #     return(self.args)



class ToxifyModel:
    def __init__(self):
        return

    def build(self):

        return

    def train(self):

        return

    def predict(self):

        return

# --------------------
TRAIN_MODE = True
# --------------------
def main():

    global TRAIN_MODE

    print(' >>> starting main ')
    # print(tox_args)
    # if hasattr(tox_args,"sequences"):
    #     # print(tox_args.sequences)
    #     """
    #     HERE needs to be a new way of converting fasta proteins to atchley factors, seq2window funcs
    #     """
    #     predictions_dir = tox_args.sequences +"_toxify_predictions"
    #     model_dir = tox_args.model
    #     model_len = int(model_dir.split("max_len_")[1].split("/")[0])
    #     if not os.path.exists(predictions_dir):
    #         os.makedirs(predictions_dir)
    #     protein_pd = sw.fa2pd(tox_args.sequences,0,model_len)
    #     fa_mat = []
    #     for seq  in protein_pd["sequences"]:
    #         fa_mat.append(sw.seq2atchley(seq,0,model_len))
    #     fa_np = np.array(fa_mat)
    #     # this will produce np array of fifteenmer seqs
    #     print("saving to ",predictions_dir+"/protein_vectors.npy")
    #     np.save(predictions_dir+"/protein_vectors.npy",fa_np)
    #     os.system("saved_model_cli run --dir "+model_dir+"  --tag_set serve --signature_def serving_default --inputs inputs="+predictions_dir+"/protein_vectors.npy  --outdir "+predictions_dir)
    #     prediction_np = np.load(predictions_dir+"/predictions.npy")
    #     print(prediction_np.shape,fa_np.shape)
    #     protein_pd["pred"] = prediction_np[:,0]
    #     print(list(protein_pd))
    #     # print(protein_pd.drop())
    #
    #     protein_pd.drop(["sequences"],axis=1).to_csv(predictions_dir+"/predictions.csv",index=False)
    #     use15mer = False
    #
    #
    #     if use15mer:
    #
    #         proteins = fm.ProteinWindows(tox_args.sequences)
    #         protein_15mer = proteins.data
    #
    #         protein_vectors_np = pf.ProteinVectors(protein_15mer).data
    #         np.save(predictions_dir+"/protein_vectors.npy",protein_vectors_np)
    #
    #         os.system("saved_model_cli run --dir "+model_dir+"  --tag_set serve --signature_def serving_default --inputs inputs="+predictions_dir+"/protein_vectors.npy  --outdir "+predictions_dir)
    #
    #         prediction_np = np.load(predictions_dir+"/predictions.npy")
    #         prediction_15mer = np.hstack((protein_15mer,prediction_np))
    #         prediction_15mer_df = pd.DataFrame(prediction_15mer).drop(4,1)
    #         prediction_15mer_df.columns = [ 'header','15mer','sequence','venom_probability']
    #         columnsTitles=['header','15mer','venom_probability','sequence']
    #         prediction_15mer_df=prediction_15mer_df.reindex(columns=columnsTitles)
    #         prediction_15mer_outfile = predictions_dir+"/predictions_15mer.csv"
    #         prediction_15mer_df.to_csv(prediction_15mer_outfile,index=False)
    #         prediction_proteins = fm.regenerate(prediction_15mer_df)
    #         prediction_proteins_outfile = predictions_dir+"/predictions_proteins.csv"
    #         prediction_proteins.to_csv(prediction_proteins_outfile,index=False)
    #

    # here we are given a list of positive fasta files and a list of negative fasta files
    if TRAIN_MODE :

        maxLen = 150
        window = 15
        units = 270
        epochs = 1
        lr = 0.01

        hyperparams = [ maxLen, window, units, epochs ]
        str_hyperparams= [ str(_) for _ in hyperparams ]
        model_signature = 'toxify_'+ '_'.join(str_hyperparams)
        max_seq_len = maxLen
        window_size = window
        N_units = units
        lr = lr
        epochs = epochs

        # Here we are given a list of positive fasta files and a list of negative fasta files
        list_pos_samples_fasta = ['./../sequence_data/training_data/pre.venom.csv']
        list_neg_samples_fasta = ['./../sequence_data/training_data/pre.NOT.venom.csv']

        (train_seqs,test_seqs) = sw.seqs2train(
            list_pos_samples_fasta,
            list_neg_samples_fasta,
            window_size,
            max_seq_len
        )

        # Write training data
        training_data_dir = './../model_training_data'

        if not os.path.exists(training_data_dir):
            os.mkdir(training_data_dir)
        training_data_loc = os.path.join(training_data_dir,model_signature)
        if not os.path.exists(training_data_loc):
            os.makedirs(training_data_loc)

        print(" Writing to: " + training_data_loc + "/testSeqs.csv")
        test_seqs_pd = pd.DataFrame(test_seqs)

        if window_size:
            test_seqs_pd.columns = ['header', 'kmer','sequence','label']
        else:
            test_seqs_pd.columns = ['header', 'sequence','label']

        test_seqs_pd.to_csv(
            os.path.join(
                training_data_loc ,
                'testSeqs.csv'
            ),
            index= False
        )

        test_mat = []
        test_label_mat = []

        for row in test_seqs:
            seq = row[-2]
            label = float(row[-1])
            if label:
                test_label_mat.append([1,0])
            else:
                test_label_mat.append([0,1])
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
                train_label_mat.append([1,0])
            else:
                train_label_mat.append([0,1])

        train_label_np = np.array(train_label_mat)
        train_np = np.array(train_mat)

        # Save data
        np.save(training_data_loc+"testData.npy",test_np)
        np.save(training_data_loc+"testLabels.npy",test_label_np)
        np.save(training_data_loc+"trainData.npy",train_np)
        np.save(training_data_loc+"trainLabels.npy",train_label_np)

        # Load data
        test_X = np.load(training_data_loc+"testData.npy")
        test_Y = np.load(training_data_loc+"testLabels.npy")
        train_X = np.load(training_data_loc+"trainData.npy")
        train_Y = np.load(training_data_loc+"trainLabels.npy")

        print("train_X.shape:",train_X.shape)
        print("train_Y.shape:",train_Y.shape)
        # Parameters
        n = train_X.shape[0]  # Number of training sequences
        print(n) #7352
        n_test = train_Y.shape[0]  # Number of test sequences
        # print(n_test) #7352
        m = train_Y.shape[1]  # Output dimension
        print(m) #6
        d = train_X.shape[2]  # Input dimension
        print(d) #9
        T = train_X.shape[1]  # Sequence length

        '''
        ----------------- TF Model ---------------------
        '''

        # batch_size = 256
        # Learning rate

        # Placeholders
        inputs = tf.placeholder(tf.float32, [None, None, d])
        target = tf.placeholder(tf.float32, [None, m])

        # Network architecture

        rnn_units = tf.nn.rnn_cell.GRUCell(N_units)
        rnn_output, _ = tf.nn.dynamic_rnn(rnn_units, inputs, dtype=tf.float32)

        # Ignore all but the last timesteps
        last = tf.gather(rnn_output, T - 1, axis=1)

        # Fully connected layer
        logits = tf.layers.dense(last, m, activation=None)
        # Output mapped to probabilities by softmax
        prediction = tf.nn.softmax(logits)
        # Error function
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=target, logits=logits)
        )
        # 0-1 loss; compute most likely class and compare with target
        accuracy = tf.equal(tf.argmax(logits, 1), tf.argmax(target, 1))
        # Average 0-1 loss
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        # Optimizer
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)

        merged_summary_op = tf.summary.merge_all()
        model_dir = os.path.join("models",model_signature)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)


        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            summary_writer = tf.summary.FileWriter(
                model_dir,
                graph=tf.get_default_graph()
            )
            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print(ckpt.model_checkpoint_path)
                i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            else:
                print('No checkpoint file found!')
                i_stopped = 0

            '''
            Train model 
            '''
            batch_size  = 512
            num_batches = n//batch_size+1
            print('Number of batches ',num_batches)
            for i in range(epochs):

                # sess.run(
                #     train_step,
                #     feed_dict={
                #         inputs: train_X,
                #         target: train_Y
                #     }
                # )
                for _b in range(num_batches):
                    if _b == num_batches-1:
                        _train_x = train_X[_b*batch_size:]
                        _train_y = train_Y[_b*batch_size:]
                    else:
                        _train_x = train_X[_b*batch_size : (_b+1)*batch_size]
                        _train_y = train_Y[_b * batch_size: (_b+1)*batch_size]

                _, c, summary = sess.run(
                    [train_step, loss, merged_summary_op],
                    feed_dict={
                        inputs: _train_x,
                        target: _train_y
                    }
                )
                summary_writer.add_summary(summary, epochs)
                if (i + 1) % 10 == 0:
                    tmp_loss, tmp_acc = sess.run(
                        [loss, accuracy],
                        feed_dict=
                        {inputs: train_X, target: train_Y}
                    )
                    tmp_acc_test = sess.run(
                        accuracy,
                        feed_dict={
                            inputs: test_X,
                            target: test_Y
                        }
                    )
                    print(
                        i + 1,
                        ' Loss:', tmp_loss,
                        ' Accuracy, train:', tmp_acc,
                        ' Accuracy, test:', tmp_acc_test
                    )

                    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=i)

            save_dir = model_dir + "/saved_model/" + str(time.time()).split('.')[0]
            tf.saved_model.simple_save(
                sess,
                save_dir ,
                inputs={
                    "inputs":inputs,
                    "target":target},
                outputs={"predictions":prediction}
            )
            sess.graph.finalize()

        print('end of training model')
        '''
        Test the model 
        '''



main()
