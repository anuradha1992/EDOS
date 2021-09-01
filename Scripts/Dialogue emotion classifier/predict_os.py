import time
import numpy as np
import pandas as pd
import tensorflow as tf
from os import mkdir
from os.path import exists
from math import ceil
from tqdm import tqdm
from pytorch_transformers import RobertaTokenizer
from sklearn.metrics import precision_recall_fscore_support

from optimize import CustomSchedule
from model_utils import *
from model_emobert import EmoBERT, loss_function
from datasets import create_osed_dataset


# Some hyper-parameters
num_layers = 12
d_model = 768
num_heads = 12
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 514
num_emotions = 41  # Number of emotion categories

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size

max_length = 100  # Maximum number of tokens
batch_size = 512
peak_lr = 2e-5
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

id2emot = {}
with open('ebp_labels.txt', 'r') as f:
    for line in f:
        emot, index = line.strip().split(',')
        id2emot[int(index)] = emot


def main():
    os_dataset, N = create_osed_dataset(tokenizer, batch_size, max_length)

    # Define the model.
    emobert = EmoBERT(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
        layer_norm_eps, max_position_embed, vocab_size, num_emotions)

    # Define optimizer and metrics.
    learning_rate = peak_lr
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
        epsilon = adam_epsilon)

    # Define the checkpoint manager.
    ckpt = tf.train.Checkpoint(model = emobert, optimizer = optimizer)

    checkpoint_path = 'checkpoints/emobert_high_sim_weighted'
    restore_epoch = 5
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)
    ckpt.restore(ckpt_manager.checkpoints[restore_epoch - 1]).expect_partial()
    print('Checkpoint {} restored!!'.format(ckpt_manager.checkpoints[restore_epoch - 1]))

    preds = []
    print('Predicting the OS dialogs...')
    for inputs in tqdm(os_dataset, total = ceil(N / batch_size)):
        inp, weights = inputs
        enc_padding_mask = create_masks(inp)
        pred = emobert(inp, weights, False, enc_padding_mask)
        pred = tf.nn.softmax(pred)
        preds.append(pred.numpy())
    preds = np.concatenate(preds, axis = 0)
    print('N = {}, preds.shape = {}'.format(N, preds.shape))

    print('Saving preds to file...')
    np.save('../osed_data/preds.npy', preds)

def get_confidence_values():
    preds = np.load('../osed_data/preds.npy')
    dialog_ids = np.load('../osed_data/dialog_ids.npy')
    print('preds.shape = {}'.format(preds.shape))
    print('dialog_ids.shape = {}'.format(dialog_ids.shape))

    conf = np.max(preds, axis = 1)
    labels = np.argmax(preds, axis = 1)

    # index = np.argsort(conf)[::-1]
    # dialog_ids = dialog_ids[index]
    # labels = labels[index]
    emots = [id2emot[i] for i in labels]
    # conf = conf[index]

    data = {'dialog_id': dialog_ids, 'emotion': emots, 'confidence': conf}
    df = pd.DataFrame(data)
    df.to_csv('../osed_data/osed_prediction.csv')

def get_high_confidence_dialogs(top_n):
    print('Reading the whole OS dataset...')
    df_os = pd.read_csv('../os_2018_dialogs_emobert_reduce_freq.csv')

    df = pd.read_csv('../high_conf_data/all_conf.csv')
    df_high = df.iloc[:top_n].copy()

    dialogs = []
    for i in tqdm(range(df_high.shape[0])):
        dialog_id = df_high.iloc[i]['os_dialog_id']
        df_os_id = df_os[df_os['dialogue_id'] == dialog_id]
        dialog = ['- {}'.format(u) for u in df_os_id['text'].tolist()]
        dialog = '\n'.join(dialog)
        dialogs.append(dialog)

    df_high['os_dialog'] = dialogs
    df_high.to_csv('../high_conf_data/high_conf_top_{}.csv'.format(top_n))

def get_high_confidence_dialogs_each(top_n):
    print('Reading the whole OS dataset...')
    df_os = pd.read_csv('../os_2018_dialogs_emobert_reduce_freq.csv')

    df = pd.read_csv('../high_conf_data/all_conf.csv')

    dfs = []
    for i in tqdm(range(num_emotions)):
        df_emot = df[df['emotion'] == id2emot[i]].iloc[:top_n].copy()
        dialogs = []
        for j in range(df_emot.shape[0]):
            dialog_id = df_emot.iloc[j]['os_dialog_id']
            df_os_id = df_os[df_os['dialogue_id'] == dialog_id]
            dialog = ['- {}'.format(u) for u in df_os_id['text'].tolist()]
            dialog = '\n'.join(dialog)
            dialogs.append(dialog)
        df_emot['os_dialog'] = dialogs
        dfs.append(df_emot)

    pd.concat(dfs).to_csv('../high_conf_data/high_conf_each_{}.csv'.format(top_n))

if __name__ == '__main__':
    # main()
    get_confidence_values()
    # get_high_confidence_dialogs(10000)
    # get_high_confidence_dialogs_each(100)
