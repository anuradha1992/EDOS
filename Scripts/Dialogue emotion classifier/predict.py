import time
import numpy as np
import tensorflow as tf
from os import mkdir
from os.path import exists
from pytorch_transformers import RobertaTokenizer
from sklearn.metrics import precision_recall_fscore_support

from optimize import CustomSchedule
from model_utils import *
from model_emobert import EmoBERT, loss_function
from datasets import create_test_dataset_from_csv


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
batch_size = 256
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
    test_dataset = create_test_dataset_from_csv(tokenizer, batch_size, max_length)

    # Define the model.
    emobert = EmoBERT(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
        layer_norm_eps, max_position_embed, vocab_size, num_emotions)

    # Define optimizer and metrics.
    learning_rate = peak_lr
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
        epsilon = adam_epsilon)

    def validate(model_name):
        y_true = []
        y_pred = []
        for inputs in test_dataset:
            inp, weights, tar_emot = inputs
            enc_padding_mask = create_masks(inp)
            pred_emot = emobert(inp, weights, False, enc_padding_mask)
            pred_emot = np.argmax(pred_emot.numpy(), axis = 1)
            y_true += tar_emot.numpy().tolist()
            y_pred += pred_emot.tolist()

        # with open('metrics/test_results_{}.csv'.format(model_name), 'w') as f:
        #     f.write('y_true,y_pred\n')
        #     for i in range(len(y_true)):
        #         f.write('{},{}\n'.format(y_true[i], y_pred[i]))
        # return

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        emot_index = (y_true < 32)
        intent_index = (y_pred >= 32)
        y_true_emot = y_true[emot_index]
        y_pred_emot = y_pred[emot_index]
        y_true_intent = y_true[intent_index]
        y_pred_intent = y_pred[intent_index]

        p, r, f, _ = precision_recall_fscore_support(y_true_emot, y_pred_emot, average = 'macro')
        acc = np.mean(np.array(y_true_emot) == np.array(y_pred_emot))
        print('Emotion -- P: {:.4f}, R: {:.4f}, F: {:.4f}, A: {:.4f}'.format(p, r, f, acc))

        p, r, f, _ = precision_recall_fscore_support(y_true_intent, y_pred_intent, average = 'macro')
        acc = np.mean(np.array(y_true_intent) == np.array(y_pred_intent))
        print('Intent -- P: {:.4f}, R: {:.4f}, F: {:.4f}, A: {:.4f}'.format(p, r, f, acc))

        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average = 'macro')
        acc = np.mean(np.array(y_true) == np.array(y_pred))
        print('All -- P: {:.4f}, R: {:.4f}, F: {:.4f}, A: {:.4f}\n'.format(p, r, f, acc))

    def get_individual_scores(model_name):
        y_true = []
        y_pred = []
        for inputs in test_dataset:
            inp, weights, tar_emot = inputs
            enc_padding_mask = create_masks(inp)
            pred_emot = emobert(inp, weights, False, enc_padding_mask)
            pred_emot = np.argmax(pred_emot.numpy(), axis = 1)
            y_true += tar_emot.numpy().tolist()
            y_pred += pred_emot.tolist()
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average = None, labels = list(range(num_emotions)))
        with open('metrics/individual_scores_{}.csv'.format(model_name), 'w') as f_out:
            f_out.write('class,precision,recall,f_score\n')
            for i in range(num_emotions):
                f_out.write('{},{:.4f},{:.4f},{:.4f}\n'.format(id2emot[i], p[i], r[i], f[i]))

    # # Define the checkpoint manager.
    # ckpt = tf.train.Checkpoint(model = emobert, optimizer = optimizer)

    # checkpoint_path = 'checkpoints/emobert_weighted'
    # restore_epoch = 8
    # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)
    # ckpt.restore(ckpt_manager.checkpoints[restore_epoch - 1]).expect_partial()
    # print('Checkpoint {} restored!!'.format(ckpt_manager.checkpoints[restore_epoch - 1]))
    # validate('baseline_weighted')

    # Define the checkpoint manager.
    ckpt = tf.train.Checkpoint(model = emobert, optimizer = optimizer)

    checkpoint_path = 'checkpoints/emobert_high_sim_weighted'
    restore_epoch = 5
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)
    ckpt.restore(ckpt_manager.checkpoints[restore_epoch - 1]).expect_partial()
    print('Checkpoint {} restored!!'.format(ckpt_manager.checkpoints[restore_epoch - 1]))
    validate('extra_3k_weighted')


if __name__ == '__main__':
    main()
