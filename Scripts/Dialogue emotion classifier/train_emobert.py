import time
import numpy as np
import tensorflow as tf
from os import mkdir
from os.path import exists
from pytorch_transformers import RobertaTokenizer

from optimize import CustomSchedule
from model_utils import *
from model_emobert import EmoBERT, loss_function
from datasets import create_datasets_from_csv, create_test_dataset_from_csv


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
buffer_size = 100000
batch_size = 256
num_epochs = 10
peak_lr = 2e-5
# warmup_steps = 44
# total_steps = 440
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

checkpoint_path = 'checkpoints/emobert_high_sim_weighted'
log_path = 'log/emobert_high_sim_weighted.log'
data_path = '../MTurkHitsAll_results.csv'
extra_paths = ['../similar_dialogs_3k.csv', '../high_conf_data/high_conf_each_100.csv', '../embedding_2/similar_dialogs_090.csv']


def main():
    if not exists('log'):
        mkdir('log')
    f = open(log_path, 'a', encoding = 'utf-8')

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        train_dataset, val_dataset = create_datasets_from_csv(tokenizer, data_path, 
            buffer_size, batch_size, max_length, extra_paths = extra_paths)
        test_dataset = create_test_dataset_from_csv(tokenizer, batch_size, max_length)
        train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)

        # Define the model.
        emobert = EmoBERT(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
            layer_norm_eps, max_position_embed, vocab_size, num_emotions)

        # Build the model and initialize weights from PlainTransformer pre-trained on OpenSubtitles.
        build_model(emobert, max_length, vocab_size)
        emobert.load_weights('../weights/roberta2emobert_ebp.h5')
        print('Weights initialized from RoBERTa.')
        f.write('Weights initialized from RoBERTa.\n')

        # Define optimizer and metrics.
        # learning_rate = CustomSchedule(peak_lr, total_steps, warmup_steps)
        learning_rate = peak_lr
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
            epsilon = adam_epsilon)

        train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        val_loss = tf.keras.metrics.Mean(name = 'val_loss')
        test_loss = tf.keras.metrics.Mean(name = 'test_loss')


        # Define the checkpoint manager.
        ckpt = tf.train.Checkpoint(model = emobert, optimizer = optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

        # If a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
            f.write('Latest checkpoint restored!!\n')

        @tf.function
        def train_step(dist_inputs):
            def step_fn(inputs):
                # inp.shape == (batch_size, seq_len)
                # tar_emot.shape == (batch_size,)
                inp, weights, tar_emot = inputs
                enc_padding_mask = create_masks(inp)

                with tf.GradientTape() as tape:
                    pred_emot = emobert(inp, weights, True, enc_padding_mask)  # (batch_size, num_emotions)
                    losses_per_examples = loss_function(tar_emot, pred_emot)
                    loss = tf.reduce_sum(losses_per_examples) * (1.0 / batch_size)

                gradients = tape.gradient(loss, emobert.trainable_variables)
                optimizer.apply_gradients(zip(gradients, emobert.trainable_variables))
                return loss

            losses_per_replica = mirrored_strategy.run(step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses_per_replica, axis = None)

            train_loss(mean_loss)
            return mean_loss

        @tf.function
        def valid_step(dist_inputs, my_loss):
            def step_fn(inputs):
                # inp.shape == (batch_size, seq_len)
                # tar_emot.shape == (batch_size,)
                inp, weights, tar_emot = inputs
                enc_padding_mask = create_masks(inp)

                pred_emot = emobert(inp, weights, False, enc_padding_mask)  # (batch_size, num_emotions)
                losses_per_examples = loss_function(tar_emot, pred_emot)
                loss = tf.reduce_sum(losses_per_examples) * (1.0 / batch_size)

                return loss

            losses_per_replica = mirrored_strategy.run(step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses_per_replica, axis = None)
            my_loss(mean_loss)

        def validate(dataset):
            accuracy = []
            for (batch, inputs) in enumerate(dataset):
                inp, weights, tar_emot = inputs
                enc_padding_mask = create_masks(inp)
                pred_emot = emobert(inp, weights, False, enc_padding_mask)
                pred_emot = np.argmax(pred_emot.numpy(), axis = 1)
                accuracy += (tar_emot.numpy() == pred_emot).tolist()
            return np.mean(accuracy)

        # Start training
        for epoch in range(num_epochs):
            start = time.time()

            train_loss.reset_states()

            for (batch, inputs) in enumerate(train_dataset):
                current_loss = train_step(inputs)
                current_mean_loss = train_loss.result()
                print('Epoch {} Batch {} Mean Loss {:.4f} Loss {:.4f}'.format(
                    epoch + 1, batch, current_mean_loss, current_loss))
                f.write('Epoch {} Batch {} Mean Loss {:.4f} Loss {:.4f}\n'.format(
                    epoch + 1, batch, current_mean_loss, current_loss))

            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
            f.write('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

            epoch_loss = train_loss.result()
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, epoch_loss))
            f.write('Epoch {} Loss {:.4f}\n'.format(epoch + 1, epoch_loss))

            current_time = time.time()
            print('Time taken for 1 epoch: {} secs'.format(current_time - start))
            f.write('Time taken for 1 epoch: {} secs\n'.format(current_time - start))

            val_loss.reset_states()
            for inputs in val_dataset:
                valid_step(inputs, val_loss)
            epoch_val_loss = val_loss.result()
            print('Epoch {} Validation loss {:.4f}'.format(epoch + 1, epoch_val_loss))
            f.write('Epoch {} Validation loss {:.4f}\n'.format(epoch + 1, epoch_val_loss))

            val_ac = validate(val_dataset)
            print('Epoch {} Validation accuracy {:.4f}'.format(epoch + 1, val_ac))
            f.write('Epoch {} Validation accuracy {:.4f}\n'.format(epoch + 1, val_ac))

            test_loss.reset_states()
            for inputs in test_dataset:
                valid_step(inputs, test_loss)
            test_val_loss = test_loss.result()
            print('Epoch {} Test loss {:.4f}'.format(epoch + 1, test_val_loss))
            f.write('Epoch {} Test loss {:.4f}\n'.format(epoch + 1, test_val_loss))

            test_ac = validate(test_dataset)
            print('Epoch {} Test accuracy {:.4f}\n'.format(epoch + 1, test_ac))
            f.write('Epoch {} Test accuracy {:.4f}\n\n'.format(epoch + 1, test_ac))

    f.close()

if __name__ == '__main__':
    main()
