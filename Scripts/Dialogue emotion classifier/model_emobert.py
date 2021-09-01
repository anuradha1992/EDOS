import tensorflow as tf
from model_basics import *


def loss_function(real_emot, pred_emot):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits = True, reduction = 'none')
    loss_ = scce(real_emot, pred_emot)
    return loss_

class EmoBERT(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
                 layer_norm_eps, max_position_embed, vocab_size, num_emotions):
        super().__init__(name = 'emo_bert')

        self.padding_idx = 1

        # Embedding layers
        self.word_embeddings = tf.keras.layers.Embedding(vocab_size, d_model, name = 'word_embed')
        self.pos_embeddings = tf.keras.layers.Embedding(max_position_embed, d_model, name = 'pos_embed')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon = layer_norm_eps,
            name = 'layernorm_embed')
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name = 'dropout_embed')

        # Encoder layers
        self.num_layers = num_layers
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, hidden_act, dropout_rate, layer_norm_eps, i)
            for i in range(num_layers)
        ]

        # Output layers
        self.attention_v = tf.keras.layers.Dense(1, use_bias = False, name = 'attention_v')
        self.attention_layer = tf.keras.layers.Dense(d_model, activation = 'tanh', name = 'attention_layer')
        self.hidden_layer = tf.keras.layers.Dense(d_model, activation = 'tanh', name = 'hidden_layer')
        self.output_layer = tf.keras.layers.Dense(num_emotions, name = 'output_layer')

    def call(self, x, weights, training, mask):
        # x.shape == (batch_size, seq_len)

        seq_len = tf.shape(x)[1]

        # Add word embedding and position embedding.
        pos = tf.range(self.padding_idx + 1, seq_len + self.padding_idx + 1)
        pos = tf.broadcast_to(pos, tf.shape(x))
        x = self.word_embeddings(x)  # (batch_size, seq_len, d_model)
        x = x * tf.expand_dims(weights, 2)
        x += self.pos_embeddings(pos)

        x = self.layernorm(x)
        x = self.dropout(x, training = training)

        # x.shape == (batch_size, seq_len, d_model)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        # Compute the attention scores
        projected = self.attention_layer(x)  # (batch_size, seq_len, d_model)
        logits = tf.squeeze(self.attention_v(projected), 2)  # (batch_size, seq_len)
        logits += (tf.squeeze(mask) * -1e9)  # Mask out the padding positions
        scores = tf.expand_dims(tf.nn.softmax(logits), 1)  # (batch_size, 1, seq_len)

        # x.shape == (batch_size, d_model)
        x = tf.squeeze(tf.matmul(scores, x), 1)

        x = self.hidden_layer(x)
        x = self.output_layer(x)

        return x  # (batch_size, num_emotions)
