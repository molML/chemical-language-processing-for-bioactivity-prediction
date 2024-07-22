import tensorflow as tf
from tensorflow import keras

from emb_library.models.utils import get_embedding_layer


class TransformerBlock(keras.layers.Layer):
    def __init__(
        self, model_dim: int, num_heads: int, ff_dim: int, dropout_rate: float
    ):
        super().__init__()
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=model_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(model_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(
        self, embedding_type: str, vocab_size: int, model_dim: int, maxlen: int
    ):
        super().__init__()
        self.token_emb = get_embedding_layer(
            embedding_type, vocab_size, model_dim, maxlen
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=model_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class Transformer(keras.Model):
    def __init__(
        self,
        embedding_type: str,
        vocab_size: int,
        embedding_dim: int,
        maxlen: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        learning_rate: float,
        batch_size: int,
        combination_idx: int,
        is_regressor: bool = True,
    ):
        super().__init__()
        self.embedding_type = embedding_type
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.combination_idx = combination_idx
        self.is_regressor = is_regressor

        self.embedding = TokenAndPositionEmbedding(
            embedding_type, vocab_size, embedding_dim, maxlen
        )
        self.transformer_block = TransformerBlock(
            embedding_dim, n_heads, ff_dim, dropout
        )
        self.pooling = keras.layers.GlobalAveragePooling1D()
        self.dropout = keras.layers.Dropout(dropout)
        self.dense1 = keras.layers.Dense(ff_dim * 2, activation="relu")
        self.dense2 = keras.layers.Dense(ff_dim, activation="relu")
        self.dense3 = keras.layers.Dense(ff_dim // 2, activation="relu")
        if is_regressor:
            output_activation = "linear"
        else:
            output_activation = "sigmoid"
        self.output_layer = keras.layers.Dense(1, activation=output_activation)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.transformer_block(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        return self.output_layer(x)
