import keras
from keras import ops

from deepclp.models.utils import get_embedding_layer


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

    def call(self, inputs):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(
        self, token_encoding: str, vocab_size: int, model_dim: int, maxlen: int
    ):
        super().__init__()
        self.token_emb = get_embedding_layer(token_encoding, vocab_size, model_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=model_dim)

    def call(self, x):
        maxlen = x.shape[-1]
        positions = ops.arange(0, maxlen, dtype="int32")
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class Transformer(keras.Model):
    def __init__(
        self,
        token_encoding: str,
        embedding_dim: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        vocab_size: int,
        maxlen: int,
        is_classification: bool,
    ):
        """
        Initialize the Transformer model.
        Args:
            token_encoding (str): Type of embedding to use. One
                of "random", "onehot", or "learnable".
            embedding_dim (int): Dimension of the token embeddings.
            n_layers (int): Number of transformer blocks.
            n_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feedforward layer.
            dropout (float): Dropout rate.
            vocab_size (int): Size of the vocabulary.
            maxlen (int): Maximum length of the input sequences.
            is_classification (bool): Whether the model is used for
                classification or regression.

        Returns:
            keras.Model: Transformer model
        """
        super().__init__()
        self.token_encoding = token_encoding
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.is_classification = is_classification

        self.embedding = TokenAndPositionEmbedding(
            token_encoding, vocab_size, embedding_dim, maxlen
        )
        self.transformer_block = TransformerBlock(
            embedding_dim, n_heads, ff_dim, dropout
        )
        self.pooling = keras.layers.GlobalAveragePooling1D()
        self.dropout = keras.layers.Dropout(dropout)
        self.dense1 = keras.layers.Dense(ff_dim * 2, activation="relu")
        self.dense2 = keras.layers.Dense(ff_dim, activation="relu")
        self.dense3 = keras.layers.Dense(ff_dim // 2, activation="relu")
        if is_classification:
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
