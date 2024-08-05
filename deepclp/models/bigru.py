import keras

from deepclp.models.utils import get_embedding_layer


class BiGRU(keras.Model):
    def __init__(
        self,
        token_encoding: str,
        embedding_dim: int,
        n_layers: int,
        gru_dim: int,
        dense_layer_size: int,
        dropout: float,
        vocab_size: int,
        maxlen: int,
        is_classification: bool,
    ):
        """Bidirectional GRU model.

        Args:
            token_encoding (str): Type of token encoding to use. Choose from {"onehot", "embedding, "random"}.
            embedding_dim (int): Dimension of the token embeddings.
            n_layers (int): Number of GRU layers.
            gru_dim (int): Dimension of the GRU layers.
            dense_layer_size (int): Size of the dense layer.
            dropout (float): Dropout rate.
            vocab_size (int): Size of the vocabulary.
            maxlen (int): Maximum length of the input sequences.
            is_classification (bool): Whether the model is used for classification
                or regression.

        Returns:
            keras.Model: BiGRU model
        """
        super().__init__()
        self.token_encoding = token_encoding
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.n_layers = n_layers
        self.gru_dim = gru_dim
        self.dense_layer_size = dense_layer_size
        self.dropout = dropout
        self.is_classification = is_classification

        self.embedding = get_embedding_layer(
            token_encoding,
            vocab_size,
            embedding_dim,
        )
        self.grus = [
            keras.layers.Bidirectional(
                keras.layers.GRU(
                    gru_dim,
                    return_sequences=layer_ix < n_layers - 1,
                    return_state=False,
                )
            )
            for layer_ix in range(n_layers)
        ]
        self.dropout = keras.layers.Dropout(dropout)
        self.dense1 = keras.layers.Dense(
            dense_layer_size,
            activation="relu",
        )
        self.dense2 = keras.layers.Dense(
            dense_layer_size // 2,
            activation="relu",
        )
        self.dense3 = keras.layers.Dense(
            dense_layer_size // 4,
            activation="relu",
        )
        if is_classification:
            output_activation = "linear"
        else:
            output_activation = "sigmoid"
        self.output_layer = keras.layers.Dense(1, activation=output_activation)

    def call(self, inputs):
        x = self.embedding(inputs)
        for gru in self.grus:
            x = gru(x)
            x = self.dropout(x)

        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        return self.output_layer(x)
