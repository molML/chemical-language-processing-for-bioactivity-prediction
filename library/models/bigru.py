from tensorflow import keras

from emb_library.models.utils import get_embedding_layer


class BiGRU(keras.Model):
    def __init__(
        self,
        embedding_type: str,
        vocab_size: int,
        embedding_dim: int,
        maxlen: int,
        n_layers: int,
        gru_dim: int,
        dense_layer_size: int,
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
        self.gru_dim = gru_dim
        self.dense_layer_size = dense_layer_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.combination_idx = combination_idx
        self.is_regressor = is_regressor

        self.embedding = get_embedding_layer(
            embedding_type,
            vocab_size,
            embedding_dim,
            maxlen,
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
        if is_regressor:
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
