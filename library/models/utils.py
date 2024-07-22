from tensorflow import keras


def get_embedding_layer(
    embedding_type: str,
    vocab_size: int,
    embedding_dim: int,
    maxlen: int,
):
    if embedding_type == "random":
        return keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=maxlen,
            mask_zero=True,
            trainable=False,
        )

    if embedding_type == "onehot":
        return keras.layers.Lambda(lambda x: keras.backend.one_hot(x, vocab_size))

    if embedding_type == "learnable":
        return keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=maxlen,
            mask_zero=True,
            trainable=True,
        )

    raise ValueError(f"Unknown embedding type: {embedding_type}")
