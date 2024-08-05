import keras


def get_embedding_layer(
    token_encoding: str,
    vocab_size: int,
    embedding_dim: int,
):
    """Get an embedding layer.

    Args:
        token_encoding (str): Type of embedding to use. Choose from {"onehot", "embedding, "random"}.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the token embeddings.

    Returns:
        keras.layers.Layer: Embedding layer
    """
    if token_encoding == "random":
        return keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            mask_zero=True,
            trainable=False,
        )

    if token_encoding == "onehot":
        return keras.layers.CategoryEncoding(
            num_tokens=vocab_size, output_mode="one_hot"
        )

    if token_encoding == "learnable":
        return keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            mask_zero=True,
            trainable=True,
        )

    raise ValueError(f"Unknown embedding type: {token_encoding}")
