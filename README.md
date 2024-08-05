# A Hitchhiker's Guide to Deep Chemical Language Processing for Bioactivity Prediction

:thumbsdown: :oncoming_automobile: :stop_sign: :oncoming_automobile: 

Jump on board hitchhiker!

Welcome to our galaxy, tha galaxy of deep chemical language processing (`deepclp`). We as the citizens of `deepclp` are known to be friendly and prepared a guide for you to hitchhike through galaxy :night_with_stars:

This guide walks you through the steps of training a bioactivity prediction model, *i.e.,* predicting the binding between a small molecule and a target protein, using `deepclp`. By the end, you will able to train and evaluate bioactivity prediction models across different representations, encodings, and architectures with minimal code. Because this is what the `deepclp` galaxy is all about!


We have already trained quite a few such models, compared them, and shared our insights. If you want to read into them, check out our [paper](https://arxiv.org/abs/2407.12152) :bookmark:

## :school_satchel: Packing the Bag
Hitchhikers rely extensively on their gadgets. So will you.

Before you start your journey, start a terminal and run the following to claim your bag :baggage_claim:

>[!TIP]
> We use `conda` to setup our environment. You can read [this tutorial](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html), if you are unfamiliar with `conda` :snake: 

```python
conda create -n hitchhiker python==3.9.16
conda activate hitchhiker
```

Amazing! You now have your bag. Only one step left: download this codebase (either via the green button on the top right or via `git clone https://github.com/molML/hitchhiker-guide-CLP.git`) and run the following commands on your terminal:

```python
python -m pip install -r requirements.txt  # install the required packages. make sure that you are in the root directory of the codebase
python -m pip install .  # install deepclp 
```

Perfect! The bag is packed. Now we go! :oncoming_automobile:


## :oncoming_automobile: Hitchhiking

Remember we said that deepclpeers (tiny wordplay :wink:) are very friendly? Thanks to our friendliness, we make things easy for others. So we made training a bioactivity prediction models as easy as possible:

```python
import keras 
from deepclp import models, training
keras.utils.set_random_seed(42)  # fix the randomness for reproducibility
# read the data
training_molecules, training_labels = training.csv_to_matrix("data/smiles_classification/train.csv", "smiles", maxlen=85)
validation_molecules, validation_labels = training.csv_to_matrix("data/smiles_classification/val.csv", "smiles", maxlen=85)

# define the model architecture
cnn = models.CNN(
    token_encoding="learnable",
    embedding_dim=64,
    n_layers=1,
    kernel_size=7,
    n_filters=128,
    dense_layer_size=64,
    dropout=0.25,
    vocab_size=35,
    maxlen=85,
    is_classification=True,
)

# train!
history = training.train_predictor(
    model=cnn,  # feed in the model to train
    X_train=training_molecules,  # feed in the training data
    y_train=training_labels,
    X_val=validation_molecules,  # feed in the validation data
    y_val=validation_labels,
    learning_rate=0.001,  # set the training hyperparameters (learning rate and batch size)
    batch_size=128,
    balance_loss=True,  # target class imbalance
)
```

This code trains a convolutional neural network on SMILES representations of the molecules using learnable token embeddings. The choice of the architecture, molecule representation, and token encoding are discussed in detail in our [paper](https://arxiv.org/abs/2407.12152). Don't forget to check it out! :pushpin:

Do you want to train on a different dataset? Just change the path in the `csv_to_matrix` function to your dataset.

> [!IMPORTANT]
> The dataset must have a column named `"molecule"` that contains either SMILES or SELFIES representation of the molecules and a column named `"label"` that contains the interaction labels. Check the `data` folder for example datasets.

Now that you have a model, you can also quickly evaluate it on a test set:

```python
test_molecules, test_labels = training.csv_to_matrix("data/smiles_classification/test.csv", "smiles", maxlen=85)
scores = training.evaluate_predictor(model=cnn, X_test=test_molecules, y_test=test_labels)
```

If you need the predictions on the test set, not the scores, then you can use the following code:

```python
predictions = cnn.predict(test_molecules)
```

Done! You are now a hitchhiker in the `deepclp` galaxy :tada:

Could it have been easier than this? :sunglasses: If your answer is "yes!", please let us know in the issues section :monocle_face: We are always looking for ways to make things easier for new hitchhikers!


## :telescope: Depths of `deepclp`

### :control_knobs: Representations, Encodings, and Architectures
While CNNs on SMILES and learnable token embeddings are our favorite out of the box (see the [paper](https://arxiv.org/abs/2407.12152) for why :wink:), different hitchhikers have different needs and preferences. As you might have guessed already, we have these friends covered, too :smirk:.

Our `deepclp` library implements two more models (GRU and Transformer) and two other token encodings, random and one-hot. It also supports training models using SELFIES, covering the most popular tools in the field.

You know what to read if you need more information on these models and representations :point_up:. If you want to directly jump into coding :octocat: below is a quick example of creating a GRU model with one-hot encoding and a transformer model with random encoding. Both are using SELFIES.

Keep in mind that you can always mix and match the models, representations, and encodings as you like!

```python
training_molecules, training_labels = training.csv_to_matrix("data/selfies_classification/train.csv", "selfies", maxlen=85)  # use selfies instead of smiles
# GRU with one-hot encoding
gru = models.GRU(
    token_encoding="onehot",
    embedding_dim=64,
    n_layers=1,
    hidden_size=128,
    dense_layer_size=64,
    dropout=0.25,
    vocab_size=50,  # models using SELFIES have a larger vocabulary
    maxlen=85,
    is_classification=True,
)

# Transformer with random encoding
transformer = models.Transformer(
    token_encoding="random",
    embedding_dim=64,
    n_layers=1,
    n_heads=8,
    ff_dim=128,
    dense_layer_size=64,
    dropout=0.25,
    vocab_size=50,  # models using SELFIES have a larger vocabulary
    maxlen=85,
    is_classification=True,
)
```

### :1234: Affinity Prediction (Regression)
Do you have continuous labels, *e.g.,* inhibition constants, instead of binary ones? No worries, we have this covered! Just set `is_classification=False` in the model definition and you are done!

```python
cnn = models.CNN(..., is_classification=False)
```

You can use the training and evaluation functions as before. These functions will automatically use regression loss and evaluation metrics.


### :passport_control: From Hitchhiker to a DeepCLPeer
Training, predicting, and evaluating models across representations, encodings, and architectures with minimal code. That's what `deepclp` is all about. But what if you want to go beyond that and build further?

> [!TIP]
> All architectures are `keras.Model` instances. 

What does this mean? It means that you can use all `keras` functions on the models you created, `fit`, `predict`, `save_model`, etc. You can also implement custom callbacks, losses, and metrics, just like you do for any `keras.Model`, and use them in combination with the `deepclp.models`.

Or, you can simply copy the implementation of the model (available in `deepclp.models`) and modify, *e.g.,* to add more hyperparameters, to edit the prediction head, or to add layer normalization.

The opportunities are endless. You can explore the galaxy as you like :rocket:

## :airplane: GPU Support
We know that hitchhikers are always in a hurry. We got you covered here. This time there is a caveat, though :/ 

`deepclp` uses `keras3` to implement the models, which supports `tensorflow`, `jax`, and `pytorch` backends to run models. `deepclp` selects `pytorch` backend as it optimally combines the simplicity and extensibility. However:

> [!IMPORTANT]
> To allow quick and easy installation on more devices, `deepclp` does not automatically install pytorch with GPU support. 

This means that you need to install `keras3` with GPU support explicitly. We refer you the [keras documentation](https://keras.io/getting_started/) for how. It's well-explained there.

If you want to switch to another backend, that's also doable. Just edit the value at `deepclp/__init__.py` to `"tensorflow"` or `"jax"` and rerun the installation command: `python -m pip install .` 

## :fireworks: Ends for New Beginnings
You are now a hitchhiker in the galaxy of deep chemical language processing. You have the tools to train bioactivity prediction models, to evaluate them, and to explore and expand the galaxy. 

If you have any questions, suggestions, or feedback, please let us know in the issues section. We are always here to help you :vulcan_salute:

If you end up using `deepclp` in your research, please don't forget us :people_holding_hands:

```bibtex
@article{ozccelik2024hitchhiker,
  title={A Hitchhiker's Guide to Deep Chemical Language Processing for Bioactivity Prediction},
  author={{\"O}z{\c{c}}elik, R{\i}za and Grisoni, Francesca},
  journal={arXiv preprint arXiv:2407.12152},
  year={2024}
}
```
