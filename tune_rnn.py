import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from _collections_abc import Sequence
import pandas as pd
import time
import itertools


def create_optimizer(optimizer, learning_rate):
    """
    Simply returns an optimizer based on the string refering it.
    :param optimizer: Name of the optimizer
    :param learning_rate: learning rate of the optimizer
    :return: tensorflow.keras.{optimizers}.
    """
    if optimizer.lower() == "adadelta":
        return optimizers.Adadelta(lr=learning_rate)
    elif optimizer.lower() == "adagrad":
        return optimizers.Adagrad(lr=learning_rate)
    elif optimizer.lower() == "adam":
        return optimizers.Adam(lr=learning_rate)
    elif optimizer.lower() == "adamax":
        return optimizers.Adamax(lr=learning_rate)
    elif optimizer.lower() == "nadam":
        return optimizers.Nadam(lr=learning_rate)
    elif optimizer.lower() == "rmsprop":
        return optimizers.RMSprop(lr=learning_rate)
    elif optimizer.lower() == "sgd":
        return optimizers.SGD(lr=learning_rate)


def create_model(ins, outs, optimizer, learning_rate, layers,  loss):
    """
    Create a NeuralNetwork model based on parameters.
    :param ins: Number of inputs (int).
    :param outs: Number of outputs (int).
    :param optimizer: Name of the optimizer to use (string).
    :param learning_rate: Learning_rate of the optimizer (float).
    :param layers: number neurones per layer (list of int).
    :param loss: Name of the loss function to use (string).
    :return: A compiled model of type tensorflow.keras.{Model}.
    """
    input = tf.keras.Input(shape=ins)
    layer = input
    for neurones in layers:
        layer = tf.keras.layers.Dense(neurones, activation="relu")(layer)
    layer = tf.keras.layers.Dense(outs, activation="softmax")(layer)
    model = tf.keras.Model(inputs=input, outputs=layer)
    model.compile(optimizer=create_optimizer(optimizer, learning_rate), loss=loss, metrics=["accuracy"])
    return model


def check_sequence(obj):
    """
    Checks if an instance is a list_like type, else puts it in a list.
    :param obj: Instance to check.
    :return: A list-like instance in any case.
    """
    if not isinstance(obj, (Sequence, np.ndarray)) or isinstance(obj, (str, bytes, bytearray)):
        return [obj]
    else:
        return obj


def tune_params(data_train, target_train, learning_rates=(0.00001, 0.0001, 0.001),
                n_layers=(1, 2, 4), neurones_per_layer=(4, 8, 16), architectures=None,
                optimizers=("SGD", 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'),
                batch_sizes=(512, 1024), loss=("mean_squared_error", "categorical_crossentropy"),
                decreasing_layers=True):
    """
    Tunes parameters using models and ranges of values or fuctions.
    Will test all the combinations and give the results back as a DataFrame.
    :param data_train: Training input data (ndarray).
    :param target_train: Training label data (ndarray).
    :param learning_rates: List as a range of learning rates (list of float).
    :param n_layers: List as a range of number of layers (list of int).
    :param neurones_per_layer: List as a range of number of neurons per layer (list of int).
    :param architectures: Layer and neurones architecture (list of list of int).
        If not None, it will overwite any arguments given in n_layers and neurone_per_layer.
    :param optimizers: List of optimizers (list of string).
    :param batch_sizes: List of batch sizes (list of int).
    :param loss: list of names of loss functions (list of string).
    :param decreasing_layers: If true, each layer will be of equal or smaller size as the previous one. If false, all
    combinations will be tested. This parameter is ignored if architectures is informed. (boolean)
    :return: A pandas. DataFrame containing the results of all the combinations of paramters.
    """
    # Checking that arguments are of type List
    learning_rates = check_sequence(learning_rates)
    optimizers = check_sequence(optimizers)
    batch_sizes = check_sequence(batch_sizes)
    loss = check_sequence(loss)
    if architectures is None:
        architectures = genList(n_layers, neurones_per_layer, decreasing_layers)

    # Setting up the data format
    target_encoded = to_categorical(target_train)
    ins = data_train.shape[1]
    outs = target_encoded.shape[1]

    # Setting up the search
    best_score = 0
    best_param = None
    res_list = []
    i = 0
    num = len(batch_sizes) * len(optimizers) * len(architectures) * len(neurones_per_layer) * len(loss)
    print("\nThere are %i total iterations.\n\n" % num)

    # Testing all combinations
    for architecture in architectures:
        for batch_size in batch_sizes:
            for optimizer in optimizers:
                for los in loss:
                    for learning_rate in learning_rates:
                        t = time.perf_counter()
                        model = create_model(ins, outs, optimizer, learning_rate, architecture, los)
                        model.fit(data_train, target_encoded, batch_size=batch_size)
                        score = accuracy_score(target_train, np.argmax(model.predict(data_train), axis=1))
                        res_list.append((batch_size, architecture, optimizer, los, learning_rate,
                                         time.perf_counter() - t, score))
                        if score > best_score:
                            best_score = score
                            best_param = i
                        print("[%i/%i] architecture = %s,  batch=%i, %s, %s, learning at rate "
                              "%f\n\tScore = %f , in %f secs" % (i + 1, num, architecture,  batch_size,
                                                  optimizer, los, learning_rate, score, time.perf_counter()-t))
                        i += 1
    print("\nBest (index", best_param, ") :\n", res_list[best_param])
    return pd.DataFrame(res_list, columns=["batch_size", "architecture", "optimizer", "loss_function",
                                           "learning_rate", "time", "score"])


def get_best_by(dataframe, target_col, score_col="score"):
    """
    Returns a dataframe where the best score of each category or value of a given column in a DataFrame.
    :param dataframe: pandas.DataFrame to read.
    :param target_col: Columns to differentiate.
    :param score_col: Score column.
    :return: A DataFrame showing the best score per chosen category/value.
    """
    future_df = []
    for target in dataframe[target_col].unique():
        temp_df = dataframe[dataframe["target_col"] == target].reset_index(drop=True)
        future_df.append(temp_df.iloc[temp_df[score_col].idxmax(), :].tolist())
    return pd.DataFrame(future_df, columns=dataframe.columns).sort_values(score_col, ascending=False)\
        .reset_index(drop=True)


def genList(lengths, values, decreasing_only):
    """
    Generates a list of architecture given the number of layer and each different number of neurones per layer.
    :param lengths: Number of layer (list of int).
    :param values: Neurones per layer (list of int).
    :param decreasing_only: If True, each layer positioned after another one will have less or the same amout of
        neurones. If false, every combinations will be returned.
    :return: A list of architectures, to fill in the "architecture" argument of tune_params.
    """
    ret = []
    if decreasing_only:
        for length in lengths:
            ret.extend(sorted(list(itertools.combinations_with_replacement(sorted(values, reverse=True), length))))
    else:
        for length in lengths:
            ret.extend(sorted(list(itertools.product(sorted(values, reverse=True), repeat=length))))
    return ret