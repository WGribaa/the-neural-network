import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from _collections_abc import Sequence
import pandas as pd
import time
import itertools

all_optimizers = ("adadelta", "adagrad", "adam", "adamax", "nadam", "rmsprop", "sgd")
all_activations = ("elu", "softmax", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid",
                   "exponential", "linear")


class RnnTuner:
    """
    An object which stores all the tunable parameters, the best model from the last search as well as the pandas
    DataFrame showing all the models tested and resulting accuracy scores.
    """

    last_best_param = None
    last_best_score = None
    last_best_model = None
    results = None

    def __init__(self, data_train, target_train, data_test=None, target_test=None, learning_rates=(0.0001, 0.001),
                 n_layers=(2, 4), neurones_per_layer=(4, 8, 16), architectures=None, optimizers=('RMSprop', 'Adam'),
                 batch_sizes=(512, 1024), loss=("mean_squared_error", "categorical_crossentropy"),
                 decreasing_layers=True, hidden_activations="relu", output_activations="softmax"):
        """
        Tunes parameters using models and ranges of values or functions.
        Will test all the combinations and give the results back as a DataFrame.
        :param data_train: Training input data (ndarray).
        :param target_train: Training label data (ndarray).
        :param data_test: Test input data (ndarray).
        :param target_test: Test label data (ndarray).
        :param learning_rates: List as a range of learning rates (list of float).
        :param n_layers: List as a range of number of layers (list of int).
        :param neurones_per_layer: List as a range of number of neurons per layer (list of int).
        :param architectures: Layer and neurones archit$ecture (list of list of int).
            If not None, it will overwite any arguments given in n_layers and neurone_per_layer.
        :param optimizers: List of optimizers (list of string).
        :param hidden_activations: Activation functions for hidden layers (list of string).
        :param output_activations: Activation functions for output layers (list of string).
        :param batch_sizes: List of batch sizes (list of int).
        :param loss: list of names of loss functions (list of string).
        :param decreasing_layers: If true, each layer will be of equal or smaller size as the previous one. If false,
        all combinations will be tested. This parameter is ignored if architectures is informed. (boolean)
        """
        self.data_train = data_train
        self.data_test = data_test
        self.target_train = target_train
        self.target_test = target_test
        self.learning_rates = check_sequence(learning_rates)
        self.optimizers = check_sequence(optimizers, "optimizers")
        self.hidden_activations = check_sequence(hidden_activations, "activations")
        self.output_activations = check_sequence(output_activations, "activations")
        self.batch_sizes = check_sequence(batch_sizes)
        self.loss = check_sequence(loss)
        if architectures is None:
            self.architectures = gen_architectures(n_layers, neurones_per_layer, decreasing_layers)
        self.score_on_train = data_test is None or target_test is None

    def tune(self, score_on_train=False):
        # Setting up the data format
        target_encoded = to_categorical(self.target_train)
        ins = self.data_train.shape[1]
        outs = target_encoded.shape[1]

        # Setting up the search
        best_score = 0
        res_list = []
        i = 0
        num = len(self.batch_sizes) * len(self.optimizers) * len(self.architectures) * len(self.loss) * \
            len(self.hidden_activations) * len(self.output_activations)
        print("\nThere are %i total iterations.\n\n" % num)

        # Testing all combinations
        for architecture in self.architectures:
            for batch_size in self.batch_sizes:
                for optimizer in self.optimizers:
                    for los in self.loss:
                        for hidden_activation in self.hidden_activations:
                            for output_activation in self.output_activations:
                                for learning_rate in self.learning_rates:
                                    t = time.perf_counter()
                                    model = create_model(ins, outs, optimizer, learning_rate, architecture, los,
                                                         hidden_activation=hidden_activation,
                                                         output_activation=output_activation)
                                    model.fit(self.data_train, target_encoded, batch_size=batch_size)
                                    if self.score_on_train or score_on_train:
                                        score = accuracy_score(self.target_train,
                                                               np.argmax(model.predict(self.data_train), axis=1))
                                    else:
                                        score = accuracy_score(self.target_test,
                                                               np.argmax(model.predict(self.data_test), axis=1))
                                    res_list.append((batch_size, architecture, optimizer, hidden_activation,
                                                     output_activation, los, learning_rate,
                                                     time.perf_counter() - t, score))
                                    if score > best_score:
                                        self.last_best_model = model
                                        self.last_best_score = score
                                        self.last_best_param = res_list[-1]
                                    print("[%i/%i] architecture=%s,  batch=%i, hidden activation=%s,"
                                          "output activation=%s, %s, %s, learning at rate "
                                          "%f\n\tScore = %f , in %f secs" % (i + 1, num, architecture, batch_size,
                                                                             hidden_activation, output_activation,
                                                                             optimizer, los, learning_rate, score,
                                                                             time.perf_counter() - t))
                                    i += 1
        self.results = pd.DataFrame(res_list, columns=["batch_size", "architecture", "optimizer", "hidden_activation",
                                                       "output_activation", "loss_function",
                                                       "learning_rate", "time", "score"])
        print("\nBest :\n", self.last_best_param)
        return self.results

    def get_best_by(self, target_col, score_col="score"):
        return get_best_by(self.results, target_col, score_col)

    def predict(self, test_data, test_target):
        t = time.perf_counter()
        score = accuracy_score(test_target, np.argmax(self.last_best_model.predict(test_data), axis=1))
        print("Accuracy = %f in % sec" % (score, int(1000 * (time.perf_counter() - t)) / 1000))
        return score


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


def create_model(ins, outs, optimizer, learning_rate, layers, loss, hidden_activation="relu",
                 output_activation="softmax"):
    """
    Create a NeuralNetwork model based on parameters.
    :param ins: Number of inputs (int).
    :param outs: Number of outputs (int).
    :param hidden_activation: Activation function for the hidden layers (string).
    :param output_activation: Activation function for the output layer (string).
    :param optimizer: Name of the optimizer to use (string).
    :param learning_rate: Learning_rate of the optimizer (float).
    :param layers: number neurones per layer (list of int).
    :param loss: Name of the loss function to use (string).
    :return: A compiled model of type tensorflow.keras.{Model}.
    """
    input = tf.keras.Input(shape=ins)
    layer = input
    for neurones in layers:
        layer = tf.keras.layers.Dense(neurones, activation=hidden_activation)(layer)
    layer = tf.keras.layers.Dense(outs, activation=output_activation)(layer)
    model = tf.keras.Model(inputs=input, outputs=layer)
    model.compile(optimizer=create_optimizer(optimizer, learning_rate), loss=loss, metrics=["accuracy"])
    return model


def check_sequence(obj, allable=None):
    """
    Checks if an instance is a list_like type, else puts it in a list.
    :param obj: Instance to check.
    :param allable: Type of the object if it can take the value "all".
    :return: A list-like instance in any case.
    """
    if not isinstance(obj, (Sequence, np.ndarray)) or isinstance(obj, (str, bytes, bytearray)):
        if obj == "all" and allable == "optimizers":
            return all_optimizers
        elif obj == "all" and allable == "activations":
            return all_activations
        else:
            return [obj]
    else:
        return obj


def tune_params(data_train, target_train, data_test=None, target_test=None, learning_rates=(0.0001, 0.001),
                n_layers=(2, 4), neurones_per_layer=(4, 8, 16), architectures=None,
                optimizers=('RMSprop', 'Adam'),
                batch_sizes=(512, 1024), loss=("mean_squared_error", "categorical_crossentropy"),
                decreasing_layers=True, hidden_activations="relu", output_activations="softmax",
                score_on_train=False):
    """
    Tunes parameters using models and ranges of values or functions.
    Will test all the combinations and give the results back as a DataFrame.
    :param data_train: Training input data (ndarray).
    :param target_train: Training label data (ndarray).
    :param data_test: Test input data (ndarray).
    :param target_test: Test label data (ndarray).
    :param learning_rates: List as a range of learning rates (list of float).
    :param n_layers: List as a range of number of layers (list of int).
    :param neurones_per_layer: List as a range of number of neurons per layer (list of int).
    :param architectures: Layer and neurones archit$ecture (list of list of int).
        If not None, it will overwite any arguments given in n_layers and neurone_per_layer.
    :param optimizers: List of optimizers (list of string).
    :param hidden_activations: Activation functions for hidden layers (list of string).
    :param output_activations: Activation functions for output layers (list of string).
    :param batch_sizes: List of batch sizes (list of int).
    :param loss: list of names of loss functions (list of string).
    :param score_on_train: If true, the score will be predicted from the train data. Sets to True automatically if
        data_test or  target_test is None.
    :param decreasing_layers: If true, each layer will be of equal or smaller size as the previous one. If false, all
    combinations will be tested. This parameter is ignored if architectures is informed. (boolean)
    :return: A pandas. DataFrame containing the results of all the combinations of paramters.
    """
    # Checking that arguments are of type List
    learning_rates = check_sequence(learning_rates)
    optimizers = check_sequence(optimizers, "optimizers")
    hidden_activations = check_sequence(hidden_activations, "activations")
    output_activations = check_sequence(output_activations, "activations")
    batch_sizes = check_sequence(batch_sizes)
    loss = check_sequence(loss)
    if architectures is None:
        architectures = gen_architectures(n_layers, neurones_per_layer, decreasing_layers)
    if not score_on_train and (data_test is None or target_test is None):
        score_on_train = True

    # Setting up the data format
    target_encoded = to_categorical(target_train)
    ins = data_train.shape[1]
    outs = target_encoded.shape[1]

    # Setting up the search
    best_score = 0
    best_param = None
    res_list = []
    i = 0
    num = len(batch_sizes) * len(optimizers) * len(architectures) * len(loss) * len(hidden_activations) \
        * len(output_activations)
    print("\nThere are %i total iterations.\n\n" % num)

    # Testing all combinations
    for architecture in architectures:
        for batch_size in batch_sizes:
            for optimizer in optimizers:
                for los in loss:
                    for hidden_activation in hidden_activations:
                        for output_activation in output_activations:
                            for learning_rate in learning_rates:
                                t = time.perf_counter()
                                model = create_model(ins, outs, optimizer, learning_rate, architecture, los,
                                                     hidden_activation=hidden_activation,
                                                     output_activation=output_activation)
                                model.fit(data_train, target_encoded, batch_size=batch_size)
                                if score_on_train:
                                    score = accuracy_score(target_train, np.argmax(model.predict(data_train), axis=1))
                                else:
                                    score = accuracy_score(target_test, np.argmax(model.predict(data_test), axis=1))
                                res_list.append((batch_size, architecture, optimizer, hidden_activation,
                                                 output_activation, los, learning_rate,
                                                 time.perf_counter() - t, score))
                                if score > best_score:
                                    best_score = score
                                    best_param = i
                                print("[%i/%i] architecture=%s,  batch=%i, hidden activation=%s,"
                                      "output activation=%s, %s, %s, learning at rate "
                                      "%f\n\tScore = %f , in %f secs" % (i + 1, num, architecture, batch_size,
                                                                         hidden_activation, output_activation,
                                                                         optimizer, los, learning_rate, score,
                                                                         time.perf_counter() - t))
                                i += 1
    print("\nBest (index", best_param, ") :\n", res_list[best_param])
    return pd.DataFrame(res_list, columns=["batch_size", "architecture", "optimizer", "hidden_activation",
                                           "output_activation", "loss_function",
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
        temp_df = dataframe[dataframe[target_col] == target].reset_index(drop=True)
        future_df.append(temp_df.iloc[temp_df[score_col].idxmax(), :].tolist())
    return pd.DataFrame(future_df, columns=dataframe.columns).sort_values(score_col, ascending=False) \
        .reset_index(drop=True)


def gen_architectures(lengths, values, decreasing_only=True):
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
