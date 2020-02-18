# the-neural-network
Several tools to learn and experiment neural networks training.

*******************
I tune_rnn:
*******************
A module that take care of finding the best architecture and functions for a regular neural network.
-
Tuned parameters :
1- architectures : number hidden layers and number neurones per each hidden layer. Two ways are available to inform it :
  a) Inform directly the "architectures" to test : each list is the number of nodes for a layer, the length of the list is the number of hidden, layer.
  b) Inform a list of hidden layers "n_layers", the possible number of nodes per layer "neurones_per_layer", and the strategy to build the architectures "decreasing_layers".
    An available function genList will generate each architecture from all possible combinations.
2- batch_sizes : self-explanatory. By default, it will test 512 and 1024.
3- optimizers : currently available : SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam. By default, will test all of them.
4- loss (functions) : check the available loss function here : https://keras.io/losses/ . By default, will test mean_squared_error_ and categorical_crossentropy. 
5- learning_rates : self-explanatory. by default : 0.00001, 0.0001 and 0.001.
-
General informations :
1- Works only for multi-categorical regular neural network (at the moment anyways).
2- Needs tensorflow with keras, sklearn.metrics and pandas.
3- The main function, tune_params, return a pandas DataFrame with:
  a) All (tuned or not) parameters.
  b) The score for each combination.
  c) The total time to process each combination.
-
Extras :
1- get_best_by is a simple function that takes a resulting dataframe and shows the best scoring combinations per relative to a specific parameter.
  Exemple : Best combination per each optimizer.
  
  

*******************
II Mnist_reader:
*******************
A module that reads the Mnist DataSets idx files. Check and download here : http://yann.lecun.com/exdb/mnist/
-
General informations :
1- read_image_file and read_label_file take the idx files and return numpy arrays.
  flatten argument available to flatten the resulting array into a one-dimensional-list.
2- Requires numpy.

Thanks to: 
Yann LeCun, Professor
The Courant Institute of Mathematical Sciences
New York University
yann@cs.nyu.edu

Corinna Cortes, Research Scientist
Google Labs, New York
corinna at google dot com 
