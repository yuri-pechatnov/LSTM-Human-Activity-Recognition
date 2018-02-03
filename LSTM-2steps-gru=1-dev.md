
# LSTMs for Human Activity Recognition

Human activity recognition using smartphones dataset and an LSTM RNN. Classifying the type of movement amongst six categories:
- WALKING,
- WALKING_UPSTAIRS,
- WALKING_DOWNSTAIRS,
- SITTING,
- STANDING,
- LAYING.

Compared to a classical approach, using a Recurrent Neural Networks (RNN) with Long Short-Term Memory cells (LSTMs) require no or almost no feature engineering. Data can be fed directly into the neural network who acts like a black box, modeling the problem correctly. Other research on the activity recognition dataset used mostly use a big amount of feature engineering, which is rather a signal processing approach combined with classical data science techniques. The approach here is rather very simple in terms of how much did the data was preprocessed. 

## Video dataset overview

Follow this link to see a video of the 6 activities recorded in the experiment with one of the participants:

<p align="center">
  <a href="http://www.youtube.com/watch?feature=player_embedded&v=XOEN9W05_4A
" target="_blank"><img src="http://img.youtube.com/vi/XOEN9W05_4A/0.jpg" 
alt="Video of the experiment" width="400" height="300" border="10" /></a>
  <a href="https://youtu.be/XOEN9W05_4A"><center>[Watch video]</center></a>
</p>

## Details about input data

I will be using an LSTM on the data to learn (as a cellphone attached on the waist) to recognise the type of activity that the user is doing. The dataset's description goes like this:

> The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. 

That said, I will use the almost raw data: only the gravity effect has been filtered out of the accelerometer  as a preprocessing step for another 3D feature as an input to help learning. 

## What is an RNN?

As explained in [this article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), an RNN takes many input vectors to process them and output other vectors. It can be roughly pictured like in the image below, imagining each rectangle has a vectorial depth and other special hidden quirks in the image below. **In our case, the "many to one" architecture is used**: we accept time series of feature vectors (one vector per time step) to convert them to a probability vector at the output for classification. Note that a "one to one" architecture would be a standard feedforward neural network. 

<img src="http://karpathy.github.io/assets/rnn/diags.jpeg" />

An LSTM is an improved RNN. It is more complex, but easier to train, avoiding what is called the vanishing gradient problem. 


## Results 

Scroll on! Nice visuals awaits. 


```python
# All Includes

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics

import os
```


```python
# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
] 

```

## Let's start by downloading the data: 


```python
# Note: Linux bash commands start with a "!" inside those "ipython notebook" cells

DATA_PATH = "data/"

#!pwd && ls
#os.chdir(DATA_PATH)
#!pwd && ls

#!python download_dataset.py

#!pwd && ls
#os.chdir("..")
#!pwd && ls

DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
print("\n" + "Dataset is now located at: " + DATASET_PATH)

```

    
    Dataset is now located at: data/UCI HAR Dataset/


## Preparing dataset:


```python
TRAIN = "train/"
TEST = "test/"


# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []
    
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()
    
    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], 
        dtype=np.int32
    )
    file.close()
    
    # Substract 1 to each output class for friendly 0-based indexing 
    return y_ - 1

make_y_easier_m = [0, 1, 1, 3, 3, 5]
def make_y_easier(y):
    return np.array([make_y_easier_m[int(i)] for i in y])

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_easier_train = make_y_easier(y_train)
y_test = load_y(y_test_path)
y_easier_test = make_y_easier(y_test)

```

## Additionnal Parameters:

Here are some core parameter definitions for the training. 

The whole neural network's structure could be summarised by enumerating those parameters and the fact an LSTM is used. 


```python
# Input Data 

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep


# LSTM Neural Network's internal structure

n_hidden = 16 # Hidden layer num of features
n_classes = 6 # Total classes (should go up, or should go down)
n_easier_classes = max(make_y_easier_m) + 1

# Training 

learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 2000
display_iter = 30000  # To show test set accuracy during training


# Some debugging info

print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

```

    Some useful info to get an insight on dataset's shape and normalisation:
    (X shape, y shape, every X's mean, every X's standard deviation)
    (2947, 128, 9) (2947, 1) 0.0991399 0.395671
    The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.


## Utility functions for training:


```python
def LSTM_RNN(_X):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters. 
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network. 
    # Note, some code of this notebook is inspired from an slightly different 
    # RNN architecture used on another dataset, some of the credits goes to 
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) 
    # new shape: (n_steps*batch_size, n_input)
    
    # Linear activation
    _X = tf.layers.dense(_X, n_hidden, activation=tf.nn.relu)
    
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 
    # new shape: n_steps * (batch_size, n_hidden)

    gru_cell_1 = tf.contrib.rnn.GRUCell(n_hidden, activation=tf.nn.relu)
    #gru_cell_2 = tf.contrib.rnn.GRUCell(n_hidden, activation=tf.nn.relu)
    rnn_cells = tf.contrib.rnn.MultiRNNCell([gru_cell_1], state_is_tuple=True)
    
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cells, _X, dtype=tf.float32)

    rnn_last_output = outputs[-1]
    
    before_split = tf.layers.dense(rnn_last_output, n_classes * 2)
    
    return tf.layers.dense(before_split, n_classes), tf.layers.dense(rnn_last_output, n_classes)
    
    

def extract_batch(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_, n_values=n_classes):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

```

## Let's get serious and build the neural network:


```python

# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

pred, pred_easier = LSTM_RNN(x)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) #+ l2 # Softmax loss
cost_easier = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred_easier)) #+ l2 # Softmax loss
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer_easier = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost_easier)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

correct_pred_easier = tf.equal(tf.argmax(pred_easier,1), tf.argmax(y,1))
accuracy_easier = tf.reduce_mean(tf.cast(correct_pred_easier, tf.float32))
```


```python
def extract_batch_xy(x, y, step, batch_size):
    return extract_batch(x, step, batch_size), one_hot(extract_batch(y, step, batch_size))
```

## Hooray, now train the neural network:


```python
from IPython.display import clear_output
#To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

def learn_easier(bx, by):
    _, loss, acc = sess.run(
        [optimizer_easier, cost_easier, accuracy_easier],
        feed_dict={
            x: bx, 
            y: by
        }
    )
    return loss, acc

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
diter = 0
while step * batch_size <= training_iters:
    batch_xs, batch_ys = extract_batch_xy(X_train, y_easier_train, step, batch_size)

    # Fit training using batch data
    loss, acc = learn_easier(batch_xs, batch_ys)
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        diter += 1
        # To not spam console, show training accuracy/loss in this "if"
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss2, acc2 = sess.run(
            [cost_easier, accuracy_easier], 
            feed_dict={
                x: X_test,
                y: one_hot(y_easier_test)
            }
        )
        test_losses.append(loss2)
        test_accuracies.append(acc2)
        clear_output(True)

        width = 12
        height = 12
        plt.figure(figsize=(width, height))
        
        ixs = np.array(range(diter))
        plt.plot(ixs, np.array(test_losses),     "b-", label="Test losses")
        plt.plot(ixs, np.array(test_accuracies), "g-", label="Test accuracies")

        plt.axhline(y=1.0, c='r')
        plt.axhline(y=0.9, c='orange')

        plt.title("Training session's progress over iterations")
        plt.legend(loc='upper right', shadow=True)
        plt.ylabel('Training Progress (Loss or Accuracy values)')
        plt.xlabel('Training iteration')

        plt.show()
        
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))

        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss2) + \
              ", Accuracy = {}".format(acc2))
        
    step += 1

print("Optimization Finished!")

# Accuracy for test data

one_hot_predictions, accuracyv, final_loss = sess.run(
    [pred_easier, accuracy_easier, cost_easier],
    feed_dict={
        x: X_test,
        y: one_hot(y_easier_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracyv)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracyv))

```


![png](LSTM-2steps-gru%3D1-dev_files/LSTM-2steps-gru%3D1-dev_15_0.png)


    Training iter #2190000:   Batch Loss = 0.000671, Accuracy = 1.0000001192092896
    PERFORMANCE ON TEST SET: Batch Loss = 0.25734850764274597, Accuracy = 0.9694603085517883
    Optimization Finished!
    FINAL RESULT: Batch Loss = 0.3364391624927521, Accuracy = 0.9653884172439575



```python
from IPython.display import clear_output
#To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
#sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
#init = tf.global_variables_initializer()
#sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
diter = 0
training_iters2 = training_iters * 2
while step * batch_size <= training_iters2:
    batch_xs, batch_ys = extract_batch_xy(X_train, y_train, step, batch_size)

    learn_easier(batch_xs, batch_ys)
    
    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters2):
        diter += 1
        # To not spam console, show training accuracy/loss in this "if"
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss2, acc2 = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss2)
        test_accuracies.append(acc2)
        clear_output(True)

        width = 12
        height = 12
        plt.figure(figsize=(width, height))
        
        ixs = np.array(range(diter))
        plt.plot(ixs, np.array(test_losses),     "b-", label="Test losses")
        plt.plot(ixs, np.array(test_accuracies), "g-", label="Test accuracies")

        plt.axhline(y=1.0, c='r')
        plt.axhline(y=0.9, c='orange')

        plt.title("Training session's progress over iterations")
        plt.legend(loc='upper right', shadow=True)
        plt.ylabel('Training Progress (Loss or Accuracy values)')
        plt.xlabel('Training iteration')
        plt.ylim(0, 2)
        plt.show()
        
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))

        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss2) + \
              ", Accuracy = {}".format(acc2))
        
    step += 1

print("Optimization Finished!")

# Accuracy for test data

one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))

```


![png](LSTM-2steps-gru%3D1-dev_files/LSTM-2steps-gru%3D1-dev_16_0.png)


    Training iter #4410000:   Batch Loss = 0.029866, Accuracy = 0.9854999780654907
    PERFORMANCE ON TEST SET: Batch Loss = 1.1519465446472168, Accuracy = 0.8944689035415649
    Optimization Finished!
    FINAL RESULT: Batch Loss = 1.1519466638565063, Accuracy = 0.8944689035415649



```python
list(zip(one_hot_predictions.argmax(axis=1), y_test.reshape(-1,), one_hot_predictions))
```




    [(4, 4, array([-1.36432242, -2.44424891, -5.19298553, -0.06043893,  8.72373009,
             -9.39953423], dtype=float32)),
     (4, 4, array([-1.36360598, -2.00036669, -4.60944271, -0.62437171,  8.94321346,
             -9.40503407], dtype=float32)),
     (4, 4, array([-1.59412026, -2.03066611, -4.37662649, -0.68194026,  9.07898521,
             -9.36316395], dtype=float32)),
     (4, 4, array([-1.80591536, -2.03214979, -4.24927664, -0.60739684,  8.78366947,
             -9.26470661], dtype=float32)),
     (4, 4, array([-1.75418425, -1.92673147, -4.23812389, -0.61924237,  8.63965607,
             -9.3277359 ], dtype=float32)),
     (4, 4, array([-1.77565432, -2.03147769, -4.36998558, -0.55507874,  8.73183632,
             -9.39223862], dtype=float32)),
     (4, 4, array([-1.75968885, -1.92394221, -4.23460388, -0.64117223,  8.69697952,
             -9.34314823], dtype=float32)),
     (4, 4, array([-1.82887077, -1.91275954, -4.22139025, -0.60684007,  8.56014824,
             -9.36584091], dtype=float32)),
     (4, 4, array([-1.70712256, -1.8711791 , -4.22897053, -0.68114847,  8.68591595,
             -9.39596653], dtype=float32)),
     (4, 4, array([-1.67480397, -1.91796541, -4.30052662, -0.67405057,  8.8072567 ,
             -9.43527794], dtype=float32)),
     (4, 4, array([-1.70583963, -1.90778756, -4.28892756, -0.67141843,  8.74955368,
             -9.44805336], dtype=float32)),
     (4, 4, array([-1.72668529, -1.88421786, -4.26232243, -0.66386062,  8.66296101,
             -9.4194212 ], dtype=float32)),
     (4, 4, array([-1.69850707, -1.80180359, -4.15255308, -0.75739962,  8.69900799,
             -9.42965508], dtype=float32)),
     (4, 4, array([-1.79930091, -1.81795013, -4.14907122, -0.69309431,  8.53510666,
             -9.44368553], dtype=float32)),
     (4, 4, array([-1.60672164, -1.8730278 , -4.40846062, -0.68811846,  8.74429321,
             -9.60415745], dtype=float32)),
     (4, 4, array([-2.02780867, -2.48672247, -4.50447035, -0.43201402,  8.84800148,
             -9.29810143], dtype=float32)),
     (4, 4, array([-2.6004715 , -1.19450808, -3.15437031, -0.77482951,  6.99064922,
             -9.32094288], dtype=float32)),
     (4, 4, array([-2.37205982, -1.38497329, -3.52431059, -0.77406311,  7.57985687,
             -9.55634975], dtype=float32)),
     (4, 4, array([-2.35759306, -1.40505099, -3.48628569, -0.79952246,  7.66567993,
             -9.45937824], dtype=float32)),
     (4, 4, array([-2.45870137, -1.30607653, -3.20208549, -1.01566756,  7.78777409,
             -9.55217457], dtype=float32)),
     (4, 4, array([-2.66675687, -1.09699678, -2.90100789, -0.90888816,  6.94527435,
             -9.24243069], dtype=float32)),
     (4, 4, array([-2.29666948, -1.15390468, -3.41593933, -0.96887439,  7.49930668,
             -9.81342793], dtype=float32)),
     (4, 4, array([ -1.72830367,  -1.29559374,  -4.09766674,  -0.98565286,
               7.9847393 , -10.00571537], dtype=float32)),
     (4, 4, array([-2.14017487, -1.26522422, -3.52496481, -0.91759259,  7.64054489,
             -9.56191349], dtype=float32)),
     (4, 4, array([-2.20466399, -1.34268057, -3.54218483, -0.90475225,  7.77405453,
             -9.56586838], dtype=float32)),
     (4, 4, array([-2.34602261, -1.24152696, -3.28132105, -0.91159171,  7.43516254,
             -9.37321663], dtype=float32)),
     (4, 4, array([-2.27801394, -1.33680522, -3.55594349, -0.84466714,  7.58250141,
             -9.63103199], dtype=float32)),
     (4, 4, array([-2.26694822, -1.3306042 , -3.43893099, -0.90425342,  7.68746662,
             -9.48720551], dtype=float32)),
     (4, 4, array([-2.34901023, -1.35186148, -3.44914865, -0.8337422 ,  7.5463953 ,
             -9.5105896 ], dtype=float32)),
     (4, 4, array([-2.15933681, -1.4123832 , -3.70798874, -0.81267786,  7.76750278,
             -9.61753845], dtype=float32)),
     (4, 4, array([-2.34783268, -1.4737035 , -3.46899557, -0.78695256,  7.65864468,
             -9.38364697], dtype=float32)),
     (3, 3, array([-1.7337172 , -6.75622034, -9.52362442,  5.76810408,  4.99275589,
             -6.85783815], dtype=float32)),
     (3, 3, array([-2.03420281, -6.68519163, -9.10721397,  5.59188652,  5.24499702,
             -7.05271339], dtype=float32)),
     (4, 3, array([-2.23180795, -6.68228006, -8.94910145,  5.47971582,  5.54127789,
             -6.91032219], dtype=float32)),
     (3, 3, array([-2.09164929, -6.7488327 , -9.22904396,  5.62376451,  5.39046097,
             -7.09131336], dtype=float32)),
     (3, 3, array([-1.62508178, -6.82138157, -9.9037056 ,  5.97374058,  4.80789948,
             -6.75000715], dtype=float32)),
     (3, 3, array([-1.76380277, -6.78915453, -9.62314415,  5.85326624,  4.93156242,
             -6.83102703], dtype=float32)),
     (3, 3, array([-1.9755435 , -6.7126708 , -9.25379372,  5.64918756,  5.19168854,
             -7.07198524], dtype=float32)),
     (3, 3, array([-1.69702053, -6.70279789, -9.57666302,  5.75544357,  4.951931  ,
             -7.12192869], dtype=float32)),
     (3, 3, array([-1.76369762, -6.71310997, -9.50449181,  5.73996162,  4.9926548 ,
             -7.05840874], dtype=float32)),
     (3, 3, array([-1.69883037, -6.68618774, -9.55074024,  5.73330975,  4.96210861,
             -7.17152739], dtype=float32)),
     (3, 3, array([-1.75704312, -6.75759983, -9.59876442,  5.7815361 ,  5.03328705,
             -7.01529074], dtype=float32)),
     (3, 3, array([-1.62518454, -6.72019148, -9.7054081 ,  5.81505966,  4.88055992,
             -7.07189894], dtype=float32)),
     (3, 3, array([-1.77182198, -6.83178902, -9.66387367,  5.86798143,  4.9936924 ,
             -6.74454546], dtype=float32)),
     (3, 3, array([-12.57903957, -12.03937149,  -7.86445332,  11.44035625,
               5.77464962,  -1.81856811], dtype=float32)),
     (4, 3, array([-9.04166889, -7.53844738, -4.56732035,  5.62391567,  6.99867916,
             -3.33575773], dtype=float32)),
     (4, 3, array([-2.96286893, -3.54862404, -4.93306398,  1.06003213,  8.6596899 ,
             -7.40548325], dtype=float32)),
     (4, 3, array([-2.91519976, -3.74205589, -5.17860365,  1.19544637,  8.83837223,
             -7.46769857], dtype=float32)),
     (4, 3, array([-3.06740332, -3.9442544 , -5.2258625 ,  1.43424141,  8.56943893,
             -7.15567493], dtype=float32)),
     (4, 3, array([-3.01680446, -4.01776695, -5.39707136,  1.54105067,  8.66998482,
             -7.30597258], dtype=float32)),
     (4, 3, array([-2.89414525, -3.43983006, -4.74284124,  0.86692804,  8.75740814,
             -7.5645318 ], dtype=float32)),
     (4, 3, array([-2.96077442, -3.52192616, -4.91620874,  0.94456428,  8.78200817,
             -7.62048435], dtype=float32)),
     (4, 3, array([-2.87890077, -3.45324636, -4.90340948,  0.91042894,  8.71820641,
             -7.59717274], dtype=float32)),
     (4, 3, array([-2.73555779, -3.45008755, -4.99191666,  0.84570897,  8.90779781,
             -7.76071548], dtype=float32)),
     (4, 3, array([-2.94724631, -3.52074361, -4.87374783,  0.92468542,  8.78925896,
             -7.59422922], dtype=float32)),
     (5, 5, array([ -9.1538868 ,  -3.85069394,  -9.86553288,   3.86970544,
               5.8895731 ,  45.14921188], dtype=float32)),
     (5, 5, array([ -7.66817379,  -1.43614662,  -7.97542763,   2.00519633,
               4.24953365,  48.79193878], dtype=float32)),
     (5, 5, array([ -7.81050014,  -1.71040189,  -8.11762905,   2.22782564,
               4.2421217 ,  48.50715637], dtype=float32)),
     (5, 5, array([ -7.51515961,  -1.18546581,  -7.88213778,   1.8063854 ,
               4.0796175 ,  49.35813522], dtype=float32)),
     (5, 5, array([ -7.51004457,  -1.1737957 ,  -7.76983976,   1.79886782,
               3.96462393,  49.61617661], dtype=float32)),
     (5, 5, array([ -7.59300375,  -1.2878989 ,  -7.79551125,   1.88492954,
               3.98884702,  49.54959488], dtype=float32)),
     (5, 5, array([ -7.48482084,  -1.10134017,  -7.62150002,   1.73878145,
               3.86773777,  49.92754745], dtype=float32)),
     (5, 5, array([ -7.65062618,  -1.35922277,  -7.71682405,   1.94423461,
               3.92005086,  49.53238678], dtype=float32)),
     (5, 5, array([ -7.64775658,  -1.35988462,  -7.74145412,   1.94450831,
               3.95806527,  49.46117783], dtype=float32)),
     (5, 5, array([ -7.73168278,  -1.5553484 ,  -8.05408955,   2.10689735,
               4.16810036,  49.02593613], dtype=float32)),
     (5, 5, array([ -7.5527029 ,  -1.14422417,  -7.64053822,   1.75931084,
               3.94350505,  49.63923264], dtype=float32)),
     (5, 5, array([ -9.06393242,  -3.19724178,  -9.52902985,   3.2446537 ,
               6.23517227,  48.15481186], dtype=float32)),
     (5, 5, array([ -8.24050236,  -2.15954828,  -9.28807068,   2.55181646,
               5.59404087,  47.65047073], dtype=float32)),
     (5, 5, array([ -9.0042696 ,  -3.37386179,  -8.91187763,   3.48625135,
               5.10509491,  45.43495941], dtype=float32)),
     (5, 5, array([ -8.89280415,  -3.28964233,  -9.16201591,   3.4303627 ,
               5.22885418,  44.89690781], dtype=float32)),
     (5, 5, array([ -8.43861294,  -2.52565312,  -8.8940382 ,   2.82038498,
               5.04999161,  45.62585449], dtype=float32)),
     (5, 5, array([ -8.56479549,  -2.71947241,  -8.9669342 ,   2.98208046,
               5.06826782,  45.66680527], dtype=float32)),
     (5, 5, array([ -8.63580322,  -2.88786554,  -9.07736969,   3.11454225,
               5.14019012,  45.25717926], dtype=float32)),
     (5, 5, array([ -8.71148109,  -2.91663623,  -9.02581024,   3.12747669,
               5.14624977,  45.33195114], dtype=float32)),
     (5, 5, array([ -8.77612591,  -3.0629921 ,  -9.09156322,   3.25738645,
               5.13524532,  45.46144104], dtype=float32)),
     (5, 5, array([ -8.5411129 ,  -2.72994685,  -8.9799099 ,   2.99216676,
               5.07248974,  45.58857346], dtype=float32)),
     (5, 5, array([ -8.64013767,  -2.81568885,  -9.00512981,   3.04997945,
               5.1269083 ,  45.50825119], dtype=float32)),
     (5, 5, array([ -8.63071728,  -2.79927659,  -9.00222969,   3.04237151,
               5.13183784,  45.65777969], dtype=float32)),
     (5, 5, array([ -8.5747776 ,  -2.77101731,  -9.0551405 ,   3.02239919,
               5.16305065,  45.48487091], dtype=float32)),
     (0, 0, array([ 31.40832138,  19.58455086, -13.96309853, -21.03125763,
              -2.96720791, -41.52288818], dtype=float32)),
     (0, 0, array([ 36.99245071,  23.05081749, -17.35885429, -23.94374657,
              -2.43141794, -48.46120071], dtype=float32)),
     (0, 0, array([ 36.83475876,  21.20799828, -19.84428215, -22.07657051,
              -1.74320507, -49.61985016], dtype=float32)),
     (0, 0, array([ 35.7481842 ,  20.37976646, -19.21481133, -21.38192558,
              -1.86720109, -47.35061646], dtype=float32)),
     (0, 0, array([ 41.27149582,  25.77448654, -18.42383003, -26.11396408,
              -4.26464558, -51.22088623], dtype=float32)),
     (0, 0, array([ 36.25235367,  21.52720261, -18.38019562, -22.96733856,
              -0.1177488 , -41.40549469], dtype=float32)),
     (0, 0, array([ 40.02241135,  22.97274971, -21.30882454, -24.0493145 ,
              -1.53549969, -53.42024994], dtype=float32)),
     (0, 0, array([ 34.781353  ,  18.69581985, -21.47536278, -19.85656738,
              -0.5277763 , -48.30625153], dtype=float32)),
     (0, 0, array([ 35.61507034,  19.499403  , -21.10813141, -20.62750244,
              -1.02936971, -48.98287582], dtype=float32)),
     (0, 0, array([ 33.75501633,  17.47941399, -21.59065628, -18.76400948,
               0.05449414, -46.64997864], dtype=float32)),
     (0, 0, array([ 33.49942017,  17.29649544, -20.63165665, -18.92729759,
              -0.23468018, -45.85984039], dtype=float32)),
     (0, 0, array([ 37.40554428,  22.57278633, -17.87522697, -23.43187714,
              -3.47287893, -49.40035248], dtype=float32)),
     (0, 0, array([ 41.085495  ,  25.15383339, -19.06693649, -25.5239563 ,
              -5.32498455, -54.78799057], dtype=float32)),
     (0, 0, array([ 37.54917908,  22.41003609, -18.70510101, -23.17687988,
              -2.60581684, -48.07717514], dtype=float32)),
     (0, 0, array([ 40.90301514,  24.58294106, -20.30561638, -25.68161964,
              -1.2234304 , -44.47761917], dtype=float32)),
     (0, 0, array([ 35.85858536,  20.69976997, -20.33579063, -21.58795547,
              -1.4843843 , -48.95610809], dtype=float32)),
     (0, 0, array([ 40.9798317 ,  23.12135696, -22.68642807, -24.15841675,
              -2.80900574, -55.57029724], dtype=float32)),
     (0, 0, array([ 42.36946106,  24.6464138 , -21.85896873, -25.55758858,
              -2.26493192, -51.37463379], dtype=float32)),
     (0, 0, array([ 39.08594513,  23.56248665, -19.00021935, -24.48101044,
              -2.79951644, -51.6437149 ], dtype=float32)),
     (0, 0, array([ 35.94478989,  20.48714447, -19.89002228, -21.69644928,
              -1.62784326, -49.29579544], dtype=float32)),
     (0, 0, array([ 44.25767517,  27.70232773, -19.3595047 , -27.77204323,
              -5.31986046, -55.48496246], dtype=float32)),
     (0, 0, array([ 40.18008804,  26.10468674, -17.4013443 , -26.51415253,
              -4.17428207, -46.23726273], dtype=float32)),
     (0, 0, array([ 43.00065613,  24.35341072, -23.02976799, -25.32665634,
              -2.24296474, -57.18689346], dtype=float32)),
     (0, 0, array([ 35.55701447,  19.22682571, -21.63263321, -20.45334244,
              -1.12466168, -50.25073242], dtype=float32)),
     (0, 0, array([ 37.62402725,  20.65312767, -21.65686417, -21.87736893,
              -1.6151135 , -48.79347992], dtype=float32)),
     (0, 0, array([ 37.75954437,  21.4250946 , -20.1978302 , -22.63084602,
              -1.34498656, -45.59383011], dtype=float32)),
     (0, 0, array([ 37.76112366,  22.18035126, -19.38470268, -23.24510193,
              -1.54621387, -48.62468719], dtype=float32)),
     (0, 0, array([ 39.9813652 ,  23.5338459 , -20.42029953, -24.30797958,
              -3.95071816, -51.35637283], dtype=float32)),
     (0, 0, array([ 40.92631531,  24.20866585, -20.52775192, -24.78090858,
              -3.51775336, -52.59357452], dtype=float32)),
     (0, 0, array([ 35.92821121,  21.02352715, -18.79716301, -21.97062683,
              -2.05132627, -48.26623535], dtype=float32)),
     (2, 2, array([-12.20008469,  -2.07277989,  14.3273344 ,  -1.45369506,
              -3.41478848,  -2.82623625], dtype=float32)),
     (2, 2, array([ -9.41444397,   2.08883047,  15.51540565,  -5.30924129,
              -3.48201847,  -3.47331285], dtype=float32)),
     (2, 2, array([-14.18280506,   1.06840563,  20.52820396,  -5.23182821,
              -5.10435009,  -6.75991869], dtype=float32)),
     (2, 2, array([-10.91359425,  -0.70305938,  14.36386108,  -2.77253652,
              -2.55773473,  -1.47781992], dtype=float32)),
     (2, 2, array([-11.28842735,   1.76924407,  18.61260414,  -5.69642591,
              -4.64817524,  -5.97473764], dtype=float32)),
     (2, 2, array([-15.7901659 ,  -2.69350553,  20.43751526,  -2.32143545,
              -5.37604809,  -1.40464854], dtype=float32)),
     (2, 2, array([-11.00234795,   1.75357842,  18.9719429 ,  -5.93190718,
              -2.53240824,  -3.35598946], dtype=float32)),
     (2, 2, array([-13.62423897,   5.85544491,  22.11669159, -10.39313126,
              -4.26071262, -13.15171623], dtype=float32)),
     (2, 2, array([-13.81545258,   8.44763279,  26.37011528, -12.70039177,
              -7.10781765, -15.58416462], dtype=float32)),
     (2, 2, array([-16.21246529,  -1.60669708,  20.78289795,  -3.40922022,
              -6.24953651,  -1.59401155], dtype=float32)),
     (2, 2, array([-13.35681152,   0.93698406,  19.23741722,  -5.66098595,
              -3.01245475,  -4.7272706 ], dtype=float32)),
     (2, 2, array([-15.30611897,   4.5649147 ,  25.81530952, -10.02667236,
              -7.05829239,  -8.24194241], dtype=float32)),
     (2, 2, array([-13.26259804,  -1.09260881,  18.14414215,  -3.51977038,
              -1.90491867,  -4.2016592 ], dtype=float32)),
     (2, 2, array([-13.81282902,  -0.31310984,  19.80675316,  -4.49540043,
              -3.57997727,  -2.57142806], dtype=float32)),
     (2, 2, array([-11.93849182,   0.84178495,  20.09792137,  -5.21511841,
              -3.30769181,  -4.24211311], dtype=float32)),
     (2, 2, array([ -8.9987545 ,   6.99816561,  23.57057381, -12.02029419,
              -4.57849216, -12.01815224], dtype=float32)),
     (2, 2, array([-17.98702621,  -2.84097767,  20.38753128,  -2.77345705,
              -2.77106118,  -3.54893637], dtype=float32)),
     (2, 2, array([-17.13259888,  -3.25446534,  20.25381851,  -2.19478416,
              -4.26999664,   0.33811915], dtype=float32)),
     (2, 2, array([-15.38621712,   1.19612837,  22.37555885,  -6.71106577,
              -2.95765567,  -3.28098583], dtype=float32)),
     (2, 2, array([-17.78882217,  -3.31514239,  20.08849907,  -1.99340868,
              -2.61381459,  -0.59251541], dtype=float32)),
     (2, 2, array([-19.25378227,   4.52307415,  29.29776382, -10.35569286,
              -8.06318665, -11.61495781], dtype=float32)),
     (2, 2, array([-18.45793533,  -1.5321418 ,  26.55428123,  -4.81441259,
              -5.90202141,  -4.43676519], dtype=float32)),
     (2, 2, array([-17.44347954,  -0.03464416,  21.1440258 ,  -6.26432467,
              -3.29119563, -10.36095524], dtype=float32)),
     (2, 2, array([-15.20008755,  -2.03415155,  18.94372368,  -2.8011167 ,
              -2.29252791,  -3.77064419], dtype=float32)),
     (1, 1, array([ 34.3283844 ,  54.46724319,  20.51753807, -48.37730026,
             -24.59009361, -33.1439209 ], dtype=float32)),
     (1, 1, array([ 32.68041992,  54.17919922,  20.45892143, -48.20977402,
             -23.41052246, -30.03754807], dtype=float32)),
     (1, 1, array([ 32.8051033 ,  54.54433441,  21.70787048, -47.97754288,
             -25.57560349, -36.10567856], dtype=float32)),
     (1, 1, array([ 12.03895378,  21.89972496,   9.72760773, -22.97083092,
              -8.98209476, -35.3567009 ], dtype=float32)),
     (1, 1, array([ 28.50145721,  44.05276871,  19.16930389, -40.5715065 ,
             -23.12352562, -44.1265831 ], dtype=float32)),
     (1, 1, array([ 20.73297691,  41.76972198,  13.39635181, -35.50537491,
             -16.56872177, -27.49490929], dtype=float32)),
     (1, 1, array([  2.19381118,  12.87295723,   6.35165262, -12.14465618,
              -5.79714966, -15.36124134], dtype=float32)),
     (1, 1, array([ 33.41875076,  53.39314651,  27.02847481, -49.94277573,
             -31.91272545, -58.88213348], dtype=float32)),
     (1, 1, array([ 23.77099419,  42.474823  ,  18.05208778, -38.74256897,
             -19.91186523, -45.32327652], dtype=float32)),
     (1, 1, array([ 23.87774086,  42.99436188,  22.32432747, -39.79074478,
             -23.22303391, -42.82200623], dtype=float32)),
     (1, 1, array([ 25.28220749,  48.2447319 ,  16.51226997, -42.06747437,
             -18.38051414, -16.43544769], dtype=float32)),
     (1, 1, array([ 37.76555634,  65.33777618,  29.70804214, -57.85050964,
             -30.87773705, -36.23323441], dtype=float32)),
     (1, 1, array([ 12.37435055,  27.39870834,   6.16515732, -24.35560799,
              -6.88815022,  -3.83387089], dtype=float32)),
     (1, 1, array([ 17.74428368,  36.87284851,   6.0139122 , -31.66082954,
             -10.36579132,  -6.77661514], dtype=float32)),
     (1, 1, array([ 11.29753017,  28.03928757,   5.22679329, -24.00016975,
             -10.47308731,   2.56492901], dtype=float32)),
     (1, 1, array([  8.06213188,  25.28384209,   4.89684677, -21.92800522,
              -6.988554  ,   2.97507334], dtype=float32)),
     (1, 1, array([ 19.74564171,  41.18262863,   9.85128689, -34.62687302,
             -12.97885704,  -9.95431519], dtype=float32)),
     (1, 1, array([ 24.12992668,  42.52707291,  19.07276154, -39.25893402,
             -21.21410179, -34.90359116], dtype=float32)),
     (1, 1, array([ 23.17162704,  46.11686707,  10.30212307, -39.41080475,
             -14.62152481, -14.41265392], dtype=float32)),
     (1, 1, array([ 23.34774017,  43.58005905,  15.74524498, -38.59669876,
             -13.89187908, -18.09972   ], dtype=float32)),
     (1, 1, array([ 16.52061844,  27.72528839,  12.77445221, -26.22550011,
             -13.49844456, -18.58908844], dtype=float32)),
     (1, 1, array([ 26.52888489,  48.43610001,  17.54397202, -42.44103622,
             -19.36121559, -20.83739853], dtype=float32)),
     (1, 1, array([  8.50411892,  20.47863579,   4.75991726, -18.65428352,
              -5.95410061,  -3.86299372], dtype=float32)),
     (1, 1, array([ 16.6749115 ,  39.0089035 ,   9.71779728, -32.87199783,
             -11.16813183, -10.43894768], dtype=float32)),
     (1, 1, array([ 31.94086075,  54.56192017,  25.03560257, -48.85461426,
             -27.90690231, -37.55994797], dtype=float32)),
     (4, 4, array([ -1.00419319,  -1.49498165,  -4.79521799,  -0.85550624,
               8.2607336 , -10.07397747], dtype=float32)),
     (4, 4, array([ -0.95468283,  -1.44421351,  -4.61494446,  -1.09367275,
               8.72826576, -10.08972549], dtype=float32)),
     (4, 4, array([-1.37359667, -1.40289509, -4.11998034, -1.09072638,  8.48000526,
             -9.81669712], dtype=float32)),
     (4, 4, array([-1.13762164, -1.36788809, -4.25859118, -1.20825517,  8.86618328,
             -9.9951601 ], dtype=float32)),
     (4, 4, array([-1.37013102, -1.39756668, -4.11555338, -1.121333  ,  8.57107735,
             -9.90410519], dtype=float32)),
     (4, 4, array([ -1.07969522,  -1.45804095,  -4.36881351,  -1.18828773,
               8.99625397, -10.03521633], dtype=float32)),
     (4, 4, array([-1.22618878, -1.45253527, -4.25849962, -1.15247273,  8.84375095,
             -9.95333672], dtype=float32)),
     (4, 4, array([ -1.09026289,  -1.46363783,  -4.38082027,  -1.16154158,
               8.93681622, -10.01014233], dtype=float32)),
     (4, 4, array([-1.17289662, -1.45392084, -4.31394958, -1.14404309,  8.83550453,
             -9.93888378], dtype=float32)),
     (4, 4, array([ -1.05535364,  -1.42038512,  -4.38309479,  -1.16893828,
               8.83519936, -10.01590633], dtype=float32)),
     (4, 4, array([-1.21182942, -1.34942746, -4.13443899, -1.22391319,  8.72816086,
             -9.90569401], dtype=float32)),
     (4, 4, array([ -1.33975315,  -1.10816741,  -4.13076067,  -1.19212508,
               8.01116753, -10.16913891], dtype=float32)),
     (4, 4, array([ -1.23479843,  -1.11659813,  -4.14933062,  -1.28856421,
               8.31306171, -10.23900509], dtype=float32)),
     (4, 4, array([ -1.2714448 ,  -1.09669173,  -4.00377798,  -1.34985697,
               8.42166805, -10.18331051], dtype=float32)),
     (4, 4, array([ -1.25275278,  -1.1502012 ,  -4.08108187,  -1.32433331,
               8.48741055, -10.1842308 ], dtype=float32)),
     (4, 4, array([ -1.3117063 ,  -1.06551826,  -3.91663289,  -1.35456681,
               8.33657932, -10.09797764], dtype=float32)),
     (4, 4, array([ -1.33251631,  -1.16054535,  -4.06142521,  -1.26900625,
               8.34963322, -10.16173744], dtype=float32)),
     (4, 4, array([ -1.36116672,  -1.15280926,  -3.92300963,  -1.31024945,
               8.38173389, -10.04423523], dtype=float32)),
     (4, 4, array([ -1.34921455,  -1.02734768,  -3.83583045,  -1.38186538,
               8.29683113, -10.08706188], dtype=float32)),
     (4, 4, array([ -1.44874406,  -1.10603833,  -3.90808368,  -1.27670288,
               8.18740559, -10.08965969], dtype=float32)),
     (4, 4, array([ -1.36146057,  -1.21390307,  -4.16003609,  -1.19223845,
               8.28542614, -10.18854904], dtype=float32)),
     (4, 4, array([ -1.36322188,  -1.13317728,  -3.97586417,  -1.29659843,
               8.35596371, -10.13301659], dtype=float32)),
     (4, 4, array([-1.59994006, -1.47588408, -4.00423574, -1.09398949,  8.38249493,
             -9.95565414], dtype=float32)),
     (3, 3, array([-5.8734355 , -7.26335096, -5.99086285,  7.76224566,  0.9470734 ,
              0.01625985], dtype=float32)),
     (3, 3, array([-12.20435238, -14.94777584, -10.57449818,  16.78876686,
              -1.13788044,   5.84359789], dtype=float32)),
     (3, 3, array([-16.31517792, -19.52822304, -13.55782223,  21.81499863,
              -0.89126611,   8.25166512], dtype=float32)),
     (3, 3, array([-17.13615227, -20.40143013, -14.0894537 ,  22.77375221,
              -0.84887695,   8.64300442], dtype=float32)),
     (3, 3, array([-17.12840652, -20.37085342, -14.05047798,  22.70878601,
              -0.79537344,   8.57073593], dtype=float32)),
     (3, 3, array([-17.18132401, -20.43400383, -14.10291576,  22.76312065,
              -0.76601088,   8.59579086], dtype=float32)),
     (3, 3, array([-17.58402634, -20.860075  , -14.38663673,  23.23162651,
              -0.69970822,   8.71738529], dtype=float32)),
     (3, 3, array([-17.70656586, -20.98324776, -14.43639374,  23.37368202,
              -0.73731863,   8.83422565], dtype=float32)),
     (3, 3, array([-17.7612915 , -21.05315018, -14.49050713,  23.44245148,
              -0.71926785,   8.86314201], dtype=float32)),
     (3, 3, array([-17.68439674, -20.94670868, -14.38703156,  23.342659  ,
              -0.78902173,   8.77993107], dtype=float32)),
     (3, 3, array([-17.65770721, -21.01141548, -14.53964806,  23.43059349,
              -0.80677664,   8.91941738], dtype=float32)),
     (4, 3, array([-7.34815121, -4.28822613, -1.33599663,  2.35581303,  4.8993845 ,
             -5.55857849], dtype=float32)),
     (4, 3, array([ -0.81398189,  -1.24596107,  -5.35786009,  -0.46041894,
               6.25936985, -10.05220413], dtype=float32)),
     (4, 3, array([-0.98936075, -3.6075685 , -7.03302813,  1.29088056,  8.27803326,
             -8.97696495], dtype=float32)),
     (3, 3, array([ -1.25576556,  -6.74719286, -10.03709698,   5.89278078,
               4.72529793,  -7.03647423], dtype=float32)),
     (3, 3, array([-1.8338511 , -6.38911295, -9.00790596,  5.48291349,  4.75447369,
             -5.91766405], dtype=float32)),
     (3, 3, array([-1.65180206, -5.84157801, -8.52916145,  4.9191618 ,  4.76403904,
             -6.0441227 ], dtype=float32)),
     (4, 3, array([-1.5875504 , -5.66217756, -8.38933754,  4.75582409,  4.75605392,
             -5.82465696], dtype=float32)),
     (3, 3, array([-1.66545606, -6.17022753, -8.83817959,  5.26854038,  4.70258808,
             -6.14097548], dtype=float32)),
     (4, 3, array([-1.43730736, -5.69370127, -8.54468727,  4.76129007,  4.81420231,
             -6.41712761], dtype=float32)),
     (4, 3, array([-1.60067105, -5.26423883, -7.9381752 ,  4.30943775,  4.79301739,
             -5.98949146], dtype=float32)),
     (4, 3, array([-1.45605969, -5.38312292, -8.26139259,  4.39087915,  4.87490368,
             -6.5888052 ], dtype=float32)),
     (5, 5, array([-10.18642521,  -5.45628834, -11.73655605,   4.91780806,
               8.30007744,  40.24479675], dtype=float32)),
     (5, 5, array([ -4.55221605,   4.59923887,  -4.72292089,  -2.92732835,
               2.9145174 ,  58.69921875], dtype=float32)),
     (5, 5, array([ -7.61873913,  -1.02303636,  -8.04324913,   1.62961578,
               4.6552    ,  48.98825455], dtype=float32)),
     (5, 5, array([ -7.56536293,  -0.99090648,  -8.31943226,   1.609972  ,
               4.8266573 ,  48.19052887], dtype=float32)),
     (5, 5, array([ -7.27737474,  -0.58902973,  -8.27805424,   1.29706419,
               4.76413155,  48.39089203], dtype=float32)),
     (5, 5, array([ -7.06927061,  -0.24421816,  -8.07539368,   1.0126617 ,
               4.65221786,  48.64664459], dtype=float32)),
     (5, 5, array([ -7.24505472,  -0.48122716,  -8.14596844,   1.20195186,
               4.68877506,  48.65769577], dtype=float32)),
     (5, 5, array([ -7.23665285,  -0.48962343,  -8.1164093 ,   1.21151674,
               4.6429348 ,  48.76728058], dtype=float32)),
     (5, 5, array([ -7.1822834 ,  -0.3692629 ,  -8.00517845,   1.10803092,
               4.60680103,  48.84456253], dtype=float32)),
     (5, 5, array([ -7.19241333,  -0.39131862,  -8.05321598,   1.13565314,
               4.60956764,  48.96144485], dtype=float32)),
     (5, 5, array([ -7.22732449,  -0.47222805,  -8.16047382,   1.19736457,
               4.69295025,  48.71012115], dtype=float32)),
     (5, 5, array([ -8.38005066,  -1.92705548,  -9.08803558,   2.28981447,
               5.83911991,  47.0537796 ], dtype=float32)),
     (5, 5, array([ -9.51050949,  -3.98460579,  -9.70731831,   3.92954421,
               5.99598598,  42.68208694], dtype=float32)),
     (5, 5, array([ -8.86067963,  -2.95318437,  -8.96636581,   3.08827734,
               5.48558331,  42.51499557], dtype=float32)),
     (5, 5, array([ -8.77865791,  -2.80601668,  -8.92016125,   2.9640038 ,
               5.42382145,  42.53111649], dtype=float32)),
     (5, 5, array([ -8.17033577,  -1.80432868,  -7.96190834,   2.14988327,
               4.75529766,  43.65251541], dtype=float32)),
     (5, 5, array([ -8.783144  ,  -2.79419804,  -8.87316704,   2.95387793,
               5.36743069,  42.75235748], dtype=float32)),
     (5, 5, array([ -8.53560925,  -2.43783021,  -8.60807228,   2.67119455,
               5.16543674,  43.06939697], dtype=float32)),
     (5, 5, array([ -8.66365147,  -2.59483552,  -8.67869186,   2.79383373,
               5.24964523,  43.11405563], dtype=float32)),
     (5, 5, array([ -8.91599178,  -3.02079058,  -8.91143894,   3.1363728 ,
               5.36411476,  42.72414017], dtype=float32)),
     (5, 5, array([ -8.55317688,  -2.47878766,  -8.50662136,   2.70641685,
               5.0864439 ,  43.19000626], dtype=float32)),
     (5, 5, array([ -8.88727188,  -2.97248602,  -8.87373829,   3.09863353,
               5.36375999,  42.77876282], dtype=float32)),
     (5, 5, array([ -8.8171978 ,  -2.86854291,  -8.91242027,   3.01853752,
               5.37602711,  42.82793045], dtype=float32)),
     (5, 5, array([ -8.67400646,  -2.4280808 ,  -8.22278023,   2.60537839,
               5.15018272,  42.43615341], dtype=float32)),
     (0, 0, array([ 36.19002533,  21.55040359, -18.81935501, -22.46234703,
              -3.21548915, -49.48733521], dtype=float32)),
     (0, 0, array([ 41.44735336,  26.05696869, -19.06323814, -27.3290062 ,
              -1.75775218, -49.16355515], dtype=float32)),
     (0, 0, array([ 38.35762405,  21.88285446, -20.52260017, -23.13683319,
              -2.19137645, -52.37680817], dtype=float32)),
     (0, 0, array([ 39.40815353,  23.2621727 , -20.21283531, -24.04858208,
              -1.88052702, -45.797966  ], dtype=float32)),
     (0, 0, array([  3.43731041e+01,   2.02090187e+01,  -1.82094402e+01,
              -2.15373020e+01,   2.45916843e-02,  -3.73225632e+01], dtype=float32)),
     (0, 0, array([ 35.85203171,  19.20560265, -21.79808807, -20.50688171,
              -1.02611113, -49.7420578 ], dtype=float32)),
     (0, 0, array([ 37.8899498 ,  22.46690559, -20.06675148, -23.25948334,
              -2.22402477, -49.09023666], dtype=float32)),
     (0, 0, array([ 38.76636124,  22.96423721, -19.95895386, -24.02103424,
              -1.79576492, -46.58614731], dtype=float32)),
     (0, 0, array([ 43.27601624,  26.45094109, -20.48225403, -26.92250061,
              -3.21636152, -53.75460052], dtype=float32)),
     (0, 0, array([ 34.70629501,  19.20561028, -19.97992897, -20.5191803 ,
              -0.73716533, -48.31716156], dtype=float32)),
     (0, 0, array([ 39.27939987,  23.71287346, -19.71779251, -24.55924606,
              -2.02332211, -45.12490845], dtype=float32)),
     (0, 0, array([ 36.51589203,  21.95247841, -18.67235756, -23.0260601 ,
              -0.93172228, -40.53171539], dtype=float32)),
     (0, 0, array([ 33.57143784,  16.64393044, -22.28654289, -18.23267555,
               0.81506062, -46.4086647 ], dtype=float32)),
     (0, 0, array([ 35.82614517,  20.79420662, -19.66383171, -21.42224884,
              -1.73665285, -48.25600052], dtype=float32)),
     (0, 0, array([ 33.04921341,  17.91464424, -19.70161057, -19.10621452,
              -0.3623541 , -42.18790436], dtype=float32)),
     (0, 0, array([ 39.47898102,  21.92791939, -22.2513237 , -23.0448761 ,
              -2.97020388, -53.35420227], dtype=float32)),
     (0, 0, array([ 40.11741638,  25.85904694, -17.69273186, -27.02145386,
              -2.53869891, -49.66207886], dtype=float32)),
     (0, 0, array([ 39.60837173,  23.62000084, -19.84559822, -24.75007439,
              -2.11885905, -51.73375702], dtype=float32)),
     (0, 0, array([ 38.13145828,  22.32972527, -19.93603325, -23.22918129,
              -1.64632761, -45.3603363 ], dtype=float32)),
     (0, 0, array([  3.96725426e+01,   2.32570362e+01,  -2.06875267e+01,
              -2.46897335e+01,   2.19441354e-02,  -4.50943108e+01], dtype=float32)),
     (0, 0, array([ 41.29982758,  23.05013466, -22.95604515, -24.33030891,
              -1.78337526, -56.5211525 ], dtype=float32)),
     (0, 0, array([ 38.37497711,  21.83742905, -20.4353466 , -23.0603714 ,
              -1.47774303, -47.04444122], dtype=float32)),
     (0, 0, array([ 41.85447311,  26.43645287, -19.10486984, -26.89810371,
              -5.07009315, -58.01638031], dtype=float32)),
     (0, 0, array([ 41.90632248,  23.72146988, -22.80205345, -24.67275047,
              -1.73697186, -52.09655762], dtype=float32)),
     (0, 0, array([ 35.32475281,  19.89508057, -19.51572609, -21.03556442,
              -0.52475309, -43.18516541], dtype=float32)),
     (0, 0, array([ 39.66968918,  23.73040199, -19.8890667 , -24.78425407,
              -2.29680681, -53.51426315], dtype=float32)),
     (0, 0, array([ 39.63557434,  22.3765583 , -22.00874138, -23.44545174,
              -1.99414372, -53.87852478], dtype=float32)),
     (0, 0, array([ 39.17713928,  23.79041672, -19.64975166, -24.54549599,
              -2.3415432 , -47.52437973], dtype=float32)),
     (0, 0, array([ 35.4442749 ,  22.36515617, -17.04436302, -23.35400581,
              -0.95571446, -39.50049973], dtype=float32)),
     (2, 2, array([ -8.31611633,   3.66672516,  18.36260986,  -7.80221176,
              -3.22875881,  -5.66367245], dtype=float32)),
     (2, 2, array([-11.49060059,   1.2057842 ,  18.22592354,  -5.5218668 ,
              -2.09886003,  -1.38685   ], dtype=float32)),
     (2, 2, array([-15.60733318,  -2.34173417,  19.99662971,  -2.77350807,
              -4.73675346,  -1.51990747], dtype=float32)),
     (2, 2, array([-12.00449753,   2.34875202,  21.04161835,  -7.30343342,
              -2.92236328,  -4.54764271], dtype=float32)),
     (2, 2, array([-19.249897  ,   1.91686165,  24.20410347,  -8.50777817,
              -5.01638794, -10.66941357], dtype=float32)),
     (2, 2, array([-15.4485302 ,   0.10027033,  23.5557766 ,  -5.3875041 ,
              -5.85709858,  -2.27099323], dtype=float32)),
     (2, 2, array([-13.82736111,   2.28463483,  21.82324219,  -6.54390097,
              -5.21209526,  -5.7251687 ], dtype=float32)),
     (2, 2, array([-11.85459137,   0.71266574,  18.95480347,  -4.91232538,
              -2.92606997,  -2.82741594], dtype=float32)),
     (2, 2, array([-14.16115761,   0.56244183,  23.09861946,  -6.50034904,
              -2.30447578,  -3.18665457], dtype=float32)),
     (2, 2, array([-15.87559891,   0.40504062,  22.27766037,  -6.18379927,
              -2.88677168,  -3.51034212], dtype=float32)),
     (2, 2, array([-21.99209785,  -0.80040562,  29.01768494,  -6.13883591,
              -5.84358978,  -7.88560677], dtype=float32)),
     (2, 2, array([-14.09525681,   0.45542681,  21.76084709,  -5.83765697,
              -1.74627256,  -4.5120163 ], dtype=float32)),
     (2, 2, array([-16.30594063,  -2.83567858,  20.0755558 ,  -2.13821173,
              -4.10468197,  -2.56561565], dtype=float32)),
     (2, 2, array([-13.46440887,   1.84138715,  20.62680626,  -7.85683918,
              -3.0565455 , -10.55507565], dtype=float32)),
     (2, 2, array([-12.52262592,   0.74782658,  20.4954586 ,  -5.30109549,
              -4.67962265,  -2.06908512], dtype=float32)),
     (2, 2, array([-19.4259758 ,  -2.5138154 ,  18.97631836,  -2.97186208,
              -3.74430203,  -6.30985117], dtype=float32)),
     (2, 2, array([-13.24769878,   1.74826777,  20.95702553,  -7.34502554,
              -4.06462002,  -6.84224463], dtype=float32)),
     (2, 2, array([ -7.34178114,   4.72218466,  15.59430695,  -7.709723  ,
              -2.98067641,  -6.16964102], dtype=float32)),
     (2, 2, array([ -9.0758419 ,   2.30478787,  14.34010029,  -5.3695364 ,
              -2.49176288,  -1.38997567], dtype=float32)),
     (2, 2, array([-20.44824219,  -5.69614315,  20.53644371,  -0.65422159,
              -3.00009465,  -3.6844244 ], dtype=float32)),
     (2, 2, array([-12.13637066,   5.06429482,  21.31976128,  -8.86205387,
              -3.38402843,  -9.44691944], dtype=float32)),
     (2, 2, array([-13.29911804,   0.34519869,  20.3758049 ,  -5.11775064,
              -2.521415  ,  -1.23372054], dtype=float32)),
     (2, 2, array([-10.70722103,   5.58046675,  23.7114296 , -10.248353  ,
              -6.03059769,  -6.17374182], dtype=float32)),
     (1, 1, array([ 57.58644867,  98.14303589,  54.02313232, -89.38446045,
             -58.50010681, -93.99213409], dtype=float32)),
     (1, 1, array([ 41.57154083,  73.46563721,  29.76293755, -64.40169525,
             -32.10776138, -37.56787872], dtype=float32)),
     (1, 1, array([ 38.50042725,  63.77639771,  31.70083618, -58.76958847,
             -38.59400177, -71.45942688], dtype=float32)),
     (1, 1, array([ 30.47236061,  55.39054489,  18.0439415 , -48.05651093,
             -21.41327858, -24.75797844], dtype=float32)),
     (1, 1, array([ 32.6684761 ,  60.20855331,  22.78142548, -52.48014832,
             -23.97144699, -27.5390873 ], dtype=float32)),
     (1, 1, array([ 41.23412704,  67.82164001,  27.20943069, -59.6642189 ,
             -33.59209061, -44.78540802], dtype=float32)),
     (1, 1, array([ 30.65688896,  55.25972748,  19.48416328, -48.20792007,
             -21.46486664, -25.12159538], dtype=float32)),
     (1, 1, array([ 35.7586441 ,  64.31305695,  18.06964302, -54.69532013,
             -25.57052422, -24.00846481], dtype=float32)),
     (1, 1, array([ 18.03683662,  38.81575394,   5.99332476, -32.22621155,
             -11.26361847, -15.83606243], dtype=float32)),
     (1, 1, array([ 46.12770462,  80.45635986,  36.25855255, -71.7395401 ,
             -42.30000687, -80.43431091], dtype=float32)),
     (1, 1, array([ 10.73085976,  30.19003868,  10.58594704, -26.10675621,
             -12.41709995, -22.87292671], dtype=float32)),
     (1, 1, array([ 24.70210648,  41.7744751 ,  20.30577278, -38.59915161,
             -21.05685425, -41.90965652], dtype=float32)),
     (1, 1, array([ 24.04178429,  43.49345016,  14.45361137, -37.7784462 ,
             -17.34584808, -31.8630867 ], dtype=float32)),
     (1, 1, array([ 23.83522034,  39.85754395,  11.88788414, -35.60577774,
             -15.36897373, -28.33474922], dtype=float32)),
     (1, 1, array([ 23.97958183,  42.24903488,   9.48528481, -36.49775696,
             -15.90104008, -28.85147858], dtype=float32)),
     (1, 1, array([ 23.24908066,  44.76205063,  14.35832787, -38.05437088,
             -19.49879456, -36.05379868], dtype=float32)),
     (1, 1, array([ 49.11666107,  85.42845154,  39.88471603, -75.89891052,
             -41.67573547, -50.49775314], dtype=float32)),
     (1, 1, array([ 36.46893311,  65.22368622,  26.66182137, -57.40586853,
             -28.75614166, -34.94884109], dtype=float32)),
     (1, 1, array([ 23.73451042,  47.09808731,  11.05777264, -40.83174896,
             -12.73107815, -15.82173347], dtype=float32)),
     (1, 1, array([ 17.18586922,  37.81356812,   7.68791676, -31.89573479,
             -11.6534071 , -11.49711514], dtype=float32)),
     (1, 1, array([ 28.26583862,  51.78106689,  15.27152729, -44.35026932,
             -18.95432663, -18.90974426], dtype=float32)),
     (1, 1, array([ 10.29204941,  25.18552017,   6.58521986, -21.98760033,
              -6.40594196,  -4.28795815], dtype=float32)),
     (1, 1, array([ 20.69415474,  44.29125977,  11.31931305, -37.52948761,
             -18.03756714, -30.2892704 ], dtype=float32)),
     (4, 4, array([-0.55171096,  2.76913095,  0.69767201, -3.84900641,  4.17546272,
             -7.19650507], dtype=float32)),
     (4, 4, array([  2.24448013,  -1.90237486,  -9.15888977,  -0.16632867,
               7.20202827, -11.90844727], dtype=float32)),
     (4, 4, array([-1.81439734, -6.78212595, -9.74480152,  5.50129509,  5.74074173,
             -7.02241945], dtype=float32)),
     (3, 4, array([-2.30886626, -6.99075651, -9.51609898,  5.91028929,  5.43637657,
             -5.71420145], dtype=float32)),
     (3, 4, array([ -0.6355257 ,  -6.48455763, -10.62019062,   5.78011084,
               4.41337967,  -7.33382416], dtype=float32)),
     (3, 4, array([-2.43273926, -6.0077095 , -8.03248692,  5.35861874,  4.45558262,
             -5.31351137], dtype=float32)),
     (3, 4, array([-2.88641143, -5.56340885, -6.90672398,  4.83254433,  4.24789619,
             -4.39713955], dtype=float32)),
     (3, 4, array([-2.6997931 , -6.65404987, -8.62901592,  6.02605247,  4.32043743,
             -4.15878296], dtype=float32)),
     (3, 4, array([-3.24652672, -6.60164118, -7.9484024 ,  6.13149214,  3.90767813,
             -4.10892248], dtype=float32)),
     (3, 4, array([-2.58014464, -6.53632927, -8.5548439 ,  5.8659482 ,  4.40026379,
             -4.6563077 ], dtype=float32)),
     (3, 4, array([-2.56892323, -6.4922452 , -8.49209118,  5.82680893,  4.38992786,
             -4.78647614], dtype=float32)),
     (3, 4, array([-2.32093644, -6.42499304, -8.6131649 ,  5.57410908,  4.70178699,
             -5.3647213 ], dtype=float32)),
     (3, 4, array([-2.53237867, -6.86543894, -9.08977413,  6.28734636,  4.29791164,
             -4.5868268 ], dtype=float32)),
     (3, 4, array([-2.38127351, -6.44985199, -8.59490204,  5.70950508,  4.48439026,
             -5.39853907], dtype=float32)),
     (3, 4, array([-2.32759356, -6.9344573 , -9.40801525,  6.34419489,  4.33300877,
             -4.86977148], dtype=float32)),
     (4, 4, array([-4.44970894, -5.86238146, -6.14216566,  3.52800155,  6.42338467,
             -7.72882318], dtype=float32)),
     (3, 4, array([-2.36232662, -6.81367731, -9.34171295,  5.86951542,  5.15029907,
             -6.84966278], dtype=float32)),
     (3, 4, array([-2.19998527, -7.0110755 , -9.77020931,  6.15934801,  5.01338387,
             -6.29174232], dtype=float32)),
     (3, 4, array([-2.11961484, -6.88944197, -9.50257683,  6.01121569,  4.92552662,
             -6.2419219 ], dtype=float32)),
     (3, 4, array([-2.47259641, -6.85091496, -9.03856277,  5.79417801,  5.3243885 ,
             -6.01962042], dtype=float32)),
     (3, 4, array([-1.89043808, -6.97729445, -9.83376026,  6.11616468,  4.88398361,
             -6.34106636], dtype=float32)),
     (3, 4, array([-1.79372263, -6.94506693, -9.99602795,  6.12108469,  4.85669899,
             -6.3888979 ], dtype=float32)),
     (3, 4, array([-2.34650373, -6.96600246, -9.30271435,  5.90265703,  5.2955389 ,
             -5.81561232], dtype=float32)),
     (3, 4, array([-2.33815265, -6.79517698, -9.01669788,  5.77451229,  5.16071701,
             -6.27173042], dtype=float32)),
     (3, 4, array([-1.71181273, -6.93181801, -9.97959805,  6.07332277,  4.86994362,
             -6.44846821], dtype=float32)),
     (3, 4, array([-1.89792585, -6.97226763, -9.84584045,  6.05578661,  5.0473671 ,
             -6.57141924], dtype=float32)),
     (3, 4, array([-1.7636652 , -6.69485998, -9.53874207,  5.50552797,  5.45688057,
             -6.36577988], dtype=float32)),
     (4, 4, array([-0.71331704, -2.28460789, -6.13264322,  0.37746733,  6.5958252 ,
             -9.97271729], dtype=float32)),
     (3, 3, array([-22.9221859 , -26.93676758, -19.15019989,  29.91573334,
               0.3997024 ,  11.5279789 ], dtype=float32)),
     (3, 3, array([-24.87662888, -28.85521126, -20.12420082,  31.98354912,
               0.4574883 ,  11.95340061], dtype=float32)),
     (3, 3, array([-25.15572166, -29.08695602, -20.21409035,  32.24141693,
               0.45854479,  11.90577698], dtype=float32)),
     (3, 3, array([-25.33184433, -29.25309944, -20.30642319,  32.4140625 ,
               0.48938709,  11.95822239], dtype=float32)),
     (3, 3, array([-25.60692024, -29.44164276, -20.34121323,  32.58151627,
               0.57775402,  11.83588982], dtype=float32)),
     (3, 3, array([-25.57218933, -29.44246101, -20.3648777 ,  32.61385345,
               0.5318327 ,  11.90387344], dtype=float32)),
     (3, 3, array([-25.50245285, -29.29776192, -20.23986244,  32.45467377,
               0.59325981,  11.68169975], dtype=float32)),
     (3, 3, array([-25.48640633, -29.28832245, -20.25862122,  32.45458984,
               0.59610516,  11.71856976], dtype=float32)),
     (3, 3, array([-25.60600853, -29.36223602, -20.26925659,  32.50118637,
               0.6537022 ,  11.65474892], dtype=float32)),
     (3, 3, array([-25.63108826, -29.43150139, -20.33000946,  32.59780884,
               0.6022495 ,  11.72529793], dtype=float32)),
     (3, 3, array([-25.60481262, -29.43318939, -20.35662651,  32.60842896,
               0.58630013,  11.79832172], dtype=float32)),
     (3, 3, array([-25.6227684 , -29.48409653, -20.45631027,  32.65551376,
               0.64234316,  11.92437458], dtype=float32)),
     (3, 3, array([ -9.98872852, -11.62016773,  -9.40824032,  12.110466  ,
               2.62031078,  -1.09284925], dtype=float32)),
     (3, 3, array([-6.69546986, -9.39647961, -9.17545509,  9.34967041,  3.6459074 ,
             -2.52598715], dtype=float32)),
     (3, 3, array([ -2.43016315,  -7.10280943, -10.06380081,   6.30544662,
               5.15830994,  -6.42095613], dtype=float32)),
     (3, 3, array([-2.84089136, -7.21258545, -9.63579369,  6.45496702,  5.10080814,
             -5.43459558], dtype=float32)),
     (3, 3, array([-2.71451926, -7.01895046, -9.55422306,  6.33793306,  4.92186928,
             -5.42345953], dtype=float32)),
     (3, 3, array([-2.6625793 , -6.99332905, -9.5878315 ,  6.27502251,  4.98907566,
             -5.48999786], dtype=float32)),
     (3, 3, array([-2.80706525, -7.04858828, -9.49989986,  6.28968334,  5.08919907,
             -5.4244957 ], dtype=float32)),
     (3, 3, array([-2.70001531, -7.13543272, -9.72420406,  6.42120314,  4.98780823,
             -5.38440609], dtype=float32)),
     (3, 3, array([-2.54584312, -7.1581645 , -9.99203491,  6.46299124,  4.99130344,
             -5.45346355], dtype=float32)),
     (3, 3, array([-2.97901607, -7.30539942, -9.80662632,  6.68586206,  4.9133606 ,
             -4.71722698], dtype=float32)),
     (3, 3, array([-6.9197135 , -9.7539959 , -9.03117657,  9.97245884,  2.83258414,
             -1.23228049], dtype=float32)),
     (3, 3, array([-6.47047186, -9.32077026, -8.78525352,  9.38014698,  3.14472437,
             -1.88316369], dtype=float32)),
     (5, 5, array([-12.26075268,  -7.78467178, -10.82828426,   6.23471546,
               9.17009354,  36.8335228 ], dtype=float32)),
     (5, 5, array([-10.40911579,  -5.38071394,  -9.02834129,   4.96058178,
               5.49180317,  40.89200974], dtype=float32)),
     (5, 5, array([-11.37287045,  -6.65380478,  -9.07536507,   5.88303661,
               5.6241436 ,  39.68205261], dtype=float32)),
     (5, 5, array([-11.06305408,  -6.20220423,  -8.75869083,   5.5417099 ,
               5.2655592 ,  39.42456818], dtype=float32)),
     (5, 5, array([-10.7684021 ,  -5.80932856,  -8.51491833,   5.21888208,
               5.09392071,  39.17765808], dtype=float32)),
     (5, 5, array([-10.80257988,  -5.79872131,  -8.55150032,   5.21438551,
               5.08633232,  39.46353149], dtype=float32)),
     (5, 5, array([-10.67834282,  -5.69227791,  -8.42664909,   5.12973785,
               4.99996853,  39.32739639], dtype=float32)),
     (5, 5, array([-10.76403618,  -5.77988768,  -8.59172058,   5.2177186 ,
               5.06437778,  39.68693542], dtype=float32)),
     (5, 5, array([-10.73688984,  -5.77457237,  -8.51043892,   5.19821787,
               5.07008648,  39.34613037], dtype=float32)),
     (5, 5, array([-10.85540867,  -5.90030766,  -8.63284397,   5.30448055,
               5.11516953,  39.62858582], dtype=float32)),
     (5, 5, array([-10.7032795 ,  -5.68383646,  -8.30196285,   5.11556578,
               4.95678139,  39.28152084], dtype=float32)),
     (5, 5, array([-10.86116886,  -5.94530487,  -8.69406509,   5.3477788 ,
               5.13887596,  39.62137985], dtype=float32)),
     (5, 5, array([-10.6939497 ,  -5.65755701,  -8.33543396,   5.10634518,
               4.92955494,  39.60527039], dtype=float32)),
     (5, 5, array([-10.88868332,  -5.95588732,  -8.65769672,   5.33557701,
               5.21268272,  39.22554779], dtype=float32)),
     (5, 5, array([-10.86800289,  -5.93184471,  -8.61481094,   5.33362198,
               5.07428265,  39.64427948], dtype=float32)),
     (5, 5, array([-10.67854023,  -5.78851223,  -8.55750561,   5.25003958,
               4.92547607,  39.96915817], dtype=float32)),
     (5, 5, array([-11.5561552 ,  -6.87650728, -10.47961617,   5.95200729,
               7.30718803,  39.80793762], dtype=float32)),
     (5, 5, array([-10.61329269,  -5.81471539, -11.19043922,   5.34880829,
               7.2866888 ,  39.34413528], dtype=float32)),
     (5, 5, array([-11.11703873,  -6.25769472,  -8.41781521,   5.57959175,
               5.04898739,  39.61547089], dtype=float32)),
     (5, 5, array([-10.99158764,  -6.08999538,  -8.32744312,   5.41023636,
               5.05857658,  38.56174088], dtype=float32)),
     (5, 5, array([-10.89267349,  -5.89052629,  -8.29146767,   5.26873159,
               4.92054081,  39.17419434], dtype=float32)),
     (5, 5, array([-10.73914337,  -5.71751928,  -7.9528904 ,   5.12342834,
               4.67146778,  39.03487015], dtype=float32)),
     (5, 5, array([-10.93894386,  -5.97412872,  -8.4146347 ,   5.34440565,
               4.97654533,  39.35787964], dtype=float32)),
     (5, 5, array([-10.92741489,  -6.01164484,  -8.23416519,   5.36051369,
               4.89315796,  38.95082092], dtype=float32)),
     (5, 5, array([-11.02495384,  -6.10387468,  -8.45821762,   5.44380093,
               5.02433014,  39.2149353 ], dtype=float32)),
     (5, 5, array([-10.95605755,  -6.01813459,  -8.24625206,   5.36238098,
               4.92010021,  38.95508194], dtype=float32)),
     (5, 5, array([-10.96088886,  -6.03542709,  -8.44211006,   5.39480543,
               5.01161098,  39.27127075], dtype=float32)),
     (5, 5, array([-10.94286823,  -5.9737668 ,  -8.27035904,   5.33719063,
               4.90182686,  39.14473343], dtype=float32)),
     (5, 5, array([-10.9879446 ,  -6.06413984,  -8.52514648,   5.41390896,
               5.09321117,  39.2333374 ], dtype=float32)),
     (5, 5, array([-10.99257946,  -5.97477531,  -8.23422527,   5.32734108,
               4.93359852,  39.06478119], dtype=float32)),
     (0, 0, array([ 17.08390617,   7.23778296, -15.52332878,  -8.70501328,
               3.74835181, -28.13025856], dtype=float32)),
     (0, 0, array([ 20.84195137,   7.98178768, -18.19022942,  -9.7967701 ,
               4.14323378, -26.85235214], dtype=float32)),
     (0, 0, array([ 23.36121178,   8.53159809, -20.17650032,  -9.92652416,
               3.17628074, -33.29695511], dtype=float32)),
     (0, 0, array([ 18.26270103,   3.719661  , -19.74065208,  -5.1641283 ,
               3.71413541, -25.31798935], dtype=float32)),
     (0, 0, array([ 22.2824173 ,   9.13753986, -18.03681946, -10.60797501,
               3.31834722, -32.59884644], dtype=float32)),
     (0, 0, array([ 25.13984299,   8.86869144, -22.57683945,  -9.98466682,
               2.86182928, -34.67696762], dtype=float32)),
     (0, 0, array([ 26.66765022,  12.51108265, -19.41193008, -14.13735294,
               2.9142046 , -34.04718399], dtype=float32)),
     (0, 0, array([ 28.15550995,  11.6009016 , -21.68833923, -12.40227699,
               0.70749736, -39.00621796], dtype=float32)),
     (0, 0, array([ 23.85008812,   7.33579445, -22.48127174,  -8.76322079,
               3.76987076, -26.92749786], dtype=float32)),
     (0, 0, array([ 23.02490807,   8.62670422, -20.24394417,  -9.9762125 ,
               3.13560224, -33.59138107], dtype=float32)),
     (0, 0, array([ 17.42402267,   5.34009886, -17.61891365,  -7.48032808,
               4.93456841, -26.41831398], dtype=float32)),
     (0, 0, array([ 23.38104248,   9.2340889 , -19.32282257, -11.24387264,
               4.66329479, -30.18690681], dtype=float32)),
     (0, 0, array([ 20.05272865,   3.063977  , -23.1332531 ,  -4.30464935,
               3.60837555, -28.82343102], dtype=float32)),
     (0, 0, array([ 17.8942318 ,   1.34549999, -22.44093704,  -2.60467386,
               3.75081778, -24.45668411], dtype=float32)),
     (0, 0, array([ 18.10634232,   4.06372833, -19.81075478,  -5.53290415,
               4.5494051 , -24.51803207], dtype=float32)),
     (0, 0, array([ 18.09972954,   3.65532398, -20.22575188,  -4.4995079 ,
               2.4800849 , -27.83104706], dtype=float32)),
     (0, 0, array([ 20.18344688,   7.87688637, -17.14629364,  -9.35114956,
               3.50706077, -28.55614853], dtype=float32)),
     (0, 0, array([ 23.23415947,   8.2628088 , -20.92663956, -10.23565292,
               4.4408989 , -33.12098312], dtype=float32)),
     (0, 0, array([ 22.64467621,   5.43466282, -23.35481834,  -6.83875227,
               3.44766831, -31.19970703], dtype=float32)),
     (0, 0, array([ 25.28737068,   9.03502941, -21.90707588, -10.51397705,
               3.08277583, -35.58947372], dtype=float32)),
     (0, 0, array([ 24.05681992,   9.48468781, -19.91516113, -11.15986347,
               2.86527991, -34.25822067], dtype=float32)),
     (0, 0, array([ 25.47972488,  10.92513371, -20.13793373, -12.73541737,
               4.13617325, -35.1194458 ], dtype=float32)),
     (0, 0, array([ 21.28140068,   7.19583941, -19.789011  ,  -9.4355402 ,
               4.61240482, -31.1831131 ], dtype=float32)),
     (0, 0, array([ 26.53211594,  11.09303188, -20.77522469, -12.65316486,
               3.31496954, -36.58188248], dtype=float32)),
     (0, 0, array([ 25.29804993,   8.80875874, -21.95280266, -10.02444363,
               2.18398762, -33.52023315], dtype=float32)),
     (0, 0, array([ 23.00911713,   6.90358448, -22.10909843,  -8.27085018,
               4.0694952 , -30.0615406 ], dtype=float32)),
     (0, 0, array([ 19.7714386 ,   2.57662487, -23.18143463,  -3.57595491,
               2.91196585, -31.19180679], dtype=float32)),
     (0, 0, array([ 20.89570045,   7.43328428, -18.51335716,  -9.79340458,
               4.55595875, -27.42800713], dtype=float32)),
     (0, 0, array([ 23.80883026,   8.50555038, -20.82629395, -10.33697605,
               3.84497571, -34.64973831], dtype=float32)),
     (0, 0, array([ 19.0759716 ,   2.99262571, -21.95939636,  -4.34436655,
               3.91310287, -24.85625267], dtype=float32)),
     (0, 0, array([ 16.6950779 ,   3.25354838, -19.12204933,  -4.8804822 ,
               4.40297508, -26.32382393], dtype=float32)),
     (2, 2, array([-15.03154278,  -0.23742092,  16.37713051,  -3.6628747 ,
              -4.06822205,  -0.24089581], dtype=float32)),
     (2, 2, array([-12.32919884,   2.11400628,  12.78683281,  -4.86459684,
              -1.97862601,   2.64609909], dtype=float32)),
     (2, 2, array([-14.36485863,  -0.28817418,  15.82719898,  -4.06541443,
              -1.60451543,  -3.67717218], dtype=float32)),
     (2, 2, array([-10.92732239,   1.36491573,  17.57439613,  -5.15288115,
              -4.66149616,   1.54635513], dtype=float32)),
     (2, 2, array([-12.04094601,   4.01729298,  14.47982025,  -6.86532021,
              -3.469877  ,  -2.10700846], dtype=float32)),
     (2, 2, array([-2.19569325,  6.57306194,  9.37354946, -7.54377365, -3.80486465,
              1.43658006], dtype=float32)),
     (2, 2, array([ -8.56435394,   1.95927083,  14.03424454,  -4.53599024,
              -4.36144638,   2.53933764], dtype=float32)),
     (2, 2, array([-10.37435627,   0.09687352,   8.54300022,  -2.79498291,
              -1.66395545,   2.67106295], dtype=float32)),
     (2, 2, array([ -8.50664997,   7.14457512,  18.27249146, -10.36366272,
              -3.75725007, -12.94818497], dtype=float32)),
     (2, 2, array([-10.37409115,   2.84570885,  15.41078663,  -6.50989294,
              -2.13214135,  -6.6015563 ], dtype=float32)),
     (2, 2, array([-8.86983204,  3.16441274,  8.68310452, -5.0167675 , -2.24095464,
              4.40485334], dtype=float32)),
     (2, 2, array([ -8.90671253,   2.76873159,  15.55282593,  -5.82883406,
              -2.78700566,   0.83577347], dtype=float32)),
     (2, 2, array([ -9.63226318,   3.38670254,  13.60105324,  -6.18491364,
              -3.27711821,   2.72681379], dtype=float32)),
     (2, 2, array([ -9.7637701 ,   5.29400587,  10.47618961,  -6.80376434,
              -2.55526638,   3.29794788], dtype=float32)),
     (2, 2, array([-14.90177727,   3.27180338,  17.71344566,  -7.38690424,
              -4.38813686,  -0.13048321], dtype=float32)),
     (2, 2, array([ -9.63718224,   4.73811865,  20.01467514,  -8.55241299,
              -4.4361124 ,  -3.02943659], dtype=float32)),
     (2, 2, array([-6.36248016,  6.81947947,  8.99561501, -8.36158657, -3.05088282,
              1.8635937 ], dtype=float32)),
     (2, 2, array([ -9.55223083,   1.12399244,  10.32152462,  -4.24314117,
              -1.49599433,   0.53873909], dtype=float32)),
     (2, 2, array([ -9.56446075,   1.95915926,  14.74110699,  -5.27043533,
              -2.2282145 ,   1.27404022], dtype=float32)),
     (2, 2, array([ -7.93782425,   4.85285711,  10.80161667,  -6.79126501,
              -3.4397471 ,   1.40251076], dtype=float32)),
     (2, 2, array([ -7.85132122,   5.04528284,  14.04159355,  -7.15184212,
              -3.65541482,   2.55182695], dtype=float32)),
     (2, 2, array([-11.41800976,   5.92093897,  17.10561943,  -9.57489395,
              -2.97274542,  -4.67697811], dtype=float32)),
     (2, 2, array([-13.06803703,  -0.13623792,  16.73019028,  -3.83976674,
              -3.84299302,   3.60716963], dtype=float32)),
     (1, 1, array([ 31.65692902,  48.7271843 ,  15.80688   , -41.87897873,
             -22.40621758, -31.57033348], dtype=float32)),
     (1, 1, array([ 14.28443718,  28.0657444 ,   7.85997868, -24.34110641,
              -8.64201069, -21.30473137], dtype=float32)),
     (1, 1, array([ 15.97102547,  26.5256176 ,   5.92735052, -23.08804512,
              -9.20820713, -22.35129929], dtype=float32)),
     (1, 1, array([ 21.47616005,  35.52138138,  12.14407444, -31.31727791,
             -15.61921978, -31.92417526], dtype=float32)),
     (1, 1, array([ 13.86083698,  28.3465519 ,   5.17955971, -24.48166466,
              -6.43563557, -14.28340912], dtype=float32)),
     (1, 1, array([ 18.12249184,  35.80367279,   6.66058207, -30.49608421,
             -10.62360859, -10.99551582], dtype=float32)),
     (1, 1, array([ 12.7398777 ,  22.78836441,   3.9110949 , -20.34179497,
              -3.87438035, -13.74249458], dtype=float32)),
     (1, 1, array([ 18.63375854,  32.48674393,   6.74019241, -28.55390358,
             -11.98156643, -23.00902367], dtype=float32)),
     (1, 1, array([ 30.6704464 ,  46.81384659,  12.52902889, -41.1660347 ,
             -22.25779915, -38.23180389], dtype=float32)),
     (1, 1, array([ 25.57121277,  42.18540192,  16.63568306, -37.98379517,
             -21.74685669, -37.63954926], dtype=float32)),
     (1, 1, array([ 13.41983604,  30.63416862,   6.60705853, -25.92532158,
              -9.56782436, -14.31430435], dtype=float32)),
     (1, 1, array([ 14.31925964,  28.9760704 ,   7.1572237 , -25.29997635,
             -10.63648987, -15.97837448], dtype=float32)),
     (1, 1, array([ 19.40082169,  35.61540222,  10.50535297, -31.54383087,
             -13.87275028, -24.35209846], dtype=float32)),
     (1, 1, array([ 19.53152084,  34.72525406,   8.85436153, -29.98431015,
             -13.67648029, -21.32529068], dtype=float32)),
     (1, 1, array([ 20.75848961,  33.44119644,  14.69167519, -31.30474091,
             -14.69443607, -33.00931168], dtype=float32)),
     (1, 1, array([ 19.66094017,  34.13346863,   8.80290699, -29.56027794,
             -14.03911304, -24.87745285], dtype=float32)),
     (1, 1, array([ 14.97581768,  31.03879547,  13.48637199, -27.92286301,
             -10.90337467, -25.86013222], dtype=float32)),
     (1, 1, array([ 24.00387573,  38.89958954,  15.71029282, -35.32792282,
             -17.2948494 , -34.6371727 ], dtype=float32)),
     (1, 1, array([ 15.65784836,  33.70145798,   3.5089128 , -28.03948402,
             -10.82416153, -16.68078804], dtype=float32)),
     (1, 1, array([ 18.14425278,  36.50126266,   8.27361488, -30.87662315,
             -13.03063011, -22.97102928], dtype=float32)),
     (1, 1, array([ 42.09512711,  66.2875061 ,  32.02936554, -60.79971695,
             -38.12692642, -70.44392395], dtype=float32)),
     (1, 1, array([ 28.03192329,  48.87007141,  14.17941189, -42.17631531,
             -19.5343647 , -26.99969482], dtype=float32)),
     (1, 1, array([ 41.9758606 ,  65.31382751,  31.36070061, -59.62259674,
             -37.79243088, -67.52014923], dtype=float32)),
     (1, 1, array([ 39.28900146,  60.47770309,  21.64756775, -53.33390808,
             -26.92838478, -39.26977539], dtype=float32)),
     (1, 1, array([ 31.14910889,  52.54169846,  19.64717102, -45.77556992,
             -23.47047806, -30.36875153], dtype=float32)),
     (1, 1, array([ 41.53960037,  62.13248825,  26.9884758 , -56.14383698,
             -32.36367035, -49.40262604], dtype=float32)),
     (1, 1, array([ 33.63019943,  58.96973419,  18.18281555, -50.79818344,
             -25.12169266, -32.95002747], dtype=float32)),
     (1, 1, array([ 20.19761848,  38.95478439,   6.03975677, -32.75110245,
             -11.93184376, -14.27011108], dtype=float32)),
     (4, 4, array([-1.29806495, -1.20440388, -4.82760239, -0.35435134,  6.1499815 ,
             -8.27516174], dtype=float32)),
     (3, 4, array([-2.41518378, -6.274755  , -8.33543777,  5.43908978,  4.53689098,
             -4.95781469], dtype=float32)),
     (4, 4, array([-2.22571826, -6.37147093, -8.92992592,  4.96124983,  6.1989193 ,
             -7.13518667], dtype=float32)),
     (3, 4, array([ -2.24189997,  -7.22006273, -10.08121109,   6.29190302,
               5.20868015,  -5.48992443], dtype=float32)),
     (3, 4, array([-2.24190307, -7.25003815, -9.91125202,  6.56601381,  4.61688519,
             -5.75464964], dtype=float32)),
     (3, 4, array([-2.61581683, -7.22995424, -9.64266872,  6.38485718,  5.07900333,
             -5.35855484], dtype=float32)),
     (3, 4, array([-1.82359743, -6.79908466, -9.66147614,  5.76735306,  5.19573212,
             -7.15717649], dtype=float32)),
     (3, 4, array([-2.15816689, -6.84794521, -9.55323982,  5.93251657,  5.10233212,
             -6.99955988], dtype=float32)),
     (3, 4, array([-2.11323357, -7.01703691, -9.85342884,  6.1868248 ,  4.94087124,
             -6.48211908], dtype=float32)),
     (3, 4, array([-2.09673691, -7.03611851, -9.87009907,  6.10695505,  5.17926311,
             -6.60975552], dtype=float32)),
     (3, 4, array([-2.17599082, -7.1742835 , -9.98355484,  6.25828505,  5.14746666,
             -6.18301678], dtype=float32)),
     (3, 4, array([ -2.0436008 ,  -7.18434381, -10.14836788,   6.33152103,
               5.01647282,  -6.12233782], dtype=float32)),
     (3, 4, array([-2.30357528, -7.21197224, -9.91670322,  6.28072453,  5.19690418,
             -5.99102926], dtype=float32)),
     (3, 4, array([-2.22013879, -6.95120382, -9.56089306,  5.97325039,  5.19834518,
             -6.23445129], dtype=float32)),
     (3, 4, array([-2.21112084, -7.19407511, -9.99481392,  6.31425476,  5.09391308,
             -6.08938456], dtype=float32)),
     (3, 4, array([-3.25421834, -6.87555647, -8.45632935,  5.7142787 ,  5.26362705,
             -7.09033394], dtype=float32)),
     (4, 4, array([-3.07855701, -4.6981473 , -6.69145441,  2.84298062,  6.95633316,
             -7.13125992], dtype=float32)),
     (4, 4, array([-3.09608221, -5.44314861, -7.33079624,  3.40047574,  8.00851917,
             -6.72640038], dtype=float32)),
     (4, 4, array([-3.31893659, -6.83592796, -8.45215416,  5.26002407,  6.52590942,
             -6.06805325], dtype=float32)),
     (4, 4, array([-2.5826242 , -5.8245883 , -7.87867451,  4.35120201,  6.48381901,
             -7.17328024], dtype=float32)),
     (3, 4, array([-2.05568385, -6.93210888, -9.82476425,  5.87110996,  5.44812107,
             -7.07874632], dtype=float32)),
     (3, 4, array([-2.65751791, -7.1049757 , -9.37362576,  5.97069216,  5.63053131,
             -5.79737329], dtype=float32)),
     (3, 4, array([-2.48230362, -6.930058  , -9.28047562,  5.85103941,  5.4885273 ,
             -6.23885536], dtype=float32)),
     (3, 4, array([-1.99166584, -6.86525202, -9.81282902,  5.77500868,  5.49316216,
             -7.2261219 ], dtype=float32)),
     (3, 4, array([-2.3426075 , -6.98182344, -9.81747627,  6.12117004,  5.12921715,
             -6.44105625], dtype=float32)),
     (3, 4, array([-2.61388898, -7.00968218, -9.47658157,  5.95020866,  5.58904362,
             -6.31655788], dtype=float32)),
     (3, 4, array([-2.16864157, -7.06822634, -9.8694191 ,  6.09028006,  5.31075001,
             -6.7441926 ], dtype=float32)),
     (4, 4, array([-2.29712343, -6.87892818, -9.39003849,  5.60542488,  5.63157177,
             -5.95387173], dtype=float32)),
     (3, 3, array([-20.43653488, -23.06622505, -15.22762299,  25.49097824,
              -0.23242682,  10.60274601], dtype=float32)),
     (3, 3, array([-23.97110939, -27.43946457, -18.89650536,  30.51028252,
               0.45556325,  11.22477436], dtype=float32)),
     (3, 3, array([-23.9815731 , -27.52502632, -19.06925774,  30.62033463,
               0.46069357,  11.09806061], dtype=float32)),
     (3, 3, array([-24.36936378, -27.90326118, -19.2543354 ,  30.99672508,
               0.5213266 ,  11.15742111], dtype=float32)),
     (3, 3, array([-24.4805851 , -28.04374695, -19.42858505,  31.15766716,
               0.57169777,  11.06049919], dtype=float32)),
     (3, 3, array([-24.41326714, -27.89284325, -19.25114059,  30.98849678,
               0.57701331,  10.89682388], dtype=float32)),
     (3, 3, array([-24.31793022, -27.8263588 , -19.27703094,  30.91942024,
               0.57935774,  11.06023121], dtype=float32)),
     (3, 3, array([-24.44709206, -27.98149872, -19.38194275,  31.07982063,
               0.59341055,  11.05727482], dtype=float32)),
     (3, 3, array([-24.52900505, -28.1065464 , -19.48552704,  31.2092514 ,
               0.59961605,  11.15174198], dtype=float32)),
     (3, 3, array([-24.55391312, -28.08998871, -19.41500282,  31.21045494,
               0.52530223,  11.12632656], dtype=float32)),
     (3, 3, array([-24.31880188, -27.83711243, -19.30514145,  30.93273926,
               0.57836318,  10.99468803], dtype=float32)),
     (3, 3, array([-24.34170914, -27.81568909, -19.20575333,  30.89082527,
               0.58335936,  10.99360466], dtype=float32)),
     (3, 3, array([-24.38419914, -27.92431259, -19.34379768,  31.03764153,
               0.54060674,  11.06216812], dtype=float32)),
     (3, 3, array([-24.33618736, -27.83191681, -19.23348999,  30.92001343,
               0.56104648,  11.02189159], dtype=float32)),
     (3, 3, array([-13.36742115, -14.19983482, -10.73441982,  14.77965927,
               3.22459912,  -0.49490371], dtype=float32)),
     (3, 3, array([-11.1221447 , -12.61508465,  -9.93432426,  13.43973255,
               2.26407433,   0.29183   ], dtype=float32)),
     (3, 3, array([-3.28360748, -7.06774044, -9.07859898,  6.07015753,  5.50841808,
             -6.1806922 ], dtype=float32)),
     (4, 3, array([-3.393291  , -7.00298309, -8.87679482,  5.69359684,  6.34020042,
             -6.12492323], dtype=float32)),
     (3, 3, array([-2.98361158, -6.95895863, -9.32033825,  5.99032402,  5.58289242,
             -6.43360996], dtype=float32)),
     (3, 3, array([-2.45652747, -7.09530497, -9.96759129,  6.22411394,  5.2596426 ,
             -6.59664583], dtype=float32)),
     (3, 3, array([-2.40162158, -6.98348284, -9.87426186,  5.97166967,  5.60855103,
             -7.10553694], dtype=float32)),
     (3, 3, array([ -2.51889277,  -7.38294983, -10.47656536,   6.52948189,
               5.48383713,  -6.30987597], dtype=float32)),
     (3, 3, array([ -3.69450855,  -7.98938417, -10.28087139,   7.45204401,
               5.04455376,  -4.0584507 ], dtype=float32)),
     (3, 3, array([-16.06505966, -17.84835052, -11.805233  ,  19.81298637,
               0.51850986,   2.99854326], dtype=float32)),
     (3, 3, array([-15.43031406, -17.78610802, -12.30105209,  19.88323975,
               0.18370046,   5.00111866], dtype=float32)),
     (4, 3, array([-2.34484649, -6.57309484, -9.27807999,  5.46626282,  5.60317421,
             -6.61710691], dtype=float32)),
     (5, 5, array([-14.57540703,  -9.01901436,  -5.89761209,   6.33773041,
               8.13795662,  31.32512093], dtype=float32)),
     (5, 5, array([-13.55297375,  -8.8722496 ,  -7.27016354,   7.28625631,
               5.1425581 ,  36.34066772], dtype=float32)),
     (5, 5, array([-13.45857334,  -8.5307045 ,  -5.65852022,   6.9285512 ,
               4.13634872,  35.29695129], dtype=float32)),
     (5, 5, array([-13.30404854,  -8.24571514,  -5.19907522,   6.65121126,
               3.8635838 ,  34.41679382], dtype=float32)),
     (5, 5, array([-12.70152855,  -7.44563866,  -4.36193609,   6.05236816,
               3.00241756,  34.45210266], dtype=float32)),
     (5, 5, array([-12.9811573 ,  -7.92272186,  -5.17411947,   6.4440589 ,
               3.64669704,  33.97556305], dtype=float32)),
     (5, 5, array([-12.71722507,  -7.4280076 ,  -4.16797256,   6.01244068,
               2.88169026,  34.33132553], dtype=float32)),
     (5, 5, array([-12.48962879,  -7.14646626,  -3.8123548 ,   5.80055666,
               2.58634806,  34.30504227], dtype=float32)),
     (5, 5, array([-12.7918396 ,  -7.55651283,  -4.29314423,   6.1137104 ,
               2.99322891,  34.38736343], dtype=float32)),
     (5, 5, array([-12.63665676,  -7.33646631,  -4.09177542,   5.95033741,
               2.8158803 ,  34.47294235], dtype=float32)),
     (5, 5, array([-12.91787529,  -7.7440567 ,  -4.57956171,   6.26364946,
               3.23776436,  34.41507721], dtype=float32)),
     (5, 5, array([-12.88119698,  -7.66132832,  -4.29209137,   6.21017885,
               2.93916607,  34.69367218], dtype=float32)),
     (5, 5, array([-13.81690025,  -9.0560894 ,  -8.31141758,   7.01589298,
               7.71403313,  35.02416992], dtype=float32)),
     (5, 5, array([-12.80466557,  -8.29246712,  -7.67567682,   6.9771986 ,
               5.07345104,  36.8984108 ], dtype=float32)),
     (5, 5, array([-13.14221478,  -8.31872654,  -6.13369894,   6.8511076 ,
               4.22301435,  35.91259384], dtype=float32)),
     (5, 5, array([-13.04973412,  -8.1266861 ,  -5.9292407 ,   6.65777969,
               4.09119606,  35.29684067], dtype=float32)),
     (5, 5, array([-12.62776184,  -7.55727911,  -5.23895311,   6.23270941,
               3.38641644,  35.4136467 ], dtype=float32)),
     (5, 5, array([-12.52841568,  -7.4505291 ,  -5.0932498 ,   6.13670397,
               3.29070926,  35.27022171], dtype=float32)),
     (5, 5, array([-12.70288944,  -7.64485645,  -5.36898661,   6.28832483,
               3.5208385 ,  35.35940552], dtype=float32)),
     (5, 5, array([-12.72471714,  -7.67256975,  -5.2922945 ,   6.30310154,
               3.47369027,  35.30273438], dtype=float32)),
     (5, 5, array([-12.82063103,  -7.81984997,  -5.47187662,   6.43344355,
               3.56872487,  35.59986115], dtype=float32)),
     (5, 5, array([-12.55377865,  -7.45036411,  -5.01492882,   6.14150381,
               3.23783326,  35.40673447], dtype=float32)),
     (5, 5, array([-12.76577282,  -7.75729942,  -5.38856983,   6.37315083,
               3.55502319,  35.34741592], dtype=float32)),
     (5, 5, array([-12.76331711,  -7.75169611,  -5.40989923,   6.38561964,
               3.49980235,  35.68296432], dtype=float32)),
     (0, 0, array([ 19.71160698,   6.42181587, -19.01261711,  -7.1991291 ,
               2.42997241, -30.05787659], dtype=float32)),
     (0, 0, array([ 19.01787949,   7.21712589, -16.72907448,  -9.53575611,
               5.00520992, -27.87426758], dtype=float32)),
     (0, 0, array([ 26.87549973,  13.49379539, -18.08024597, -14.73376369,
               0.89667231, -35.14496994], dtype=float32)),
     (0, 0, array([ 21.39911461,   5.43948984, -21.84592628,  -7.07186699,
               4.45007706, -29.76232147], dtype=float32)),
     (0, 0, array([ 23.99745941,   9.57837009, -19.99636078, -11.55976009,
               3.69992709, -34.23202133], dtype=float32)),
     (0, 0, array([ 25.89693642,  12.7354269 , -18.32479858, -14.03483772,
               1.80062234, -37.13855362], dtype=float32)),
     (0, 0, array([ 24.99740791,  11.18340492, -18.8651123 , -12.43102455,
               1.98575532, -33.27002716], dtype=float32)),
     (0, 0, array([ 14.60263824,   3.18824625, -17.06378365,  -5.34384298,
               6.53976822, -18.06380844], dtype=float32)),
     (0, 0, array([ 21.93241692,   9.66599941, -17.83973885, -11.10473251,
               2.99628067, -30.41266441], dtype=float32)),
     (0, 0, array([ 24.74406815,  11.32207966, -18.44234848, -13.10393333,
               3.35269761, -33.3950882 ], dtype=float32)),
     (0, 0, array([ 28.04376411,  13.76791573, -19.09334373, -14.51249313,
               0.26042992, -36.03073502], dtype=float32)),
     (0, 0, array([ 19.12513542,   3.62967229, -21.41578293,  -5.12439489,
               4.65816021, -24.64909172], dtype=float32)),
     (0, 0, array([ 25.62187004,  10.20458984, -20.87187958, -12.0764246 ,
               3.15202188, -36.20550156], dtype=float32)),
     (0, 0, array([ 20.47322655,   7.65553522, -18.39219856, -10.02762032,
               5.87939548, -27.54801941], dtype=float32)),
     (0, 0, array([ 20.25065613,  14.99919224, -10.53550625, -17.80377579,
               0.29361257, -39.55696106], dtype=float32)),
     (0, 0, array([ 24.53416443,  10.5922842 , -19.69695663, -12.04766083,
               3.78756452, -33.37007523], dtype=float32)),
     (0, 0, array([ 21.45437813,   7.6067977 , -19.08919525,  -9.94253731,
               4.24642658, -30.57871056], dtype=float32)),
     (0, 0, array([ 31.41705704,  15.6604166 , -21.23133659, -16.80440712,
               0.7251249 , -42.67049408], dtype=float32)),
     (0, 0, array([ 22.25020981,   5.81582785, -22.5265274 ,  -7.08488703,
               3.72058105, -25.75486183], dtype=float32)),
     (0, 0, array([ 23.76836777,   8.82060146, -21.28367996, -10.53675747,
               3.06180859, -35.63357925], dtype=float32)),
     (0, 0, array([ 27.14291763,  13.30920029, -18.66879463, -14.70373249,
               1.62166393, -35.91525269], dtype=float32)),
     (0, 0, array([ 29.10379791,  12.87731934, -21.41981697, -13.9546814 ,
               1.15066695, -37.94398117], dtype=float32)),
     (0, 0, array([ 23.70018959,   9.99352646, -19.00809479, -11.63533783,
               3.85751724, -31.78442383], dtype=float32)),
     (0, 0, array([ 28.36886406,  13.56604958, -20.00988197, -15.0726099 ,
               1.90560031, -37.39332581], dtype=float32)),
     (0, 0, array([ 26.55825996,  12.64535618, -19.05152893, -13.65847111,
               1.74274015, -36.82419968], dtype=float32)),
     (0, 0, array([ 22.23578644,   9.1938324 , -18.14866066, -10.85294724,
               3.58101869, -29.58874893], dtype=float32)),
     (0, 0, array([ 26.01940918,  11.02838135, -20.60843277, -12.60405445,
               3.01689506, -35.81843567], dtype=float32)),
     (0, 0, array([ 24.74934006,   9.85530758, -20.73816109, -11.79927063,
               3.96044922, -33.27204895], dtype=float32)),
     (0, 0, array([ 26.3735218 ,  13.16551208, -18.0758419 , -14.63918972,
               3.20427418, -33.49664307], dtype=float32)),
     (2, 2, array([-11.6889267 ,   3.05287099,  18.46024704,  -7.03487921,
              -3.87466407,  -5.16864538], dtype=float32)),
     (2, 2, array([-12.52489758,   1.01203978,  14.85391426,  -4.61101913,
              -2.73963404,   1.36353409], dtype=float32)),
     (2, 2, array([-10.61220837,   1.86886942,  18.7200222 ,  -5.80854225,
              -3.74704099,  -4.75831127], dtype=float32)),
     (2, 2, array([-13.253335  ,  -1.22315812,  16.60288811,  -3.35014224,
              -1.02499259,   0.89463162], dtype=float32)),
     (2, 2, array([-13.3924017 ,  -0.96280897,  15.94701958,  -3.23922014,
              -1.85657382,   1.04095173], dtype=float32)),
     (2, 2, array([-11.51145649,   1.92365468,  19.35219193,  -5.6886096 ,
              -4.39424515,   1.62080681], dtype=float32)),
     (2, 2, array([-14.32968235,   5.5105958 ,  23.72860336, -10.22934628,
              -5.10288811,  -4.87621069], dtype=float32)),
     (2, 2, array([-14.13454056,  -0.08370447,  14.3539362 ,  -3.82250786,
              -2.59809399,   3.93028784], dtype=float32)),
     (2, 2, array([-12.04879856,   0.36352193,  15.45333862,  -4.24649858,
              -2.6503396 ,   0.24371588], dtype=float32)),
     (2, 2, array([-11.32834148,  -1.36276984,  11.33439827,  -1.93851268,
              -1.79875135,   0.54204452], dtype=float32)),
     (2, 2, array([-12.50815964,   3.14001656,  17.23619652,  -7.02188206,
              -3.2423718 ,  -8.27132416], dtype=float32)),
     (2, 2, array([-13.69561005,   0.55653816,  18.49987221,  -5.86418533,
              -1.86784077,  -5.91766119], dtype=float32)),
     (2, 2, array([ -2.33734155,  11.09590816,  16.13440704, -13.20380592,
              -4.19191074,  -5.6499896 ], dtype=float32)),
     (2, 2, array([-11.57431889,   2.90365529,  16.49663353,  -6.33245945,
              -3.20075846,   0.87063777], dtype=float32)),
     (2, 2, array([-13.85773277,   2.42062473,  20.90755272,  -7.01810789,
              -4.96934223,   0.51887131], dtype=float32)),
     (2, 2, array([-12.29791641,   0.04338373,  18.14807129,  -4.28817797,
              -3.20942402,   0.74924064], dtype=float32)),
     (2, 2, array([-17.72086143,  -1.49287474,  16.57856178,  -3.56115317,
              -2.09969759,  -1.07460713], dtype=float32)),
     (2, 2, array([-10.88834286,   5.21372747,  19.08778572,  -9.38246822,
              -2.86211085,  -9.71050549], dtype=float32)),
     (2, 2, array([-12.46190453,   0.67226654,  17.16798401,  -5.12619638,
              -2.16827202,  -1.04058611], dtype=float32)),
     (2, 2, array([ -8.59222412,   4.16536856,  17.51250458,  -7.68068838,
              -3.16892648,  -2.07822132], dtype=float32)),
     (1, 2, array([ 14.39118862,  36.75730133,  26.79358673, -35.94325638,
             -16.65042305, -36.88114166], dtype=float32)),
     (2, 2, array([-14.63322258,  -1.06783676,  16.11301041,  -3.20641422,
              -3.65537739,   5.01316452], dtype=float32)),
     (1, 1, array([ 20.15573311,  37.63087845,  12.16819859, -32.48308945,
             -14.42450047, -32.29094696], dtype=float32)),
     (1, 1, array([ 27.71871185,  43.95853424,  17.17435074, -38.96903992,
             -21.67215538, -41.91098022], dtype=float32)),
     (1, 1, array([ 29.33295822,  45.85100937,  17.36987686, -41.14565659,
             -21.18880653, -42.89637756], dtype=float32)),
     (1, 1, array([ 18.34778404,  30.74611282,  11.02688313, -27.93613243,
             -12.22966576, -27.493536  ], dtype=float32)),
     (1, 1, array([ 27.78816223,  43.96225357,  17.99368668, -38.99966431,
             -21.37365723, -35.19794846], dtype=float32)),
     (1, 1, array([ 18.16444016,  31.26779556,   7.18150568, -27.8002491 ,
              -9.68793201, -17.54457474], dtype=float32)),
     (1, 1, array([ 19.42410278,  32.95795822,   7.64480734, -29.29590797,
              -9.88056469, -17.62096977], dtype=float32)),
     (1, 1, array([ 18.54232597,  32.10018921,   8.49768257, -28.16830826,
             -12.65246677, -24.98501015], dtype=float32)),
     (1, 1, array([ 20.71517944,  38.66347885,  17.58551598, -34.99399948,
             -18.43964958, -42.38251877], dtype=float32)),
     (1, 1, array([ 39.38611984,  62.46554184,  27.59708023, -55.47162628,
             -32.4879837 , -51.54523468], dtype=float32)),
     (1, 1, array([ 36.70420456,  58.54900742,  26.46512604, -52.92648697,
             -33.99548721, -64.32849121], dtype=float32)),
     (1, 1, array([ 31.05679703,  50.80008316,  15.21223068, -43.97766495,
             -20.59863853, -29.57112312], dtype=float32)),
     (1, 1, array([ 30.20324516,  46.33600235,  17.69415283, -40.96676254,
             -23.02838707, -36.03630447], dtype=float32)),
     (1, 1, array([ 15.96666336,  31.38980293,   5.54952335, -27.05335617,
              -7.43436337, -11.46841717], dtype=float32)),
     (1, 1, array([ 21.37829971,  40.79598236,   7.83450317, -34.95035934,
             -12.84802151, -18.79756737], dtype=float32)),
     (1, 1, array([ 17.5527916 ,  37.29235077,   7.08239651, -31.23692322,
             -10.76006699, -14.1586771 ], dtype=float32)),
     (1, 1, array([ 28.95768929,  51.28170013,  15.65259838, -44.22015762,
             -19.8036232 , -29.54859352], dtype=float32)),
     (1, 1, array([ 20.61408234,  40.82217026,   4.96009874, -34.11755371,
             -11.17739296, -14.44233227], dtype=float32)),
     (1, 1, array([ 21.95535278,  43.34460449,   8.51365566, -37.13045502,
             -11.68610001, -15.89853191], dtype=float32)),
     (1, 1, array([ 12.11891079,  25.61278152,   6.69266796, -22.08337021,
              -6.72092628, -16.19017029], dtype=float32)),
     (1, 1, array([ 14.81982517,  29.95629311,  10.09941196, -26.42816734,
              -9.24893284, -22.14165878], dtype=float32)),
     (1, 1, array([ 17.92648506,  30.11426163,  12.96747112, -27.96073914,
             -12.74608421, -27.60263062], dtype=float32)),
     (1, 1, array([ 14.07488632,  27.68919563,   5.81365442, -24.43697739,
              -6.88316345, -12.19328594], dtype=float32)),
     (1, 1, array([ 21.85788536,  37.36907578,  12.28055668, -32.97908401,
             -15.27223492, -23.75003624], dtype=float32)),
     (3, 4, array([-2.8549974 , -6.8408742 , -8.79253674,  6.03255558,  4.93642902,
             -5.26223135], dtype=float32)),
     (3, 4, array([-2.23130131, -6.70417213, -9.27719212,  5.83222389,  4.99016571,
             -5.90505838], dtype=float32)),
     (3, 4, array([-2.45562887, -7.00360394, -9.49069595,  6.16222858,  5.03023815,
             -5.61540127], dtype=float32)),
     (3, 4, array([-2.27989626, -6.83135033, -9.40444374,  5.95851707,  5.00505161,
             -5.87388706], dtype=float32)),
     (3, 4, array([-2.26067138, -6.93607759, -9.54079533,  6.11606407,  4.86843395,
             -5.82687998], dtype=float32)),
     (3, 4, array([-2.15737963, -6.97165537, -9.75067806,  6.12609339,  4.97174072,
             -6.17495346], dtype=float32)),
     (3, 4, array([-2.37852359, -7.03833199, -9.62681198,  6.14387465,  5.14654446,
             -6.17880297], dtype=float32)),
     (3, 4, array([-2.2801187 , -6.90156364, -9.53056526,  5.98852825,  5.12957954,
             -5.99647093], dtype=float32)),
     (3, 4, array([-2.39139032, -7.03590345, -9.6096611 ,  6.17539024,  5.06985569,
             -5.85691404], dtype=float32)),
     (3, 4, array([-2.40735245, -6.91055107, -9.41294289,  6.04215288,  5.06328297,
             -5.94552755], dtype=float32)),
     (3, 4, array([-1.84886861, -6.83445263, -9.94487858,  6.15136623,  4.60196114,
             -6.52997875], dtype=float32)),
     (3, 4, array([-2.31386209, -7.05020428, -9.83374596,  6.13886023,  5.2401638 ,
             -6.0748868 ], dtype=float32)),
     (3, 4, array([-2.32336187, -7.04670954, -9.7882452 ,  6.19849968,  5.09850788,
             -5.95663595], dtype=float32)),
     (3, 4, array([-2.35246491, -6.98340082, -9.66507149,  6.19098997,  4.96059227,
             -5.78944492], dtype=float32)),
     (3, 4, array([-2.40277624, -6.94210005, -9.52302265,  6.15876007,  4.90275002,
             -5.558496  ], dtype=float32)),
     (3, 4, array([-2.55860257, -6.93448925, -9.3705492 ,  6.15804291,  4.92739105,
             -5.5353303 ], dtype=float32)),
     (3, 4, array([-2.46713901, -6.94723272, -9.45849895,  6.28983545,  4.64996052,
             -5.63662958], dtype=float32)),
     (4, 4, array([-3.90577984, -6.1771512 , -7.15004015,  3.87972593,  6.89239883,
             -7.26369619], dtype=float32)),
     (4, 4, array([-2.66715407, -6.42893982, -8.60780048,  4.91784906,  6.35287952,
             -6.51154995], dtype=float32)),
     (4, 4, array([-2.3812592 , -6.69457006, -9.21744919,  5.28080893,  6.24163723,
             -7.00496578], dtype=float32)),
     (3, 4, array([ -2.05132508,  -7.22548199, -10.35835361,   6.17354727,
               5.51230335,  -5.81995869], dtype=float32)),
     (3, 4, array([-3.04297256, -7.29211378, -9.34196663,  6.50279951,  5.06181049,
             -5.068923  ], dtype=float32)),
     (3, 3, array([-12.83835506, -15.57974625, -11.40573978,  17.21253777,
               0.37760532,   4.37581873], dtype=float32)),
     (3, 3, array([-10.44075108, -12.903409  ,  -9.89934254,  14.05050945,
               0.96698773,   2.31724691], dtype=float32)),
     (3, 3, array([-12.25131989, -14.77347469, -10.88033295,  16.24965286,
               0.64993703,   3.64802837], dtype=float32)),
     (3, 3, array([ -7.5310297 , -10.19929695,  -8.9221859 ,  10.61686516,
               2.26954484,  -0.44323313], dtype=float32)),
     (3, 3, array([-6.41106844, -9.30697155, -8.80685425,  9.42554092,  2.90986538,
             -1.55895138], dtype=float32)),
     (3, 3, array([-6.41368675, -9.30995941, -8.80681896,  9.44765568,  2.86696601,
             -1.47912252], dtype=float32)),
     (3, 3, array([ -7.23385143,  -9.94384193,  -8.84683895,  10.28063774,
               2.42708135,  -0.77631104], dtype=float32)),
     (3, 3, array([-5.39495659, -8.4942522 , -8.72432518,  8.37937737,  3.44189358,
             -2.44723845], dtype=float32)),
     (3, 3, array([-5.4163003 , -8.53257179, -8.75342751,  8.40585804,  3.47417164,
             -2.4791224 ], dtype=float32)),
     (3, 3, array([-5.49530649, -8.59964848, -8.77137566,  8.49288559,  3.42711759,
             -2.34196854], dtype=float32)),
     (3, 3, array([ -7.00562143,  -9.8172245 ,  -8.95114994,  10.1024847 ,
               2.59707475,  -1.05617309], dtype=float32)),
     (3, 3, array([-13.10361862, -14.30057812,  -9.63005733,  15.69499779,
               0.8886528 ,   2.16043043], dtype=float32)),
     (3, 3, array([-7.32824469, -9.80752182, -8.99088287,  9.86357307,  3.46404552,
             -2.1932199 ], dtype=float32)),
     (3, 3, array([-4.76663351, -8.08568764, -9.23550987,  7.69401217,  4.52634907,
             -4.52603483], dtype=float32)),
     (3, 3, array([-13.13491249, -15.34385014, -11.34610367,  16.84091187,
               1.39290178,   2.94040728], dtype=float32)),
     (3, 3, array([-12.46453857, -14.68640423, -11.00172806,  16.08823013,
               1.45312548,   2.45687771], dtype=float32)),
     (3, 3, array([-12.30864239, -14.26413727, -10.46277809,  15.57404327,
               1.45008302,   2.15256023], dtype=float32)),
     (3, 3, array([-2.63701582, -6.940341  , -9.84326172,  6.17788172,  5.2140007 ,
             -6.45066309], dtype=float32)),
     (3, 3, array([-7.45399427, -9.91507339, -9.06835651,  9.9964304 ,  3.52782726,
             -2.13118052], dtype=float32)),
     (3, 3, array([-13.37966537, -15.61426258, -11.5252924 ,  17.12057114,
               1.44345796,   3.02695942], dtype=float32)),
     (3, 3, array([-14.61911964, -16.7548027 , -11.92483425,  18.53364182,
               1.05809665,   3.85836267], dtype=float32)),
     (3, 3, array([ -7.52039385, -10.17064762,  -9.42139626,  10.38684273,
               3.29957843,  -1.75695395], dtype=float32)),
     (3, 3, array([-12.95488167, -15.12919712, -11.2377882 ,  16.57757187,
               1.4820025 ,   2.73175287], dtype=float32)),
     (3, 3, array([-14.47619534, -16.66481781, -11.95008755,  18.39993095,
               1.16657662,   3.8224659 ], dtype=float32)),
     (3, 3, array([-14.12204838, -16.28110504, -11.80875778,  17.92298889,
               1.32698452,   3.52534771], dtype=float32)),
     (5, 5, array([-14.13974857,  -5.10259819,   2.90002966,   2.01921463,
               3.91303515,  23.36264801], dtype=float32)),
     (5, 5, array([ -2.5753777 ,   6.19747353,  -8.95930958,  -3.82554507,
               5.22369003,  52.95002365], dtype=float32)),
     (5, 5, array([ -0.68938315,   8.43470669,  -5.89826155,  -5.53246593,
               2.68074203,  55.61137772], dtype=float32)),
     (5, 5, array([  1.35878253,  13.40937424,   0.17473882,  -9.9244175 ,
              -0.77993858,  69.06189728], dtype=float32)),
     (5, 5, array([  1.18488121,  12.85984802,  -0.77726305,  -9.45099068,
              -0.26490057,  66.33824158], dtype=float32)),
     (5, 5, array([  1.86193967,  14.27110386,   1.53346109, -10.66184521,
              -1.80370855,  70.13604736], dtype=float32)),
     (5, 5, array([  1.48562551,  13.46383572,   0.23582   ,  -9.96565437,
              -0.96129668,  67.83298492], dtype=float32)),
     (5, 5, array([  1.56750965,  13.64791775,   0.42186266, -10.11799526,
              -1.09204292,  68.43460846], dtype=float32)),
     (5, 5, array([  1.54037499,  13.5804863 ,   0.40816402, -10.06519699,
              -1.07061028,  68.14963531], dtype=float32)),
     (5, 5, array([  1.51670218,  13.44007969,   0.17628407,  -9.93203163,
              -0.98082149,  67.68332672], dtype=float32)),
     (5, 5, array([  1.35217476,  13.17315578,  -0.18970728,  -9.71253395,
              -0.6901598 ,  67.05969238], dtype=float32)),
     (5, 5, array([  1.48241925,  13.46766186,   0.18194735,  -9.9630909 ,
              -0.94319201,  67.9865799 ], dtype=float32)),
     (5, 5, array([ -7.36018229,  -1.63987029, -12.04733658,   2.00232077,
               8.31338882,  42.54077911], dtype=float32)),
     (5, 5, array([  0.89951473,  13.66270733,   0.84629327, -10.19394684,
              -0.76342547,  73.6680603 ], dtype=float32)),
     (5, 5, array([  1.25525212,  13.50050545,   0.42383724, -10.01237583,
              -0.8221159 ,  69.86520386], dtype=float32)),
     (5, 5, array([  1.78384733,  14.2474575 ,   1.49145496, -10.62176037,
              -1.77145743,  70.789505  ], dtype=float32)),
     (5, 5, array([  1.71179652,  13.98441982,   1.04472351, -10.40366364,
              -1.47359371,  69.53540039], dtype=float32)),
     (5, 5, array([  1.87744641,  14.36331558,   1.61891115, -10.73493385,
              -1.86006403,  70.34618378], dtype=float32)),
     (5, 5, array([  1.85098529,  14.25156689,   1.50644279, -10.64068413,
              -1.80638075,  70.05310059], dtype=float32)),
     (5, 5, array([  1.67770314,  14.27735138,   1.43215561, -10.6775198 ,
              -1.6093688 ,  71.24263   ], dtype=float32)),
     (5, 5, array([  1.9286685 ,  14.1676321 ,   1.2844156 , -10.54999924,
              -1.7150712 ,  69.29657745], dtype=float32)),
     (5, 5, array([  1.93127728,  14.33127022,   1.46731842, -10.69902992,
              -1.7861011 ,  70.09320831], dtype=float32)),
     (5, 5, array([  1.99452901,  14.55930328,   2.00607014, -10.89917374,
              -2.11942554,  71.02263641], dtype=float32)),
     (5, 5, array([  1.98635387,  14.53524876,   1.84365714, -10.86985588,
              -2.03313565,  71.008461  ], dtype=float32)),
     (0, 0, array([ 46.10355759,  29.35794449, -21.65879631, -29.2947464 ,
              -5.79713917, -57.43291855], dtype=float32)),
     (0, 0, array([ 36.70115662,  20.90624428, -20.514328  , -22.6570797 ,
               0.94686353, -40.45426178], dtype=float32)),
     (0, 0, array([ 38.85457611,  19.8390007 , -25.26515007, -20.92247581,
              -0.10741267, -53.24740601], dtype=float32)),
     (0, 0, array([ 37.27557755,  18.40699577, -24.61720848, -20.00424004,
               0.84680301, -51.546978  ], dtype=float32)),
     (0, 0, array([ 49.46268082,  30.63051605, -23.40113068, -30.18390083,
              -7.03807163, -60.38410568], dtype=float32)),
     (0, 0, array([ 42.75983429,  25.55458069, -22.21539688, -25.72918129,
              -2.83210802, -55.68645096], dtype=float32)),
     (0, 0, array([ 42.57017899,  24.02151299, -23.66996574, -24.2381649 ,
              -3.46870136, -51.34521484], dtype=float32)),
     (0, 0, array([ 27.79852676,  14.90303612, -18.23393059, -16.32052231,
               1.99806082, -36.12585068], dtype=float32)),
     (0, 0, array([ 23.16505432,   9.95173359, -18.8052578 , -12.67346478,
               4.46876717, -35.18380356], dtype=float32)),
     (0, 0, array([ 27.87586784,  14.82287884, -18.01495361, -16.18581963,
               0.60500664, -36.460186  ], dtype=float32)),
     (0, 0, array([ 30.20874214,  16.30895615, -18.80003166, -17.72792244,
               1.68193889, -36.86647034], dtype=float32)),
     (0, 0, array([ 28.54294777,  12.939888  , -21.70100975, -14.74443531,
               2.57104015, -41.84537125], dtype=float32)),
     (0, 0, array([ 34.6656189 ,  19.17152596, -20.90707016, -20.20454407,
               0.11231829, -42.94657516], dtype=float32)),
     (0, 0, array([ 28.89374542,  17.45290756, -16.15572548, -18.84547043,
               0.44476682, -38.63742828], dtype=float32)),
     (0, 0, array([ 39.266922  ,  23.09196281, -20.24375534, -24.03683853,
              -2.64821434, -49.10497665], dtype=float32)),
     (0, 0, array([ 33.36832047,  19.63739014, -17.95719337, -20.73046303,
              -0.7136687 , -37.81036758], dtype=float32)),
     (0, 0, array([ 35.03881454,  17.83206558, -22.38894463, -18.81040764,
              -0.06449121, -46.23107147], dtype=float32)),
     (0, 0, array([ 35.94598389,  17.7677784 , -24.16424751, -19.32386398,
               0.80328661, -46.11257172], dtype=float32)),
     (0, 0, array([ 43.22141266,  23.97795296, -24.3068161 , -25.08116341,
              -1.51417613, -57.23268509], dtype=float32)),
     (0, 0, array([ 35.6827774 ,  22.20789337, -17.23246956, -23.44944954,
              -1.67327845, -43.02000809], dtype=float32)),
     (0, 0, array([ 44.99055099,  26.0966301 , -24.91638374, -26.01724434,
              -4.11317062, -59.19828415], dtype=float32)),
     (0, 0, array([ 41.4762764 ,  22.80509949, -23.60585976, -23.52737427,
              -2.02418041, -52.87939453], dtype=float32)),
     (0, 0, array([ 37.26687241,  19.63344955, -22.64363289, -21.01766014,
              -0.22313398, -51.01510239], dtype=float32)),
     (0, 0, array([ 27.67651176,  11.75451279, -21.68126869, -14.02257156,
               2.95075154, -38.19836807], dtype=float32)),
     (0, 0, array([ 32.86296463,  18.42539597, -19.47027779, -19.68944359,
               1.19767594, -36.71645737], dtype=float32)),
     (0, 0, array([ 33.01707458,  16.86130905, -21.31502342, -17.67507172,
              -0.11934429, -45.33287048], dtype=float32)),
     (2, 2, array([-18.31088066,   2.52850199,  24.74433327,  -7.92770338,
              -6.13315105,  -7.50969887], dtype=float32)),
     (2, 2, array([-17.78030968,  -1.07326102,  23.47395706,  -4.45027018,
              -5.27864361,   2.45462751], dtype=float32)),
     (2, 2, array([-19.90474892,   3.45257783,  30.55620575,  -9.82756996,
              -6.81613159,  -8.50781727], dtype=float32)),
     (2, 2, array([-15.09344673,   2.24970675,  27.05620575,  -7.22972536,
              -7.50652885,  -0.86723304], dtype=float32)),
     (2, 2, array([-12.71801376,   1.94195187,  22.91207123,  -6.5653367 ,
              -5.46527481,   1.47365916], dtype=float32)),
     (2, 2, array([-20.62313652,   0.89490032,  26.08116531,  -7.59140348,
              -7.06635952,  -9.04691219], dtype=float32)),
     (2, 2, array([-17.37133026,   0.27466869,  27.8563118 ,  -6.19180727,
              -7.32894039,   1.09012282], dtype=float32)),
     (2, 1, array([-3.54142237,  4.66851568,  7.27074909, -5.68455935, -1.75479937,
             -0.18319649], dtype=float32)),
     (1, 1, array([  7.72652292,  12.70409203,   3.56185889, -12.05681229,
              -3.70529532,  -4.28813839], dtype=float32)),
     (1, 1, array([ 12.20024872,  18.52423668,   4.07216597, -16.97799301,
              -5.10014725,  -7.80680275], dtype=float32)),
     (1, 1, array([  4.9709816 ,  10.99658775,   0.67687786, -10.39399719,
              -1.52489781, -10.39662552], dtype=float32)),
     (1, 1, array([  3.48894453,  10.05126381,   2.96713424,  -8.98618126,
              -2.61570358,   2.33081341], dtype=float32)),
     (1, 1, array([  9.31129551,  23.61898613,   5.68460274, -20.34392929,
              -6.29809189, -11.99715137], dtype=float32)),
     (1, 1, array([  1.71058059,  11.53974152,   4.33412027, -10.83212566,
              -2.91510391, -10.37411404], dtype=float32)),
     (1, 1, array([  2.17816591,  11.5939579 ,   5.36693287, -11.11680698,
              -2.99370217,  -7.1558013 ], dtype=float32)),
     (1, 1, array([ 10.19616318,  19.73882484,   7.49229956, -18.75212479,
              -8.47971916, -17.6781044 ], dtype=float32)),
     (1, 1, array([  8.26264572,  21.56210899,  10.86670876, -19.14565277,
              -7.64906979,  -7.66133022], dtype=float32)),
     (2, 2, array([-20.22360229,  -2.92234349,  23.75009727,  -2.96368098,
              -5.11909008,   0.36189473], dtype=float32)),
     (2, 2, array([-19.31816864,  -2.15115929,  21.43229294,  -3.23483825,
              -4.53986835,   0.55015111], dtype=float32)),
     (2, 2, array([-22.36902237,   2.0445323 ,  30.5684433 ,  -8.96996975,
              -8.29015827,  -6.8095479 ], dtype=float32)),
     (2, 2, array([-17.60367203,  -2.60480547,  24.25780106,  -3.44654155,
              -4.86098194,   0.96017873], dtype=float32)),
     (2, 2, array([-19.08260345,   4.90904093,  33.24900818, -11.92936707,
              -8.48817825,  -8.24113369], dtype=float32)),
     (2, 2, array([-15.74311638,   0.3123309 ,  27.22553825,  -6.20214748,
              -6.05102921,   1.46491063], dtype=float32)),
     (1, 1, array([ 15.75155258,  26.10996056,  13.84784794, -25.75993347,
             -14.03111172, -28.13953781], dtype=float32)),
     (1, 1, array([  8.24346638,  20.24269867,   4.36166191, -17.41485214,
              -5.91886425,  -4.61050463], dtype=float32)),
     (1, 1, array([  6.52742338,  17.95022011,   6.43444681, -16.26464844,
              -3.18286657,  -3.51118994], dtype=float32)),
     (1, 1, array([  3.46658468,  14.46245861,   4.71865845, -13.0263052 ,
              -2.57807064,  -9.80966187], dtype=float32)),
     (1, 1, array([  7.30513525,  15.95398617,   3.8779757 , -14.2694273 ,
              -3.04576516,  -3.51158381], dtype=float32)),
     (1, 1, array([  5.80122566,  16.03723907,   6.59538364, -14.28922462,
              -3.52600169,  -5.36907101], dtype=float32)),
     (1, 1, array([  5.44599056,  15.55898762,   7.52067614, -15.16531467,
              -5.03816319, -17.68967819], dtype=float32)),
     (1, 1, array([  5.02016926,  15.06636906,   4.9679265 , -14.08553219,
              -4.14901733, -12.06570911], dtype=float32)),
     (2, 2, array([-21.72959328,  -2.68578506,  20.85103607,  -3.57524323,
              -2.79224706,  -4.74059439], dtype=float32)),
     (2, 2, array([-17.57047844,   1.57867599,  29.00841522,  -7.96219921,
              -5.8863945 ,  -2.07168078], dtype=float32)),
     (2, 2, array([-19.58679581,  -3.19677186,  25.76379395,  -3.42652583,
              -5.09231663,   2.04073572], dtype=float32)),
     (2, 2, array([-14.4701519 ,   4.7941041 ,  30.26068306, -11.846241  ,
              -6.44577026,  -9.08930969], dtype=float32)),
     (2, 2, array([-19.34991074,   3.44301653,  29.4762764 ,  -9.82000065,
              -6.25618267,  -9.41456318], dtype=float32)),
     (2, 2, array([-14.85280609,  -0.6304512 ,  22.70477676,  -4.66319799,
              -4.27000999,  -0.67358464], dtype=float32)),
     (2, 2, array([-18.99139023,   2.54448843,  26.82171249,  -8.55448246,
              -5.85499191,  -5.82725334], dtype=float32)),
     (2, 2, array([-17.19629478,  -0.74504614,  27.01623535,  -5.83566904,
              -4.1307869 ,  -1.69621503], dtype=float32)),
     (2, 1, array([ -2.12020826,   6.84678984,   7.60710812,  -8.8588686 ,
              -2.87600279, -12.78670311], dtype=float32)),
     (1, 1, array([  4.08887959,  12.12391663,   1.25242126, -11.43134689,
              -2.2953527 , -10.1900568 ], dtype=float32)),
     (1, 1, array([ 36.41629791,  50.68096924,  19.92910957, -46.61903763,
             -25.55322838, -51.59242249], dtype=float32)),
     (1, 1, array([  8.68985367,  17.35283852,   2.99815011, -15.27264404,
              -2.43650675,  -1.0139221 ], dtype=float32)),
     (1, 1, array([  5.83340645,  16.93249702,   2.76533198, -14.74512959,
              -4.12125397,   2.10060167], dtype=float32)),
     (1, 1, array([  4.42776489,  12.76060963,   3.94317317, -11.75419521,
              -0.70034873,  -3.34851646], dtype=float32)),
     (1, 1, array([  3.60610104,  11.03588963,   3.68060899, -10.37904167,
              -2.2532506 ,  -2.28216791], dtype=float32)),
     (1, 1, array([  3.51257133,  10.64021683,   3.08082819,  -9.58983898,
              -1.64021993,   0.8767302 ], dtype=float32)),
     (4, 4, array([-5.86278582, -6.9352231 , -6.40797806,  5.20412588,  6.42324448,
             -4.39185238], dtype=float32)),
     (4, 4, array([-7.35732031, -7.50447273, -6.09414816,  5.74571323,  7.30629539,
             -3.56564474], dtype=float32)),
     (3, 4, array([-6.78994751, -8.28433514, -7.7414937 ,  8.1955595 ,  3.86895871,
             -4.01213217], dtype=float32)),
     (3, 4, array([-5.36699295, -7.54538631, -7.97315454,  6.85187435,  5.07893944,
             -4.61551237], dtype=float32)),
     (3, 4, array([-5.0897522 , -7.28017235, -7.85248375,  6.22589207,  5.9178772 ,
             -4.68535566], dtype=float32)),
     (3, 4, array([-4.76507568, -7.29679966, -8.28447914,  6.63949299,  5.01982212,
             -5.11180258], dtype=float32)),
     (4, 4, array([-4.50198889, -7.14651299, -8.21978569,  5.92045259,  6.26759911,
             -5.03906298], dtype=float32)),
     (4, 4, array([-4.5016613 , -6.95971155, -7.88739014,  5.54988909,  6.66388035,
             -4.9458437 ], dtype=float32)),
     (4, 4, array([-4.60002184, -7.06402349, -8.00893593,  5.78165865,  6.49768734,
             -4.97339869], dtype=float32)),
     (4, 4, array([-4.48123884, -7.14598799, -8.20233631,  5.86408758,  6.46391869,
             -5.10446453], dtype=float32)),
     (4, 4, array([-4.56110764, -7.09321451, -8.03189754,  5.73487997,  6.6296587 ,
             -4.94100237], dtype=float32)),
     (4, 4, array([-3.90486479, -6.72093105, -7.7544899 ,  5.45038605,  6.09608459,
             -4.92450428], dtype=float32)),
     (3, 4, array([-2.95882106, -6.83320236, -9.16967583,  5.86293221,  5.6131897 ,
             -6.40949821], dtype=float32)),
     (4, 4, array([-3.13062596, -7.01672554, -9.20489979,  5.83063173,  6.09940529,
             -6.22792053], dtype=float32)),
     (4, 4, array([-3.88850284, -7.05162239, -8.63119698,  5.62197828,  6.76902676,
             -5.57575941], dtype=float32)),
     (4, 4, array([-3.82522035, -6.9961586 , -8.66908073,  5.69587088,  6.46184921,
             -5.59309483], dtype=float32)),
     (4, 4, array([-4.05868721, -7.10479641, -8.59471989,  5.80604744,  6.46350765,
             -5.40817833], dtype=float32)),
     (4, 4, array([-3.84350133, -7.07906008, -8.79875946,  5.95554113,  6.04516983,
             -5.59411049], dtype=float32)),
     (4, 4, array([-3.60984182, -7.01691198, -8.95625973,  5.83994436,  6.2211256 ,
             -5.88104296], dtype=float32)),
     (4, 4, array([-3.56345749, -7.06130981, -9.00389004,  5.85777521,  6.23675346,
             -5.90545368], dtype=float32)),
     (4, 4, array([-3.60436869, -7.01090002, -8.88587856,  5.77135944,  6.29238701,
             -5.81228065], dtype=float32)),
     (4, 4, array([-3.76890182, -7.02297974, -8.7614069 ,  5.79654598,  6.29581165,
             -5.626966  ], dtype=float32)),
     (4, 4, array([-3.61750054, -7.03520393, -8.92520618,  5.81824875,  6.25791168,
             -5.78371906], dtype=float32)),
     (4, 3, array([-3.69699907, -3.41777539, -4.1886344 ,  1.33482325,  7.36056328,
             -6.74104261], dtype=float32)),
     (4, 3, array([-2.93706489, -4.23872614, -6.24204206,  1.86009049,  8.63596058,
             -7.35952759], dtype=float32)),
     (4, 3, array([-3.21953297, -4.79171276, -6.44393539,  2.46054316,  8.49764061,
             -6.9734807 ], dtype=float32)),
     (4, 3, array([-1.5900023 ,  1.17655861, -0.46102279, -2.38982558,  5.21186352,
             -5.85863352], dtype=float32)),
     (1, 3, array([  2.24966621,   2.59359717,  -3.3549192 ,  -3.0044167 ,
               1.36258602, -11.78398323], dtype=float32)),
     (1, 3, array([-0.4859049 ,  7.20949411,  6.20773411, -7.56528473,  0.05449376,
             -3.19746685], dtype=float32)),
     (2, 3, array([-1.70518684,  7.80029917,  7.94289398, -8.46676064, -0.76704514,
             -2.01372623], dtype=float32)),
     (2, 3, array([-3.25426555,  6.59321117,  8.35181808, -7.25560474, -2.02092862,
             -2.10877347], dtype=float32)),
     (1, 3, array([  2.57618308,  11.41012287,   5.15378189, -10.44471455,
              -3.49371886,  -5.10955381], dtype=float32)),
     (1, 3, array([  4.16228533,  12.26158619,   4.34000397, -11.22125149,
              -2.72910762,  -6.73207045], dtype=float32)),
     (1, 3, array([  3.69041491,  12.48826981,   5.00383902, -11.40299892,
              -3.10476112,  -4.78895044], dtype=float32)),
     (1, 3, array([  3.54558611,  12.64440155,   5.20957708, -11.52733326,
              -3.22308373,  -4.20120192], dtype=float32)),
     (4, 3, array([-6.87185907, -7.06195307, -5.39790344,  5.37798786,  6.76221085,
             -3.27975035], dtype=float32)),
     (3, 3, array([-8.4765358 , -9.13551712, -6.81911755,  8.79338932,  4.4720335 ,
             -3.03725195], dtype=float32)),
     (3, 3, array([-5.41036892, -7.25235939, -7.36363173,  6.35220432,  5.56284428,
             -4.48216772], dtype=float32)),
     (3, 3, array([-5.25483084, -7.44806337, -7.97672939,  6.69630861,  5.27311134,
             -4.81111193], dtype=float32)),
     (3, 3, array([-4.98850918, -7.22776556, -7.88317394,  6.19469643,  5.91636372,
             -4.80200386], dtype=float32)),
     (3, 3, array([-4.87550116, -7.31869221, -8.18420506,  6.34174776,  5.76338673,
             -4.95902634], dtype=float32)),
     (4, 3, array([-4.92595863, -7.13797426, -7.81685495,  5.87997341,  6.48700428,
             -4.75641441], dtype=float32)),
     (4, 3, array([-4.89313173, -7.21361351, -7.95126057,  6.08824015,  6.10809422,
             -4.82471371], dtype=float32)),
     (4, 3, array([-4.6353159 , -7.09635353, -8.20018864,  5.69987631,  6.95368099,
             -5.57660103], dtype=float32)),
     (3, 3, array([-4.22715282, -7.12921524, -8.36152363,  5.99890852,  5.95357227,
             -5.07140207], dtype=float32)),
     (4, 3, array([-4.38111544, -7.0069437 , -8.14044476,  5.75637388,  6.41790199,
             -5.10275316], dtype=float32)),
     (3, 3, array([-4.02087927, -7.17746973, -8.80755424,  6.16140127,  5.86402321,
             -5.69398499], dtype=float32)),
     (4, 3, array([-4.00659227, -7.08187246, -8.57695484,  5.77556467,  6.50974464,
             -5.52385855], dtype=float32)),
     (5, 5, array([ -5.79586077,   1.71706712,  -5.87559652,  -1.79757965,
               3.61968637,  23.75738525], dtype=float32)),
     (5, 5, array([ -6.51496315,   0.87762535,  -5.86150122,  -1.01101267,
               3.6131587 ,  25.05655098], dtype=float32)),
     (5, 5, array([ -5.73665905,   1.74850023,  -6.09261417,  -1.72486997,
               3.69853377,  24.96664619], dtype=float32)),
     (5, 5, array([ -6.5072875 ,   1.05736959,  -5.83716822,  -1.16327941,
               3.79476643,  25.29940605], dtype=float32)),
     (5, 5, array([ -5.69417286,   1.81014895,  -5.99575806,  -1.77234066,
               3.60608673,  24.63710594], dtype=float32)),
     (5, 5, array([ -5.7741046 ,   1.88395464,  -6.25882721,  -1.84573162,
               3.89227343,  25.3399353 ], dtype=float32)),
     (5, 5, array([ -6.86845255,   0.43840963,  -6.37159157,  -0.67756319,
               4.19305229,  26.1777935 ], dtype=float32)),
     (5, 5, array([ -6.27041435,   1.12037182,  -6.04019785,  -1.25085413,
               3.93474507,  25.15912437], dtype=float32)),
     (5, 5, array([ -6.92607784,   0.36481923,  -6.11064243,  -0.57824826,
               3.90473843,  25.77517891], dtype=float32)),
     (5, 5, array([ -5.5287571 ,   1.81329572,  -6.63146114,  -1.66335344,
               3.87900829,  25.53673172], dtype=float32)),
     (5, 5, array([ -6.44690132,   0.92652154,  -6.68671703,  -0.9258967 ,
               3.9901967 ,  26.51852226], dtype=float32)),
     (5, 5, array([ -7.19230175,   0.47748882,  -6.15676022,  -0.91629314,
               4.10687065,  24.85087776], dtype=float32)),
     (5, 5, array([ -6.0894556 ,   1.10516119,  -6.6899581 ,  -1.10552251,
               3.98919344,  25.63508987], dtype=float32)),
     (5, 5, array([ -6.90524769,   0.1038691 ,  -6.40546989,  -0.24089442,
               4.05426598,  26.3694706 ], dtype=float32)),
     (5, 5, array([ -5.60409069,   1.62088895,  -6.81719971,  -1.42703235,
               3.95349312,  25.98649025], dtype=float32)),
     (5, 5, array([ -6.24325466,   0.93608212,  -7.40453243,  -0.83006829,
               4.1922245 ,  27.31903267], dtype=float32)),
     (5, 5, array([ -5.79789686,   1.39135969,  -8.10800076,  -1.19756794,
               4.85611725,  28.02437019], dtype=float32)),
     (5, 5, array([ -7.56348658,  -0.75160462,  -6.53232527,   0.39710692,
               4.51888084,  26.87423515], dtype=float32)),
     (5, 5, array([ -6.48372793,   0.61859781,  -7.79968882,  -0.44644397,
               4.58281898,  28.03351021], dtype=float32)),
     (5, 5, array([ -6.55002165,   0.18890847,  -7.65293312,  -0.14338994,
               4.67469215,  27.82254601], dtype=float32)),
     (5, 5, array([ -5.87676191,   1.11275303,  -8.22007942,  -0.80630666,
               4.82430077,  28.11816216], dtype=float32)),
     (5, 5, array([ -6.51668072,   0.29220808,  -7.83117962,  -0.17958722,
               4.69908714,  28.08227921], dtype=float32)),
     (5, 5, array([ -6.09350729,   0.96192789,  -8.10068512,  -0.71362978,
               4.77047443,  28.32554817], dtype=float32)),
     (5, 5, array([ -6.81398249e+00,  -6.27858937e-03,  -7.53960609e+00,
               6.99044764e-03,   4.77500916e+00,   2.76731663e+01], dtype=float32)),
     (5, 5, array([ -6.00155115,   0.96799242,  -7.42099667,  -0.76125681,
               4.51139545,  27.23348808], dtype=float32)),
     (5, 5, array([ -6.61859989,   0.82311493,  -6.4766736 ,  -0.79960132,
               4.25861835,  26.16432381], dtype=float32)),
     (1, 0, array([  3.18147564,   6.98153353,  -5.25100613,  -7.62330198,
               2.58618593, -15.93899822], dtype=float32)),
     (0, 0, array([ 18.43375397,  10.21328449, -17.09004974, -11.48530293,
               2.21065187, -33.46766281], dtype=float32)),
     (1, 0, array([  8.47960186,  21.16101646,   6.96448088, -18.55157661,
              -6.62508583, -10.16653824], dtype=float32)),
     (1, 0, array([ 12.05056572,  23.11559486,   3.57149696, -20.0350666 ,
              -6.01307869,  -6.70143175], dtype=float32)),
     (1, 0, array([ 11.93971729,  21.9154911 ,   4.53864479, -19.42464447,
              -8.13576794, -26.21090698], dtype=float32)),
     (0, 0, array([ 24.56555557,  17.21839142, -13.37269402, -18.16467667,
              -0.46310383, -35.55896759], dtype=float32)),
     (1, 0, array([  5.25137854,  16.0234108 ,   4.56863403, -14.4576683 ,
              -2.94928002,  -9.72231007], dtype=float32)),
     (1, 0, array([ 22.03830528,  23.18451118,  -2.28890252, -22.80624008,
              -4.47617626, -17.8643856 ], dtype=float32)),
     (1, 0, array([  9.85727596,  23.12394524,   5.90979958, -20.19928551,
              -5.47806549, -18.19977188], dtype=float32)),
     (1, 0, array([ 22.56427574,  32.9862709 ,  11.98115921, -29.87674141,
             -14.41139317, -25.07489395], dtype=float32)),
     (1, 0, array([  6.19194984,  15.89309216,   0.83321774, -13.86854744,
              -4.16190815,   1.66351211], dtype=float32)),
     (0, 0, array([ 33.35726166,  19.72335434, -17.51464844, -20.72554779,
              -2.71904516, -44.18755722], dtype=float32)),
     (0, 0, array([ 28.49176407,  18.82829094, -14.8074522 , -20.03429794,
              -0.42712861, -32.50583649], dtype=float32)),
     (1, 0, array([ 14.77777958,  17.3130703 ,   3.33329248, -18.21470261,
              -6.32937145, -26.81951332], dtype=float32)),
     (0, 0, array([ 33.67416382,  23.56144905, -13.62388229, -24.50527   ,
              -2.16483021, -37.80212784], dtype=float32)),
     (0, 0, array([ 15.21994495,  12.93896675,  -8.92860317, -13.2702446 ,
              -1.52128637, -31.13385582], dtype=float32)),
     (0, 0, array([ 37.74805069,  24.91653633, -16.65828323, -25.01130676,
              -4.39084148, -42.78100586], dtype=float32)),
     (0, 0, array([ 36.56568909,  24.69444084, -16.22258949, -24.5518055 ,
              -5.36624908, -48.04607773], dtype=float32)),
     (1, 0, array([ 14.63615894,  23.26201439,   4.38601637, -21.3065052 ,
              -6.41034317, -12.64958382], dtype=float32)),
     (0, 0, array([ 36.06804657,  21.78756905, -20.06615257, -22.55854416,
              -2.08138394, -51.57223129], dtype=float32)),
     (0, 0, array([ 35.01490402,  22.14283562, -18.24060631, -22.56969643,
              -2.96028352, -43.08388138], dtype=float32)),
     (0, 0, array([ 33.70062637,  23.43355751, -14.8149786 , -24.44260979,
              -1.76617932, -42.39606476], dtype=float32)),
     (0, 0, array([ 30.70608521,  18.1620369 , -16.79570961, -19.43836403,
              -0.68242556, -42.03546524], dtype=float32)),
     (0, 0, array([ 18.72410202,   7.73166895, -17.40261459, -10.96993637,
               6.71172619, -21.62676811], dtype=float32)),
     (1, 0, array([ 20.36638451,  31.61625099,  13.01555061, -29.70545578,
             -15.58462811, -35.44199371], dtype=float32)),
     (1, 0, array([  8.58533382,  17.62311363,   1.9208734 , -15.42576694,
              -4.09230328,   1.62131798], dtype=float32)),
     (2, 2, array([ -1.01341715e+01,   4.28194952e+00,   1.53582411e+01,
              -6.96912575e+00,  -4.62942410e+00,   1.29753351e-03], dtype=float32)),
     (2, 2, array([-12.53637314,   0.84974819,  19.09942627,  -5.34252357,
              -3.4126091 ,  -0.7041893 ], dtype=float32)),
     (2, 2, array([-11.31588173,  -0.04237415,  18.7071476 ,  -4.39818811,
              -5.30269909,  -0.35137457], dtype=float32)),
     (2, 2, array([-13.90988636,   1.17808151,  21.0631218 ,  -6.62731266,
              -2.92576122,  -4.15731239], dtype=float32)),
     (2, 2, array([-11.9154129 ,   2.55195546,  16.40012932,  -6.6301074 ,
              -2.56102872,  -3.72433424], dtype=float32)),
     (2, 2, array([ -7.09104872,   7.26515961,  19.36372757, -10.38364506,
              -4.20251083,  -3.16432476], dtype=float32)),
     (2, 2, array([-13.74505806,   1.74614477,  22.81572151,  -7.42612982,
              -5.58573151,  -6.88852692], dtype=float32)),
     (2, 2, array([ -8.57806492,   8.95490837,  22.44664574, -12.18072701,
              -7.29747486,  -1.94270539], dtype=float32)),
     (2, 2, array([-15.81957531,  -0.0578647 ,  23.07957649,  -5.5857563 ,
              -4.31488323,  -0.30726305], dtype=float32)),
     (2, 2, array([-10.37168407,   8.19472218,  20.7972908 , -12.65586376,
              -3.67548561, -11.11762619], dtype=float32)),
     (2, 2, array([-11.57079506,   1.99052441,  21.68776894,  -6.74416876,
              -4.84990597,  -2.48713326], dtype=float32)),
     (2, 2, array([-12.76969242,   0.89128876,  20.02044868,  -5.42634439,
              -4.68863201,   0.65780652], dtype=float32)),
     (2, 2, array([-12.41066933,   4.62699318,  23.72626305, -10.89718819,
              -6.20730877, -13.01772213], dtype=float32)),
     (2, 2, array([-12.99455547,  -0.66314089,  19.96048164,  -4.09262133,
              -4.20304871,   0.08210373], dtype=float32)),
     (2, 1, array([  0.33625895,  12.45687008,  15.22357655, -14.17711735,
              -5.59926224,  -9.84207439], dtype=float32)),
     (2, 1, array([-8.67605495,  0.77152348,  6.84309816, -3.44178557,  0.10027975,
             -4.47000742], dtype=float32)),
     (1, 1, array([  7.66980791,  19.32032204,  10.40703106, -18.59950829,
              -7.57301903, -16.57173729], dtype=float32)),
     (1, 1, array([  9.90645313,  21.12178421,   7.04030085, -18.48838615,
              -7.83812046,  -1.39052296], dtype=float32)),
     (2, 1, array([ -4.48551369,   7.22943306,  11.50232792,  -8.39066124,
              -4.47774601,   0.87988365], dtype=float32)),
     (1, 1, array([  4.54049253,  14.06561375,   4.25300026, -12.73115921,
              -3.68293142,  -9.41359901], dtype=float32)),
     (2, 1, array([ -3.58239245,   7.61055946,  10.42283249,  -9.58919048,
              -2.54035497,  -6.66039181], dtype=float32)),
     (1, 1, array([ 10.72457027,  22.28796577,  13.94957829, -22.11915398,
             -10.42895317, -17.18565178], dtype=float32)),
     (2, 1, array([ -1.81012559,  10.82356644,  15.28318501, -12.69563961,
              -4.98346138,  -6.74413538], dtype=float32)),
     (1, 1, array([-3.3359828 ,  7.41235495,  7.3237257 , -8.63630867, -2.90535283,
             -8.24455929], dtype=float32)),
     (1, 1, array([ 18.7332859 ,  35.50422287,  11.93593121, -31.14222145,
             -13.47914791, -22.75243759], dtype=float32)),
     (1, 1, array([  6.1046586 ,  15.86240101,   2.75423431, -14.09191322,
              -3.73028398,   1.74814641], dtype=float32)),
     (1, 1, array([  6.47667456,  20.08406448,   4.89342737, -17.7676239 ,
              -5.75594139,   0.16463846], dtype=float32)),
     (1, 1, array([ 11.86650276,  23.72350121,  12.72616005, -22.75506401,
             -12.19482327, -28.38998413], dtype=float32)),
     (1, 1, array([  5.85917854,  16.8933773 ,  12.19474792, -17.6575737 ,
              -6.44543552, -12.78725243], dtype=float32)),
     (2, 1, array([-2.21275997,  8.14458942,  9.77234459, -9.49976444, -3.63888502,
             -1.2708205 ], dtype=float32)),
     (1, 1, array([ 16.97790718,  33.42998123,  18.47667313, -30.79387283,
             -15.15814114, -16.79416847], dtype=float32)),
     (1, 1, array([ 18.1017437 ,  35.95835114,   6.9761672 , -30.52868843,
              -9.19679832, -10.21310043], dtype=float32)),
     (1, 1, array([  6.98443365,  19.63238907,   6.5825572 , -17.64908218,
              -6.8175602 ,  -0.95023465], dtype=float32)),
     (1, 1, array([  6.73886538,  20.83774376,   4.61967754, -18.40212631,
              -5.99622345,   2.22724962], dtype=float32)),
     (1, 1, array([  5.89945459,  17.01238823,   7.54975986, -16.04847145,
              -6.79556847, -16.14944458], dtype=float32)),
     (1, 1, array([  7.56743765,  20.81776237,  14.71339893, -20.94192696,
              -8.64189434, -14.90144253], dtype=float32)),
     (2, 1, array([-2.22609067,  6.06236315,  8.70008278, -7.79237318, -2.72998071,
             -3.52140474], dtype=float32)),
     (2, 2, array([-13.01107216,   1.11476052,  20.0882206 ,  -5.57271814,
              -4.05518913,  -3.49252224], dtype=float32)),
     (2, 2, array([-13.93024063,  -0.82566714,  17.83161926,  -3.97252369,
              -2.13192081,  -0.17497489], dtype=float32)),
     (2, 2, array([-11.66021442,  -1.15225387,  13.9492836 ,  -2.70596933,
              -1.81695294,   0.34219009], dtype=float32)),
     (2, 2, array([ -7.37690163,  11.02919292,  26.61070442, -16.58364487,
              -7.40704155, -14.89847469], dtype=float32)),
     (2, 2, array([ -9.53981686,   8.65035915,  21.48195267, -12.97932625,
              -6.45126534,  -8.59032154], dtype=float32)),
     (2, 2, array([ -7.76701593,   7.60793829,  21.56240273, -10.98861504,
              -5.41228294,  -4.06904078], dtype=float32)),
     (2, 2, array([-12.58475113,   4.30070448,  26.05599022, -10.68287563,
              -6.16495705,  -9.42101097], dtype=float32)),
     (4, 4, array([-2.61278081, -3.12551951, -4.77678776,  0.4122065 ,  8.84186363,
             -8.85536003], dtype=float32)),
     (4, 4, array([-2.21249676, -3.13070178, -4.76095057,  0.52002537,  8.68834782,
             -8.01502323], dtype=float32)),
     (4, 4, array([-2.35541606, -2.92519665, -4.43435812,  0.24778099,  8.87567234,
             -8.17510891], dtype=float32)),
     (4, 4, array([-2.48024631, -2.81742001, -4.25928307,  0.17692374,  8.81633568,
             -8.21867752], dtype=float32)),
     (4, 4, array([-2.14907813, -2.68394995, -4.48588085,  0.08148694,  8.99964809,
             -8.44962883], dtype=float32)),
     (4, 4, array([-2.15897679, -2.93310499, -4.75054932,  0.22511525,  9.10948086,
             -8.40665054], dtype=float32)),
     (4, 4, array([-2.30284142, -2.97798371, -4.65013504,  0.28527427,  8.99254608,
             -8.28543949], dtype=float32)),
     (4, 4, array([-2.35239959, -3.18038273, -4.85778332,  0.55205107,  8.84009647,
             -8.16954041], dtype=float32)),
     (4, 4, array([-2.2533567 , -2.9671526 , -4.61484861,  0.2947115 ,  8.99753475,
             -8.21393681], dtype=float32)),
     (4, 4, array([-2.28536773, -2.81401372, -4.41646242,  0.12308924,  8.98626137,
             -8.3322649 ], dtype=float32)),
     (4, 4, array([-2.24187732, -2.78912854, -4.44735384,  0.11890495,  8.92805481,
             -8.31254578], dtype=float32)),
     (4, 4, array([-2.24612164, -2.85251141, -4.56737137,  0.17995067,  8.96586037,
             -8.35361958], dtype=float32)),
     (4, 4, array([-7.8743844 , -5.76287365, -3.21386623,  3.7825799 ,  6.6041441 ,
             -4.62408018], dtype=float32)),
     (4, 4, array([-4.6787324 , -3.63379097, -3.80903649,  1.53553617,  7.48125744,
             -6.65475178], dtype=float32)),
     (4, 4, array([-4.19393444, -3.54297471, -4.17704821,  1.31857824,  7.74159813,
             -6.98489285], dtype=float32)),
     (4, 4, array([-4.45683575, -3.61246181, -4.07005119,  1.45459163,  7.59147739,
             -6.86645937], dtype=float32)),
     (4, 4, array([-4.77113581, -3.77152228, -3.86670899,  1.6336273 ,  7.51053143,
             -6.60522509], dtype=float32)),
     (4, 4, array([-4.94191122, -3.79471874, -3.71689367,  1.67865026,  7.39435005,
             -6.51732445], dtype=float32)),
     (4, 4, array([-5.06800795, -3.93319559, -3.91295791,  1.82201135,  7.39803505,
             -6.58708143], dtype=float32)),
     (4, 4, array([-5.00807285, -3.83613491, -3.77774334,  1.75111294,  7.35446167,
             -6.52696848], dtype=float32)),
     (4, 4, array([-4.95392275, -3.79931211, -3.8023212 ,  1.70399356,  7.37970543,
             -6.59704924], dtype=float32)),
     (4, 4, array([-4.57285786, -3.54962373, -3.56200337,  1.48174417,  7.18621349,
             -6.46015215], dtype=float32)),
     (4, 4, array([-3.62878776, -3.19415188, -4.20049953,  0.93860519,  8.16470146,
             -7.61750555], dtype=float32)),
     (3, 3, array([ -2.19241848e+01,  -2.51125355e+01,  -1.69086819e+01,
               2.75576057e+01,   1.10603273e-02,   1.14845600e+01], dtype=float32)),
     (3, 3, array([-25.24728203, -29.16556931, -20.20990181,  32.24285889,
               0.35526946,  12.38418293], dtype=float32)),
     (3, 3, array([-25.02660942, -28.98429489, -20.14479828,  32.06329727,
               0.2795088 ,  12.38940144], dtype=float32)),
     (3, 3, array([-25.31566238, -29.25671768, -20.29336739,  32.34041977,
               0.32788396,  12.30554581], dtype=float32)),
     (3, 3, array([-25.51778412, -29.43866348, -20.36935806,  32.52749252,
               0.35191053,  12.34039402], dtype=float32)),
     (3, 3, array([-25.51554489, -29.4199543 , -20.36421967,  32.5037384 ,
               0.38186032,  12.25752068], dtype=float32)),
     (3, 3, array([-25.55358505, -29.4498539 , -20.35559082,  32.53356552,
               0.37574768,  12.29496098], dtype=float32)),
     (3, 3, array([-25.51438332, -29.44441414, -20.41521645,  32.53959656,
               0.37061098,  12.26920223], dtype=float32)),
     (3, 3, array([-25.47916412, -29.40073013, -20.37330437,  32.48612213,
               0.37554795,  12.27608299], dtype=float32)),
     (3, 3, array([-25.53798294, -29.47454834, -20.42175674,  32.5688858 ,
               0.36019188,  12.33765984], dtype=float32)),
     (3, 3, array([-25.48568916, -29.4124527 , -20.38713264,  32.50101089,
               0.36822662,  12.28781891], dtype=float32)),
     (3, 3, array([-25.47987747, -29.40462494, -20.36863136,  32.49473572,
               0.35546443,  12.29819107], dtype=float32)),
     (3, 3, array([-17.22843361, -19.53504562, -14.99636459,  21.35790253,
               2.0474534 ,   4.9577179 ], dtype=float32)),
     (3, 3, array([-21.56337929, -24.55342484, -17.65276718,  27.17972374,
               1.48187482,   8.34086323], dtype=float32)),
     (3, 3, array([-24.04763222, -27.37165833, -19.13864708,  30.40627289,
               1.00831878,  10.18371487], dtype=float32)),
     (3, 3, array([-24.68750381, -28.16936684, -19.49144936,  31.26970291,
               0.83932084,  10.79871082], dtype=float32)),
     (3, 3, array([-25.01090813, -28.55696297, -19.70423698,  31.69098663,
               0.78420967,  11.02178574], dtype=float32)),
     (3, 3, array([-25.17242622, -28.71681595, -19.73828697,  31.87169838,
               0.74058634,  11.08117199], dtype=float32)),
     (3, 3, array([-25.10655212, -28.64468193, -19.6947155 ,  31.8057251 ,
               0.72562683,  11.0595417 ], dtype=float32)),
     (3, 3, array([-25.03939056, -28.53918076, -19.61560059,  31.68738937,
               0.75328672,  10.92884541], dtype=float32)),
     (3, 3, array([-24.93919182, -28.45526505, -19.60639191,  31.60266876,
               0.75823987,  10.93621349], dtype=float32)),
     (3, 3, array([-24.97723007, -28.47909737, -19.60326195,  31.62127876,
               0.76115608,  10.9408083 ], dtype=float32)),
     (3, 3, array([-25.03645897, -28.57273865, -19.68663216,  31.72746468,
               0.75848293,  11.04203892], dtype=float32)),
     (3, 3, array([-24.97301292, -28.47936249, -19.61496162,  31.61649895,
               0.77479511,  10.92679405], dtype=float32)),
     (5, 5, array([-11.63474941,  -7.07394791,  -9.55787945,   5.97881937,
               6.83391476,  34.61569214], dtype=float32)),
     (5, 5, array([-12.16102219,  -7.52809048,  -7.37175894,   6.48897648,
               4.41765404,  38.33563232], dtype=float32)),
     (5, 5, array([-12.64413548,  -7.89143848,  -7.35420942,   6.60943222,
               4.84682655,  36.66305542], dtype=float32)),
     (5, 5, array([-12.07471752,  -7.18499565,  -6.75742292,   6.10361433,
               4.13209438,  36.67422867], dtype=float32)),
     (5, 5, array([-11.97346878,  -7.0352273 ,  -6.63410187,   5.97764921,
               4.04846001,  36.69989014], dtype=float32)),
     (5, 5, array([-11.89059925,  -6.95835733,  -6.53019476,   5.92805958,
               3.9245162 ,  36.89704895], dtype=float32)),
     (5, 5, array([-12.02435112,  -7.12322426,  -6.69622612,   6.04939318,
               4.09126091,  36.87296677], dtype=float32)),
     (5, 5, array([-11.96732616,  -7.07021666,  -6.70200968,   6.02342749,
               4.05772591,  37.03867722], dtype=float32)),
     (5, 5, array([-12.10821056,  -7.25107098,  -6.7558732 ,   6.15698957,
               4.12349558,  36.8598938 ], dtype=float32)),
     (5, 5, array([-12.09100437,  -7.23227644,  -6.8199544 ,   6.14396   ,
               4.1802845 ,  36.90808868], dtype=float32)),
     (5, 5, array([-12.03471851,  -7.17109394,  -6.73154211,   6.09997463,
               4.08258486,  36.97810364], dtype=float32)),
     (5, 5, array([-12.13288403,  -7.26948643,  -6.77757835,   6.16528034,
               4.1569109 ,  36.95001602], dtype=float32)),
     (5, 5, array([-12.09220695,  -7.24008751,  -6.81402588,   6.15341473,
               4.15545702,  36.91752243], dtype=float32)),
     (5, 5, array([-12.02989674,  -7.16956043,  -6.69135904,   6.09895992,
               4.0427556 ,  37.00059128], dtype=float32)),
     (5, 5, array([-12.13174915,  -7.25418377,  -6.75953436,   6.1451149 ,
               4.17308092,  36.86798477], dtype=float32)),
     (5, 5, array([-16.61418724,  -7.18479729,   2.13525653,   4.18139315,
              -0.93305457,  22.9787693 ], dtype=float32)),
     (5, 5, array([-17.3167572 ,  -8.43755436,   5.52576733,   5.5085535 ,
              -1.38516688,  17.25167084], dtype=float32)),
     (5, 5, array([-17.51677513,  -9.5175209 ,   2.79898095,   6.77214479,
               0.08427973,  15.46407413], dtype=float32)),
     (5, 5, array([-17.4451313 , -10.06694126,   1.19917357,   7.45956945,
               0.46257627,  13.46677494], dtype=float32)),
     (5, 5, array([-17.29661179, -10.09607697,   0.81435782,   7.55494308,
               0.49644321,  13.27175236], dtype=float32)),
     (5, 5, array([-17.04548073,  -9.86196804,   1.03814232,   7.33023024,
               0.40028712,  13.49128628], dtype=float32)),
     (5, 5, array([-17.28554916,  -9.64830208,   1.96784282,   6.99475431,
               0.23080349,  14.30160904], dtype=float32)),
     (5, 5, array([-17.32542992,  -9.81525993,   1.69977665,   7.23493862,
              -0.07244909,  14.33368206], dtype=float32)),
     (5, 5, array([-17.09095001,  -9.73560715,   1.52031755,   7.20864105,
              -0.04415295,  13.8190136 ], dtype=float32)),
     (5, 5, array([-17.27364922,  -9.89483738,   1.413975  ,   7.32276249,
               0.12431625,  13.66373634], dtype=float32)),
     (5, 5, array([-17.27560997,  -9.89796448,   1.44197023,   7.32594109,
               0.09183374,  13.65298176], dtype=float32)),
     (5, 5, array([-17.31985855,  -9.92354488,   1.44642389,   7.33859348,
               0.12427706,  13.60532379], dtype=float32)),
     (5, 5, array([-17.21718407,  -9.8678751 ,   1.41520953,   7.31364965,
               0.07407735,  13.72168446], dtype=float32)),
     (5, 5, array([-17.23428154,  -9.86362839,   1.46482682,   7.28676081,
               0.13092718,  13.53729534], dtype=float32)),
     (5, 5, array([-17.35545921,  -9.94240952,   1.46504688,   7.3526268 ,
               0.09879316,  13.62753677], dtype=float32)),
     (5, 5, array([-17.2218647 ,  -9.71815109,   1.75103164,   7.13367319,
              -0.02143392,  12.89364815], dtype=float32)),
     (0, 0, array([ 35.52032089,  21.68677521, -18.01387787, -22.71012497,
              -1.37883079, -43.43668747], dtype=float32)),
     (0, 0, array([ 36.51570892,  23.8247509 , -17.60820198, -24.69031715,
              -4.19098854, -54.84356689], dtype=float32)),
     (0, 0, array([ 40.48688889,  25.49831581, -18.79808044, -25.83737183,
              -4.65713692, -52.83087158], dtype=float32)),
     (0, 0, array([ 37.01844406,  24.85581398, -16.55970573, -25.63104248,
              -2.27907252, -42.63372803], dtype=float32)),
     (0, 0, array([ 36.48239517,  22.02676582, -19.6580143 , -22.60879517,
              -2.99580145, -51.48577881], dtype=float32)),
     (0, 0, array([ 19.62678528,  12.19141293, -16.19358253, -12.86679459,
               1.99173307, -34.46933746], dtype=float32)),
     (0, 0, array([ 22.7310257 ,  11.81981373, -18.41381264, -13.51467228,
               3.22588563, -35.63949966], dtype=float32)),
     (0, 0, array([ 24.19131088,  14.55827999, -15.19483185, -16.15433121,
               2.17189384, -36.49385452], dtype=float32)),
     (0, 0, array([ 30.55203056,  17.63044167, -18.72635651, -18.95141983,
              -0.36918437, -45.31063461], dtype=float32)),
     (0, 0, array([ 30.66974831,  21.1645298 , -13.96113873, -22.23468018,
              -1.72470641, -38.62413025], dtype=float32)),
     (0, 0, array([ 24.19604874,  17.93313408, -10.31396294, -19.17616463,
               1.85920775, -26.48731422], dtype=float32)),
     (0, 0, array([ 29.59510994,  17.92398262, -16.58745766, -19.52083588,
               1.74416709, -34.63047409], dtype=float32)),
     (0, 0, array([ 36.88531113,  22.42542458, -18.43730927, -23.00525093,
              -2.68903685, -49.72757721], dtype=float32)),
     (0, 0, array([ 26.54528999,  16.96637726, -16.96396255, -17.63068199,
               0.08250287, -41.24230194], dtype=float32)),
     (0, 0, array([ 40.78387833,  25.99541473, -18.80815315, -26.01384163,
              -3.34709787, -45.59712601], dtype=float32)),
     ...]




```python
import pandas as pd
def show_serie(i):
    display(LABELS[int(y_train[i])])
    sm = pd.ewma(pd.DataFrame(X_train[i]), halflife=5)
    
    pd.ewma(pd.DataFrame(X_train[i]), halflife=7).iloc[:, :3].plot(figsize=(10, 10))
    plt.show()
    
show_serie(0)
show_serie(101)
```


    'STANDING'


    /usr/local/lib/python3.5/dist-packages/ipykernel_launcher.py:4: FutureWarning: pd.ewm_mean is deprecated for DataFrame and will be removed in a future version, replace with 
    	DataFrame.ewm(halflife=5,ignore_na=False,adjust=True,min_periods=0).mean()
      after removing the cwd from sys.path.
    /usr/local/lib/python3.5/dist-packages/ipykernel_launcher.py:6: FutureWarning: pd.ewm_mean is deprecated for DataFrame and will be removed in a future version, replace with 
    	DataFrame.ewm(halflife=7,ignore_na=False,adjust=True,min_periods=0).mean()
      



![png](LSTM-2steps-gru%3D1-dev_files/LSTM-2steps-gru%3D1-dev_18_2.png)



    'WALKING'


    /usr/local/lib/python3.5/dist-packages/ipykernel_launcher.py:4: FutureWarning: pd.ewm_mean is deprecated for DataFrame and will be removed in a future version, replace with 
    	DataFrame.ewm(halflife=5,ignore_na=False,adjust=True,min_periods=0).mean()
      after removing the cwd from sys.path.
    /usr/local/lib/python3.5/dist-packages/ipykernel_launcher.py:6: FutureWarning: pd.ewm_mean is deprecated for DataFrame and will be removed in a future version, replace with 
    	DataFrame.ewm(halflife=7,ignore_na=False,adjust=True,min_periods=0).mean()
      



![png](LSTM-2steps-gru%3D1-dev_files/LSTM-2steps-gru%3D1-dev_18_5.png)



```python
tf.trainable_variables()
```




    [<tf.Variable 'dense/kernel:0' shape=(9, 16) dtype=float32_ref>,
     <tf.Variable 'dense/bias:0' shape=(16,) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel:0' shape=(32, 32) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias:0' shape=(32,) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel:0' shape=(32, 16) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias:0' shape=(16,) dtype=float32_ref>,
     <tf.Variable 'dense_1/kernel:0' shape=(16, 12) dtype=float32_ref>,
     <tf.Variable 'dense_1/bias:0' shape=(12,) dtype=float32_ref>,
     <tf.Variable 'dense_2/kernel:0' shape=(12, 6) dtype=float32_ref>,
     <tf.Variable 'dense_2/bias:0' shape=(6,) dtype=float32_ref>,
     <tf.Variable 'dense_3/kernel:0' shape=(16, 6) dtype=float32_ref>,
     <tf.Variable 'dense_3/bias:0' shape=(6,) dtype=float32_ref>]



## Training is good, but having visual insight is even better:

Okay, let's plot this simply in the notebook for now.


```python
# (Inline plots: )
%matplotlib inline

font = {
    'weight' : 'bold',
    'size'   : 18
}
matplotlib.rc('font', **font)

width = 12
height = 12
plt.figure(figsize=(width, height))

indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

indep_test_axis = np.append(
    np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    [training_iters]
)
plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

plt.axhline(y=1.0, c='r')

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training iteration')
plt.ylim(0, 2)
plt.show()
```


![png](LSTM-2steps-gru%3D1-dev_files/LSTM-2steps-gru%3D1-dev_21_0.png)


## And finally, the multi-class confusion matrix and metrics!


```python
# Results

predictions = one_hot_predictions.argmax(1)

print("Testing Accuracy: {}%".format(100*accuracy))

print("")
print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)
print("Note: training and testing data is not equally distributed amongst classes, ")
print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
```

    Testing Accuracy: 89.4468903541565%
    
    Precision: 89.6285265917526%
    Recall: 89.44689514760775%
    f1_score: 89.48575079996284%
    
    Confusion Matrix:
    [[445  39   8   0   2   2]
     [ 20 422  24   0   1   4]
     [  0  12 408   0   0   0]
     [  0   6   2 408  75   0]
     [  0   0   0  89 443   0]
     [  0  27   0   0   0 510]]
    
    Confusion matrix (normalised to % of total test data):
    [[ 15.10010242   1.32337964   0.2714625    0.           0.06786563
        0.06786563]
     [  0.67865622  14.31964684   0.8143875    0.           0.03393281
        0.13573125]
     [  0.           0.40719375  13.84458828   0.           0.           0.        ]
     [  0.           0.20359688   0.06786563  13.84458828   2.54496098   0.        ]
     [  0.           0.           0.           3.02002048  15.0322361    0.        ]
     [  0.           0.91618598   0.           0.           0.          17.30573463]]
    Note: training and testing data is not equally distributed amongst classes, 
    so it is normal that more than a 6th of the data is correctly classifier in the last category.



![png](LSTM-2steps-gru%3D1-dev_files/LSTM-2steps-gru%3D1-dev_23_1.png)



```python
sess.close()
```

## Conclusion

Outstandingly, **the final accuracy is of 91%**! And it can peak to values such as 92.73%, at some moments of luck during the training, depending on how the neural network's weights got initialized at the start of the training, randomly. 

This means that the neural networks is almost always able to correctly identify the movement type! Remember, the phone is attached on the waist and each series to classify has just a 128 sample window of two internal sensors (a.k.a. 2.56 seconds at 50 FPS), so those predictions are extremely accurate.

I specially did not expect such good results for guessing between "SITTING" and "STANDING". Those are seemingly almost the same thing from the point of view of a device placed at waist level according to how the dataset was gathered. Thought, it is still possible to see a little cluster on the matrix between those classes, which drifts away from the identity. This is great.

It is also possible to see that there was a slight difficulty in doing the difference between "WALKING", "WALKING_UPSTAIRS" and "WALKING_DOWNSTAIRS". Obviously, those activities are quite similar in terms of movements. 

I also tried my code without the gyroscope, using only the two 3D accelerometer's features (and not changing the training hyperparameters), and got an accuracy of 87%. In general, gyroscopes consumes more power than accelerometers, so it is preferable to turn them off. 


## Improvements

In [another open-source repository of mine](https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs), the accuracy is pushed up to 94% using a special deep LSTM architecture which combines the concepts of bidirectional RNNs, residual connections and stacked cells. This architecture is also tested on another similar activity dataset. It resembles to the architecture used in "[Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)" without an attention mechanism and with just the encoder part - still as a "many to one" architecture which is adapted to the Human Activity Recognition (HAR) problem.

If you want to learn more about deep learning, I have also built a list of the learning ressources for deep learning which have revealed to be the most useful to me [here](https://github.com/guillaume-chevalier/awesome-deep-learning-resources). You could as well learn to [learn to learn by gradient descent by gradient descent](https://arxiv.org/pdf/1606.04474.pdf) (not for the faint of heart). Ok, I pushed the joke deep enough... 


## References

The [dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) can be found on the UCI Machine Learning Repository. 

> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

To cite my work, point to the URL of the GitHub repository: 
> Guillaume Chevalier, LSTMs for Human Activity Recognition, 2016
> https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

My code is available under the [MIT License](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/LICENSE). 

## Connect with me

- https://ca.linkedin.com/in/chevalierg 
- https://twitter.com/guillaume_che
- https://github.com/guillaume-chevalier/



```python
# Let's convert this notebook to a README for the GitHub project's title page:
!jupyter nbconvert --to markdown LSTM-2steps-gru\=1-dev.ipynb

```

    [NbConvertApp] Converting notebook LSTM-2steps-gru=1-dev.ipynb to markdown
    [NbConvertApp] Support files will be in LSTM-2steps-gru=1-dev_files/
    [NbConvertApp] Making directory LSTM-2steps-gru=1-dev_files
    [NbConvertApp] Making directory LSTM-2steps-gru=1-dev_files
    [NbConvertApp] Making directory LSTM-2steps-gru=1-dev_files
    [NbConvertApp] Making directory LSTM-2steps-gru=1-dev_files
    [NbConvertApp] Making directory LSTM-2steps-gru=1-dev_files
    [NbConvertApp] Making directory LSTM-2steps-gru=1-dev_files
    [NbConvertApp] Writing 161447 bytes to LSTM-2steps-gru=1-dev.md



```python
__file__
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-19-358d5687b810> in <module>()
    ----> 1 __file__
    

    NameError: name '__file__' is not defined

