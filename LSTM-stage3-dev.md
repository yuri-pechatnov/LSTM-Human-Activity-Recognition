
# GRU for Human Activity Recognition

Human activity recognition using smartphones dataset and an LSTM RNN. Classifying the type of movement amongst six categories:
- WALKING,
- WALKING_UPSTAIRS,
- WALKING_DOWNSTAIRS,
- SITTING,
- STANDING,
- LAYING.

Compared to a classical approach, using a Recurrent Neural Networks (RNN) with Gatet Recurent Unit cells (GRUs) require no or almost no feature engineering. Data can be fed directly into the neural network who acts like a black box, modeling the problem correctly. Other research on the activity recognition dataset used mostly use a big amount of feature engineering, which is rather a signal processing approach combined with classical data science techniques. The approach here is rather very simple in terms of how much did the data was preprocessed. 


## Details about input data

I will be using an GRU on the data to learn (as a cellphone attached on the waist) to recognise the type of activity that the user is doing. The dataset's description goes like this:

> The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. 

That said, I will use the almost raw data: only the gravity effect has been filtered out of the accelerometer  as a preprocessing step for another 3D feature as an input to help learning. 

## What is an RNN?

As explained in [this article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), an RNN takes many input vectors to process them and output other vectors. It can be roughly pictured like in the image below, imagining each rectangle has a vectorial depth and other special hidden quirks in the image below. **In our case, the "many to one" architecture is used**: we accept time series of feature vectors (one vector per time step) to convert them to a probability vector at the output for classification. Note that a "one to one" architecture would be a standard feedforward neural network. 

<img src="http://karpathy.github.io/assets/rnn/diags.jpeg" />

An GRU is an improved RNN. It is more complex, but easier to train, avoiding what is called the vanishing gradient problem. 



```python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
from time import time
import os
import sys
```


```python
# some reflection

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
if not is_notebook():
    def get_ipython():
        class Mock:
            def run_cell_magic(*args):
                pass
            def system(self, arg):
                os.system(arg)
                
        return Mock()
```


```python
def notify(msg):
    print(msg)
    try:
        get_ipython().system('ypnotify "' + msg + '"')
    except:
        print("can't notify")
```


```python
def find_nb_name():
    from http.server import BaseHTTPRequestHandler, HTTPServer # python3
    class HandleRequests(BaseHTTPRequestHandler):
        def do_GET(self):
            global nb_name
            nb_name = self.requestline.split()[1][1:]
            print("name is found: " + nb_name)

    import socket
    from contextlib import closing

    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('',0))
            return s.getsockname()[1]

    host = ''
    port = find_free_port()
    server = HTTPServer((host, port), HandleRequests)
    server.server_activate()#.serve_forever()
    get_ipython().run_cell_magic('javascript', '', 'var i = document.createElement("img");' + 
                                 ' i.src = "http://localhost:' + str(port) + 
                                '/" + IPython.notebook.notebook_name;')
    server.handle_request()

if is_notebook():
    find_nb_name()
```


    <IPython.core.display.Javascript object>


    name is found: LSTM-stage3-dev.ipynb



```python

# standartize arguments
if is_notebook():
    sys.argv = [nb_name]
sys.argv

import argparse

parser = argparse.ArgumentParser(description='HAR task solution with limited RAM and CPU')

parser.add_argument('--nhidden', dest='n_hidden', type=int, default=16,
                    help='Amount of hidden variables in recurrent unit')
parser.add_argument('--lr', dest='lr', type=float, default=0.0015,
                    help='Learning rate')
parser.add_argument('--bsize', dest='batch_size', type=int, default=400,
                    help='batch size')
parser.add_argument('--training_iterate_dataset_times', dest='training_iterate_dataset_times', 
                    type=int, default=400, help='Loop <training_iterate_dataset_times> times on the dataset')
parser.add_argument('--ewma', dest='ewma_halflife', 
                    type=int, default=4, help='halflife in exponential smoothing and splitting to average and noise')

parser.add_argument('--use_adam', dest="use_adam", action='store_true')
parser.add_argument('--use_rmsprop', dest="use_rmsprop", action='store_true')

args = parser.parse_args()
print(args)


```

    Namespace(batch_size=400, ewma_halflife=4, lr=0.0015, n_hidden=16, training_iterate_dataset_times=400, use_adam=False, use_rmsprop=False)



```python
# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "total_acc_x_", "total_acc_y_", "total_acc_z_"
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

os.chdir(DATA_PATH)
!python download_dataset.py
os.chdir("..")

DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
print("\n" + "Dataset is now located at: " + DATASET_PATH)

```

    
    Downloading...
    Dataset already downloaded. Did not download twice.
    
    Extracting...
    Dataset already extracted. Did not extract twice.
    
    
    Dataset is now located at: data/UCI HAR Dataset/



```python
pd.read_csv("data/UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt", 
            header=None, sep=' ', skipinitialspace=True).values.shape
```




    (2947, 128)



## Preparing dataset:


```python
# Load "X" (the neural network's training and testing inputs)
def load_X(signals_paths):
    X_signals = [pd.read_csv(path, header=None, sep=' ', skipinitialspace=True).values
                 for path in signals_paths] 
    return np.transpose(np.array(X_signals), (1, 2, 0)).astype(np.float32)

X_train = load_X([DATASET_PATH + "train/Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES])
X_test = load_X([DATASET_PATH + "test/Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES])
X_my_test = load_X(["../prepared_my_dataset/accel%s.txt" % s for s in ['X', 'Y', 'Z']])

# Load "y" (the neural network's training and testing outputs)
def load_y(path):
    return pd.read_csv(path, header=None, sep=' ', skipinitialspace=True).values - 1

make_y_easier_m = [0, 1, 2, 3, 3, 3]
def make_y_easier(y):
    return np.array([make_y_easier_m[int(i)] for i in y])

y_train = load_y(DATASET_PATH + "train/y_train.txt")
y_easier_train = make_y_easier(y_train)
y_test = load_y(DATASET_PATH + "test/y_test.txt")
y_easier_test = make_y_easier(y_test)
y_my_test = load_y("../prepared_my_dataset/activity.txt")
y_easier_my_test = make_y_easier(y_my_test)
```


```python
def shuffle_all(*args):
    perm = np.random.permutation(len(args[0]))
    return [np.array(arg)[perm] for arg in args]

X_train, y_train, y_easier_train = shuffle_all(X_train, y_train, y_easier_train)
```


```python
from sklearn.decomposition import PCA

def calc_rotation_zero_not_first(h):
    h = np.array(h)
    
    # max axis to first position
    gci = np.argmax(np.abs(h))
    fmax_m = np.matrix(np.diag([1, 1, 1]))
    if gci != 0:
        fmax_m[0], fmax_m[gci] = np.matrix(fmax_m[gci]), np.matrix(fmax_m[0])
        fmax_m *= -1
    h = np.array(fmax_m * np.matrix(h).T).ravel()
    
    #make first axis positive
    fsign_m = np.matrix(np.diag([(1 if h[0] > 0 else -1)] * 2 + [1]))
    h = np.array(fsign_m * np.matrix(h).T).ravel()
    
    # zero second
    phi = np.arctan2(h[2], h[0])
    s, c = np.sin(phi), np.cos(phi)
    zero2nd_m = np.matrix(np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ]))
    h = np.array(zero2nd_m * np.matrix(h).T).ravel()
    
    # zero first
    phi = np.arctan2(h[1], h[0])
    s, c = np.sin(phi), np.cos(phi)
    zero1st_m = np.matrix(np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1],
    ]))
    h = np.array(zero1st_m * np.matrix(h).T).ravel()
    
    final_rotation = zero1st_m * zero2nd_m * fsign_m * fmax_m
    
    return final_rotation / ((h ** 2).sum() ** 0.5 + 1e-6)
    
#print(calc_rotation_zero_not_first(np.array([1, 2, 3])))

def align_0th(xyz):
    xyz = np.array(xyz)
    h = xyz.mean(axis=0)
    m = calc_rotation_zero_not_first(h)
    return np.array(m * np.matrix(xyz).T).T

def align_determ(xyz):
    xyz = align_0th(xyz)
    pcator = PCA(n_components=2)
    pcator.fit(xyz[:, 1:3])
    xyz[:, 1:3] = pcator.transform(xyz[:, 1:3])
    return xyz

def split_avg_noise(xyz):
    smoothed = pd.ewma(pd.DataFrame(xyz), halflife=args.ewma_halflife).values
    return smoothed, xyz - smoothed
```


```python
if is_notebook():
    def show_serie(i):
        j = np.where(y_train.ravel() == i)[0][5]
        display(LABELS[int(i)])
        pd.DataFrame(X_train[j, :, :3]).plot(figsize=(10, 6))
        plt.show()
#         k = np.where(y_my_test.ravel() == i)[0][0]
#         display(LABELS[int(i)])
#         print(X_my_test.shape)
#         pd.DataFrame(X_my_test[k, :, 0:3]).plot(figsize=(10, 6))
#         plt.show()
   
    show_serie(1)
    show_serie(2)
    for i in range(6):
        inp = len(X_train[0][0])
        print((X_train[y_train.ravel() == i] ** 2).reshape((-1, inp)).mean(axis=0)[:3],
              (X_train[y_train.ravel() == i] ** 2).reshape((-1, inp)).std(axis=0)[:3])
    

```


    'WALKING_UPSTAIRS'



![png](LSTM-stage3-dev_files/LSTM-stage3-dev_14_1.png)



    'WALKING_DOWNSTAIRS'



![png](LSTM-stage3-dev_files/LSTM-stage3-dev_14_3.png)


    [ 1.04405189  0.06873626  0.05026194] [ 0.48435602  0.10794329  0.09266992]
    [ 0.97028708  0.11439437  0.08883563] [ 0.5416224   0.13520329  0.14329197]
    [ 1.12598574  0.06569687  0.05364963] [ 0.87020081  0.12652464  0.11900949]
    [ 0.91490334  0.05576444  0.07207287] [ 0.17270103  0.08080186  0.09085945]
    [ 1.00355542  0.0369462   0.02634041] [ 0.04999202  0.03245921  0.03872563]
    [ 0.02509746  0.5229187   0.4671647 ] [ 0.02688421  0.2738589   0.27586567]



```python
def align_X(X):
    X = np.array(X).astype(np.float)
    if args.ewma_halflife != 0:
        Y = np.array(X)
        for i in range(X.shape[0]):
            X[i] = align_determ(X[i])
            X[i], Y[i] = split_avg_noise(X[i])
        X = np.concatenate([X, Y], axis=2)
    else:
        for i in range(X.shape[0]):
            X[i] = align_determ(X[i])
    return X
    
X_train = align_X(X_train)
# print(X_test.shape, X_my_test.shape)
X_test = align_X(X_test)
X_my_test = align_X(X_my_test)
#print(X_my_test)

# print(X_my_test[-1,:,:])
# print(align_0th(X_my_test[-1,:,:]))
```

    LSTM-stage3-dev.ipynb:58: FutureWarning: pd.ewm_mean is deprecated for DataFrame and will be removed in a future version, replace with 
    	DataFrame.ewm(ignore_na=False,adjust=True,min_periods=0,halflife=4).mean()
      "outputs": [],



```python
X_my_test[0]
```




    array([[  1.00388135e+00,  -3.87861463e-02,  -3.56764976e-03,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  1.00371654e+00,  -3.40527790e-02,  -9.20274803e-03,
             -1.38587658e-04,   3.98027154e-03,  -4.73853394e-03],
           [  1.00654173e+00,  -3.35557645e-02,  -4.44787782e-03,
              4.37341479e-03,   7.69380109e-04,   7.36055429e-03],
           [  1.00564939e+00,  -3.18996304e-02,  -4.84006496e-03,
             -1.91194947e-03,   3.54844400e-03,  -8.40302820e-04],
           [  1.00403576e+00,  -2.87111542e-02,  -5.46267944e-03,
             -4.26417824e-03,   8.42588874e-03,  -1.64532522e-03],
           [  1.00477857e+00,  -2.76662045e-02,  -5.91124404e-03,
              2.27527253e-03,   3.20073855e-03,  -1.37397805e-03],
           [  1.00329927e+00,  -2.45927052e-02,  -4.50019025e-03,
             -5.05420845e-03,   1.05009434e-02,   4.82101817e-03],
           [  1.00214453e+00,  -2.33049794e-02,  -3.81309309e-03,
             -4.28859298e-03,   4.78249767e-03,   2.55181710e-03],
           [  1.00318672e+00,  -2.31406267e-02,  -4.61517782e-03,
              4.13116172e-03,   6.51479358e-04,  -3.17939178e-03],
           [  1.00386164e+00,  -2.26166147e-02,  -4.85542967e-03,
              2.81718093e-03,   2.18729658e-03,  -1.00284346e-03],
           [  1.00335881e+00,  -2.01208770e-02,  -5.19711262e-03,
             -2.18774968e-03,   1.08587321e-02,  -1.48663208e-03],
           [  1.00093996e+00,  -1.61926016e-02,  -5.52683817e-03,
             -1.08837598e-02,   1.76755181e-02,  -1.48362049e-03],
           [  1.00171092e+00,  -1.69199660e-02,  -4.18479569e-03,
              3.56533351e-03,  -3.36374175e-03,   6.20635842e-03],
           [  1.00151711e+00,  -1.64621858e-02,  -4.39484264e-03,
             -9.16622731e-04,   2.16515134e-03,  -9.93453586e-04],
           [  1.00242067e+00,  -1.56856739e-02,  -2.82138777e-03,
              4.35337214e-03,   3.74128254e-03,   7.58100345e-03],
           [  1.00343200e+00,  -1.59081123e-02,  -2.29135334e-03,
              4.94784420e-03,  -1.08825506e-03,   2.59313392e-03],
           [  1.00170536e+00,  -1.48821326e-02,  -2.70334154e-03,
             -8.55533277e-03,   5.08361432e-03,  -2.04135523e-03],
           [  1.00081730e+00,  -1.48911224e-02,  -1.79530117e-03,
             -4.44688601e-03,  -4.50161496e-05,   4.54696100e-03],
           [  1.00015574e+00,  -1.44447923e-02,  -1.74539769e-03,
             -3.34195809e-03,   2.25469834e-03,   2.52094304e-04],
           [  9.99700851e-01,  -1.49207334e-02,  -1.04362477e-03,
             -2.31485474e-03,  -2.42196928e-03,   3.57118245e-03],
           [  9.99418643e-01,  -1.53168519e-02,  -2.90620469e-03,
             -1.44491651e-03,  -2.02814706e-03,  -9.53650339e-03],
           [  1.00011930e+00,  -1.76790026e-02,  -2.58797788e-03,
              3.60581702e-03,  -1.21564035e-02,   1.63769971e-03],
           [  1.00059324e+00,  -2.03357399e-02,  -1.48391972e-03,
              2.44949216e-03,  -1.37311492e-02,   5.70624255e-03],
           [  1.00015874e+00,  -2.01706471e-02,  -3.04193787e-03,
             -2.25371668e-03,   8.56337505e-04,  -8.08145111e-03],
           [  1.00072390e+00,  -2.19080787e-02,  -2.92190210e-03,
              2.94032765e-03,  -9.03921748e-03,   6.24501920e-04],
           [  1.00127637e+00,  -2.22327652e-02,  -2.62972299e-03,
              2.88155187e-03,  -1.69349057e-03,   1.52393936e-03],
           [  1.00285746e+00,  -2.28250058e-02,  -2.98228386e-03,
              8.26406094e-03,  -3.09553456e-03,  -1.84277207e-03],
           [  1.00285733e+00,  -2.42742073e-02,  -3.22481277e-03,
             -6.65465346e-07,  -7.58817904e-03,  -1.26990812e-03],
           [  1.00471929e+00,  -2.53103637e-02,  -3.16497816e-03,
              9.76396474e-03,  -5.43352410e-03,   3.13768083e-04],
           [  1.00590529e+00,  -2.27498989e-02,  -4.32907334e-03,
              6.22706300e-03,   1.34437008e-02,  -6.11207278e-03],
           [  1.00467571e+00,  -2.11220942e-02,  -5.03899367e-03,
             -6.46267174e-03,   8.55576860e-03,  -3.73135299e-03],
           [  1.00293707e+00,  -2.03478881e-02,  -5.29522222e-03,
             -9.14642207e-03,   4.07283650e-03,  -1.34793177e-03],
           [  1.00326685e+00,  -2.23001002e-02,  -3.27881817e-03,
              1.73615310e-03,  -1.02775539e-02,   1.06154965e-02],
           [  1.00294804e+00,  -2.27725149e-02,  -3.47551815e-03,
             -1.67943405e-03,  -2.48861101e-03,  -1.03618654e-03],
           [  1.00389367e+00,  -2.11569235e-02,  -3.74578389e-03,
              4.98407757e-03,   8.51516049e-03,  -1.42446668e-03],
           [  1.00356644e+00,  -1.98954196e-02,  -2.84066200e-03,
             -1.72548928e-03,   6.65183140e-03,   4.77265132e-03],
           [  1.00145136e+00,  -1.60095478e-02,  -4.62493668e-03,
             -1.11567923e-02,   2.04975494e-02,  -9.41185412e-03],
           [  1.00158598e+00,  -1.37560062e-02,  -3.91388295e-03,
              7.10315186e-04,   1.18908871e-02,   3.75189862e-03],
           [  1.00067005e+00,  -1.22840505e-02,  -3.44670843e-03,
             -4.83421803e-03,   7.76885577e-03,   2.46570706e-03],
           [  9.99785654e-01,  -9.91468112e-03,  -1.52096950e-03,
             -4.66878476e-03,   1.25080803e-02,   1.01661214e-02],
           [  9.98911721e-01,  -9.81512098e-03,  -1.92892203e-03,
             -4.61441309e-03,   5.25682724e-04,  -2.15401065e-03],
           [  9.99095644e-01,  -9.87777123e-03,  -1.86438417e-03,
              9.71273956e-04,  -3.30848032e-04,   3.40816268e-04],
           [  9.99967501e-01,  -9.88777747e-03,  -1.53317307e-03,
              4.60476575e-03,  -5.28486281e-05,   1.74931255e-03],
           [  1.00110281e+00,  -1.16275483e-02,  -1.25489561e-03,
              5.99685706e-03,  -9.18972105e-03,   1.46990178e-03],
           [  9.99875278e-01,  -1.27215212e-02,  -1.11182333e-03,
             -6.48458866e-03,  -5.77905706e-03,   7.55798318e-04],
           [  1.00003573e+00,  -1.43374949e-02,  -5.36430124e-04,
              8.47650521e-04,  -8.53725940e-03,   3.03982732e-03],
           [  1.00127193e+00,  -1.58772137e-02,   9.65615855e-04,
              6.53137409e-03,  -8.13493263e-03,   7.93589275e-03],
           [  1.00085562e+00,  -1.61365659e-02,   7.83731139e-04,
             -2.19966611e-03,  -1.37033387e-03,  -9.61020460e-04],
           [  1.00287857e+00,  -1.68473388e-02,  -3.32792413e-04,
              1.06890877e-02,  -3.75566955e-03,  -5.89962467e-03],
           [  1.00281873e+00,  -1.93819306e-02,  -1.34570108e-04,
             -3.16211795e-04,  -1.33931087e-02,   1.04743212e-03],
           [  1.00236234e+00,  -1.95556208e-02,  -1.23720269e-03,
             -2.41169585e-03,  -9.17830975e-04,  -5.82664255e-03],
           [  1.00231986e+00,  -1.98558923e-02,  -2.95455864e-03,
             -2.24478153e-04,  -1.58676883e-03,  -9.07527526e-03],
           [  1.00320734e+00,  -2.03449432e-02,  -3.14622033e-03,
              4.68994172e-03,  -2.58442279e-03,  -1.01284933e-03],
           [  1.00287053e+00,  -2.01159772e-02,  -4.44286259e-03,
             -1.77988252e-03,   1.21000966e-03,  -6.85232772e-03],
           [  1.00407661e+00,  -2.04691011e-02,  -5.64212472e-03,
              6.37382545e-03,  -1.86617385e-03,  -6.33780932e-03],
           [  1.00578309e+00,  -2.16859710e-02,  -4.87292740e-03,
              9.01846770e-03,  -6.43095074e-03,   4.06507701e-03],
           [  1.00329729e+00,  -2.09957579e-02,  -5.79117335e-03,
             -1.31372124e-02,   3.64770093e-03,  -4.85282968e-03],
           [  1.00226710e+00,  -2.08374060e-02,  -5.65608157e-03,
             -5.44450605e-03,   8.36881025e-04,   7.13952267e-04],
           [  1.00130045e+00,  -2.06342915e-02,  -4.08690097e-03,
             -5.10872898e-03,   1.07345721e-03,   8.29309653e-03],
           [  1.00190813e+00,  -1.96783539e-02,  -3.43044563e-03,
              3.21160499e-03,   5.05215106e-03,   3.46938075e-03],
           [  1.00252238e+00,  -1.90921717e-02,  -3.47424693e-03,
              3.24635111e-03,   3.09800353e-03,  -2.31492197e-04],
           [  1.00320921e+00,  -1.82022488e-02,  -3.01007816e-03,
              3.62993293e-03,   4.70331155e-03,   2.45316814e-03],
           [  1.00518411e+00,  -1.75842810e-02,  -1.83734220e-03,
              1.04375576e-02,   3.26602128e-03,   6.19802616e-03],
           [  1.00400522e+00,  -1.55285276e-02,  -3.30269839e-03,
             -6.23056961e-03,   1.08648983e-02,  -7.74457979e-03],
           [  1.00272763e+00,  -1.44656574e-02,  -3.71003773e-03,
             -6.75221454e-03,   5.61741020e-03,  -2.15284253e-03],
           [  1.00151550e+00,  -1.11888806e-02,  -4.09449803e-03,
             -6.40628449e-03,   1.73182430e-02,  -2.03192870e-03],
           [  1.00188160e+00,  -1.02402566e-02,  -3.72708228e-03,
              1.93489142e-03,   5.01362627e-03,   1.94184976e-03],
           [  1.00168158e+00,  -8.51035020e-03,  -4.64018617e-03,
             -1.05715101e-03,   9.14284178e-03,  -4.82590525e-03],
           [  1.00127307e+00,  -8.77106006e-03,  -4.48968808e-03,
             -2.15902335e-03,  -1.37789678e-03,   7.95408457e-04],
           [  1.00034320e+00,  -9.56934661e-03,  -4.20307694e-03,
             -4.91453092e-03,  -4.21908777e-03,   1.51479135e-03],
           [  9.99933113e-01,  -1.02779969e-02,  -5.12877894e-03,
             -2.16740281e-03,  -3.74534787e-03,  -4.89250628e-03],
           [  1.00213617e+00,  -1.02940453e-02,  -5.76298426e-03,
              1.16435759e-02,  -8.48190947e-05,  -3.35189532e-03],
           [  1.00349769e+00,  -9.07520416e-03,  -4.30655787e-03,
              7.19588665e-03,   6.44181133e-03,   7.69749506e-03],
           [  1.00053926e+00,  -6.26179513e-03,  -2.58505503e-03,
             -1.56358958e-02,   1.48694197e-02,   9.09848087e-03],
           [  9.96606812e-01,  -3.83368116e-03,  -5.16824602e-04,
             -2.07837515e-02,   1.28330661e-02,   1.09310099e-02],
           [  9.92372632e-01,  -2.15370704e-03,   1.94007939e-03,
             -2.23784942e-02,   8.87900175e-03,   1.29852327e-02],
           [  9.88973038e-01,  -6.38250092e-03,   2.32504114e-03,
             -1.79675482e-02,  -2.23500359e-02,   2.03460116e-03],
           [  1.00154075e+00,  -1.76672027e-02,   2.21372205e-03,
              6.64229312e-02,  -5.96419625e-02,  -5.88344197e-04],
           [  1.01228088e+00,  -2.07058621e-02,  -1.60473589e-03,
              5.67638178e-02,  -1.60599422e-02,  -2.01813383e-02],
           [  1.01914731e+00,  -1.43441547e-02,  -8.11336721e-04,
              3.62904789e-02,   3.36229436e-02,   4.19327924e-03],
           [  1.01769277e+00,  -1.97277781e-03,  -1.69372626e-03,
             -7.68752995e-03,   6.53853060e-02,  -4.66361268e-03],
           [  1.01019001e+00,   1.08508379e-02,  -5.42894209e-03,
             -3.96536475e-02,   6.77754924e-02,  -1.97413973e-02],
           [  9.95688643e-01,   2.39713471e-02,  -6.91439924e-03,
             -7.66427749e-02,   6.93446459e-02,  -7.85095288e-03],
           [  9.81167855e-01,   3.58038130e-02,  -4.07722818e-03,
             -7.67454234e-02,   6.25370733e-02,   1.49950463e-02],
           [  9.71390010e-01,   4.38042035e-02,   3.98034779e-04,
             -5.16779744e-02,   4.22837517e-02,   2.36527089e-02],
           [  9.71194173e-01,   4.25821385e-02,   6.27649633e-03,
             -1.03503591e-03,  -6.45887169e-03,   3.10689120e-02],
           [  9.81230666e-01,   3.34950424e-02,   1.18021812e-02,
              5.30449887e-02,  -4.80272271e-02,   2.92044143e-02],
           [  9.95032250e-01,   2.44938545e-02,   1.54264661e-02,
              7.29442977e-02,  -4.75731864e-02,   1.91551142e-02],
           [  1.01123758e+00,   1.70027287e-02,   1.54140108e-02,
              8.56485921e-02,  -3.95921900e-02,  -6.58288796e-05],
           [  1.01905886e+00,   1.46986825e-02,   1.16959926e-02,
              4.13371471e-02,  -1.21773735e-02,  -1.96505159e-02],
           [  1.01652274e+00,   1.72916753e-02,   7.04597645e-03,
             -1.34039453e-02,   1.37045184e-02,  -2.45763243e-02],
           [  1.00610324e+00,   2.29818709e-02,   2.41737412e-03,
             -5.50692444e-02,   3.00738942e-02,  -2.44631481e-02],
           [  9.93130983e-01,   2.80334988e-02,  -2.27544287e-03,
             -6.85611631e-02,   2.66989289e-02,  -2.48025368e-02],
           [  9.87039295e-01,   3.07132896e-02,  -4.66945109e-03,
             -3.21958696e-02,   1.41632653e-02,  -1.26528433e-02],
           [  9.86810950e-01,   2.95354291e-02,  -1.62814930e-03,
             -1.20685098e-03,  -6.22524401e-03,   1.60739280e-02],
           [  9.89240935e-01,   2.51647708e-02,   4.15829403e-03,
              1.28429907e-02,  -2.30998606e-02,   3.05825863e-02],
           [  9.93894780e-01,   1.77819137e-02,   9.65397966e-03,
              2.45965618e-02,  -3.90199739e-02,   2.90458702e-02],
           [  9.99826005e-01,   1.24854448e-02,   1.34807379e-02,
              3.13477899e-02,  -2.79929674e-02,   2.02252335e-02],
           [  1.00382592e+00,   8.99695362e-03,   1.20599171e-02,
              2.11403804e-02,  -1.84374200e-02,  -7.50934094e-03],
           [  1.00494818e+00,   1.09923613e-02,   9.44014636e-03,
              5.93139214e-03,   1.05461552e-02,  -1.38460474e-02],
           [  1.00158335e+00,   1.35216304e-02,   6.07327499e-03,
             -1.77838520e-02,   1.33677270e-02,  -1.77946335e-02],
           [  9.97736746e-01,   1.78374119e-02,   2.18652467e-03,
             -2.03300988e-02,   2.28098262e-02,  -2.05423048e-02],
           [  9.94587975e-01,   2.08037773e-02,   1.24884472e-03,
             -1.66419272e-02,   1.56778741e-02,  -4.95583862e-03],
           [  9.92749121e-01,   2.03865425e-02,   1.53146216e-03,
             -9.71873539e-03,  -2.20517524e-03,   1.49369345e-03],
           [  9.97814282e-01,   2.35097105e-02,   1.52197536e-03,
              2.67704547e-02,   1.65066096e-02,  -5.01397264e-05],
           [  1.00128794e+00,   2.51305353e-02,   4.16689782e-03,
              1.83590392e-02,   8.56640510e-03,   1.39789797e-02],
           [  9.98522650e-01,   2.53125964e-02,   3.65406056e-03,
             -1.46151657e-02,   9.62231408e-04,  -2.71045440e-03],
           [  9.96380159e-01,   2.33351106e-02,   9.06070429e-04,
             -1.13235178e-02,  -1.04514344e-02,  -1.45237144e-02],
           [  9.95790759e-01,   2.50798261e-02,  -1.30377242e-03,
             -3.11510875e-03,   9.22119398e-03,  -1.16794912e-02],
           [  9.96752696e-01,   2.76247989e-02,  -1.47724872e-03,
              5.08404224e-03,   1.34507244e-02,  -9.16859261e-04],
           [  9.99411972e-01,   2.81431991e-02,   2.22104784e-04,
              1.40548425e-02,   2.73985566e-03,   8.98144605e-03],
           [  9.97610627e-01,   2.68242917e-02,   2.78999563e-03,
             -9.52049458e-03,  -6.97070731e-03,   1.35718513e-02],
           [  9.95017480e-01,   2.73656815e-02,   5.84971698e-03,
             -1.37053347e-02,   2.86136085e-03,   1.61712806e-02],
           [  9.91851625e-01,   2.73562412e-02,   9.68707158e-03,
             -1.67322173e-02,  -4.98937437e-05,   2.02812383e-02],
           [  9.89768797e-01,   2.57260482e-02,   1.24063628e-02,
             -1.10081895e-02,  -8.61591802e-03,   1.43720344e-02],
           [  9.92386001e-01,   2.55004688e-02,   1.36158864e-02,
              1.38324821e-02,  -1.19223569e-03,   6.39259084e-03],
           [  9.97389352e-01,   2.44857810e-02,   1.07594309e-02,
              2.64437789e-02,  -5.36284170e-03,  -1.50969775e-02],
           [  1.00067456e+00,   2.37900518e-02,   7.51474364e-03,
              1.73630222e-02,  -3.67707735e-03,  -1.71488646e-02],
           [  1.00182565e+00,   2.37405341e-02,   4.59751538e-03,
              6.08376261e-03,  -2.61711399e-04,  -1.54181742e-02],
           [  1.00154162e+00,   2.62636865e-02,   2.64237296e-03,
             -1.50118052e-03,   1.33353993e-02,  -1.03333451e-02],
           [  9.99297808e-01,   2.65809967e-02,   9.38309241e-04,
             -1.18590083e-02,   1.67705225e-03,  -9.00634058e-03],
           [  9.96642297e-01,   2.66223629e-02,   6.60961110e-04,
             -1.40349446e-02,   2.18629208e-04,  -1.46584408e-03],
           [  9.96608816e-01,   2.61747913e-02,   2.71076989e-03,
             -1.76955023e-04,  -2.36551178e-03,   1.08336770e-02],
           [  9.97735638e-01,   2.59470576e-02,   3.63639673e-03,
              5.95549586e-03,  -1.20362116e-03,   4.89213548e-03],
           [  9.98094332e-01,   2.45321077e-02,   4.09003881e-03,
              1.89577539e-03,  -7.47831257e-03,   2.39759529e-03],
           [  9.98846543e-01,   2.39384164e-02,   5.43503703e-03,
              3.97559791e-03,  -3.13778486e-03,   7.10860272e-03],
           [  9.97237086e-01,   2.51649348e-02,   3.75063390e-03,
             -8.50632841e-03,   6.48241126e-03,  -8.90243016e-03],
           [  9.97185317e-01,   2.45000579e-02,   3.08507611e-03,
             -2.73606806e-04,  -3.51401636e-03,  -3.51761500e-03]])



## Additionnal Parameters:

Here are some core parameter definitions for the training. 

The whole neural network's structure could be summarised by enumerating those parameters and the fact an LSTM is used. 


```python
# Input Data 

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 3 input parameters per timestep


# LSTM Neural Network's internal structure

n_hidden = args.n_hidden # Hidden layer num of features
n_classes = y_test.max() + 1 # 6 - total classes (should go up, or should go down)
n_easier_classes = y_easier_test.max() + 1 # also 6

# Training 

learning_rate = args.lr
lambda_loss_amount = 0.0015
training_iters = training_data_count * args.training_iterate_dataset_times
batch_size = args.batch_size
display_iter = 50  # To show test set accuracy during training

# Some debugging info

print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

```

    Some useful info to get an insight on dataset's shape and normalisation:
    (X shape, y shape, every X's mean, every X's standard deviation)
    (2947, 128, 6) (2947, 1) 0.166666503831 0.383176222723
    The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.


## Utility functions for training:


```python
def MAKE_RNN(_X, tdep, reuse):
    # Function returns a tensorflow GRU (RNN) artificial neural network from given parameters. 
    # input shape: (batch_size, n_steps, n_input)
    # common layers options
    clo = {"reuse": reuse, "activation": tf.nn.relu6}
    # Linear activation
    _X = tf.layers.dense(_X, n_hidden, name="input_dense", **clo)
    
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.unstack(_X, _X.shape[1], 1)
    
    gru_cell_1 = tf.contrib.rnn.GRUCell(n_hidden, **clo)
    rnn_cells = tf.contrib.rnn.MultiRNNCell([gru_cell_1], state_is_tuple=True)
    
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cells, _X, dtype=tf.float32)

    #rnn_last_output = outputs[-1]
    
    rnn_output = tf.stack(outputs, axis=1)
    before_split = tf.layers.dense(rnn_output, n_classes * 2, name="output_dense", **clo)
    result = tf.layers.dense(before_split, n_classes, name="normal_task_dense", reuse=reuse)
    
    
    pred = tf.transpose(result, [0, 2, 1])  # permute n_steps and batch_size
    pred = pred * tdep 
    pred = tf.reduce_sum(pred, 2)
    
    pred_easier = tf.layers.dense(outputs[-1], n_classes, name="easier_task_dense", reuse=reuse)
    
    return pred, pred_easier, result
    

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

def extract_batch_xy(x, y, step, batch_size):
    return extract_batch(x, step, batch_size), one_hot(extract_batch(y, step, batch_size))
```

## Let's get serious and build the neural network:


```python
from collections import namedtuple

class TfVars:
    def __init__(self):
        crossentropy = tf.nn.softmax_cross_entropy_with_logits
        self.OptClass = tf.train.RMSPropOptimizer
        if args.use_adam:
            self.OptClass = tf.train.AdamOptimizer
    
        # Graph input/output
        self.x = tf.placeholder(tf.float32, [None, n_steps, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_classes])
        self.lr = tf.placeholder(tf.float32, shape=[])
        
        self.tdep = tf.Variable(tf.constant(0.6 * (0.8 ** np.arange(0, n_steps, 1))[::-1], dtype=tf.float32), 
                                trainable=False, name="reversed_exp_weights")
        
        pred, pred_easier, self.normal_pred_log = MAKE_RNN(self.x, self.tdep, reuse=None) 
        
        self.LearnVariant = namedtuple('LearnVariant', ['cost', 'optimizer', "accuracy", "tfv", "pred"])
        
        def cost_opt_acc(pred):
            cost = tf.reduce_mean(crossentropy(labels=self.y, logits=pred))
            opt = self.OptClass(learning_rate=self.lr).minimize(cost)
            pred_c = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
            acc = tf.reduce_mean(tf.cast(pred_c, tf.float32))
            return self.LearnVariant(cost=cost, optimizer=opt, accuracy=acc, tfv = self, pred=pred)
        
        self.normal = cost_opt_acc(pred)
        self.easier = cost_opt_acc(pred_easier)
        
tfv = TfVars()
```

## Hooray, now train the neural network:
### First stage of training (easier):


```python
# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
init = tf.global_variables_initializer()
sess.run(init)
```


```python
start_learning_time = time()

from IPython.display import clear_output
#To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []


def learn(variant, bx, by):
    _, loss, acc = sess.run(
        [variant.optimizer, variant.cost, variant.accuracy],
        feed_dict={
            variant.tfv.x: bx, 
            variant.tfv.y: by,
            variant.tfv.lr: learning_rate
        }
    )
    return loss, acc

def estimate_on_test(variant, ans):
    return sess.run(
        [variant.cost, variant.accuracy], 
        feed_dict={
            variant.tfv.x: X_test,
            variant.tfv.y: one_hot(ans)
        }
    )

def init_default_plot():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    
    plt.axhline(y=1.0, c='r')
    plt.axhline(y=0.9, c='orange')
    plt.ylim(0, 2)
    
    ax.set_yticks(list(ax.get_yticks()) + [0.9])


    
# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
diter = 0
while step * batch_size <= training_iters:
    batch_xs, batch_ys = extract_batch_xy(X_train, y_easier_train, step, batch_size)

    # Fit training using batch data
    loss, acc = learn(tfv.easier, batch_xs, batch_ys)
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if is_notebook() and step % display_iter == 0:
        diter += 1
        # To not spam console, show training accuracy/loss in this "if"
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        test_loss, test_acc = estimate_on_test(tfv.easier, y_easier_test)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        clear_output(True)

        init_default_plot()
        
        ixs = np.array(range(diter))
        plt.plot(ixs, np.array(test_losses),     "b-", label="Test losses")
        plt.plot(ixs, np.array(test_accuracies), "g-", label="Test accuracies")

        plt.title("Training session's progress over iterations")
        plt.legend(loc='upper right', shadow=True)
        plt.ylabel('Training Progress (Loss or Accuracy values)')
        plt.xlabel('Training iteration')

        plt.show()
        
        print("Training iter #" + str(step * batch_size) + \
              "Accuracy = {}".format(acc))

        print("PERFORMANCE ON TEST SET: " + \
              "Accuracy = {}".format(test_acc))
        
        if (test_acc > 0.97):
            break
    
    if step % int(X_train.shape[0] / batch_size) == 0:
        X_train, y_train, y_easier_train = shuffle_all(X_train, y_train, y_easier_train)
        
    step += 1

print("Optimization Finished!")

# Accuracy for test data

test_loss, test_acc = estimate_on_test(tfv.easier, y_easier_test)
test_losses.append(test_loss)
test_accuracies.append(test_acc)

notify("INTERMEDIATE RESULT: " + \
      "Accuracy = {}".format(test_acc))
```


![png](LSTM-stage3-dev_files/LSTM-stage3-dev_25_0.png)


    Training iter #1120000Accuracy = 0.9775000810623169
    PERFORMANCE ON TEST SET: Accuracy = 0.9742110371589661
    Optimization Finished!
    INTERMEDIATE RESULT: Accuracy = 0.9742110371589661
    Notification SUCCESS


### Second stage (normal)


```python
test_losses = []
test_accuracies = []
test_losses_easier = []
test_accuracies_easier = []
train_losses = []
train_accuracies = []

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
diter = 0
training_iters2 = training_iters

while step * batch_size <= training_iters2:
    batch_xs, batch_ys = extract_batch_xy(X_train, y_train, step, batch_size)
    #poses = ((batch_ys.argmax(axis=1) == 3) | (batch_ys.argmax(axis=1) == 4) | (batch_ys.argmax(axis=1) == 5))
    #print(poses.shape, batch_xs.shape)
    #batch_xs = batch_xs[poses]
    #batch_ys = batch_ys[poses]
    # period of moving from easy task to normal task
    if step * batch_size <= training_iters2 / 2:
        _, batch_ys_easier = extract_batch_xy(X_train, y_easier_train, step, batch_size)
        learn(tfv.easier, batch_xs, batch_ys_easier)
    
    # Fit training using batch data
    loss, acc = learn(tfv.normal, batch_xs, batch_ys)
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if is_notebook() and step % display_iter == 0:
        diter += 1
        # To not spam console, show training accuracy/loss in this "if"
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        test_loss, test_acc = estimate_on_test(tfv.normal, y_test)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        test_loss_easier, test_acc_easier = estimate_on_test(tfv.easier, y_easier_test)
        test_losses_easier.append(test_loss_easier)
        test_accuracies_easier.append(test_acc_easier)
        
        clear_output(True)

        init_default_plot()
        
        ixs = np.array(range(diter))
        plt.plot(ixs, np.array(test_losses),     "b-", label="Test losses")
        plt.plot(ixs, np.array(test_accuracies), "g-", label="Test accuracies")
        
        plt.plot(ixs, np.array(test_accuracies_easier), "magenta", label="Test accuracies easier")

        plt.title("Training session's progress over iterations")
        plt.legend(loc='upper right', shadow=True)
        plt.ylabel('Training Progress (Loss or Accuracy values)')
        plt.xlabel('Training iteration')

        plt.show()
        
        print("Training iter #" + str(step * batch_size) +
              " Accuracy = {}".format(acc))

        print("PERFORMANCE ON TEST SET: " +
              "Accuracy = {}".format(test_acc) +
              ", Accuracy easier = {}".format(test_acc_easier))
    
    if step % int(X_train.shape[0] / batch_size) == 0:
        X_train, y_train, y_easier_train = shuffle_all(X_train, y_train, y_easier_train)
        
    step += 1

print("Optimization Finished!")

# Accuracy for test data

test_loss, test_acc = estimate_on_test(tfv.normal, y_test)
test_losses.append(test_loss)
test_accuracies.append(test_acc)

notify("FINAL RESULT: " + \
      "Accuracy = {}".format(test_acc))
```


![png](LSTM-stage3-dev_files/LSTM-stage3-dev_27_0.png)


    Training iter #2940000 Accuracy = 0.6775000095367432
    PERFORMANCE ON TEST SET: Accuracy = 0.649134635925293, Accuracy easier = 0.969799816608429
    Optimization Finished!
    FINAL RESULT: Accuracy = 0.6189345121383667
    Notification SUCCESS


## If notebook is run as .py file, then it is time to finish


```python
if not is_notebook():
    test_loss, test_acc = estimate_on_test(tfv.normal, y_test)
    
    argsdict = dict(args.__dict__)
    argsdict.update({
        "accuracy": float(test_acc),
        "time": float(time() - start_learning_time),
        "memory": sum(int(np.prod(var.shape)) for var in tf.trainable_variables()) * 4,
    })
    print("RESULT " + str(argsdict))
    exit(0)
```

## Let's see how class probabilities change in time window


```python
test_pred_log = sess.run(
    [tfv.normal_pred_log], 
    feed_dict={
        tfv.x: X_test
    }
)[0]

def watch_log(i):
    pd.DataFrame(test_pred_log[i]).plot(figsize=(10, 6))
    plt.title("#" + str(i) + "  true label" + str(y_test[i]))
    plt.show()
    
#print(np.where(y_test == 1)[0][:10])
    
for i in range(131, 135):
    watch_log(i)
```


![png](LSTM-stage3-dev_files/LSTM-stage3-dev_31_0.png)



![png](LSTM-stage3-dev_files/LSTM-stage3-dev_31_1.png)



![png](LSTM-stage3-dev_files/LSTM-stage3-dev_31_2.png)



![png](LSTM-stage3-dev_files/LSTM-stage3-dev_31_3.png)


## Let's plot training and test losses and accuracies:


```python
# (Inline plots: )
# %matplotlib inline

# font = {
#     'weight' : 'bold',
#     'size'   : 18
# }
# matplotlib.rc('font', **font)

# width = 12
# height = 12
# plt.figure(figsize=(width, height))

# indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
# plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
# plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

# indep_test_axis = range(len(test_losses))
# plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
# plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

# plt.axhline(y=1.0, c='r')
# plt.axhline(y=0.9, c='orange')

# plt.title("Training session's progress over iterations")
# plt.legend(loc='upper right', shadow=True)
# plt.ylabel('Training Progress (Loss or Accuracy values)')
# plt.xlabel('Training iteration')
# plt.ylim(0, 2)
# plt.show()
```

## Multi-class confusion matrix and metrics! (For UCI HAR test)


```python
# Results

one_hot_predictions = np.array(sess.run(
    [tfv.normal.pred], 
    feed_dict={
        tfv.x: X_test,
        tfv.y: one_hot(y_test)
    }
))
one_hot_predictions.shape = one_hot_predictions.shape[1:]

print(one_hot_predictions.shape)
predictions = one_hot_predictions.argmax(axis=1)

print("Testing Accuracy: {}%".format(100*test_acc))

print(predictions.shape)

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

    (2947, 6)
    Testing Accuracy: 61.89345121383667%
    (2947,)
    
    Precision: 56.13112636868496%
    Recall: 61.89345096708517%
    f1_score: 53.26915256057313%
    
    Confusion Matrix:
    [[484   8   4   0   0   0]
     [ 43 424   4   0   0   0]
     [  1  35 384   0   0   0]
     [  0   0   0   0 487   4]
     [  0   0   0   0 528   4]
     [  0   0   0   0 533   4]]
    
    Confusion matrix (normalised to % of total test data):
    [[ 16.42348099   0.2714625    0.13573125   0.           0.           0.        ]
     [  1.45911098  14.38751221   0.13573125   0.           0.           0.        ]
     [  0.03393281   1.18764842  13.0302       0.           0.           0.        ]
     [  0.           0.           0.           0.          16.52528
        0.13573125]
     [  0.           0.           0.           0.          17.91652679
        0.13573125]
     [  0.           0.           0.           0.          18.08618927
        0.13573125]]
    Note: training and testing data is not equally distributed amongst classes, 
    so it is normal that more than a 6th of the data is correctly classifier in the last category.


    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)



![png](LSTM-stage3-dev_files/LSTM-stage3-dev_35_2.png)


## Multi-class confusion matrix and metrics! (For MY test)


```python
# Results

one_hot_predictions = np.array(sess.run(
    [tfv.normal.pred], 
    feed_dict={
        tfv.x: X_my_test,
        tfv.y: one_hot(y_my_test)
    }
))

easier_accuracy = sess.run(
        [tfv.easier.accuracy], 
        feed_dict={
            tfv.x: X_my_test,
            tfv.y: one_hot(y_easier_my_test)
        }
    )[0]

one_hot_predictions.shape = one_hot_predictions.shape[1:]

print(one_hot_predictions.shape)
predictions = one_hot_predictions.argmax(axis=1)

print("Testing Accuracy: {}%".format(100*metrics.accuracy_score(y_my_test, predictions)))
print("Testing Accuracy EASIER: {}%".format(100*easier_accuracy))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_my_test, predictions)
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

    (454, 6)
    Testing Accuracy: 21.80616740088106%
    Testing Accuracy EASIER: 28.19383144378662%
    
    Confusion Matrix:
    [[ 92   4 300   0  28   2]
     [  3   2   2   0   0   0]
     [  3   1   2   0   0   0]
     [  0   1   0   0   3   0]
     [  0   0   0   0   3   0]
     [  0   0   0   0   8   0]]
    
    Confusion matrix (normalised to % of total test data):
    [[ 20.26431656   0.88105726  66.0792923    0.           6.16740084
        0.44052863]
     [  0.66079295   0.44052863   0.44052863   0.           0.           0.        ]
     [  0.66079295   0.22026432   0.44052863   0.           0.           0.        ]
     [  0.           0.22026432   0.           0.           0.66079295   0.        ]
     [  0.           0.           0.           0.           0.66079295   0.        ]
     [  0.           0.           0.           0.           1.76211452   0.        ]]
    Note: training and testing data is not equally distributed amongst classes, 
    so it is normal that more than a 6th of the data is correctly classifier in the last category.



![png](LSTM-stage3-dev_files/LSTM-stage3-dev_37_1.png)


## Run net on REAL DATA


```python
df = pd.read_csv('../arduino_data/parsed/D-1-0041', header=None)
#display(df.head())
mX = df.values[20000:20500, 1:4].astype(np.float)
#display(mX[:5])
mX = mX.reshape((1, len(mX), len(mX[0])))
#display(mX[0][:5])
mX = align_X(mX)
#display(mX[0][:5])
```

    LSTM-stage3-dev.ipynb:58: FutureWarning: pd.ewm_mean is deprecated for DataFrame and will be removed in a future version, replace with 
    	DataFrame.ewm(ignore_na=False,adjust=True,min_periods=0,halflife=4).mean()
      "outputs": [],



```python
class TfVarsMy:
    def __init__(self):
        crossentropy = tf.nn.softmax_cross_entropy_with_logits
        self.OptClass = tf.train.RMSPropOptimizer
        if args.use_adam:
            self.OptClass = tf.train.AdamOptimizer
    
        # Graph input/output
        self.x = tf.placeholder(tf.float32, [None, mX.shape[1], n_input], name="my_x_placeholder")
        self.y = tf.placeholder(tf.float32, [None, n_classes], name="my_y_placeholder")
        self.lr = tf.placeholder(tf.float32, shape=[], name="my_lr")
        
        self.tdep = tf.Variable(tf.constant(0.6 * (0.8 ** np.arange(0, mX.shape[1], 1))[::-1], dtype=tf.float32), 
                                trainable=False, name="my_reversed_exp_weights")
        
        pred, pred_easier, self.normal_pred_log = MAKE_RNN(self.x, self.tdep, reuse=True) 
        
        self.LearnVariant = namedtuple('LearnVariant', ["accuracy", "tfv", "pred"])
        
        def cost_opt_acc(pred):
            pred_c = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
            acc = tf.reduce_mean(tf.cast(pred_c, tf.float32))
            return self.LearnVariant(accuracy=acc, tfv = self, pred=pred)
        
        self.normal = cost_opt_acc(pred)
        self.easier = cost_opt_acc(pred_easier)
        
my_tfv = TfVarsMy()
```


```python

mX.shape
```




    (1, 500, 6)




```python
test_pred_log = sess.run(
    [my_tfv.normal_pred_log], 
    feed_dict={
        my_tfv.x: mX
    }
)[0]

pd.DataFrame(test_pred_log[0]).plot(figsize=(10, 6))
pd.DataFrame(mX[0, :, 0:3]).plot(figsize=(10, 6))
#plt.ylim(-3, 4)
#plt.xlim(50, 80)
plt.show()
```


![png](LSTM-stage3-dev_files/LSTM-stage3-dev_42_0.png)



![png](LSTM-stage3-dev_files/LSTM-stage3-dev_42_1.png)



```python
pd.DataFrame(test_pred_log[0, :500]).plot(figsize=(10, 6))
pd.DataFrame(mX[0, :500, 0:3]).plot(figsize=(10, 6))
#plt.ylim(-3, 4)
#plt.xlim(50, 80)
plt.show()
```


![png](LSTM-stage3-dev_files/LSTM-stage3-dev_43_0.png)



![png](LSTM-stage3-dev_files/LSTM-stage3-dev_43_1.png)


## Export net to cpp code


```python
tf.global_variables()
```




    [<tf.Variable 'reversed_exp_weights:0' shape=(128,) dtype=float32_ref>,
     <tf.Variable 'input_dense/kernel:0' shape=(6, 16) dtype=float32_ref>,
     <tf.Variable 'input_dense/bias:0' shape=(16,) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel:0' shape=(32, 32) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias:0' shape=(32,) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel:0' shape=(32, 16) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias:0' shape=(16,) dtype=float32_ref>,
     <tf.Variable 'output_dense/kernel:0' shape=(16, 12) dtype=float32_ref>,
     <tf.Variable 'output_dense/bias:0' shape=(12,) dtype=float32_ref>,
     <tf.Variable 'normal_task_dense/kernel:0' shape=(12, 6) dtype=float32_ref>,
     <tf.Variable 'normal_task_dense/bias:0' shape=(6,) dtype=float32_ref>,
     <tf.Variable 'easier_task_dense/kernel:0' shape=(16, 6) dtype=float32_ref>,
     <tf.Variable 'easier_task_dense/bias:0' shape=(6,) dtype=float32_ref>,
     <tf.Variable 'input_dense/kernel/RMSProp:0' shape=(6, 16) dtype=float32_ref>,
     <tf.Variable 'input_dense/kernel/RMSProp_1:0' shape=(6, 16) dtype=float32_ref>,
     <tf.Variable 'input_dense/bias/RMSProp:0' shape=(16,) dtype=float32_ref>,
     <tf.Variable 'input_dense/bias/RMSProp_1:0' shape=(16,) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel/RMSProp:0' shape=(32, 32) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel/RMSProp_1:0' shape=(32, 32) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias/RMSProp:0' shape=(32,) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias/RMSProp_1:0' shape=(32,) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel/RMSProp:0' shape=(32, 16) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel/RMSProp_1:0' shape=(32, 16) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias/RMSProp:0' shape=(16,) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias/RMSProp_1:0' shape=(16,) dtype=float32_ref>,
     <tf.Variable 'output_dense/kernel/RMSProp:0' shape=(16, 12) dtype=float32_ref>,
     <tf.Variable 'output_dense/kernel/RMSProp_1:0' shape=(16, 12) dtype=float32_ref>,
     <tf.Variable 'output_dense/bias/RMSProp:0' shape=(12,) dtype=float32_ref>,
     <tf.Variable 'output_dense/bias/RMSProp_1:0' shape=(12,) dtype=float32_ref>,
     <tf.Variable 'normal_task_dense/kernel/RMSProp:0' shape=(12, 6) dtype=float32_ref>,
     <tf.Variable 'normal_task_dense/kernel/RMSProp_1:0' shape=(12, 6) dtype=float32_ref>,
     <tf.Variable 'normal_task_dense/bias/RMSProp:0' shape=(6,) dtype=float32_ref>,
     <tf.Variable 'normal_task_dense/bias/RMSProp_1:0' shape=(6,) dtype=float32_ref>,
     <tf.Variable 'input_dense/kernel/RMSProp_2:0' shape=(6, 16) dtype=float32_ref>,
     <tf.Variable 'input_dense/kernel/RMSProp_3:0' shape=(6, 16) dtype=float32_ref>,
     <tf.Variable 'input_dense/bias/RMSProp_2:0' shape=(16,) dtype=float32_ref>,
     <tf.Variable 'input_dense/bias/RMSProp_3:0' shape=(16,) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel/RMSProp_2:0' shape=(32, 32) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel/RMSProp_3:0' shape=(32, 32) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias/RMSProp_2:0' shape=(32,) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias/RMSProp_3:0' shape=(32,) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel/RMSProp_2:0' shape=(32, 16) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel/RMSProp_3:0' shape=(32, 16) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias/RMSProp_2:0' shape=(16,) dtype=float32_ref>,
     <tf.Variable 'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias/RMSProp_3:0' shape=(16,) dtype=float32_ref>,
     <tf.Variable 'easier_task_dense/kernel/RMSProp:0' shape=(16, 6) dtype=float32_ref>,
     <tf.Variable 'easier_task_dense/kernel/RMSProp_1:0' shape=(16, 6) dtype=float32_ref>,
     <tf.Variable 'easier_task_dense/bias/RMSProp:0' shape=(6,) dtype=float32_ref>,
     <tf.Variable 'easier_task_dense/bias/RMSProp_1:0' shape=(6,) dtype=float32_ref>,
     <tf.Variable 'my_reversed_exp_weights:0' shape=(10000,) dtype=float32_ref>]




```python
def to_cpp_initializer(array, depth=0):
    if type(array) == np.ndarray:
        sep = "," if depth != 0 or len(array.shape) == 1 else ",\n"
        return "{" + sep.join(to_cpp_initializer(x) for x in array) + "}"
    else:
        return "%.8ff" % array

cpp_class_name = 'TGruHarNet';

def to_cpp_decl(array, name):
    TRet = namedtuple("CppTensorDecl", ["declaration", "definition"])
    shape_s = "".join('[' + str(dim) + ']' for dim in array.shape) 
    return TRet(
        declaration="static const float %s%s PROGMEM;\n" % (name, shape_s),
        definition="const float %s::%s%s PROGMEM = \n%s;\n" % 
            (cpp_class_name, name, shape_s, to_cpp_initializer(array))
    )

def make_cpp_decls(tf2cpp_name):
    result = []
    for s in tf.global_variables():
        if s.name in tf2cpp_name:
            result.append(to_cpp_decl(s.eval(), tf2cpp_name[s.name]))
    return result

decls = make_cpp_decls({
    'input_dense/kernel:0': 'InputDenseKernel',
    'input_dense/bias:0': 'InputDenseBias',
    'rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel:0': 'GruCellGatesKernel',
    'rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias:0': 'GruCellGatesBias',
    'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel:0': 'GruCellCandidateKernel',
    'rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias:0': 'GruCellCandidateBias',
    'output_dense/kernel:0': 'OutputDenseKernel',
    'output_dense/bias:0': 'OutputDenseBias',
    'normal_task_dense/kernel:0': 'ToClassesDenseKernel',
    'normal_task_dense/bias:0': 'ToClassesDenseBias',
})

with open('arduino/gru_tensors_declarations.h', 'w') as f:
    f.write('static const int NHidden = %d;\n' % n_hidden)
    f.write('static const int NInput = %d;\n' % n_input)
    f.write('static const float AccelSmoothLambda = %0.8ff;\n' % (0.5 ** (1.0 / args.ewma_halflife)))
    f.write(''.join(decl.declaration for decl in decls))
with open('arduino/gru_tensors_definitions.h', 'w') as f:
    f.write(''.join(decl.definition for decl in decls))
#print(to_cpp_decl(s.eval(), "InputDense").declaration)
#print(to_cpp_decl(s.eval(), "InputDense").definition)
```


```python
#debug
raise KeyboardInterrupt("do not automatically run after it")
sess.close()
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-32-a25d321b1025> in <module>()
          1 #debug
    ----> 2 raise KeyboardInterrupt("do not automatically run after it")
          3 sess.close()


    KeyboardInterrupt: do not automatically run after it


## Conclusion

Outstandingly, **the final accuracy is of about 90%** (depends of launch)!

This means that the neural networks is almost always able to correctly identify the movement type! Remember, the phone is attached on the waist and each series to classify has just a 128 sample window of two internal sensors (a.k.a. 2.56 seconds at 50 FPS), so those predictions are extremely accurate.

I specially did not expect such good results for guessing between "SITTING" and "STANDING". Those are seemingly almost the same thing from the point of view of a device placed at waist level according to how the dataset was gathered. Thought, it is still possible to see a little cluster on the matrix between those classes, which drifts away from the identity. This is great.

It is also possible to see that there was a slight difficulty in doing the difference between "WALKING", "WALKING_UPSTAIRS" and "WALKING_DOWNSTAIRS". Obviously, those activities are quite similar in terms of movements. 


```python
get_ipython().system("jupyter nbconvert --to markdown " + nb_name)
get_ipython().system("jupyter nbconvert --to python " + nb_name)
display(nb_name)
pyname = nb_name[:-6] + ".py"
display(pyname)
```

    [NbConvertApp] Converting notebook LSTM-stage3-dev.ipynb to markdown
    [NbConvertApp] Support files will be in LSTM-stage3-dev_files/
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Making directory LSTM-stage3-dev_files
    [NbConvertApp] Writing 80223 bytes to LSTM-stage3-dev.md
    [NbConvertApp] Converting notebook LSTM-stage3-dev.ipynb to python
    [NbConvertApp] Writing 52289 bytes to LSTM-stage3-dev.py



    'LSTM-stage3-dev.ipynb'



    'LSTM-stage3-dev.py'



```python
raise KeyboardInterrupt("do not automatically run after it")
```

## Parameters selection


```python
import pandas as pd
def dict_protocol_to_dataframe(protocol):
    return pd.read_json('[' + protocol.replace("'", '"').replace('\n', ',')
                                      .replace('True', 'true').replace('False', 'false')+ ']')

dict_protocol_to_dataframe("""\
{'time': 548.2270517349243, 'n_hidden': 1, 'lr': 0.0025, 'accuracy': 0.6878181099891663, 'memory': 532, 'batch_size': 2000, 'training_iterate_dataset_times': 200}
{'lr': 0.0025, 'training_iterate_dataset_times': 200, 'n_hidden': 2, 'batch_size': 2000, 'accuracy': 0.7770613431930542, 'memory': 728, 'time': 566.6135125160217}\
""")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>batch_size</th>
      <th>lr</th>
      <th>memory</th>
      <th>n_hidden</th>
      <th>time</th>
      <th>training_iterate_dataset_times</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.687818</td>
      <td>2000</td>
      <td>0.0025</td>
      <td>532</td>
      <td>1</td>
      <td>548.227052</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.777061</td>
      <td>2000</td>
      <td>0.0025</td>
      <td>728</td>
      <td>2</td>
      <td>566.613513</td>
      <td>200</td>
    </tr>
  </tbody>
</table>
</div>




```python
#bench hhidden
bench_cmd = """
for i in {1..25}; do \
    python3 %s \
    --nhidden $i --training_iterate_dataset_times 200\
    2> full_bench.err \
    | tee -a full_bench.log \
    | grep -e "^RESULT" | cut -c 8- \
    | tee -a bench.log; \
done
""" % pyname
get_ipython().system("bash -c '" + bench_cmd + "'")
```


```python
bench_nhidden_protocol = """\
{'n_hidden': 1, 'memory': 508, 'accuracy': 0.6627078056335449, 'batch_size': 800, 'use_rmsprop': False, 'use_adam': False, 'training_iterate_dataset_times': 200, 'lr': 0.0025, 'time': 772.0379655361176}
{'use_rmsprop': False, 'training_iterate_dataset_times': 200, 'use_adam': False, 'accuracy': 0.5758398175239563, 'n_hidden': 2, 'time': 720.3474073410034, 'batch_size': 800, 'memory': 680, 'lr': 0.0025}
{'memory': 900, 'accuracy': 0.6057007312774658, 'n_hidden': 3, 'batch_size': 800, 'use_rmsprop': False, 'lr': 0.0025, 'training_iterate_dataset_times': 200, 'use_adam': False, 'time': 662.5361175537109}
{'accuracy': 0.6674584150314331, 'time': 655.1994025707245, 'training_iterate_dataset_times': 200, 'memory': 1168, 'use_adam': False, 'lr': 0.0025, 'batch_size': 800, 'use_rmsprop': False, 'n_hidden': 4}
{'batch_size': 800, 'memory': 1484, 'accuracy': 0.5341024398803711, 'n_hidden': 5, 'training_iterate_dataset_times': 200, 'lr': 0.0025, 'time': 678.890563249588, 'use_rmsprop': False, 'use_adam': False}
{'n_hidden': 6, 'use_adam': False, 'accuracy': 0.6016287207603455, 'memory': 1848, 'use_rmsprop': False, 'lr': 0.0025, 'training_iterate_dataset_times': 200, 'batch_size': 800, 'time': 674.8454222679138}
{'batch_size': 800, 'use_adam': False, 'use_rmsprop': False, 'lr': 0.0025, 'time': 669.8843955993652, 'accuracy': 0.7003732323646545, 'training_iterate_dataset_times': 200, 'n_hidden': 7, 'memory': 2260}
{'accuracy': 0.751272439956665, 'memory': 2720, 'use_adam': False, 'training_iterate_dataset_times': 200, 'time': 660.9931690692902, 'n_hidden': 8, 'use_rmsprop': False, 'batch_size': 800, 'lr': 0.0025}
{'accuracy': 0.5656599998474121, 'time': 707.3295772075653, 'lr': 0.0025, 'batch_size': 800, 'training_iterate_dataset_times': 200, 'use_adam': False, 'n_hidden': 9, 'memory': 3228, 'use_rmsprop': False}
{'batch_size': 800, 'training_iterate_dataset_times': 200, 'n_hidden': 10, 'time': 715.7435595989227, 'lr': 0.0025, 'memory': 3784, 'accuracy': 0.6973192691802979, 'use_rmsprop': False, 'use_adam': False}
{'training_iterate_dataset_times': 200, 'lr': 0.0025, 'time': 732.3185958862305, 'use_rmsprop': False, 'n_hidden': 11, 'accuracy': 0.6430268287658691, 'use_adam': False, 'memory': 4388, 'batch_size': 800}
{'batch_size': 800, 'accuracy': 0.7360026836395264, 'n_hidden': 12, 'use_adam': False, 'lr': 0.0025, 'use_rmsprop': False, 'memory': 5040, 'time': 742.9097394943237, 'training_iterate_dataset_times': 200}
{'use_adam': False, 'memory': 5740, 'time': 760.5904695987701, 'training_iterate_dataset_times': 200, 'batch_size': 800, 'lr': 0.0025, 'use_rmsprop': False, 'accuracy': 0.6942652463912964, 'n_hidden': 13}
{'use_rmsprop': False, 'training_iterate_dataset_times': 200, 'memory': 6488, 'lr': 0.0025, 'batch_size': 800, 'time': 776.510192155838, 'n_hidden': 14, 'accuracy': 0.8008143901824951, 'use_adam': False}
{'lr': 0.0025, 'n_hidden': 15, 'time': 790.9602189064026, 'memory': 7284, 'batch_size': 800, 'training_iterate_dataset_times': 200, 'use_adam': False, 'accuracy': 0.7488970756530762, 'use_rmsprop': False}
{'lr': 0.0025, 'n_hidden': 16, 'use_rmsprop': False, 'memory': 8128, 'use_adam': False, 'batch_size': 800, 'time': 796.3274857997894, 'training_iterate_dataset_times': 200, 'accuracy': 0.7492364645004272}
{'use_adam': False, 'time': 979.4974310398102, 'training_iterate_dataset_times': 200, 'n_hidden': 17, 'accuracy': 0.8154054880142212, 'memory': 9020, 'batch_size': 800, 'use_rmsprop': False, 'lr': 0.0025}
{'use_adam': False, 'use_rmsprop': False, 'n_hidden': 18, 'memory': 9960, 'batch_size': 800, 'accuracy': 0.819477379322052, 'lr': 0.0025, 'training_iterate_dataset_times': 200, 'time': 998.5558381080627}
{'memory': 10948, 'n_hidden': 19, 'use_adam': False, 'training_iterate_dataset_times': 200, 'lr': 0.0025, 'accuracy': 0.7227687835693359, 'time': 1023.3071684837341, 'batch_size': 800, 'use_rmsprop': False}
{'accuracy': 0.6416694521903992, 'memory': 11984, 'lr': 0.0025, 'use_rmsprop': False, 'time': 1031.0319290161133, 'batch_size': 800, 'training_iterate_dataset_times': 200, 'use_adam': False, 'n_hidden': 20}
{'lr': 0.0025, 'batch_size': 800, 'memory': 13068, 'time': 1062.1775381565094, 'training_iterate_dataset_times': 200, 'accuracy': 0.6420088410377502, 'use_rmsprop': False, 'n_hidden': 21, 'use_adam': False}
{'use_adam': False, 'lr': 0.0025, 'training_iterate_dataset_times': 200, 'time': 1144.7650830745697, 'memory': 14200, 'n_hidden': 22, 'batch_size': 800, 'use_rmsprop': False, 'accuracy': 0.6321682333946228}
{'training_iterate_dataset_times': 200, 'use_adam': False, 'lr': 0.0025, 'memory': 15380, 'time': 1155.4969289302826, 'n_hidden': 23, 'accuracy': 0.7919918298721313, 'batch_size': 800, 'use_rmsprop': False}
{'n_hidden': 24, 'use_rmsprop': False, 'memory': 16608, 'batch_size': 800, 'training_iterate_dataset_times': 200, 'accuracy': 0.634882926940918, 'time': 1172.9154741764069, 'lr': 0.0025, 'use_adam': False}
{'time': 1240.9481527805328, 'use_rmsprop': False, 'memory': 17884, 'accuracy': 0.831693172454834, 'batch_size': 800, 'training_iterate_dataset_times': 200, 'n_hidden': 25, 'lr': 0.0025, 'use_adam': False}"""

data = dict_protocol_to_dataframe(bench_nhidden_protocol)
data = data.set_index('n_hidden')

data[["accuracy", "memory", "time"]].plot(subplots=True, figsize=(10,15), title="Selecting nhidden")
plt.show()
```


```python
#bench hhidden
bench_cmd = """
for i in %s; do \
    python3 %s \
    --lr $i --training_iterate_dataset_times 200\
    2> full_bench.err \
    | tee -a full_bench.log \
    | grep -e "^RESULT" | cut -c 8- \
    | tee -a bench.log; \
done
""" % (" ".join(map(str, np.arange(0.0005, 0.005, 0.0002))), pyname)
get_ipython().system("bash -c '" + bench_cmd + "'")
```


```python
bench_lr_protocol = """\
{'training_iterate_dataset_times': 200, 'time': 794.8192670345306, 'n_hidden': 16, 'use_rmsprop': False, 'memory': 8128, 'lr': 0.0005, 'batch_size': 800, 'use_adam': False, 'accuracy': 0.7037665247917175}
{'n_hidden': 16, 'use_rmsprop': False, 'time': 791.1197946071625, 'lr': 0.0007, 'batch_size': 800, 'training_iterate_dataset_times': 200, 'accuracy': 0.7231082320213318, 'memory': 8128, 'use_adam': False}
{'lr': 0.0009, 'batch_size': 800, 'use_adam': False, 'training_iterate_dataset_times': 200, 'accuracy': 0.809297502040863, 'memory': 8128, 'use_rmsprop': False, 'n_hidden': 16, 'time': 795.8083786964417}
{'batch_size': 800, 'memory': 8128, 'time': 794.4841299057007, 'use_rmsprop': False, 'training_iterate_dataset_times': 200, 'accuracy': 0.7885985374450684, 'n_hidden': 16, 'use_adam': False, 'lr': 0.0011}
{'memory': 8128, 'use_rmsprop': False, 'lr': 0.0013, 'batch_size': 800, 'use_adam': False, 'n_hidden': 16, 'training_iterate_dataset_times': 200, 'accuracy': 0.7862231731414795, 'time': 792.6490998268127}
{'batch_size': 800, 'memory': 8128, 'use_rmsprop': False, 'time': 808.8380537033081, 'training_iterate_dataset_times': 200, 'n_hidden': 16, 'accuracy': 0.7356632947921753, 'lr': 0.0015, 'use_adam': False}
{'memory': 8128, 'use_rmsprop': False, 'training_iterate_dataset_times': 200, 'lr': 0.0017, 'time': 816.4750707149506, 'use_adam': False, 'accuracy': 0.5914489030838013, 'batch_size': 800, 'n_hidden': 16}
{'use_adam': False, 'accuracy': 0.6627078056335449, 'use_rmsprop': False, 'time': 821.1141517162323, 'training_iterate_dataset_times': 200, 'batch_size': 800, 'memory': 8128, 'n_hidden': 16, 'lr': 0.0019}
{'n_hidden': 16, 'use_rmsprop': False, 'lr': 0.0021, 'time': 810.5998747348785, 'accuracy': 0.7943670749664307, 'memory': 8128, 'use_adam': False, 'training_iterate_dataset_times': 200, 'batch_size': 800}
{'time': 816.5132377147675, 'use_adam': False, 'use_rmsprop': False, 'memory': 8128, 'training_iterate_dataset_times': 200, 'accuracy': 0.8266032934188843, 'n_hidden': 16, 'lr': 0.0023, 'batch_size': 800}
{'lr': 0.0025, 'n_hidden': 16, 'use_adam': False, 'memory': 8128, 'accuracy': 0.7315914034843445, 'use_rmsprop': False, 'batch_size': 800, 'training_iterate_dataset_times': 200, 'time': 810.8860170841217}
{'training_iterate_dataset_times': 200, 'batch_size': 800, 'time': 816.7522842884064, 'n_hidden': 16, 'use_rmsprop': False, 'lr': 0.0027, 'memory': 8128, 'accuracy': 0.6118085980415344, 'use_adam': False}
{'use_adam': False, 'memory': 8128, 'n_hidden': 16, 'use_rmsprop': False, 'accuracy': 0.8361043930053711, 'time': 815.2302830219269, 'training_iterate_dataset_times': 200, 'batch_size': 800, 'lr': 0.0029}
{'n_hidden': 16, 'time': 815.0186021327972, 'lr': 0.0031, 'training_iterate_dataset_times': 200, 'memory': 8128, 'use_rmsprop': False, 'batch_size': 800, 'use_adam': False, 'accuracy': 0.8364437818527222}
{'training_iterate_dataset_times': 200, 'lr': 0.0033, 'time': 814.5488564968109, 'accuracy': 0.6331862807273865, 'use_rmsprop': False, 'batch_size': 800, 'memory': 8128, 'n_hidden': 16, 'use_adam': False}
{'lr': 0.0035, 'batch_size': 800, 'accuracy': 0.6196131110191345, 'use_rmsprop': False, 'time': 815.6647927761078, 'training_iterate_dataset_times': 200, 'memory': 8128, 'use_adam': False, 'n_hidden': 16}
{'lr': 0.0037, 'n_hidden': 16, 'accuracy': 0.6630471348762512, 'memory': 8128, 'batch_size': 800, 'training_iterate_dataset_times': 200, 'time': 812.8441379070282, 'use_adam': False, 'use_rmsprop': False}
{'use_rmsprop': False, 'n_hidden': 16, 'time': 815.5419595241547, 'batch_size': 800, 'training_iterate_dataset_times': 200, 'lr': 0.0039, 'use_adam': False, 'memory': 8128, 'accuracy': 0.8381404280662537}
{'accuracy': 0.7933490872383118, 'n_hidden': 16, 'use_adam': False, 'use_rmsprop': False, 'lr': 0.0041, 'memory': 8128, 'batch_size': 800, 'training_iterate_dataset_times': 200, 'time': 813.244401216507}
{'lr': 0.0043, 'accuracy': 0.8523921966552734, 'batch_size': 800, 'use_rmsprop': False, 'training_iterate_dataset_times': 200, 'time': 810.2618834972382, 'n_hidden': 16, 'use_adam': False, 'memory': 8128}
{'memory': 8128, 'batch_size': 800, 'use_rmsprop': False, 'time': 814.5698442459106, 'use_adam': False, 'accuracy': 0.763488233089447, 'lr': 0.0045, 'training_iterate_dataset_times': 200, 'n_hidden': 16}
{'training_iterate_dataset_times': 200, 'use_adam': False, 'n_hidden': 16, 'use_rmsprop': False, 'memory': 8128, 'lr': 0.0047, 'accuracy': 0.8483203053474426, 'time': 815.120082616806, 'batch_size': 800}
{'n_hidden': 16, 'batch_size': 800, 'lr': 0.0049, 'memory': 8128, 'use_rmsprop': False, 'accuracy': 0.742110550403595, 'time': 814.3666989803314, 'training_iterate_dataset_times': 200, 'use_adam': False}"""

data = dict_protocol_to_dataframe(bench_lr_protocol)
data = data.set_index('lr')

data[["accuracy", "memory", "time"]].plot(subplots=True, figsize=(10,15), title="Selecting learning rate")
plt.show()
```


```python
#bench bsize
bench_cmd = """
for i in %s; do \
    python3 %s \
    --bsize $i --training_iterate_dataset_times 200\
    2> full_bench.err \
    | tee -a full_bench.log \
    | grep -e "^RESULT" | cut -c 8- \
    | tee -a bench.log; \
done
""" % (" ".join(map(str, np.arange(400, 4000, 200))), pyname)
print(bench_cmd)
get_ipython().system("bash -c '" + bench_cmd + "'")
```


```python
bench_bsize_protocol = """\
{'use_rmsprop': False, 'batch_size': 400, 'time': 1041.1525554656982, 'accuracy': 0.8276212215423584, 'n_hidden': 16, 'training_iterate_dataset_times': 200, 'memory': 8128, 'lr': 0.0025, 'use_adam': False}
{'accuracy': 0.7380385994911194, 'time': 897.4389636516571, 'lr': 0.0025, 'memory': 8128, 'training_iterate_dataset_times': 200, 'use_rmsprop': False, 'batch_size': 600, 'n_hidden': 16, 'use_adam': False}
{'accuracy': 0.6677977442741394, 'time': 811.6754739284515, 'training_iterate_dataset_times': 200, 'batch_size': 800, 'use_rmsprop': False, 'memory': 8128, 'n_hidden': 16, 'use_adam': False, 'lr': 0.0025}
{'use_adam': False, 'lr': 0.0025, 'use_rmsprop': False, 'time': 878.8416509628296, 'batch_size': 1000, 'training_iterate_dataset_times': 200, 'memory': 8128, 'n_hidden': 16, 'accuracy': 0.621309757232666}
{'memory': 8128, 'batch_size': 1200, 'training_iterate_dataset_times': 200, 'n_hidden': 16, 'accuracy': 0.7295553684234619, 'use_adam': False, 'time': 910.5316579341888, 'use_rmsprop': False, 'lr': 0.0025}
{'use_adam': False, 'time': 899.1035013198853, 'batch_size': 1400, 'lr': 0.0025, 'use_rmsprop': False, 'memory': 8128, 'training_iterate_dataset_times': 200, 'accuracy': 0.6094332933425903, 'n_hidden': 16}
{'lr': 0.0025, 'n_hidden': 16, 'training_iterate_dataset_times': 200, 'use_adam': False, 'batch_size': 1600, 'memory': 8128, 'accuracy': 0.6158804893493652, 'use_rmsprop': False, 'time': 902.1990230083466}
{'n_hidden': 16, 'training_iterate_dataset_times': 200, 'use_adam': False, 'use_rmsprop': False, 'memory': 8128, 'time': 898.3028283119202, 'batch_size': 1800, 'accuracy': 0.6420087814331055, 'lr': 0.0025}
{'n_hidden': 16, 'training_iterate_dataset_times': 200, 'lr': 0.0025, 'time': 896.1048786640167, 'batch_size': 2000, 'use_adam': False, 'use_rmsprop': False, 'accuracy': 0.5846623182296753, 'memory': 8128}
{'time': 854.5555908679962, 'n_hidden': 16, 'accuracy': 0.6420087218284607, 'batch_size': 2200, 'use_adam': False, 'training_iterate_dataset_times': 200, 'use_rmsprop': False, 'memory': 8128, 'lr': 0.0025}
{'time': 842.2896134853363, 'training_iterate_dataset_times': 200, 'accuracy': 0.5222259163856506, 'lr': 0.0025, 'batch_size': 2400, 'n_hidden': 16, 'use_adam': False, 'memory': 8128, 'use_rmsprop': False}
{'accuracy': 0.647438108921051, 'use_adam': False, 'n_hidden': 16, 'memory': 8128, 'lr': 0.0025, 'training_iterate_dataset_times': 200, 'batch_size': 2600, 'use_rmsprop': False, 'time': 850.2778007984161}
{'training_iterate_dataset_times': 200, 'memory': 8128, 'use_rmsprop': False, 'lr': 0.0025, 'batch_size': 2800, 'time': 822.6956253051758, 'n_hidden': 16, 'accuracy': 0.6915506720542908, 'use_adam': False}
{'n_hidden': 16, 'accuracy': 0.5951814651489258, 'memory': 8128, 'time': 826.1692667007446, 'lr': 0.0025, 'use_adam': False, 'training_iterate_dataset_times': 200, 'batch_size': 3000, 'use_rmsprop': False}
{'accuracy': 0.5945028066635132, 'training_iterate_dataset_times': 200, 'time': 817.3396935462952, 'use_rmsprop': False, 'memory': 8128, 'use_adam': False, 'lr': 0.0025, 'n_hidden': 16, 'batch_size': 3200}
{'memory': 8128, 'batch_size': 3400, 'training_iterate_dataset_times': 200, 'use_rmsprop': False, 'time': 824.8832528591156, 'use_adam': False, 'lr': 0.0025, 'accuracy': 0.6779775619506836, 'n_hidden': 16}
{'time': 819.2803962230682, 'n_hidden': 16, 'training_iterate_dataset_times': 200, 'use_adam': False, 'memory': 8128, 'lr': 0.0025, 'accuracy': 0.5592126846313477, 'batch_size': 3600, 'use_rmsprop': False}
{'batch_size': 3800, 'use_adam': False, 'memory': 8128, 'training_iterate_dataset_times': 200, 'time': 831.1241354942322, 'use_rmsprop': False, 'n_hidden': 16, 'lr': 0.0025, 'accuracy': 0.6311503052711487}"""
data = dict_protocol_to_dataframe(bench_bsize_protocol)
data = data.set_index('batch_size')

data[["accuracy", "memory", "time"]].plot(subplots=True, figsize=(10,15), title="Selecting bsize")
plt.show()
```


```python
#bench ewma
bench_cmd = """
for i in %s; do \
    python3 %s \
    --ewma $i --training_iterate_dataset_times 200\
    2> full_bench.err \
    | tee -a full_bench.log \
    | grep -e "^RESULT" | cut -c 8- \
    | tee -a bench.log; \
done
""" % (" ".join(map(str, np.arange(0, 10, 1))), pyname)
print(bench_cmd)
get_ipython().system("bash -c '" + bench_cmd + "'")
```

    
    for i in 0 1 2 3 4 5 6 7 8 9; do     python3 LSTM-stage3-dev.py     --ewma $i --training_iterate_dataset_times 200    2> full_bench.err     | tee -a full_bench.log     | grep -e "^RESULT" | cut -c 8-     | tee -a bench.log; done
    



```python
bench_ewma_protocol = """\
{'use_rmsprop': False, 'time': 1040.7033743858337, 'lr': 0.0025, 'memory': 8320, 'ewma_halflife': 1, 'batch_size': 400, 'use_adam': False, 'training_iterate_dataset_times': 200, 'accuracy': 0.8374617695808411, 'n_hidden': 16}
{'accuracy': 0.8707159161567688, 'lr': 0.0025, 'training_iterate_dataset_times': 200, 'batch_size': 400, 'time': 1038.3529443740845, 'memory': 8320, 'n_hidden': 16, 'ewma_halflife': 2, 'use_rmsprop': False, 'use_adam': False}
{'use_adam': False, 'accuracy': 0.8883609771728516, 'n_hidden': 16, 'lr': 0.0025, 'use_rmsprop': False, 'batch_size': 400, 'time': 1040.9371156692505, 'training_iterate_dataset_times': 200, 'memory': 8320, 'ewma_halflife': 3}
{'use_rmsprop': False, 'time': 1041.6433839797974, 'training_iterate_dataset_times': 200, 'accuracy': 0.8914148807525635, 'memory': 8320, 'use_adam': False, 'batch_size': 400, 'n_hidden': 16, 'lr': 0.0025, 'ewma_halflife': 4}
{'use_rmsprop': False, 'batch_size': 400, 'memory': 8320, 'accuracy': 0.8849676847457886, 'use_adam': False, 'lr': 0.0025, 'time': 1012.2724695205688, 'ewma_halflife': 5, 'n_hidden': 16, 'training_iterate_dataset_times': 200}
{'batch_size': 400, 'training_iterate_dataset_times': 200, 'n_hidden': 16, 'time': 1012.6590631008148, 'use_rmsprop': False, 'accuracy': 0.880895733833313, 'lr': 0.0025, 'memory': 8320, 'use_adam': False, 'ewma_halflife': 6}
{'time': 1010.5400428771973, 'n_hidden': 16, 'batch_size': 400, 'memory': 8320, 'training_iterate_dataset_times': 200, 'use_adam': False, 'ewma_halflife': 7, 'accuracy': 0.8642686605453491, 'use_rmsprop': False, 'lr': 0.0025}"""
data = dict_protocol_to_dataframe(bench_ewma_protocol)
data = data.set_index('ewma_halflife')

data[["accuracy", "memory", "time"]].plot(subplots=True, figsize=(10,15), title="Selecting bsize")
plt.show()
```


![png](LSTM-stage3-dev_files/LSTM-stage3-dev_60_0.png)

