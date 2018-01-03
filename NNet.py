import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.io as sio
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import operator

file=sio.loadmat('fdata.mat')
dataset=file['fdata']

print(type(dataset))
X= dataset[:,[0,1,2,3,5]]  #[0,1,2,3,4,5]

Y=dataset[:,[6,7]]

X_train = X[:217,:]
X_test = X[217:,:]


Y_train = Y[:217,:]
Y_test = Y[217:,:]
X_scaled = preprocessing.scale(X_train)
X_scaled_t=preprocessing.scale(X_test)

# Parameters
lrate = 0.1
n_epochs =100 #running for 10,20, 50, 100
batch_size = 100
display_step = 10

hnodes_1 = 7 #number of hidden layer nodes
hnodes_2 = 7
N1 = X_train.shape[1]
n_classes=2;

#set up dimensions for data matrix, weights and biases
#we have two layers, so weights and biases for both

X = tf.placeholder('float', [None, N1])
Y = tf.placeholder('float', [None, n_classes])


weights = {
'h1': tf.Variable(tf.random_normal([N1, hnodes_1])),
'h2': tf.Variable(tf.random_normal([hnodes_1, hnodes_2])),
'out': tf.Variable(tf.random_normal([hnodes_2, n_classes]))
        }
biases = {
'b1': tf.Variable(tf.random_normal([hnodes_1])),
'b2': tf.Variable(tf.random_normal([hnodes_2])),
'out': tf.Variable(tf.random_normal([n_classes]))
        }

#function for batch creation
def create_batch(batch_size, x,y):
    n_samples = x.shape[0]
    assert n_samples == y.shape[0]

    index = np.random.choice(range(n_samples),batch_size, replace=1)
    batch_x = x[index]
    batch_y = y[index]

    return batch_x, batch_y

#function for the NNET framework
def MLP(x,weights,biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.tanh(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.tanh(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

#model framework

logits = MLP(X,weights,biases)
prediction = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(cost)


init = tf.global_variables_initializer()

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as SESS:
    SESS.run(init)
    for epoch in range(n_epochs):

        avg_cost = 0
        n_batches = int(X_train.shape[0] / batch_size)
        for i in range(n_batches):
            batch_x, batch_y = create_batch(batch_size, X_train, Y_train)
            SESS.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
            a, c = SESS.run([accuracy,cost], feed_dict={X: batch_x, Y: batch_y})
            avg_cost += c / n_batches
            if epoch % display_step == 0 or epoch==1:
                print ('Epoch:{}, cost={}, Error = {}'  .format(epoch + 1, avg_cost,1- a))
                print('Optimization Finished!')
    print("Testing Error:", \
           1-( SESS.run(accuracy, feed_dict={X: X_test,
                                          Y: Y_test})))