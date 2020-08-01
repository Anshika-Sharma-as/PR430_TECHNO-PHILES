import tensorflow as tf


get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip     -O /tmp/cats_and_dogs_filtered.zip')


import os
import zipfile

local_zip = '/tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()



import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = r"/tmp/cats_and_dogs_filtered/train"

CATEGORIES = ["dogs", "cats"]

for category in CATEGORIES:  
    path = os.path.join(DATADIR,category) 
    for img in os.listdir(path):  
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
        plt.imshow(img_array, cmap='gray')  
        plt.show() 

        break  
    break 

print(img_array)




print(img_array.shape)


IMG_SIZE = 90

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()


new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  

        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                training_data.append([new_array, class_num])  
            except Exception as e: 
                pass

create_training_data()

print(len(training_data))


#training_data[1300] 
# Point 1: training data is containing cat and dog images in a sequence, 0 is for Dog which is up to 12501 and 
#Point 2: after that 1 is for Cat 
#Point 3: we need to be shuffle data


#data shuffling 
import random

random.shuffle(training_data)

#now data shuffled so print categories if they are also in shuffled manner
for i in range(5):
    print(training_data[i][1])  



for sample in training_data[:10]:
    print(sample[1])


plt.imshow(training_data[0][0], cmap='gray')
plt.show()



X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X)
y = np.array(y)



type(X)
X.dtype
y = np.array(y)
type(y)

print(X.shape, y.shape)
print(X.dtype, y.dtype)




X= tf.cast(X, np.float32)
print(X.dtype, y.dtype)
# Normalize images value as max value is 255
X= X / 255

#checking shape for all and unique labels in target
np.unique(y)



#display image from train set and test set
def display_image_label(featureset, labelset):
  img_no= np.random.randint(0, featureset.shape[0])
  print(img_no)
  img = featureset[img_no]
  label= labelset[img_no]
  print(f" image no:{img_no}   Label:{label}") # multiplying with 255 as we scaled with it
  plt.imshow(img, cmap=plt.cm.binary)
  plt.show()
  
print("Traing Set Sample Display:\n","="*50)
display_image_label(X, y)


#Dataset Parameters
num_classes = 2 # (0-1 digits).
image_vector_size = 8100 # (img shape: 90*90)

# Training parameters.
learning_rate = 0.001
training_steps = 6000
batch_size = 55
display_step = 100

# Network parameters.
n_hidden_1 = 128 # hidden layer1
n_hidden_2 = 256 # hidden layer2


x_train=X
x_test = X
y_train = y
# Flatten images  to feed as input
x_train=tf.reshape(x_train, [x_train.shape[0], image_vector_size])
x_test=tf.reshape(x_test, [x_test.shape[0], image_vector_size])




# Use batching of train data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

#initialize random normal weights and biases
# Store layers weight & bias

#using random_normal_initializer() of tf r2
normal = tf.random_normal_initializer()

weights = {
    'h1': tf.Variable(normal([image_vector_size, n_hidden_1])),
    'h2': tf.Variable(normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([num_classes]))
}

# Create model.
def neural_net(x):
    # Hidden fully connected layer with 128 neurons.
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Apply sigmoid to layer_1 output for non-linearity.
    layer_1 = tf.nn.sigmoid(layer_1)
    
    # Hidden fully connected layer with 256 neurons.
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Apply sigmoid to layer_2 output for non-linearity.
    layer_2 = tf.nn.sigmoid(layer_2)
    
    # Output fully connected layer with a neuron for each class.
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(out_layer)

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)



# Optimization process. 
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = neural_net(x)
        loss = cross_entropy(pred, y)
        
    # Variables to update, i.e. trainable variables.
    trainable_variables = [weights['h1'],biases['b1'],weights['h2'], biases['b2'], weights['out'], biases['out']]

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


#on training set
#collect loss and accuracy
loss_list, accuracy_list = [], []
# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = neural_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        loss_list.append(loss)
        acc = accuracy(pred, batch_y)
        accuracy_list.append(acc)
        print(f"step: {step}, loss: {loss}, accuracy: {acc}")
   
  
  
# plot loss and accuracy

def loss_plot(l1):
  plt.plot(l1, label="Loss")
  plt.legend()
  plt.show()

def acc_plot(l2):
  plt.plot(l2, label="Accuracy")
  plt.legend()
  plt.show()
  
#call plot  
loss_plot(loss_list)
acc_plot(accuracy_list)

