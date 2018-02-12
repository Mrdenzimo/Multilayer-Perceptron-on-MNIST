# Load MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('.', one_hot=True)

# Start TensorFlow InteractiveSession
import tensorflow as tf 
sess = tf.InteractiveSession()

# ------- Build a Softmax Regression Model -------

# placeholders
"""
	   x: 2D tensor of shape [None, 784]. 
	None: indicates that the batch size can be of any size.
	 784: indicates the dimensionality of a single flattened 28x28 pi MNIST image.

	   y: 2D tensor where each row is a one-hot 10-dimensional vector indicating which digit class (zero through nine)

"""
x = tf.placeholder(tf.float32, shape=[None, 784])		
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# variables
"""
	   W: weights; [784,10] matrix because of 784 input features and 10 outputs
	   b: biases; [10] vector because of 10 classes

"""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Variables must be initialized before it can be used within a session
sess.run(tf.global_variables_initializer())

# Predicted Class and Loss Function
"""
	We can now implement our regression model. We multiply the vectorized
	input images x by the weight matrix W, add the bias b.
"""
y = tf.matmul(x,W) + b

# Define a loss function
"""
	tf.nn.softmax_cross_entropy_with_logits internally applies the softmax 
	on the model's unnormalized model prediction and sums across all classes,
	and tf.reduce_mean takes the average over these sums.
"""
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Train the Model
"""
	We will use steepest gradient descent, with a step length of 0.5, to descend the cross entropy.

	We load 100 training examples in each training iteration. We then run the train_step operation,
	 using feed_dict to replace the placeholder tensors x and y_ with the training examples. Note
	 that you can replace any tensor in your computation graph using feed_dict -- it's not restricted 
	 to just placeholders.
"""
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Evaluate the Model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
