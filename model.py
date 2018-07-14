from PIL import Image
import skimage
from skimage import data, io, color
import glob
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib.layers.python.layers import batch_norm


# DataLoader class: need to customize according to your dataset
class DataLoader(object):
	def __init__(self):
		#open file and read in 100 images, and convert on the fly
		self.images = []
		self.val_images = []
		counter = 0
		for filename in glob.glob('../coast/*.jpg'):
			if counter < 360:

				im = io.imread(filename)
				im = im.astype(np.float32) / 255.

				self.images.append(im)
				self.val_images.append(im)

			else:
				break
			counter += 1

		self.load_size = 256
		self.fine_size = 224
		self.h = 224
		self.w = 224
		self.c = 3
		self._idx = 0
		self.val_idx = 0
		self.num = len(self.images) - 1
		self.val_num = len(self.val_images) - 1

	def splitImages(self, imageBatch):
		Y_images = []
		U_images = []
		V_images = []
		#IMPORTANT: changed this for loop from val_images to images
		for i in range(imageBatch.shape[0]):
			Y, U, V = tf.split(self.images[i], 3, 2)
			Y_images.append(Y)
			U_images.append(U)
			V_images.append(V)
		return Y_images, U_images, V_images

	def loadVal(self, batch_size):
		rgb_batch = np.zeros((batch_size, self.h, self.w, self.c), dtype=np.float32)
		for i in range(batch_size):
			image = self.val_images[self.val_idx]
			offset_h = (self.load_size-self.fine_size)//2
			offset_w = (self.load_size-self.fine_size)//2
			rgb_batch[i, ...] = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
			self.val_idx += 1
			if self.val_idx == self.val_num:
			 	self.val_idx = 0
		#yuv_Batch = rgb2yuv(np.array(rgb_batch))
		return rgb_batch


	def loadTrain(self):
		Y_batch = np.zeros((self.num, self.h, self.w, self.c))
		U_batch = np.zeros((self.num, self.h, self.w, self.c))
		V_batch = np.zeros((self.num, self.h, self.w, self.c))
		for i in range(self.num):
			# when your dataset is huge, you might need to load images on the fly
			# you might also want data augmentation
			L_batch[i, ...] = self.Y_images[i].reshape((self.h, self.w, self.c))
			U_batch[i, ...] = self.U_images[i].reshape((self.h, self.w, self.c))
			V_batch[i, ...] = self.V_images[i].reshape((self.h, self.w, self.c))
		return (Y_batch, U_batch, V_batch)

	def next_batch(self, batch_size):
		rgb_batch = np.zeros((batch_size, self.h, self.w, self.c), dtype=np.float32)
		for i in range(batch_size):
			image = self.images[self._idx]
			offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
			offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
			rgb_batch[i, ...] = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
			self._idx += 1
			if self._idx == self.num:
			 		self._idx = 0


		return rgb_batch
"""
Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
"""
def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img

def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb2yuv_filter = tf.constant(
        [[[[0.299, -0.169, 0.499],
           [0.587, -0.331, -0.418],
            [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])

    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)

    return temp


def yuv2rgb(yuv):
    """
    Convert YUV image into RGB https://en.wikipedia.org/wiki/YUV
    """
    yuv = tf.multiply(yuv, 255)
    yuv2rgb_filter = tf.constant(
        [[[[1., 1., 1.],
           [0., -0.34413999, 1.77199996],
            [1.40199995, -0.71414, 0.]]]])
    yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])
    temp = tf.nn.conv2d(yuv, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, yuv2rgb_bias)
    temp = tf.maximum(temp, tf.zeros(temp.get_shape(), dtype=tf.float32))
    temp = tf.minimum(temp, tf.multiply(
        tf.ones(temp.get_shape(), dtype=tf.float32), 255))
    temp = tf.div(temp, 255)
    return temp


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# def init_bias(shape):
#     return tf.Variable(tf.zeros(shape))

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)

"""
Network architecture http://tinyclouds.org/colorize/residual_encoder.png
"""
def cnn(x, train_phase):
	# print x.shape
	weights = {
	    'wc1': init_weights([3, 3, 3, 64]),
	    'wc2': init_weights([3, 3, 64, 64]),
	    'wc3': init_weights([3, 3, 64, 128]),
	    'wc4': init_weights([3, 3, 128, 128]),
	    'wc5': init_weights([3, 3, 128, 256]),
	    'wc6': init_weights([3, 3, 256, 256]),
	    'wc7': init_weights([3, 3, 256, 256]),
	    'wc8': init_weights([3, 3, 256, 512]),
	    'wc9': init_weights([3, 3, 512, 512]),
	    'wc10': init_weights([3, 3, 512, 512]),
	    'w1': init_weights([1,1, 512, 256]),
	    'w2': init_weights([3,3, 256, 128]),
	    'w3': init_weights([3,3, 128, 64]),
	    'w4': init_weights([3,3, 64, 3]),
		'w5': init_weights([3,3, 3, 3]),
		'w6': init_weights([3,3, 3, 2]),

	}

	# biases = {
	#     'bo': init_bias(100),
	#     'b1': init_bias(64),
	#     'b2': init_bias(64),
	#     'b3': init_bias(128),
	#     'b4': init_bias(128),
	#     'b5': init_bias(256),
	#     'b6': init_bias(256),
	#     'b7': init_bias(256),
	#     'b8': init_bias(512),
	#     'b9': init_bias(512),
	#     'b10': init_bias(512)
	#
	# }

	x_bn = batch_norm_layer(x, train_phase, 'xbn')
	x = tf.nn.relu(x_bn)
	# Conv + ReLU
	conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
	conv1 = batch_norm_layer(conv1, train_phase, 'b1')
	conv1 = tf.nn.relu(conv1)

	# Conv + ReLU
	conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
	conv2_bn = batch_norm_layer(conv2, train_phase, 'bn2')
	conv2 = tf.nn.relu(conv2_bn)

	#pool
	pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


	# Conv + ReLU
	conv3 = tf.nn.conv2d(pool1, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
	conv3 = batch_norm_layer(conv3, train_phase, 'b3')
	conv3 = tf.nn.relu(conv3)
	# Conv + ReLU
	conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
	conv4_bn = batch_norm_layer(conv4, train_phase, 'bn4')
	conv4 = tf.nn.relu(conv4_bn)

	#pool
	pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	# Conv + ReLU
	conv5 = tf.nn.conv2d(pool2, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
	conv5 = batch_norm_layer(conv5, train_phase, 'b5')
	conv5 = tf.nn.relu(conv5)
	# Conv + ReLU
	conv6 = tf.nn.conv2d(conv5, weights['wc6'], strides=[1, 1, 1, 1], padding='SAME')
	conv6 = batch_norm_layer(conv6, train_phase, 'b6')
	conv6 = tf.nn.relu(conv6)
	# Conv + ReLU
	conv7 = tf.nn.conv2d(conv6, weights['wc7'], strides=[1, 1, 1, 1], padding='SAME')
	conv7_bn = batch_norm_layer(conv7, train_phase, 'bn7')
	conv7 = tf.nn.relu(conv7_bn)

	#pool
	pool3 = tf.nn.max_pool(conv7, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



	# Conv + ReLU
	conv8 = tf.nn.conv2d(pool3, weights['wc8'], strides=[1, 1, 1, 1], padding='SAME')
	conv8 = batch_norm_layer(conv8, train_phase, 'b8')
	conv8 = tf.nn.relu(conv8)
	# Conv + ReLU
	conv9 = tf.nn.conv2d(conv8, weights['wc9'], strides=[1, 1, 1, 1], padding='SAME')
	conv9 = batch_norm_layer(conv9, train_phase, 'b9')
	conv9 = tf.nn.relu(conv9)
	# Conv + ReLU
	conv10 = tf.nn.conv2d(conv9, weights['wc10'], strides=[1, 1, 1, 1], padding='SAME')
	conv10_bn = batch_norm_layer(conv10, train_phase, 'bn10')



	convLHS1 = tf.nn.conv2d(conv10_bn, weights['w1'], strides=[1,1,1,1], padding = 'SAME')
	convLHS1 = tf.image.resize_bilinear(convLHS1, (56, 56))
	elementwiseSum1 = tf.add(conv7_bn, convLHS1)

	convLHS2 = tf.nn.conv2d(elementwiseSum1, weights['w2'], strides=[1,1,1,1], padding='SAME')
	convLHS2 = batch_norm_layer(convLHS2, train_phase, 'bn11')
	convLHS2 = tf.maximum(0.01 * tf.nn.relu(convLHS2), tf.nn.relu(convLHS2))
	convLHS2 = tf.image.resize_bilinear(convLHS2, (112, 112))
	elementwiseSum2 = tf.add(conv4_bn, convLHS2)

	convLHS3 = tf.nn.conv2d(elementwiseSum2, weights['w3'], strides = [1,1,1,1], padding='SAME')
	convLHS3 = batch_norm_layer(convLHS3, train_phase, 'bn12')
	convLHS3 = tf.maximum(0.01 * tf.nn.relu(convLHS3), tf.nn.relu(convLHS3))
	convLHS3 = tf.image.resize_bilinear(convLHS3, (224, 224))
	elementwiseSum3 = tf.add(conv2_bn, convLHS3)

	convLHS4 = tf.nn.conv2d(elementwiseSum3, weights['w4'], strides = [1,1,1,1], padding = 'SAME')
	convLHS4 = batch_norm_layer(convLHS4, train_phase, 'bn13')
	convLHS4 = tf.maximum(0.01 * tf.nn.relu(convLHS4), tf.nn.relu(convLHS4))



	elementwiseSum4 = tf.add(x_bn, convLHS4)
	convLHS5 = tf.nn.conv2d(elementwiseSum4, weights['w5'], strides = [1,1,1,1], padding = 'SAME')
	convLHS5 = batch_norm_layer(convLHS5, train_phase, 'bn14')
	convLHS5 = tf.maximum(0.01 * tf.nn.relu(convLHS5), tf.nn.relu(convLHS5))

	output = tf.nn.conv2d(convLHS5, weights['w6'], strides = [1,1,1,1], padding = 'SAME')
	#output = batch_norm_layer(output, train_phase, 'bn15')
	output = tf.sigmoid(output)
	#print output.shape

	return output


def trainAndEvaluate():
	# Parameters
	learning_rate = .0001
	training_iters = 100000
	batch_size = 1
	step_display = 2
	step_save = 1000
	path_save = 'convnet'
	num_results = 5

	# Network Parameters
	h = 224
	w = 224
	c = 3
	# Construct dataloader
	loader = DataLoader()
	#print "loader"


	# tf Graph input
	y = tf.placeholder(tf.float32, [batch_size, h, w, c])
	training_phase = tf.placeholder(tf.bool, name='training_phase')
	uv = tf.placeholder(tf.uint8, name='uv')
	#global_step = tf.Variable(0, name='global_step', trainable=False)

	# u = tf.placeholder(tf.float32, [None, h, w, c])
	# v = tf.placeholder(tf.float32, [None, h, w, c])

	# Construct model
	x = rgb2yuv(y)
	grayscale = tf.image.rgb_to_grayscale(y)
	grayscale_rgb = tf.image.grayscale_to_rgb(grayscale)
	grayscale_yuv = rgb2yuv(grayscale_rgb)
	grayscale = tf.concat([grayscale, grayscale, grayscale], 3)
	#print grayscale.shape


	#grayscale = tf.concat([grayscale, grayscale, grayscale], 3)

	output = cnn(grayscale, training_phase)
	pred_yuv = tf.concat([tf.split(grayscale_yuv,3, 3)[0], output], 3)
	#print "prediction", pred_yuv
	pred_rgb = yuv2rgb(pred_yuv)
	# print output.shape, x.shape


	actual = tf.concat([tf.split(x, 3, 3)[1], tf.split(x,3, 3)[2]], 3)
	# loss = tf.square(tf.subtract(output, tf.concat(
	# 	[tf.split(x, 3, 3)[1], tf.split(x,3, 3)[2]], 3)))
	loss = tf.losses.huber_loss(actual, output, reduction=tf.losses.Reduction.NONE)
	result = tf.abs(tf.subtract(output, actual))
	mask = tf.cast(result<0.01, tf.float32)
    accuracy = tf.reduce_mean(mask)

	# l2_u = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output[:, :, :, 0], x[:,:,:,1]))))
	# l2_v = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output[:, :, :, 1], v[:,:,:,2]))))
	# Define loss and optimizer
	# loss = tf.add(l2_u, l2_v)
	# train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	if uv == 1:
	    loss = tf.split(3, 2, loss)[0]
	elif uv == 2:
	    loss = tf.split(3, 2, loss)[1]
	else:
	    #print loss.get_shape()
	    loss = (tf.split(loss,2,3)[0] + tf.split(loss,2,3)[1]) / 2



	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	#train_optimizer = optimizer.minimize(loss)
	train_optimizer = optimizer.minimize(loss, gate_gradients=optimizer.GATE_NONE)

	# Evaluate model
	# correct_pred1 = tf.divide(output[:, :, :,0], x[:,:,:,1])
	# correct_pred2 = tf.divide(output[:, :, :,1], v[:,:,:,2])
	# accuracy = tf.add(tf.reduce_mean(tf.cast(correct_pred1, tf.float32)) , tf.reduce_mean(tf.cast(correct_pred2, tf.float32)))


	# define initialization
	init = tf.global_variables_initializer()

	# define saver
	saver = tf.train.Saver()

	# Launch the graph
	with tf.Session() as sess:
		# Initialization
		sess.run(init)

		step = 1
		while step < training_iters:
			# Load a batch of data
			rgb_batch = loader.next_batch(batch_size)


			# Run optimization op (backprop)
			#print "before", rgb_batch.shape
			sess.run(train_optimizer, feed_dict={y: rgb_batch, uv: 1, training_phase: True})
			sess.run(train_optimizer, feed_dict={y: rgb_batch, uv: 2, training_phase: True})

			if step % step_display == 0:
				# Calculate batch loss and accuracy while training
				a = sess.run(accuracy, feed_dict={y: rgb_batch, uv:3, training_phase: False})
				print('Iter ' + str(step) + ', Minibatch Accuracy = ' + a)
				#print colorImage, colorImage.dtype, type(colorImage)
				if step % 10 == 0:
					for i in range(num_results):
						val_rgb_batch = loader.next_batch(batch_size)
						result, grayscale_rgb_, colorimage_  = sess.run([pred_rgb, grayscale_rgb, y], feed_dict={y: val_rgb_batch, uv:3, training_phase: False})
						summary_image = concat_images(grayscale_rgb_[0], result[0])
						summary_image = concat_images(summary_image, colorimage_[0])
						plt.imsave("Huber" + str(step) + str(i), summary_image)

			step += 1


		print('Optimization Finished!')




trainAndEvaluate()
