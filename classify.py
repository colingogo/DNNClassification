import numpy as np
import matplotlib.pyplot as plt
import math
import caffe
import os
#%matplotlib inline

caffe_root = '../'
import sys
sys.path.insert(0, caffe_root + 'python')

MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = 'images/Images/IMG_8.jpg'

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

# Read the test image
input_image = caffe.io.load_image(IMAGE_FILE)
#plt.imshow(input_image)

# Preprocess the input image. Maybe size or sharpening?
resized_img = caffe.io.resize_image(im=input_image, new_dims=(256,256))
#processed_img = caffe.io.oversample(images=resized_img, crop_dims=(128,128))

prediction = net.predict([resized_img])
print 'Prediction shape:', prediction[0].shape
#plt.plot(prediction[0])
print 'Predicted class:', prediction[0].argmax()
print 'probability in the end', prediction[0][prediction[0].argmax()]

# Softmax calculation // Probability of the class
out = net.forward()
print 'Probability of the class: ', max(out['prob'][0])


# Entropy calculation
entropy = 0.0
for x in range(0,1000):
	entropy = entropy + prediction[0][x]*math.log(prediction[0][x],2)
entropy = entropy*-1
print 'Entropy:', entropy

prediction = net.predict([resized_img], oversample=False)
print 'prediction shape:', prediction[0].shape
#plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()

#%timeit net.predict([input_image])

# Resize the image to the standard (256, 256) and oversample net input sized crops.
# /////input_oversampled = caffe.io.oversample([caffe.io.resize_image(input_image, net.image_dims)], net.crop_dims)
# 'data' is the input blob name in the model definition, so we preprocess for that input.
# /////caffe_input = np.asarray([net.transformer.preprocess('data', in_) for in_ in input_oversampled])
# forward() takes keyword args for the input blobs with preprocessed input arrays.
#%timeit net.forward(data=caffe_input)

#/////caffe.set_mode_gpu()

#prediction = net.predict([input_image])
#print 'prediction shape:', prediction[0].shape
#plt.plot(prediction[0])

# Full pipeline timing.
#%timeit net.predict([input_image])

# Forward pass timing.
#%timeit net.forward(data=caffe_input)
