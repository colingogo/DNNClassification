# EFE BOZKIR, TU Munich, Department of Computer Science
import numpy as np
import matplotlib.pyplot as plt
import math
import caffe
import os

caffe_root = '../'
import sys
sys.path.insert(0, caffe_root + 'python')

MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = 'images/Images/IMG_7.jpg'

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

# Read the test image and preprocess
myInputImg = caffe.io.load_image(IMAGE_FILE)
lx,ly,lz = myInputImg.shape
croppedImg = myInputImg[lx /4:-lx/4,ly/4:-ly/4]
croppedImg = plt.imsave('images/Images/croppedIMG7.jpg',croppedImg)

# Read raw test image or preprocessed test image
input_image = caffe.io.load_image('images/Images/croppedIMG7.jpg')
#input_image = caffe.io.load_image(IMAGE_FILE) #if you want to use raw test image, uncomment this line.
prediction = net.predict([input_image])
print 'Prediction shape:', prediction[0].shape
print 'Predicted class:', prediction[0].argmax()

# Softmax calculation // Probability of the class
out = net.forward()
print 'Probability of the class: ', max(out['prob'][0])

# Entropy calculation
entropy = 0.0
for x in range(0,1000):
	entropy = entropy + prediction[0][x]*math.log(prediction[0][x],2)
entropy = entropy*-1
print 'Entropy:', entropy

