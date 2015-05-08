# DNNClassification
Deep Neural Network Classification with Caffe Framework

In this project, classification of the images which are under /Images/ directory is done using neural networks. In order to do that, a trained deep model (ImageNet) is fetched from http://caffe.berkeleyvision.org/ 

When the you run the code, classification prediction, probability of the class and entropy of the prediction is calculated and shown. For simplicity, there is no loops in the source code to read and classify multiple images. If you want to do it, it is pretty straightforward.

In order to run the project, you should install Caffe Framework from 
http://caffe.berkeleyvision.org/ 
and should fetch the trained deep net using 
/scripts/download_model_binary.py ../models/bvlc_reference_caffenet 
under Caffe's root directory. Also you should locate your classify.py file under examples folder. Please be careful with the directories because it can cause headaches :) If you want to train your own network, it will require so much time, so fetching the trained model is highly recommended.

The source code is understandable enough especially with the comments.
