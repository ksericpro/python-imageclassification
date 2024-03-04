# run
- python chatbot.py

# open browser
- http://127.0.0.1:7860

# cuda
Compute Unified Device Architecture (CUDA) is a parallel computing platform and application programming interface (API) developed by NVIDIA. It allows developers to use the power of NVIDIA GPUs (graphics processing units) for general-purpose computing, including deep learning.

CUDA provides developers with the tools and functionalities needed to harness the raw computational power of NVIDIA’s GPUs. It allows developers to direct specific computing tasks to the more efficient GPU rather than the CPU. 

- nvidia-smi

# test whether cuda is available
- import  torch
- print(torch.cuda.is_available())

# download
- https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local


# Training [For Image Classification]

## Terminology

- One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
- Total number of training examples present in a single batch.
- Iterations is the number of batches needed to complete one epoch.
- Gradient Descend = It is an iterative optimization algorithm used in machine learning to find the best results (minima of a curve). Gradient means the rate of inclination or declination of a slope.
- The learning rate is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function.

## setup environment
- set TF_ENABLE_ONEDNN_OPTS=0
- get %TF_ENABLE_ONEDNN_OPTS%
- python training.py

## model 1

_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 150, 150, 64)      1792

 max_pooling2d (MaxPooling2  (None, 75, 75, 64)        0
 D)

 conv2d_1 (Conv2D)           (None, 75, 75, 32)        18464

 max_pooling2d_1 (MaxPoolin  (None, 38, 38, 32)        0
 g2D)

 conv2d_2 (Conv2D)           (None, 38, 38, 32)        9248

 max_pooling2d_2 (MaxPoolin  (None, 19, 19, 32)        0
 g2D)

 flatten (Flatten)           (None, 11552)             0

 dense (Dense)               (None, 100)               1155300

 dense_1 (Dense)             (None, 3)                 303

=================================================================
Total params: 1185107 (4.52 MB)
Trainable params: 1185107 (4.52 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
63/63 [==============================] - 3s 51ms/step - loss: 0.6839 - accuracy: 0.7300
63/63 [==============================] - 3s 50ms/step
              precision    recall  f1-score   support

           0       0.73      0.74      0.73      1000
           1       0.73      0.72      0.73      1000

    accuracy                           0.73      2000
   macro avg       0.73      0.73      0.73      2000
weighted avg       0.73      0.73      0.73      2000

## model 2

_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 150, 150, 256)     19456

 max_pooling2d (MaxPooling2  (None, 75, 75, 256)       0
 D)

 dropout (Dropout)           (None, 75, 75, 256)       0

 conv2d_1 (Conv2D)           (None, 75, 75, 128)       819328

 max_pooling2d_1 (MaxPoolin  (None, 37, 37, 128)       0
 g2D)

 dropout_1 (Dropout)         (None, 37, 37, 128)       0

 conv2d_2 (Conv2D)           (None, 37, 37, 64)        73792

 max_pooling2d_2 (MaxPoolin  (None, 18, 18, 64)        0
 g2D)

 dropout_2 (Dropout)         (None, 18, 18, 64)        0

 conv2d_3 (Conv2D)           (None, 18, 18, 32)        18464

 max_pooling2d_3 (MaxPoolin  (None, 9, 9, 32)          0
 g2D)

 dropout_3 (Dropout)         (None, 9, 9, 32)          0

 flatten (Flatten)           (None, 2592)              0

 dense (Dense)               (None, 64)                165952

 dense_1 (Dense)             (None, 32)                2080

 dense_2 (Dense)             (None, 3)                 99

=================================================================
Total params: 1099171 (4.19 MB)
Trainable params: 1099171 (4.19 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________       
63/63 [==============================] - 51s 807ms/step - loss: 0.4739 - accuracy: 0.7715
63/63 [==============================] - 48s 765ms/step       
              precision    recall  f1-score   support

           0       0.76      0.80      0.78      1000
           1       0.79      0.74      0.77      1000

    accuracy                           0.77      2000
   macro avg       0.77      0.77      0.77      2000
weighted avg       0.77      0.77      0.77      2000


## conclusion
- As we have seen, the second CNN model was able to predict the test image correctly with a test accuracy of close to 80%.

- There is still scope for improvement in the test accuracy of the CNN model chosen here. Different architectures and optimizers can be used to build a better food classifier.

- Transfer learning can be applied to the dataset to improve accuracy. You can choose among multiple pre-trained models available in the Keras framework.

- Once the desired performance is achieved from the model, the company can use it to classify different images being uploaded to the website.

- We can further try to improve the performance of the CNN model by using some of the below techniques and see if you can increase accuracy:

- We can try hyperparameter tuning for some of the hyperparameters like the number of convolutional blocks, the number of filters in each Conv2D layer, filter size, activation function, adding/removing dropout layers, changing the dropout ratio, etc.

- Data Augmentation might help to make the model more robust and invariant toward different orientations.



# Training
- python gradio_ui_original.py -t=1

# Inference
- python gradio_ui_original.py

# Api server
- uvicorn api_server:app --reload

# references

[link] (https://www.gradio.app/guides/quickstart)
[cuda] (https://saturncloud.io/blog/what-is-assertionerror-torch-not-compiled-with-cuda-enabled/)
[GradioML] (https://towardsdatascience.com/creating-a-simple-image-classification-machine-learning-demo-with-gradioml-361a245d7b50)

[Error] (https://stackoverflow.com/questions/67553391/input-0-of-layer-conv2d-is-incompatible-with-layer-expected-axis-1-of-input-sh)