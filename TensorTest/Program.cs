using System;
using TensorFlow;
/*
[==========================================================================GROUP:Tensorflow==========================================================================]
Tensorflow Questions
 https://hanxiao.github.io/2018/04/21/Teach-Machine-to-Comprehend-Text-and-Answer-Question-with-Tensorflow/
 https://www.oreilly.com/ideas/object-detection-with-tensorflow
 https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html
 https://github.com/tensorflow/models/tree/master/research/object_detection
 Softmax
 LSTM
 True understanding of convolution
 HDF5 format vs proto and pb files
  Convert to protobuf https://stackoverflow.com/questions/36412098/convert-keras-model-to-tensorflow-protobuf
  https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#using-keras-models-with-tensorflow
 Inception vs VGG16 model
  Inception > VGG16
 Inception retraining - https://www.tensorflow.org/tutorials/image_retraining
Keras hello world
     https://github.com/fastforwardlabs/keras-hello-world/blob/master/kerashelloworld.ipynb
   Keras : VGG16 training: Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
Training a VGG16
https://slashtutorial.com/ai/tensorflow/
Neural Networking
Course: http://web.stanford.edu/class/cs20si/
TensorFlow wrappers:
 BEST Keras : Integrated into Tensorflow
 BEST TFLearn :
 MEH SKFlow : Integrated into Tensorflow
 MEH TFSlim : https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
c# TensorFlow: https://github.com/migueldeicaza/TensorFlowSharp
Watch tflearn tutorial: https://pythonprogramming.net/tflearn-machine-learning-tutorial/
Watch tensorboard tutorial: https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard
Classifier from little data: https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
Keras - Tensorflow wrapper? - https://en.wikipedia.org/wiki/Keras
 https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
Saving model
    http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
Pretrained models from imagenet
 Fully investigate and understand CNN training code
  Understand all parts and definitions
  Understand how to save and load the trained NN
 Investigate and understand VGG16 code
  VGG16 - 16 layers
  VGG19 - 19 layers
  Terms:
   ImageNet: DB of images
   Convolution:
    mathematical operation on two functions (f and g) to produce a third function, that is typically viewed as a modified version of one of the original functions
    also, weighted sum of pixel values of the image as window slides across image.
   Max-Pooling: Subsampling a convolved output
   softmax: continuous and differentiable max function
   ReLU: Rectified Linear Unit activation function
   one-hot encoding: only one element is 1 and rest are 0
   Deconvolution Network (generate data from NN)/visualizes filter
Tensorflow notes:
 Code Samples: https://github.com/BinRoot/TensorFlow-Book
 https://livebook.manning.com/#!/book/machine-learning-with-tensorflow/chapter-3/61
 Part 1)
  * Installation: https://www.tensorflow.org/get_started/eager
  * HelloWorld: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/helloworld.py
  * jupyter webapp
  * TensorBoard visualizer
 Part 2)
  * matplotlib: Plotter for python
  * Solving linear regression using a gradient descent optimizer.
  * https://en.wikipedia.org/wiki/Gradient_descent
   Gradient descent is one of the most popular algorithms to perform optimization and by far the most common way to optimize neural networks
   http://ruder.io/optimizing-gradient-descent/
  * Polynomial model solving
TensorFlow tasks:
 * Learn to use GPU
  Installation: http://www.python36.com/install-tensorflow-gpu-windows/
  EASIEST Anaconda-based: http://blog.nitishmutha.com/tensorflow/2017/01/22/TensorFlow-with-gpu-for-windows.html
  GeForce GTX1080
  2560 Cuda Cores
  Python 3.5 (upgrade pip3)
  CUDA 9.0 to work with Anaconda (doesn't work with 9.1)
  CuDNN 7
  Have to manually copy around libtensorflow dlls into bin/debug
  Use dll self check https://gist.github.com/mrry/ee5dbcfdd045fa48a27d56664411d41c
 * Setup my own trainer
 * Setup my own trainer user
Convolutional Neural Network (CNN)
Easy image training with tensorflow: http://www.wolfib.com/Image-Recognition-Intro-Part-1/
DataSet from: http://www.cs.toronto.edu/~kriz/cifar.html
https://livebook.manning.com/#!/book/machine-learning-with-tensorflow/chapter-9/126
 */
namespace TensorTest
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var session = new TFSession())
            {
                var graph = session.Graph;

                var a = graph.Const(2);
                var b = graph.Const(3);
                Console.WriteLine("a=2 b=3");

                // Add two constants
                var addingResults = session.GetRunner().Run(graph.Add(a, b));
                var addingResultValue = addingResults.GetValue();
                Console.WriteLine("a+b={0}", addingResultValue);

                // Multiply two constants
                var multiplyResults = session.GetRunner().Run(graph.Mul(a, b));
                var multiplyResultValue = multiplyResults.GetValue();
                Console.WriteLine("a*b={0}", multiplyResultValue);
            }
        }
    }
}
