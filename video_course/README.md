# Machine Learning and Deep Learning course
## some example code as practice

> [!NOTE]
> recommend running on Google Colab


## List of Machine Learning and Deep Learning exercises

* 1 - [Traingle Obtuse or Acute](https://github.com/KuoYuChang/Colab_Ipynb/blob/main/video_course/triangle_exercise.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KuoYuChang/Colab_Ipynb/blob/main/video_course/triangle_exercise.ipynb)

    A triangle formed by two vectors, determine if the triangle is obtuse or acute.

* 2 - [Co-Prime Examination](https://github.com/KuoYuChang/Colab_Ipynb/blob/main/video_course/co_prime_exercise.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KuoYuChang/Colab_Ipynb/blob/main/video_course/co_prime_exercise.ipynb)
    Pick two number from 2-10, determine if they are co-prime.

* 3 - [Tensorflow Playground Dataset training](https://github.com/KuoYuChang/Colab_Ipynb/blob/main/video_course/tf_playground_exercise.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KuoYuChang/Colab_Ipynb/blob/main/video_course/tf_playground_exercise.ipynb)

    Implement datasets on Tensorflow Playground, and train neural network to predict the dataset.

* 4 - [Linear SVM](https://github.com/KuoYuChang/Colab_Ipynb/blob/main/video_course/svm_exercise.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KuoYuChang/Colab_Ipynb/blob/main/video_course/svm_exercise.ipynb)

    Implement linear SVM with slack variables on torch, with self-defined kernel function and loss funciton.



* 5 - [K-Means](https://github.com/KuoYuChang/Colab_Ipynb/blob/main/video_course/kmeans_exercise.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KuoYuChang/Colab_Ipynb/blob/main/video_course/kmeans_exercise.ipynb)

    Implement K-Means on torch. Running on third-party package `pytorch-kmeans`.

* 6 - [Kernel SVM](https://github.com/KuoYuChang/Colab_Ipynb/blob/main/video_course/svm_kernel_exercise.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KuoYuChang/Colab_Ipynb/blob/main/video_course/svm_kernel_exercise.ipynb)

    Implement RBF kernel SVM on torch, with self-defined kernel function and loss funciton.
    
> [!WARNING]
> Experimental code, Projected Gradient Method implement in our gradient update. May not able to fully control variables still satisfied the constraints.

* 7 - [MNIST Classification Network](https://github.com/KuoYuChang/Colab_Ipynb/blob/main/video_course/mnist_torch_exercise.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KuoYuChang/Colab_Ipynb/blob/main/video_course/mnist_torch_exercise.ipynb)

    Establish simple fully connected network to classify MNIST dataset.

* 8 - [Bird Classfication Fully Connected Network](https://github.com/KuoYuChang/Colab_Ipynb/blob/main/video_course/Bird_FC.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KuoYuChang/Colab_Ipynb/blob/main/video_course/Bird_FC.ipynb)

    Train and test simple fully connected network to classify Bird dataset.


> [!WARNING]
> Reduce batch size if memory usage being too large.
> This exercise shows weakness of fully connected network. We don't recommend using fully connected network to complex image classifications.


* 9 - [Bird Classfication Convolutional Network](https://github.com/KuoYuChang/Colab_Ipynb/blob/main/video_course/Bird_Resnet.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KuoYuChang/Colab_Ipynb/blob/main/video_course/Bird_Resnet.ipynb)

    Train and test Convolutional Neural Network on image classification, here Residual Neural Network as an example.


> [!NOTE]
> To download Bird dataset, follow exercise 8: Bird Classfication Fully Connected Network.
> Or to download from [Kaggle Bird Dataset]( https://www.kaggle.com/veeralakrishna/200-bird-species-with-11788-images)



* 10 - [Yolo v9 Object Detection](https://github.com/KuoYuChang/Colab_Ipynb/blob/main/video_course/train_yolov9_object_detection_on_custom_dataset.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KuoYuChang/Colab_Ipynb/blob/main/video_course/train_yolov9_object_detection_on_custom_dataset.ipynb)

    Implement Yolo v9 object detection, simple demo.
    Then fine-tune model on custom dataset.

> [!NOTE]
> Package `Pillow` need fixed
> `pip install Pillow==9.5.0`
> Modified dataset folder path in data.yaml
> to `./football-players-detection-8/data.yaml`
> in yaml: 
> modified the following lines
> `test: ../football-players-detection-8/test/images`

> `train: ../football-players-detection-8/train/images`

> `val: ../football-players-detection-8/valid/images`
    