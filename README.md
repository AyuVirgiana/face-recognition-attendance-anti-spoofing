# WEB-BASED FACE RECOGNITION ATTENDANCE SYSTEM with ANTI SPOOFING
This GitHub repository contains a web-based Facial Recognition Attendance System built with Python language and Streamlit framework. 

The System built with Face Recognition using [Inception Resnet (V1) models](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) in pytorch, pretrained on [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) and CASIA-Webface datasets, [MiniFASNet](https://github.com/AyuVirgiana/face-recognition-attendance-anti-spoofing/blob/main/src/model_lib/MiniFASNet.py) for Anti-Spoofing models by Minivision. 

The system is designed to accurately identify and record attendance using facial recognition while incorporating measures to prevent spoofing attempts.

# Features
The application allows users to perform the following tasks:
- **Visitor Validation**: Capture an image using the camera and perform *face recognition with anti-spoof* to differentiate between real and fake faces and validate visitors. The system checks the captured face against the database of known faces and identifies the person if a match is found.
- **View Visitor History**: View the history of visitor attendance, including their ID, name, and timestamp of the visit. Additionally, the application provides an option to search and display images of specific visitors.
- **Add to Database**: Add new individuals to the database for future recognition. Users can input the person's name and either upload a picture or capture one using the camera. The system maintains 2 data storage (CSV files) for visitors information.

The demo pictures of the features are available on our **Medium** ["Web-based Face Recognition Attendance System with Anti Spoofing"](https://medium.com/@ayuvirgiana10/web-based-face-recognition-attendance-system-with-anti-spoofing-cd479d193e6b).


# System Architecture
The entire code is written in Python **this project made and tested in python 3.11.2**

**Frontend**: 
- The system utilizes **Streamlit** to create a web interface. The main UI elements include a title bar, a sidebar for navigation, and different sections for
  visitor validation, viewing visitor history, and adding to the database.

**Data Storage:**
- This project store visitor information in two CSV files, one for general user information (visitors_db.csv) and another for visitor history (visitors_history.csv).

**Face Recognition:**
- Library:
  Using facenet_pytorch for [Inception Resnet (V1) models](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py), pretrained on [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) and CASIA-Webface datasets. The datasets aligned with [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html).
- **WHY [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)?**
  The Dlib face detector misses some of the hard examples (partial occlusion, silhouettes, etc). This makes the training set too “easy” which causes the model to perform worse on other benchmarks. To solve this, we use [Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) for face landmark detector that has proven to work very well in this setting.

**Anti-Spoofing**
- Model: [MiniFASNet](https://github.com/AyuVirgiana/face-recognition-attendance-anti-spoofing/blob/main/src/model_lib/MiniFASNet.py) supported by [Silent-Face-Anti-Spoofing](https://github.com/computervisioneng/Silent-Face-Anti-Spoofing.git) developed by (https://www.minivision.cn/).

**Installation Requirements**
- [requirement.txt](https://github.com/AyuVirgiana/face-recognition-attendance-anti-spoofing/blob/main/requirements.txt) consists the version of requirements we used in this app


# Training Data 
- **Face Recognition Model**

  The following models have been ported to pytorch (with links to download pytorch state_dict's):

  |Model name|LFW accuracy (as listed [here](https://github.com/davidsandberg/facenet))|Training dataset|
  | :- | :-: | -: |
  |[20180408-102900](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt) (111MB)|0.9905|CASIA-Webface|
  |[20180402-114759](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt) (107MB)|0.9965|VGGFace2|

  There is no need to manually download the pretrained state_dict's; they are downloaded automatically on model instantiation and cached for future use in the torch cache. To use an [Inception Resnet (V1) models](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) for facial recognition/identification in pytorch, use:
  
  ```python
  from facenet_pytorch import InceptionResnetV1 
  ```
  Used in this app => [app.py](https://github.com/AyuVirgiana/face-recognition-attendance-anti-spoofing/blob/main/app.py), line 10.

  [Classifier training of inception resnet v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1) page describes how to train the Inception-Resnet-v1 model as a classifier, i.e. not using Triplet Loss as was described in the Facenet paper.

- **Anti-spoofing Model**

  cited from [Silent-Face-Anti-Spoofing](https://github.com/computervisioneng/Silent-Face-Anti-Spoofing.git)
  1. The training set is divided into three categories, and the pictures of the same category are put into a folder;
  2. Due to the multi-scale model fusion method, the original image and different patch are used to train the model, so the data is divided into the original map and the patch based on the Original picture;
     - Original picture(org_1_height**x**width),resize the original image to a fixed size (width, height), as shown in Figure 1;
     - Patch based on original(scale_height**x**width),The face detector is used to obtain the face frame, and the edge of the face frame is expanded according to a certain scale. In order to ensure the consistency of the input size of the model, the area of the face frame is resized to a fixed size (width, height). Fig. 2-4 shows the patch examples with scales of 1, 2.7 and 4;  
  ![patch demo](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/patch_demo.png)
  3. The Fourier spectrum is used as the auxiliary supervision, and the corresponding Fourier spectrum is generated online from the training set images.  


# Installation
Clone the repository:
```bash
git clone https://github.com/AyuVirgiana/face-recognition-attendance-anti-spoofing.git
cd face-recognition-attendance-anti-spoofing
```
# Install dependencies/requirements
Follow the setup instructions in the documentation to configure the system.
```bash
pip install -r requirements.txt
```
# Run the system
```bash
streamlit run [app.py] [ARGUMENTS)
```


# References
- This project is supported by [Silent-Face-Anti-Spoofing](https://github.com/computervisioneng/Silent-Face-Anti-Spoofing.git) belongs to [minivision technology](https://www.minivision.cn/).Special thanks to Minivision for providing the anti-spoofing models used in this test. 
- (https://github.com/timesler/facenet-pytorch.git) Face Recognition using Pytorch by timesler
- Pytorch model weights were initialized using parameters ported from [David Sandberg’s tensorflow facenet repo](https://github.com/davidsandberg/facenet.git).


# Contributors
- Ayu Purnama Virgiana (210040171)
- I Komang Adisaputra Gita (210040017)
