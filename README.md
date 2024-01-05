# FACE RECOGNITION SYSTEM with ANTI SPOOFING
This GitHub repository contains a comprehensive Face Recognition Attendance System with built-in Face Recognition using Pytorch and Anti-Spoofing features. The system is designed to accurately identify and record attendance using facial recognition while incorporating measures to prevent spoofing attempts.

# Overview
The application allows users to perform the following tasks:
- **Visitor Validation**: Capture an image using the camera and perform face recognition + anti-spoof to validate visitors. The system checks the captured face against the database of known faces and identifies the person if a match is found.
- **View Visitor History**: View the history of visitor attendance, including their ID, name, and timestamp of the visit. Additionally, the application provides an option to search and display images of specific visitors.
- **Add to Database**: Add new individuals to the database for future recognition. Users can input the person's name and either upload a picture or capture one using the camera.

# Key Features
- **Face Recognition**: Utilizes advanced algorithms to recognize faces in real-time accurately.
- **Anti-Spoofing**: Utilizes an anti-spoofing model to differentiate between real and fake faces during validation.
- **Attendance Logging**: Records the ID, visitor name, and timestamp when a person is recognized.
- **Data Storage Management**: The system maintains 2 data storage for visitors information.
- **User-Friendly Interface**: The Streamlit web application provides a clean and intuitive interface for users to interact with the system.

# System Architecture
The entire code is written in Python **this project made and tested in python 3.11.2**

**Frontend**: 
- The system utilizes **Streamlit** to create a web interface. The main UI elements include a title bar, a sidebar for navigation, and different sections for
  visitor validation, viewing visitor history, and adding to the database.

**Data Storage:**
- This project store visitor information in two CSV files, one for general user information (visitors_db.csv) and another for visitor history (visitors_history.csv)

**Face Recognition:**
- Library:
  Using facenet_pytorch for MTCNN and InceptionResnetV1 Models

**Anti-Spoofing**
- Model:
  MiniFASNet variants include MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, and MiniFASNetV2SE supported by Silent-Face-Anti-Spoofing developed by Minivision
  
**Webcam Integration**
- Library:
  OpenCV for accessing and capturing live webcam feed

# Installation
Clone the repository:
```bash
git clone https://github.com/AyuVirgiana/face-recognition-attendance-anti-spoofing.git
cd face-recognition-attendance-anti-spoofing
```
# Install dependencies
Follow the setup instructions in the documentation to configure the system.
```bash
pip install -r requirements.txt
```
# Run the system
```bash
streamlit run [app.py] [ARGUMENTS)
```
   
# References
- This project is supported by Silent-Face-Anti-Spoofing belongs to [minivision technology](https://www.minivision.cn/).Special thanks to Minivision for providing the anti-spoofing models used in this test. Github Link : (https://github.com/computervisioneng/Silent-Face-Anti-Spoofing.git)
- (https://github.com/timesler/facenet-pytorch.git) Face Recognition using Pytorch by timesler


# Contributors
- Ayu Purnama Virgiana (210040171)
- I Komang Adisaputra Gita (210040017)
