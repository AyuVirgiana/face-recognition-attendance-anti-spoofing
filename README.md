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
- **Database Management**: The system maintains a database of visitors, including their names and facial encodings.
- **User-Friendly Interface**: The Streamlit web application provides a clean and intuitive interface for users to interact with the system.

# System Architecture
**Frontend**: 
- Framework: Streamlit
- User Interface Components: Python with Streamlit widgets

**Data Storage:**


**Face Recognition:**
- Libraries:
  - face_recognition for face detection and recognition
  - OpenCV for real-time face detection and integration
  - Algorithm: FaceNet, dlib for accurate facial feature extraction

**Anti-Spoofing**
- Techniques: Liveness detection using a combination of methods such as texture analysis, motion analysis, and depth analysis (??)
- Libraries: OpenCV and specialized anti-spoofing libraries if available (??)

**Webcam Integration**
- Library: OpenCV for accessing and capturing live webcam feed
- Real-time Communication: WebSockets for seamless real-time data transfer (??)

# Installation
Clone the repository:
```bash
git clone https://github.com/your-username/face-recognition-attendance-anti-spoofing.git
cd face-recognition-attendance-anti-spoofing
```

# Install dependencies
Follow the setup instructions in the documentation to configure the system.
```bash
pip install -r requirements.txt
```
# Run the system
```bash
streamlit run [file_name.py] [ARGUMENTS)
```
# Lisence
- This Face Recognition Attendance System is licensed under the MIT License. Feel free to use and modify the code as needed. If you encounter any issues or have suggestions for improvement, please create an issue in the GitHub repository.
-  This project is supportd by Silent-Face-Anti-Spoofing belongs to [minivision technology](https://www.minivision.cn/).Special thanks to Minivision for providing the anti-spoofing models used in this test.

# Additional Information
- **File Structure:**
- visitor_database: Directory to store the database of visitor information.
- visitor_history: Directory to store the history of visitor attendance.
- resources/anti_spoof_models: Directory containing anti-spoofing models.

- **Note:**
The system uses the InceptionResnetV1 model for face recognition and MTCNN for face detection.
Make sure to have a compatible GPU for faster processing (optional, but recommended).

# Contributors
- Ayu Purnama Virgiana (210040171)
- I Komang Adisaputra Gita (210040017)
