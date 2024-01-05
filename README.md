# face-recognition-attendance-anti-spoofing
This GitHub repository contains a comprehensive Face Recognition Attendance System with built-in Anti-Spoofing features. The system is designed to accurately identify and record attendance using facial recognition while incorporating measures to prevent spoofing attempts.

# Key Features
- **Real-time Face Data Addition**: Capture and add face data through a live webcam feed for instant integration.
- **Data Deletion**: Effortlessly remove or update face data to maintain an accurate database.
- **Face Recognition**: Utilizes advanced algorithms to recognize faces in real-time accurately.
- **Anti-Spoofing Measures**: Implements robust anti-spoofing features to enhance system security.
- **Automatic Attendance Marking**: Marks attendance automatically upon successful face recognition.

# Datasets
```
├── datasets
    └── RGB_Images
        ├── org_1_80x60
            ├── 0
		├── aaa.png
		├── bbb.png
		└── ...
            ├── 1
		├── ddd.png
		├── eee.png
		└── ...
            └── 2
		├── ggg.png
		├── hhh.png
		└── ...
        ├── 1_80x80
        └── ...
```

# System Architecture
**Frontend**: 
- Framework: Streamlit
- User Interface Components: Python with Streamlit widgets

**Data Storage:**
- Serialization:
  - PyYAML for configuration data or settings stored in YAML format

**Face Recognition:**
- Libraries:
  - face_recognition for face detection and recognition
  - OpenCV for real-time face detection and integration
  - Algorithm: FaceNet or dlib for accurate facial feature extraction (??)

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

# Contributors
- Ayu Purnama Virgiana (210040171)
- I Komang Adisaputra Gita (210040017)
