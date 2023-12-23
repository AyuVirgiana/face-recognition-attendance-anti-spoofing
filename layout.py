# Import library yang diperlukan
import streamlit as st
import cv2
import face_recognition as frg
import yaml 
from utils import recognize, build_dataset

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(layout="wide")

# Membaca konfigurasi dari file YAML
cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']

# Menampilkan judul dan menu settings di sidebar
st.sidebar.title("Settings")

# Membuat menu bar untuk memilih jenis input (Picture atau Webcam)
menu = st.sidebar.selectbox("Menu", ["Home", "Attendance Table", "Database"])

# Slider untuk mengatur toleransi pengenalan wajah
TOLERANCE = st.sidebar.slider("Tolerance",0.0,1.0,0.5,0.01)
st.sidebar.info("Tolerance is the threshold for face recognition. The lower the tolerance, the more strict the face recognition. The higher the tolerance, the more loose the face recognition.")

# Informasi seputar mahasiswa di sidebar 
st.sidebar.title("Attender Information")
name_container = st.sidebar.empty()
id_container = st.sidebar.empty()
name_container.info('Name: Unknown')
id_container.success('ID: Unknown')

if menu == "Home":
    st.title("Face Recognition Attendance System")
    st.info("Click the button below to start attendance")
    if st.button("Start"):
        st.write(WEBCAM_PROMPT)
    
        # Pengaturan kamera
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        FRAME_WINDOW = st.image([])
        
        while True:
            # Membaca frame dari kamera
            ret, frame = cam.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                st.info("Please turn off the other app that is using the camera and restart app")
                st.stop()
            
            # Mengenali wajah pada frame dari kamera
            image, name, id = recognize(frame,TOLERANCE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Menampilkan nama dan ID orang yang dikenali
            name_container.info(f"Name: {name}")
            id_container.success(f"ID: {id}")
            FRAME_WINDOW.image(image)

elif menu == "Attendance Table":
    st.header("Attendance Table")
elif menu == "Database":
    st.header("Database")
    # Dropdown untuk memilih aksi
    action = st.selectbox("Choose Action", ["Add Data", "Delete Data"])
    if action == "Add Data":
        st.subheader("Add New Data")
        # Input nama pemilik wajah
        name = st.text_input("Name", placeholder='Enter name of face owner')

        # Membersihkan nama dari karakter yang tidak diizinkan untuk nama file
        cleaned_name = ''.join(e for e in name if e.isalnum())

        # Button untuk mengambil gambar dari webcam
        if st.button("Take Picture"):    
            img_file_buffer = st.camera_input("Take a picture")
            submit_btn = st.button("Submit", key="submit_btn")

            if img_file_buffer is not None:
                # Mengolah gambar dari webcam menggunakan OpenCV
                bytes_data = img_file_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                if submit_btn: 
                    # Memastikan nama dan ID diisi
                    if name == "" or id == "":
                        st.error("Please enter name and ID")
                    else:
                        # Memanggil fungsi submitNew untuk menambahkan data
                        ret = submitNew(name, id, cv2_img)
                        
                        # Menampilkan pesan sesuai dengan hasil operasi
                        if ret == 1: 
                            st.success("Student Added")
                        elif ret == 0: 
                            st.error("Student ID already exists")
                        elif ret == -1: 
                            st.error("There is no face in the picture")


    elif action == "Delete Data":
        st.subheader("Delete Data")

        # Dropdown untuk memilih wajah yang akan dihapus
        # name_to_remove = st.selectbox("Choose the name of face owner", known_face_names)
        name_to_delete = st.text_input("Choose the name of face owner")

        # Button untuk menghapus data wajah
        if st.button("Delete"):
            st.success(f"The data is successfully deleted!")

# Bagian Developer Section (di sidebar)
with st.sidebar.form(key='my_form'):
    st.title("Developer Section")

    # Tombol untuk membangun kembali dataset
    submit_button = st.form_submit_button(label='REBUILD DATASET')
    
    # Jika tombol ditekan, membangun ulang dataset
    if submit_button:
        with st.spinner("Rebuilding dataset..."):
            build_dataset()
        st.success("Dataset has been reset")

# Database untuk menyimpan data wajah yang telah di-training
# known_face_names = []
