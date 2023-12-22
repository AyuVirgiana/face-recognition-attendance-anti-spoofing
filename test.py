import streamlit as st
import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime
import numpy as np

# Inisialisasi webcam
def init_camera():
    return cv2.VideoCapture(0)

# Fungsi untuk mendeteksi wajah dan mengembalikan nama jika dikenali
def recognize_face(frame, known_faces):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        return name

# Fungsi untuk menambahkan data wajah ke database
def add_face_data(image, name):
    face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(image))[0]
    known_faces.append(face_encoding)
    known_face_names.append(name)

# Fungsi untuk menyimpan data absensi
def save_attendance(name):
    df = pd.DataFrame({'ID': [len(attendance) + 1],
                       'Nama': [name],
                       'Tanggal Kehadiran': [datetime.now().date()],
                       'Jam Kehadiran': [datetime.now().strftime('%H:%M:%S')]})
    attendance.append(df)
    return df

# Fungsi utama
def main():
    st.title("Face Recognition Attendance System")

    menu = st.sidebar.selectbox("Menu", ["Home", "Tabel Absensi", "Data Training"])
    
    if menu == "Home":
        #st.header("Absensi")

        # Inisialisasi webcam
        cap = init_camera()

        # Button untuk memulai absensi
        if st.button("Mulai Absen"):
            st.subheader("Absen Sedang Berlangsung...")
            while True:
                ret, frame = cap.read()

                # Ubah format BGR menjadi RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Tampilkan frame menggunakan st.image
                st.image(frame_rgb, channels="RGB")

                name = recognize_face(frame, known_faces)

                if name != "Unknown":
                    st.success(f"Halo {name}. Anda sudah diabsen")
                    attendance_df = save_attendance(name)
                    st.table(attendance_df)
                    break

            cap.release()

    elif menu == "Tabel Absensi":
        st.header("Tabel Absensi")
        if attendance:
            df = pd.concat(attendance, ignore_index=True)
            st.table(df)
        else:
            st.info("Belum ada data absensi.")

    elif menu == "Data Training":
        st.header("Data Training")

        # Dropdown untuk memilih aksi
        action = st.selectbox("Pilih Aksi", ["Tambah Data", "Hapus Data"])

        if action == "Tambah Data":
            st.subheader("Tambah Data Wajah")
            
            # Input nama pemilik wajah
            name = st.text_input("Nama Pemilik Wajah")

            # Membersihkan nama dari karakter yang tidak diizinkan untuk nama file
            cleaned_name = ''.join(e for e in name if e.isalnum())

            # Button untuk mengambil gambar dari webcam
            st.button("Ambil Gambar")


        elif action == "Hapus Data":
            st.subheader("Hapus Data Wajah")

            # Dropdown untuk memilih wajah yang akan dihapus
            name_to_remove = st.selectbox("Pilih Nama", known_face_names)

            # Button untuk menghapus data wajah
            if st.button("Hapus Data"):
                st.success(f"Data wajah telah dihapus.")

# Database untuk menyimpan data wajah yang telah di-training
known_faces = []
known_face_names = []

# Database untuk menyimpan data absensi
attendance = []

if __name__ == "__main__":
    main()
