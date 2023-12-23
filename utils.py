# Mengimpor modul yang diperlukan
import face_recognition as frg
import pickle as pkl 
import os 
import cv2 
import numpy as np
import yaml
from collections import defaultdict

# Menggunakan defaultdict untuk menyimpan informasi pengenal wajah
information = defaultdict(dict)

# Membaca konfigurasi dari file YAML
cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
DATASET_DIR = cfg['PATH']['DATASET_DIR']
PKL_PATH = cfg['PATH']['PKL_PATH']

# Fungsi untuk mendapatkan database pengenal wajah
def get_databse():
    with open(PKL_PATH,'rb') as f:
        database = pkl.load(f)
    return database

# Fungsi untuk mengenali wajah 
def recognize(image,TOLERANCE): 
    database = get_databse()
    known_encoding = [database[id]['encoding'] for id in database.keys()] 
    name = 'Unknown'
    id = 'Unknown'
    face_locations = frg.face_locations(image)
    face_encodings = frg.face_encodings(image,face_locations)
    for (top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):
        matches = frg.compare_faces(known_encoding,face_encoding,tolerance=TOLERANCE)
        distance = frg.face_distance(known_encoding,face_encoding)
        name = 'Unknown'
        id = 'Unknown'
        if True in matches:
            match_index = matches.index(True)
            name = database[match_index]['name']
            id = database[match_index]['id']
            distance = round(distance[match_index],2)
            cv2.putText(image,str(distance),(left,top-30),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
        cv2.rectangle(image,(left,top),(right,bottom),(0,255,0),2)
        cv2.putText(image,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
    return image, name, id


# Fungsi untuk mendapatkan informasi dari suatu ID
def get_info_from_name(name): 
    database = get_databse() 
    for idx, person in database.items(): 
        if person['name'] == name: 
            name = person['name']
            image = person['image']
            return name, image, idx       
    return None, None, None

# Fungsi untuk menghapus satu entitas berdasarkan ID
def deleteOne(name):
    database = get_databse()
    id = str(name)
    for key, person in database.items():
        if person['name'] == id:
            del database[key]
            break
    with open(PKL_PATH,'wb') as f:
        pkl.dump(database,f)
    return True


# Fungsi untuk membangun dataset dari gambar-gambar dalam direktori
def build_dataset():
    counter = 0
    for image in os.listdir(DATASET_DIR):
        image_path = os.path.join(DATASET_DIR,image)
        image_name = image.split('.')[0]
        parsed_name = image_name.split('_')
        person_id = parsed_name[0]
        person_name = ' '.join(parsed_name[1:])
        if not image_path.endswith('.jpg'):
            continue
        image = frg.load_image_file(image_path)
        information[counter]['image'] = image 
        information[counter]['id'] = person_id
        information[counter]['name'] = person_name
        information[counter]['encoding'] = frg.face_encodings(image)[0]
        counter += 1

    with open(os.path.join(DATASET_DIR,'database.pkl'),'wb') as f:
        pkl.dump(information,f)
        
