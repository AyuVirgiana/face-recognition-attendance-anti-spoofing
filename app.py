#######################################################
import uuid ## random id generator
from streamlit_option_menu import option_menu
import streamlit as st
import os
import shutil
import cv2
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
from test import test
import torch
import datetime
#######################################################

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VISITOR_DB = os.path.join(ROOT_DIR, "visitor_database")
VISITOR_HISTORY = os.path.join(ROOT_DIR, "visitor_history")
COLOR_DARK  = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO   = ['Name']
COLS_ENCODE = [f'v{i}' for i in range(512)]
## Database
data_path       = VISITOR_DB
file_db         = 'visitors_db.csv'         ## To store user information
file_history    = 'visitors_history.csv'    ## To store visitor history information

## Image formats allowed
allowed_image_type = ['.png', 'jpg', '.jpeg']
def initialize_data():
    if os.path.exists(os.path.join(data_path, file_db)):
        # st.info('Database Found!')
        df = pd.read_csv(os.path.join(data_path, file_db))

    else:
        # st.info('Database Not Found!')
        df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
        df.to_csv(os.path.join(data_path, file_db), index=False)

    return df

def add_data_db(df_visitor_details):
    try:
        df_all = pd.read_csv(os.path.join(data_path, file_db))

        if not df_all.empty:
            df_all = pd.concat([df_all,df_visitor_details], ignore_index=False)
            df_all.drop_duplicates(keep='first', inplace=True)
            df_all.reset_index(inplace=True, drop=True)
            df_all.to_csv(os.path.join(data_path, file_db), index=False)
            st.success('Details Added Successfully!')
        else:
            df_visitor_details.to_csv(os.path.join(data_path, file_db), index=False)
            st.success('Initiated Data Successfully!')

    except Exception as e:
        st.error(e)

def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)
def attendance(id, name):
    f_p = os.path.join(VISITOR_HISTORY, file_history)
    # st.write(f_p)

    now = datetime.datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    df_attendace_temp = pd.DataFrame(data={ "id"            : [id],
                                            "visitor_name"  : [name],
                                            "Timing"        : [dtString]
                                            })

    if not os.path.isfile(f_p):
        df_attendace_temp.to_csv(f_p, index=False)
        # st.write(df_attendace_temp)
    else:
        df_attendace = pd.read_csv(f_p)
        df_attendace = pd.concat([df_attendace,df_attendace_temp])
        df_attendace.to_csv(f_p, index=False)

def view_attendace():
    f_p = os.path.join(VISITOR_HISTORY, file_history)
    # st.write(f_p)
    df_attendace_temp = pd.DataFrame(columns=["id",
                                              "visitor_name", "Timing"])

    if not os.path.isfile(f_p):
        df_attendace_temp.to_csv(f_p, index=False)
    else:
        df_attendace_temp = pd.read_csv(f_p)

    df_attendace = df_attendace_temp.sort_values(by='Timing',
                                                 ascending=False)
    df_attendace.reset_index(inplace=True, drop=True)

    st.write(df_attendace)

    if df_attendace.shape[0]>0:
        id_chk  = df_attendace.loc[0, 'id']
        id_name = df_attendace.loc[0, 'visitor_name']

        selected_img = st.selectbox('Search Image using ID',
                                    options=['None']+list(df_attendace['id']))

        avail_files = [file for file in list(os.listdir(VISITOR_HISTORY))
                       if ((file.endswith(tuple(allowed_image_type))) &
                                                                              (file.startswith(selected_img) == True))]

        if len(avail_files)>0:
            selected_img_path = os.path.join(VISITOR_HISTORY,
                                             avail_files[0])
            #st.write(selected_img_path)

            ## Displaying Image
            st.image(Image.open(selected_img_path))

def crop_image_with_ratio(img, height,width,middle):
    h, w = img.shape[:2]
    h=h-h%4
    new_w = int(h / height)*width
    startx = middle - new_w //2
    endx=middle+new_w //2
    if startx<=0:
        cropped_img = img[0:h, 0:new_w]
    elif endx>=w:
        cropped_img = img[0:h, w-new_w:w]
    else:
        cropped_img = img[0:h, startx:endx]
    return cropped_img

################################################### Defining Static Data ###############################################

user_color      = '#000000'
title_webapp    = "Face Recognition Attendance Sytem"

html_temp = f"""
            <div style="background-color:{user_color};padding:12px">
            <h1 style="color:white;text-align:center;font-size: 38px;">{title_webapp}</h1>
            </div>
            """
st.markdown(html_temp, unsafe_allow_html=True)

###################### Defining Static Paths ###################4
if st.sidebar.button('Click to Clear out all the data'):
    ## Clearing Visitor Database
    shutil.rmtree(VISITOR_DB, ignore_errors=True)
    os.mkdir(VISITOR_DB)
    ## Clearing Visitor History
    shutil.rmtree(VISITOR_HISTORY, ignore_errors=True)
    os.mkdir(VISITOR_HISTORY)

if not os.path.exists(VISITOR_DB):
    os.mkdir(VISITOR_DB)

if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)
# st.write(VISITOR_HISTORY)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device,keep_all=True
        )
########################################################################################################################

def main():
    ###################################################
    st.sidebar.header("About")
    st.sidebar.info("This webapp gives a demo of Attendance System"
                    " using 'Face Recognition', 'Anti-spoof', and Streamlit")
    ###################################################
    selected_menu = option_menu(None,
        ['Visitor Validation', 'View Visitor History', 'Add to Database'],
        icons=['camera', "clock-history", 'person-plus'],
        ## icons from website: https://icons.getbootstrap.com/
        menu_icon="cast", default_index=0, orientation="horizontal")

    if selected_menu == 'Visitor Validation':
        ## Generates a Random ID for image storage
        visitor_id = uuid.uuid1()

        ## Reading Camera Image
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()

            # convert image from opened file to np.array
            image_array         = cv2.imdecode(np.frombuffer(bytes_data,
                                                             np.uint8),
                                               cv2.IMREAD_COLOR)
            image_array_copy    = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            # st.image(cv2_img)

            ## Saving Visitor History
            with open(os.path.join(VISITOR_HISTORY,
                                   f'{visitor_id}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())
                st.success('Image Saved Successfully!')

                ## Validating Image
                # Detect faces in the loaded image
                max_faces   = 0
                rois        = []  # region of interests (arrays of face areas)
                aligned=[]
                spoofs=[]
                can=[]
                ## To get location of Face from Image
                face_locations ,prob = mtcnn(image_array,return_prob=True)
                boxes, _ = mtcnn.detect(image_array)
                boxes_int=boxes.astype(int)
                ## To encode Image to numeric format
                if face_locations is not None:
                    for idx, (left,top, right, bottom) in enumerate(boxes_int):
                        img=crop_image_with_ratio(image_array,4,3,(left+right)//2)
                        spoof=test(img,"./resources/anti_spoof_models",device)
                        if spoof<=1:
                            spoofs.append("REAL")
                            can.append(idx)
                        else:
                            spoofs.append("FAKE")
                    print(can)
                ## Generating Rectangle Red box over the Image
                for idx,  (left,top, right, bottom) in enumerate(boxes_int):
                    # Save face's Region of Interest
                    rois.append(image_array[top:bottom, left:right].copy())

                    # Draw a box around the face and label it
                    cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
                    cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(image_array, f"#{idx} {spoofs[idx]}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

                ## Showing Image
                st.image(BGR_to_RGB(image_array), width=720)

                ## Number of Faces identified
                max_faces = len(boxes_int)

                if max_faces > 0:
                    col1, col2 = st.columns(2)

                    # select selected faces in the picture
                    face_idxs = col1.multiselect("Select face#", can,
                                                 default=can)

                    ## Filtering for similarity beyond threshold
                    similarity_threshold = col2.slider('Select Threshold for Similarity',
                                                         min_value=0.0, max_value=3.0,
                                                         value=0.5)
                                                    ## check for similarity confidence greater than this threshold

                    flag_show = False
                
                    if ((col1.checkbox('Click to proceed!')) & (len(face_idxs)>0)):
                        dataframe_new = pd.DataFrame()
                        for idx,loc in enumerate(face_locations) :
                            torch_loc = torch.stack([loc]).to(device)
                            encodesCurFrame = resnet(torch_loc).detach().cpu()
                            aligned.append(encodesCurFrame)
                        ## Iterating faces one by one
                        for face_idx in face_idxs:
                            ## Getting Region of Interest for that Face
                            # roi = rois[face_idx]
                            # st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))

                            # initial database for known faces
                            database_data = initialize_data()
                            # st.write(DB)

                            ## Getting Available information from Database
                            face_encodings  = database_data[COLS_ENCODE].values
                            dataframe       = database_data[COLS_INFO]

                            # Comparing ROI to the faces available in database and finding distances and similarities
                            # st.write(faces)

                            if len(aligned) < 1:
                                ## Face could not be processed
                                st.error(f'Please Try Again for face#{face_idx}!')
                            else:
                                face_to_compare = aligned[face_idx].numpy()
                                ## Comparing Face with available information from database
                                # dataframe['distance'] = [(e1 - face_to_compare).norm().item() for e1 in face_encodings]
                                # dataframe['distance'] = dataframe['distance'].astype(float)

                                dataframe['similarity'] = [np.linalg.norm(e1 - face_to_compare) for e1 in face_encodings]
                                dataframe['similarity'] = dataframe['similarity'].astype(float)


                                dataframe_new = dataframe.drop_duplicates(keep='first')
                                dataframe_new.reset_index(drop=True, inplace=True)
                                dataframe_new.sort_values(by="similarity", ascending=True, inplace=True)

                                dataframe_new = dataframe_new[dataframe_new['similarity'] < similarity_threshold].head(1)
                                dataframe_new.reset_index(drop=True, inplace=True)


                                if dataframe_new.shape[0]>0:
                                    (left,top, right, bottom) = (boxes_int[face_idx])

                                    ## Save Face Region of Interest information to the list
                                    rois.append(image_array_copy[top:bottom, left:right].copy())

                                    # Draw a Rectangle Red box around the face and label it
                                    cv2.rectangle(image_array_copy, (left, top), (right, bottom), COLOR_DARK, 2)
                                    cv2.rectangle(image_array_copy, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(image_array_copy, f"#{dataframe_new.loc[0, 'Name']}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

                                    ## Getting Name of Visitor
                                    name_visitor = dataframe_new.loc[0, 'Name']
                                    attendance(visitor_id, name_visitor)

                                    flag_show = True

                                else:
                                    st.error(f'No Match Found for the given Similarity Threshold! for face#{face_idx}')
                                    st.info('Please Update the database for a new person or click again!')
                                    attendance(visitor_id, 'Unknown')

                        if flag_show == True:
                            st.image(BGR_to_RGB(image_array_copy), width=720)

                else:
                    st.error('No human face detected.')

    if selected_menu == 'View Visitor History':
        view_attendace()

    if selected_menu == 'Add to Database':
        col1, col2, col3 = st.columns(3)

        face_name  = col1.text_input('Name:', '')
        pic_option = col2.radio('Upload Picture',
                                options=["Upload a Picture",
                                         "Take a Picture with Cam"])

        if pic_option == 'Upload a Picture':
            img_file_buffer = col3.file_uploader('Upload a Picture',
                                                 type=allowed_image_type)
            if img_file_buffer is not None:
                # To read image file buffer with OpenCV:
                file_bytes = np.asarray(bytearray(img_file_buffer.read()),
                                        dtype=np.uint8)

        elif pic_option == 'Take a Picture with Cam':
            img_file_buffer = col3.camera_input("Take a Picture with Cam")
            if img_file_buffer is not None:
                # To read image file buffer with OpenCV:
                file_bytes = np.frombuffer(img_file_buffer.getvalue(),
                                           np.uint8)

        if ((img_file_buffer is not None) & (len(face_name) > 1) &
                st.button('Click to Save!')):
            # convert image from opened file to np.array
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            # st.write(image_array)
            # st.image(cv2_img)

            with open(os.path.join(VISITOR_DB,
                                   f'{face_name}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())
                # st.success('Image Saved Successfully!')

            face_locations ,prob = mtcnn(image_array,return_prob=True)
            torch_loc = torch.stack([face_locations[0]]).to(device)
            encodesCurFrame = resnet(torch_loc).detach().cpu()

            df_new = pd.DataFrame(data=encodesCurFrame,
                                  columns=COLS_ENCODE)
            df_new[COLS_INFO] = face_name
            df_new = df_new[COLS_INFO + COLS_ENCODE].copy()

            # st.write(df_new)
            # initial database for known faces
            DB = initialize_data()
            add_data_db(df_new)

#######################################################
if __name__ == "__main__":
    main()
#######################################################
