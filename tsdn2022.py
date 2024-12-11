import streamlit as st
import io
from PIL import Image
from typing import Dict
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import cv2
import pandas as pd
from pathlib import Path
import tempfile
import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

def create_category_index(label_path = "model/custom_label.txt"):
    f = open(label_path)

    category_index = {}
    for i, val in enumerate(f):
        category_index.update({(i): {'id': (i+1), 'name': val.replace("\n","")}})
    f.close()
    return category_index

def make_and_show_inference(pil_image, interpreter, input_details, output_details, category_index, score_thresh):
    
    open_cv_image = np.array(pil_image) 
    img = open_cv_image[:, :, ::-1].copy() 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.resize(img, (640, 640), cv2.INTER_AREA)
    img_rgb = img_rgb.reshape([1, 640, 640, 3])
    img_rgb = (img_rgb-127.5)/127.5

    interpreter.set_tensor(input_details[0]['index'], img_rgb.astype(np.float32))
    interpreter.invoke()

    detections = {
        'detection_scores': interpreter.get_tensor(output_details[0]['index'])[0],
        'detection_boxes': interpreter.get_tensor(output_details[1]['index'])[0],
        'num_detections': interpreter.get_tensor(output_details[2]['index'])[0],
        'detection_classes': (interpreter.get_tensor(output_details[3]['index'])[0]).astype(np.int64)
    }

    scores = detections['detection_scores']
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']

    data_kelas_terdeteksi = []
    imH, imW, _ = img.shape
    for i in np.arange(0,len(classes)):
        if ((scores[i] > score_thresh) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            
            data_kelas_terdeteksi.append(classes[i])
            
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            object_name = category_index[int(classes[i])]['name']
            label = '%s' % (object_name)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
            label_ymin = max(ymin, labelSize[1] + 10) 
            cv2.rectangle(img, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                          cv2.FILLED)  
            cv2.putText(img, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                        2)  

    return(Image.fromarray(img)), data_kelas_terdeteksi

        
def main():
    interpreter = tf.lite.Interpreter(model_path="model/custom_model.tflite")
    st.set_page_config(page_title="Strawberry Counter", layout="wide")

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    category_index = create_category_index()
    input_shape = input_details[0]['shape']
    
    with st.sidebar:
        logo = Image.open('foto/logo.png')
        st.image(logo, use_container_width = True ) 
        
        choose = option_menu("Menu", ["Beranda", "Deteksi", "Kontak"],
                             icons=['house', 'robot','person lines fill'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "#000000"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},

            }
        )

    if choose == "Beranda": 
        col1, col2= st.columns( [0.9, 0.1]) 
        with col1:              
            st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">Informasi Web App</p>', unsafe_allow_html=True)    

            st.markdown("""Distrow merupakan web app yang dapat mendeteksi buah stroberi yang ada dalam sebuah gambar 
                            serta memberikan hasil pengklasifikasiannya menjadi 3 kategori, yaitu buah stroberi mentah, masak, dan rusak. 
                            Selain mendeteksi, web app ini juga menghitung jumlah stroberi yang mentah, masak, dan rusak pada gambar-gambar 
                            yang di-input-kan ke web app. Selain itu, web app juga memberikan output berupa persentase jumlah buah mentah, 
                            masak, dan rusak dari seluruh stroberi yang terdeteksi pada seluruh gambar. Di samping adalah sampel gambar stoberi 
                            untuk setiap kategorinya.""", unsafe_allow_html=True) 


            st.markdown("""Terdapat beberapa hal yang harus diperhatikan terkait kemampuan web app dan gambar yang di-input-kan pada web app untuk 
                        memperoleh hasil yang baik, yaitu:\n 1. Dalam 1 gambar, buah stroberi yang terdeteksi maksimal 50 stroberi.\n 2. Usahakan ukuran 
                        maksimal stroberi pada sebuah gambar adalah 1/9 dari gambar. \n 3. Usahakan pencahayaan/brightness pada gambar cukup. \n 4. Resolusi
                        gambar tidak terlalu rendah.""")

            st.markdown('Berikut adalah manual book web app ini. Silakan download untuk mendapatkan panduan penggunaan web app.')
            with open("panduan/manual_book_distrow.pdf", "rb") as file:
                btn = st.download_button(
                    label="Download file",
                    data=file,
                    file_name="manual_book_distrow.pdf") 
        
        with col2:
            logo = Image.open('foto/mentah.jpg')
            st.image(logo, use_container_width = True )
            st.markdown("""<p style='text-align: center;'>Stroberi mentah</p>""", unsafe_allow_html=True)
            
            st.markdown("#")
            logo = Image.open('foto/masak.jpg')
            st.image(logo, use_container_width = True )
            st.markdown("""<p style='text-align: center;'>Stroberi masak</p>""", unsafe_allow_html=True)
            
            st.markdown("#")
            logo = Image.open('foto/rusak.jpg')
            st.image(logo, use_container_width = True )
            st.markdown("""<p style='text-align: center;'>Stroberi rusak</p>""", unsafe_allow_html=True)  
            
            
    if choose == "Deteksi":
        col1, col_jarak1, col2 = st.columns([0.48, 0.02, 0.5])
        jumlah_masak_semua_gambar = 0
        jumlah_mentah_semua_gambar = 0
        jumlah_rusak_semua_gambar = 0
        data_lengkap_semua_gambar = []
        with col1:         
            st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">Unggah Foto-Foto</p>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Unggah di sini", type=['jpg','png','jpeg'], accept_multiple_files=True)
              
            
        col3, col_jarak2, col5 = st.columns([0.48, 0.02, 0.5])    

        if uploaded_file is not None:
            with col2:  
                st.markdown(""" <style> .font {
                font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
                </style> """, unsafe_allow_html=True)
                st.markdown('<p class="font">Hasil Rangkuman</p>', unsafe_allow_html=True) 
                
                placeholder = st.empty()   
                
            with col3:
                st.markdown('Hasil Deteksi',unsafe_allow_html=True)

            with col5:                
                st.markdown('<p class="font">Grafik-Grafik Rangkuman</p>', unsafe_allow_html=True)
                placeholder2 = st.empty()   

            foto_ke_x = 0                
            for uploaded_gambar in uploaded_file:
                image = Image.open(uploaded_gambar)
                image = image.convert('RGB') 
                fp = Path(uploaded_gambar.name)
                            
                try:
                    gambar_terdeteksi, data_kelas_terdeteksi_per_kelas = make_and_show_inference(image, interpreter, input_details, output_details, category_index, score_thresh=0.5)                                     
                except:
                    gambar_terdeteksi = image
                    data_kelas_terdeteksi_per_kelas = []
                jumlah_masak_per_gambar = 0
                jumlah_mentah_per_gambar = 0
                jumlah_rusak_per_gambar = 0                
                    
                for i in np.arange(0,len(data_kelas_terdeteksi_per_kelas)):  
                    if data_kelas_terdeteksi_per_kelas[i] == 0:
                        jumlah_masak_per_gambar += 1
                        jumlah_masak_semua_gambar += 1
                    if data_kelas_terdeteksi_per_kelas[i] == 1:
                        jumlah_mentah_per_gambar += 1
                        jumlah_mentah_semua_gambar += 1
                    if data_kelas_terdeteksi_per_kelas[i] == 2:
                        jumlah_rusak_per_gambar += 1
                        jumlah_rusak_semua_gambar += 1
     
                with col3:
                    foto_ke_x +=1
                    st.write('Foto ke ' + str(foto_ke_x) + ' : '+ str(fp))
                    st.write("Jumlah stroberi masak  : " + str(jumlah_masak_per_gambar) + " buah") 
                    st.write("Jumlah stroberi mentah : " + str(jumlah_mentah_per_gambar)+ " buah")
                    st.write("Jumlah stroberi rusak  : " + str(jumlah_rusak_per_gambar) + " buah")
                    data_lengkap_semua_gambar.append(jumlah_masak_per_gambar + jumlah_mentah_per_gambar +jumlah_rusak_per_gambar)
                    
                    with st.expander(str(fp)):
                        st.image(gambar_terdeteksi, use_container_width = True ) 
                    st.markdown('#')             
                   
            jumlah_objek_terdeteksi = jumlah_masak_semua_gambar + jumlah_mentah_semua_gambar + jumlah_rusak_semua_gambar                                       
                   
            with col2: 
                placeholder.empty()                    
                with placeholder.container():   
                    st.write("Jumlah stroberi terdeteksi pada seluruh gambar: " + str(jumlah_objek_terdeteksi) + " buah")
                    st.write("Jumlah stroberi masak  : " + str(jumlah_masak_semua_gambar) + " buah") 
                    st.write("Jumlah stroberi mentah : " + str(jumlah_mentah_semua_gambar)+ " buah")
                    st.write("Jumlah stroberi rusak  : " + str(jumlah_rusak_semua_gambar) + " buah")
                    st.write("#")
                    
            if jumlah_masak_semua_gambar != 0 or jumlah_mentah_semua_gambar !=0 or jumlah_rusak_semua_gambar != 0:
                with col5:
                    placeholder2.empty()
                    st.markdown('Grafik Jumlah Stroberi Seluruh Gambar',unsafe_allow_html=True)                    
                    sumbu_x = ["Masak", "Mentah", "Rusak"]
                    sumbu_y = [jumlah_masak_semua_gambar, jumlah_mentah_semua_gambar, jumlah_rusak_semua_gambar]
                    chart_data = pd.DataFrame({'Kategori Stroberi':sumbu_x, 'Jumlah Stroberi':sumbu_y}).set_index('Kategori Stroberi')
                    st.bar_chart(data=chart_data, use_container_width=True)                 
                    st.markdown('#')        
 
                    st.markdown('Grafik Persentase Setiap Kategori Stroberi Seluruh Gambar', unsafe_allow_html=True)   
                    fig = plt.figure(figsize=(8,8))
                    fig.patch.set_alpha(0)
                    ax = fig.subplots()
                    sumbu_y_baru = []
                    for ye in sumbu_y:
                        if ye != 0:
                            sumbu_y_baru.append(ye)
                            
                    if len(sumbu_y_baru) == 1:
                        ax.pie(sumbu_y_baru, labels = [' '], autopct="%.2f%%", colors = ['pink'], frame=False)
                        
                    if len(sumbu_y_baru) == 2:
                        ax.pie(sumbu_y_baru, labels = [' ', ' '], autopct="%.2f%%", colors = ['pink', 'green'], frame=False)
                        
                    if len(sumbu_y_baru) == 3:
                        ax.pie(sumbu_y_baru, labels = [' ', ' ', ' '], autopct="%.2f%%", colors = ['pink', 'green', 'orange'], frame=False)                    
                    
                    ax.legend(loc='upper right', labels = sumbu_x)
                    st.pyplot(fig)              
                    st.markdown('#')             
                    
                    st.markdown('Grafik Jumlah Stroberi Setiap Gambar',unsafe_allow_html=True)                    
                    sumbu_x = np.arange(1,len(data_lengkap_semua_gambar)+1)
                    sumbu_y = data_lengkap_semua_gambar
                    fig2 = plt.figure(figsize=(8,8))
                    fig2.patch.set_alpha(0.2)
                    ax2 = fig2.subplots()
                    ax2.patch.set_alpha(0.01)
                    ax2.plot(sumbu_x, sumbu_y, linestyle='--', marker='o', color='white') 
                    ax2.set_xlabel('Foto ke-x')                    
                    ax2.set_ylabel('Jumlah stroberi yang terdeteksi')
                    st.pyplot(fig2)                    
                    st.markdown('#')  

                
    if choose == "Kontak":
        st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
        
        st.markdown('<p class="font">Hubungi Kami!</p>', unsafe_allow_html=True)
        st.markdown("""Apabila anda memiliki pertanyaan, masukan, kritik, atau ajakan kerja sama
                silakan hubungi kami melalui nomor telephone atau email berikut.""", unsafe_allow_html=True)
        
        st.markdown("WA/Telephone: 0895401651437", unsafe_allow_html=True)
        st.markdown("Email: 632022002@student.uksw.edu", unsafe_allow_html=True)         
main()
