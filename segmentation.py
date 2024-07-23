import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from streamlit_image_comparison import image_comparison
import torch
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import SVC

model = YOLO('yolov8m-seg1.pt')

st.title("Локализация углеродных зерен в срезе металлов по микроснимкам")
st.subheader("Сегментация углеродных зерен и ванн")
st.write("Анализ микроструктуры металлов является критически важным для понимания свойств и поведения материалов. "
         "Традиционные методы анализа микроструктуры, основанные на ручном осмотре микроснимков, трудоемки, "
         "субъективны и подвержены ошибкам. Потому была разработана данная нейросетевая модель локализации углеродных"
         "зерен в срезе металла по микроснимкам для оптимизации процесса, повышения производительности и "
         "конкурентоспособности предприятий")



def save_uploadedfile(uploadedfile):
    with open(os.path.join("./media-directory/", "selfie.jpg"), "wb") as f:
        f.write(uploadedfile.getbuffer())

def convert_to_jpg(uploaded_image):
    im = Image.open(uploaded_image)
    if im.mode in ("RGBA", "P"):
        im = im.convert("RGB")
    uploaded_image_path = os.path.join(parent_media_path, "uploaded_image.jpg")
    im.save(uploaded_image_path)

st.divider()

st.markdown('')
st.markdown('##### Сегментированная часть')

## Placeholder Image
parent_media_path = "media-directory"
img_file = 'bus.jpg'

## Application States



st.sidebar.write(
      """
Это веб-приложение, которое сегментирует входной микроснимок, построенное на
мощном алгоритме обнаружения объектов YOLOv8.
 Просто загрузите микроснимок, и он будет сегментирован в режиме реального времени.
      """
    )
st.sidebar.divider()
# uploaded_file = st.sidebar.file_uploader("Upload your Image here", type=['png', 'jpeg', 'jpg'])
uploaded_file = st.sidebar.file_uploader("Загрузите JPG/PNG микроснимок", accept_multiple_files=False, type=['jpg', 'png'])
if uploaded_file is not None and uploaded_file.type != ".jpg":
    convert_to_jpg(uploaded_file)
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    new_file_name = "uploaded_image.jpg"
    with open(os.path.join(parent_media_path, new_file_name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    img_file = os.path.join(parent_media_path, new_file_name)
    st.sidebar.success("Файл успешно загружен")
    print(f"Файл успешно загружен в {os.path.abspath(os.path.join(parent_media_path, new_file_name))}")
else:
    st.sidebar.write("Вы используете изображение-плейсхолдер, загрузите свое изображение (пока в формате .jpg) для изучения")

# def make_segmentation(img_file):
results = model(img_file)
img = cv2.imread(img_file)
names_list = []
result = results[0]

seg_classes = list(result.names.values())
# seg_classes = ["door", "insulator", "wall", "window"]


for result in results:

    masks = result.masks.data
    boxes = result.boxes.cpu().numpy()
    numCols = len(boxes)
    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    conf = 0.5
    if numCols > 0:
        cols = st.columns(numCols)
    else:
        print(f"Количество найденных объектов: {numCols}")
        st.warning("Невозможно локализировать объекты - Пожалуйста, выберете фото лучшего качества")
    for mask, box in zip(result.masks.xy, result.boxes):
        predicted_name = result.names[int(box.cls[0])]
        if predicted_name == "Bath":
            points = np.int32([mask])
            segmentation_map = np.zeros_like(img)
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.fillPoly(img, points, (255, 0, 0))


        elif predicted_name == "Carbon":
            points = np.int32([mask])
            segmentation_map = np.zeros_like(img)
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.fillPoly(img, points, (255, 55, 255))
            cv2.addWeighted(img, 1, segmentation_map, 0.5, 0, img)


    # st.image(rect)
    # render image-comparison

    st.markdown('')
    st.markdown('##### Слайдер загруженного микроснимка и сегментации')
    image_comparison(
        img1=img_file,
        img2=img,
        label1="Оригинал",
        label2="Сегментированный микроснимок",
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )
    for i, box in enumerate(boxes):
        r = box.xyxy[0].astype(int)
        crop = img[r[1]:r[3], r[0]:r[2]]
        predicted_name = result.names[int(box.cls[0])]
        names_list.append(predicted_name)
        with cols[i]:
            st.write(str(predicted_name) + ".jpg")
            st.image(crop)

st.sidebar.divider()
st.sidebar.markdown('')
st.sidebar.markdown('#### Распределение выявленных объектов')

# Boolean to resize the dataframe, stored as a session state variable
if len(names_list) > 0:
    df_x = pd.DataFrame(names_list)
    summary_table = df_x[0].value_counts().rename_axis('Названия классов').reset_index(name='Количество')
    st.sidebar.dataframe(summary_table)
else:
    st.sidebar.warning("Невозможно локализировать объекты - Пожалуйста, выберете фото лучшего качества")

st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.sidebar.divider()
