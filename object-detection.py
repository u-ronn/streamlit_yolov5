import os, cv2, torch, shutil
import pandas as pd
import streamlit as st
from collections import Counter

# upload file
move_dir = "up_move"
if os.path.exists(move_dir) is False:
    os.mkdir(move_dir)
else:
    try:
        shutil.rmtree(move_dir)
        os.mkdir(move_dir)
    except:
        pass
# title
st.set_page_config(page_title="AIcoordinator v1.0",layout="wide")
st.title(":movie_camera:""VideoFile Object Detection")

# sidebar
st.sidebar.title("AI coordinator")
st.sidebar.markdown("---")

# model load
model_select = st.sidebar.selectbox(
    "model selection",
    ("yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x", "yolov5x6"))
model = torch.hub.load('ultralytics/yolov5', model_select)
st.sidebar.markdown("---")

threshold = st.sidebar.slider("threshold", min_value= 0.01, max_value=1.0, value=0.2)
st.sidebar.markdown("---")

# get class name
yolov5_list =[]
for objectname in model.names.values():
    yolov5_list.append(objectname)
yolov5_list.append("all object")

# select class
class_name = yolov5_list
class_id = []
sel_class = st.sidebar.multiselect("Select The Custom Classes", list(class_name), default="all object")
for each in sel_class:
    class_id.append(class_name.index(each))
st.sidebar.markdown("---")

col1, col2 = st.columns(2)
with col1:
    file = st.file_uploader('Please upload your video', type=['mp4', 'm4v', 'avi', 'mov'])
with col2:
    st.markdown("---")
    progress_placeholder = st.empty()
    progress_bar_placeholder = st.empty()
    progress_finish = st.empty()

col1, col2 = st.columns(2)
with col1:
    stframe = st.empty()
with col2:
    total_placeholder = st.empty()
    frame_placeholder = st.empty()
    
if file:
    # save movie
    img_path = os.path.join(move_dir + "/" + file.name)
    with open(img_path, 'wb') as f:
        f.write(file.read())

    camera = cv2.VideoCapture(img_path)
    total_frame = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

    total_cls_count_list = []
    frame_count = 0
    while True:
        # get video frame
        ret, frame = camera.read()
        frame_count += 1 
        if not ret:
            # print("Can't receive frame (stream end?). Exiting ...")
            break

        if class_id != [80]:
            model.classes = class_id
        model.conf = threshold

         # 解像度を縮小するとき
        if frame.shape[0] >= 1400:
            frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)
        elif frame.shape[0] >= 1080:
            frame = cv2.resize(frame, dsize=None, fx=0.4, fy=0.4)
        elif frame.shape[0] >= 600:
            frame = cv2.resize(frame, dsize=None, fx=0.7, fy=0.7)
        elif frame.shape[0] >= 481:
            frame = cv2.resize(frame, dsize=None, fx=0.9, fy=0.9)

        results = model(frame)
        results.render()

        save_image_flg = 0
        cls_count_list = []
        for *box, conf, cls in results.xyxy[0]:
            total_cls_count_list.append(model.names[int(cls)])
            cls_count_list.append(model.names[int(cls)])
            save_image_flg = 1

        st_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            stframe.image(st_show)
        with col2:
            # total frame count
            total_df = pd.DataFrame(Counter(total_cls_count_list).most_common(5),columns=("object","total frame count(top 5)"))
            total_df.index = total_df.index + 1

            # 1 frame count
            frame_df = pd.DataFrame(Counter(cls_count_list).most_common(3),columns=("object","1 frame count(top 3)"))
            frame_df.index = frame_df.index + 1
            
            total_placeholder.table(total_df)
            frame_placeholder.table(frame_df)

        progress_placeholder.write(str(frame_count) + "-" + str(total_frame))
        progress_bar_placeholder.progress(frame_count/total_frame)

    progress_finish.write(":mag::mag::mag::mag::mag::mag:finish:mag::mag::mag::mag::mag::mag:")