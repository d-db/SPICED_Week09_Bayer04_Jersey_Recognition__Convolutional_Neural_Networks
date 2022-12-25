import logging
import os
import pickle

import cv2
import numpy as np
import pyshine as ps

from tensorflow import keras
from keras.utils import load_img, img_to_array
from utils import write_image, key_action, init_cam

# DICT_1 is used for model 1 (see line 28)
DICT_1 = {
    0: "Home 2021/2022",
    1: "Away 2001/2002",
    2: "Home 2006/2007",
    3: "Home 2019/2020",
    4: "Home 2016/2017"
}

# DICT_2 is used for model 2 (see line 28)
DICT_2 = {
    0: "Home 2006/2007",
    1: "Home 2019/2020"
}

# User can chose between the completely self trained CNN model (1) or the pretrained VGG16 (2)
# VGG16 was only trained to distinguish the two most similar jerseys, hence DICT_2 contains only two entries
user_input = int(input("Which model would you like to classify the jerseys with (1: Keras, 2:LogReg VGG16)?: "))

if user_input == 1:
    model = keras.models.load_model("./1_data/model_new")

elif user_input == 2:
    with open('./1_data/log_reg.bin', 'rb') as f:
        log_reg = pickle.load(f)

    base_model = keras.models.load_model("./1_data/logreg_base_model")


def predict_frame(image, user_input):
    # reverse color channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply pre-processing
    image = preprocess_input(image, user_input)

    return image


def preprocess_input(image, user_input):
    img_array = img_to_array(image)
    image = np.expand_dims(img_array, axis=0)

    if user_input == 2:
        out_features_vector = base_model.predict(image)
        image = out_features_vector.reshape((1, 20 * 20 * 512))

    return image


def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def _draw_label(image, text, background_rgb, text_rgb):
    ps.putBText(image, text, text_offset_x=390, text_offset_y=600, vspace=10, hspace=10, font_scale=2.0,
                background_RGB=background_rgb, text_RGB=text_rgb)


if __name__ == "__main__":

    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    logging.getLogger().setLevel(logging.INFO)
   
    # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None

    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            # Draw a [650X650] rectangle into the frame, leave some space for the black border
            offset = 2
            width = 650
            x = 350
            y = 30
            cv2.rectangle(img=frame, 
                          pt1=(x-offset, y-offset),
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 0, 0), 
                          thickness=2
                          )
            
            # get key event
            key = key_action()

            image = frame[y:y+width, x:x+width, :]
            image = predict_frame(image, user_input)

            if user_input == 1:
                y_pred = model.predict(image)
                if np.argmax(y_pred) != 5:
                    _draw_label(frame, DICT_1[np.argmax(y_pred)], (255, 255, 255), (0, 0, 0))
            else:
                y_pred = log_reg.predict(image)
                _draw_label(frame, DICT_2[y_pred[0]], (255, 255, 255), (0, 0, 0))

            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)

    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()
