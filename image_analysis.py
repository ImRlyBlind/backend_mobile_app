import base64
import os.path

from pixellib.torchbackend.instance import instanceSegmentation
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
import numpy as np
import pixellib
import pymsgbox
import time
import cv2
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import sqlite3
from io import BytesIO
from PIL import Image
import tensorflow as tf
from uuid import uuid4
from flask_mysqldb import MySQL
import mysql.connector
from config import Config

app = Flask(__name__)
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '12345678',
    'database': 'app_record'
}

isGood = 1
path_to_image = "static/"
list_of_bad = []
global path

UPLOAD_FOLDER = 'static/'
if not os.path.exists(UPLOAD_FOLDER):
   os.makedirs(UPLOAD_FOLDER)


def brightness_analysis(image, dl_h, dl_w, level, percent, numdiff):
    # read color image
    color_img = cv2.imread(path_to_image + image, cv2.IMREAD_COLOR)

    # read gray image
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)

    imgheight = gray_img.shape[0]
    imgwidth = gray_img.shape[1]

    h = gray_img.shape[0] // dl_h
    w = gray_img.shape[1] // dl_w

    # level_range ex. level = 8 is 0-31 32-63 ... 224-255
    brightness_level = 256 // level

    # split_img = Split image
    split_img = cv2.split(gray_img)

    # score all, score light, score dark
    score_proper_light = 0
    score_under_light = 0
    score_over_light = 0

    p1_brightness = ""
    result_brightness = 0

    # 1 img = dl_h*dl_w = 36*64 pixel | h=36, w=64
    for row in range(dl_h):
        count = np.zeros(level)
        for col in range(dl_w):
            score = np.zeros(level)
            score_percent = 0

            # 1 channel = h*w = 36*64 pixel | h=36, w=64
            for row_h in range(h):
                for col_w in range(w):
                    split = split_img[0][row_h + (h * row)][col_w + (w * col)]
                    i = (split // brightness_level)
                    score[i] = score[i] + 1

            score_light = np.zeros(level, dtype=float)
            for x in range(level):
                score_light[x] = round(score[x] / (h * w), 2)

                if x <= level // 2:
                    score_under_light = score_under_light + score_light[x]

                if x > level // 2:
                    score_over_light = score_over_light + score_light[x]

                if score_light[x] >= percent:
                    score_percent = score_percent + 1

            if score_percent >= numdiff:
                score_proper_light = score_proper_light + 1

    score_all_percent = float(score_proper_light / (dl_h * dl_w)) * 100

    global list_of_bad

    if score_all_percent >= 40:
        result_brightness = 1
        p1_brightness = "Proper Light (OK)"

    elif score_over_light > score_under_light:
        result_brightness = 0
        p1_brightness = "Over Light"
        # pymsgbox.alert('Over Light!', str(image))

    else:
        result_brightness = 0
        p1_brightness = "Under Light"
        # pymsgbox.alert('Under Light!', str(image))

    print("Brightness Analysis :")
    isBadLight = "Light: " + p1_brightness
    list_of_bad.append(isBadLight)
    print(p1_brightness, "| dispersed value ", score_all_percent, "%")
    print("------------------------------------------------------------------------------")

    array_bn = {"P1_brightness": p1_brightness, "Result_brightness": result_brightness,
                "Score_all_percent": score_all_percent}
    return array_bn


def ins_segimg(image, dl_h, dl_w):
    # read people image
    peoimg = cv2.imread(path_to_image + image, cv2.IMREAD_COLOR)

    # dl_h = Draw Line Height, dl_w = Draw Line width = dl_h*dl_w = 20*20
    imgheight = peoimg.shape[0]
    imgwidth = peoimg.shape[1]

    h = peoimg.shape[0] // dl_h
    w = peoimg.shape[1] // dl_w

    # segmentation Image
    segimg = instanceSegmentation()
    segimg.load_model("pointrend_resnet50.pkl")
    target = segimg.select_target_classes(person=True, mouse=True, keyboard=True, laptop=True, tv=True)
    outputsegimg = segimg.segmentImage(path_to_image + image, segment_target_classes=target)

    # set rois, set class, set score
    set_boxes = outputsegimg[0]["boxes"]
    set_class = outputsegimg[0]["class_ids"]
    set_score = outputsegimg[0]["scores"]

    p2_peo_segimg = ""
    p2_screen_segimg = ""
    result_peo_segimg = 0
    result_screen_segimg = 0

    sr_people = []
    sr_screen = []
    sr_laptop = []
    sr_mouse = []
    sr_keyboard = []

    score_people = 0
    score_screen = 0
    score_laptop = 0
    score_mouse = 0
    score_keyboard = 0

    count_people = 0
    count_screen = 0
    count_laptop = 0

    for a in range(len(set_class)):
        # scpeo is setclass people, srpeo is setrois people
        if set_class[a] == 0: #rcnn 1 person
            if score_people < set_score[a]:
                score_people = set_score[a]
                count_people = count_people + 1
                sr_people = set_boxes[a]
                p_top = sr_people[0]
                p_left = sr_people[1]
                p_bottom = sr_people[2]
                p_right = sr_people[3]

        elif set_class[a] == 62: #rcnn 63 screen/tv
            if score_screen < set_score[a]:
                score_screen = set_score[a]
                count_screen = count_screen + 1
                sr_screen = set_boxes[a]
                screen_top = sr_screen[0]
                screen_left = sr_screen[1]
                screen_bottom = sr_screen[2]
                screen_right = sr_screen[3]

        elif set_class[a] == 63: #rcnn 64 laptop
            if score_laptop < set_score[a]:
                score_laptop = set_score[a]
                count_laptop = count_laptop + 1
                sr_laptop = set_boxes[a]
                lt_top = sr_laptop[0]
                lt_left = sr_laptop[1]
                lt_bottom = sr_laptop[2]
                lt_right = sr_laptop[3]

        elif set_class[a] == 64: #rcnn 65 mouse
            if score_mouse < set_score[a]:
                score_mouse = set_score[a]
                sr_mouse = set_boxes[a]
                m_top = sr_mouse[0]
                m_left = sr_mouse[1]
                m_bottom = sr_mouse[2]
                m_right = sr_mouse[3]

        elif set_class[a] == 66: #rcnn 67 keyboard
            if score_keyboard < set_score[a]:
                score_keyboard = set_score[a]
                sr_keyboard = set_boxes[a]
                kb_top = sr_keyboard[0]
                kb_left = sr_keyboard[1]
                kb_bottom = sr_keyboard[2]
                kb_right = sr_keyboard[3]

    global list_of_bad

    if len(sr_people) > 0 and count_people == 1:
        result_peo_segimg = 1
        p2_peo_segimg = "Found (OK)"

    elif len(sr_people) > 0 and count_people > 1:
        result_peo_segimg = 0
        p2_peo_segimg = "Found 2 People"
        # pymsgbox.alert('Found 2 People!', str(image))

    elif len(sr_people) == 0 and count_people == 0:
        result_peo_segimg = 0
        p2_peo_segimg = "People Not Found"
        # pymsgbox.alert('People Not Found!', str(image))

    if len(sr_screen) > 0 and len(sr_laptop) > 0:
        result_screen_segimg = 0
        p2_screen_segimg = "Found 2 screen"
        # pymsgbox.alert('Found 2 Screen!', str(image))

    elif len(sr_screen) > 0 and len(sr_laptop) == 0 and count_screen == 1:
        result_screen_segimg = 1
        p2_screen_segimg = "Found (OK)"

    elif len(sr_screen) == 0 and len(sr_laptop) > 0 and count_laptop == 1:
        result_screen_segimg = 1
        p2_screen_segimg = "Found (OK)"

    elif len(sr_laptop) == 0 and len(sr_screen) == 0:
        result_screen_segimg = 0
        p2_screen_segimg = "Screen Not Found"
        # pymsgbox.alert('Screen Not Found!', str(image))

    else:
        result_screen_segimg = 0
        p2_screen_segimg = "Found 2 screen"
        # pymsgbox.alert('Found 2 Screen!', str(image))

    isBadSegment = "People: " + p2_peo_segimg + "\n" + "Screeen/Laptop: " + p2_screen_segimg
    print("Instance Segmentation :")
    print(p2_peo_segimg)
    print(p2_screen_segimg)
    list_of_bad.append(isBadSegment)
    print("------------------------------------------------------------------------------")

    array_segimg = {"People": sr_people, "Screen": sr_screen, "Laptop": sr_laptop, "Mouse": sr_mouse,
                    "Keyboard": sr_keyboard, "P2_peo_segimg": p2_peo_segimg, "Result_peo_segimg": result_peo_segimg,
                    "P2_screen_segimg": p2_screen_segimg, "Result_screen_segimg": result_screen_segimg}
    return array_segimg


def human_com_location(image, dl_h, dl_w, segmentation):
    # read people image
    peoimg = cv2.imread(path_to_image + image, cv2.IMREAD_COLOR)



    # dl_h = Draw Line Height, dl_w = Draw Line width = dl_h*dl_w = 20*20
    imgheight = peoimg.shape[0]
    imgwidth = peoimg.shape[1]

    h = peoimg.shape[0] // dl_h
    w = peoimg.shape[1] // dl_w

    sr_people = []
    sr_screen = []
    sr_laptop = []

    score_people = 0
    score_screen = 0
    score_laptop = 0

    p3_peo_location = ""
    p3_screen_location = ""
    p3_lt_location = ""

    result_segimg = 0
    result_peo_location = 0
    result_screen_location = 0
    result_lt_location = 0

    sr_people = segmentation['People']
    if len(sr_people) > 0:
        p_top = sr_people[1]
        p_left = sr_people[0]
        p_bottom = sr_people[3]
        p_right = sr_people[2]

    sr_screen = segmentation['Screen']
    if len(sr_screen) > 0:
        screen_top = sr_screen[1]
        screen_left = sr_screen[0]
        screen_bottom = sr_screen[3]
        screen_right = sr_screen[2]

    sr_laptop = segmentation['Laptop']
    if len(sr_laptop) > 0:
        lt_top = sr_laptop[1]
        lt_left = sr_laptop[0]
        lt_bottom = sr_laptop[3]
        lt_right = sr_laptop[2]

    global list_of_bad

    if len(sr_people) > 0 and len(sr_laptop) > 0 and len(sr_screen) == 0:
        if abs(lt_left - p_right) < abs(lt_right - p_left):
            result_peo_location = 1
            result_lt_location = 1
            p3_peo_location = "Left (OK)"
            p3_lt_location = "Right (OK)"

        elif abs(lt_left - p_right) > abs(lt_right - p_left):
            result_peo_location = 1
            result_lt_location = 1
            p3_peo_location = "Right (OK)"
            p3_lt_location = "Left (OK)"

    elif len(sr_people) == 0 and len(sr_laptop) > 0 and len(sr_screen) == 0:
        if abs(lt_right - imgwidth) < lt_right:
            result_peo_location = 0
            result_lt_location = 1
            p3_peo_location = ""
            # pymsgbox.alert('People Not Found!', str(image))

            p3_lt_location = "Right (OK)"

        elif abs(lt_left - imgwidth) > lt_left:
            result_peo_location = 0
            result_lt_location = 1
            p3_peo_location = ""
            # pymsgbox.alert('People Not Found!', str(image))

            p3_lt_location = "Left (OK)"

    elif len(sr_people) > 0 and len(sr_laptop) == 0 and len(sr_screen) == 0:
        if p_left < abs(p_left - imgwidth) and p_right < abs(p_right - imgwidth):
            result_peo_location = 1
            p3_peo_location = "Left (OK)"
            p3_screen_location = ""
            # pymsgbox.alert('Screen Not Found!', str(image))

        else:
            result_peo_location = 1
            p3_peo_location = "Right (OK)"
            p3_screen_location = ""
            # pymsgbox.alert('Screen Not Found!', str(image))

    if len(sr_people) > 0 and len(sr_screen) > 0 and len(sr_laptop) == 0:
        if abs(screen_left - p_right) < abs(screen_right - p_left):
            result_peo_location = 1
            result_screen_location = 1
            p3_peo_location = "Left (OK)"
            p3_screen_location = "Right (OK)"

        elif abs(screen_left - p_right) > abs(screen_right - p_left):
            result_peo_location = 1
            result_screen_location = 1
            p3_peo_location = "Right (OK)"
            p3_screen_location = "Left (OK)"

    elif len(sr_people) == 0 and len(sr_screen) > 0 and len(sr_laptop) == 0:
        if abs(screen_right - imgwidth) < screen_right:
            result_peo_location = 0
            p3_peo_location = ""
            # pymsgbox.alert('People Not Found!', str(image))

            result_screen_location = 1
            p3_screen_location = "Right (OK)"

        elif abs(screen_left - imgwidth) > screen_left:
            result_peo_location = 0
            p3_peo_location = ""
            # pymsgbox.alert('People Not Found!', str(image))

            result_screen_location = 1
            p3_screen_location = "Left (OK)"

    print("Human-Computer Location Analysis :")

    if len(sr_people) > 0:
        print(p3_peo_location + "People T-B = " + str((p_top // h) + 1) + " - " + str(
            (p_bottom // h) + 1) + " | L-R = " + str((p_left // w) + 1) + " - " + str((p_right // w) + 1))

    else:
        print(p3_peo_location)

    if len(sr_screen) > 0 and len(sr_laptop) == 0:
        print(p3_screen_location + "Screen T-B = " + str((screen_top // h) + 1) + " - " + str(
            (screen_bottom // h) + 1) + " | L-R = " + str((screen_left // w) + 1) + " - " + str(
            (screen_right // w) + 1))

    elif len(sr_screen) == 0 and len(sr_laptop) > 0:
        print(p3_lt_location + "Laptop T-B = " + str((lt_top // h) + 1) + " - " + str(
            (lt_bottom // h) + 1) + " | L-R = " + str((lt_left // w) + 1) + " - " + str((lt_right // w) + 1))

    elif len(sr_screen) > 0 and len(sr_laptop) > 0:
        result_screen_location = 0
        p3_lt_location = ""
        # pymsgbox.alert('Found 2 Screen!', str(image))
        print(p3_lt_location)

    else:
        print(p3_screen_location)
    isBadLocation = "People Location: " + p3_peo_location + "\n" + "Screen Location: " + p3_screen_location + "\n" + "Laptop Location: " + p3_lt_location
    list_of_bad.append(isBadLocation)
    print("------------------------------------------------------------------------------")

    array_h_c_location = {"P3_peo_location": p3_peo_location, "Result_peo_location": result_peo_location,
                          "P3_screen_location": p3_screen_location, "Result_screen_location": result_screen_location,
                          "P3_lt_location": p3_lt_location, "Result_lt_location": result_lt_location}
    return array_h_c_location


def face_direction(image, dl_h, dl_w, segmentation):
    # read people image
    peoimg = cv2.imread(path_to_image + image, cv2.IMREAD_COLOR)

    # dl_h = Draw Line Height, dl_w = Draw Line width = dl_h*dl_w = 20*20
    imgheight = peoimg.shape[0]
    imgwidth = peoimg.shape[1]

    h = peoimg.shape[0] // dl_h
    w = peoimg.shape[1] // dl_w

    sr_screen = []
    sr_laptop = []
    setfaces = {}

    score_screen = 0
    score_laptop = 0
    result_face = 0

    p4_face = ""

    sr_screen = segmentation['Screen']
    if len(sr_screen) > 0:
        screen_top = sr_screen[1]
        screen_left = sr_screen[0]
        screen_bottom = sr_screen[3]
        screen_right = sr_screen[2]

    sr_laptop = segmentation['Laptop']
    if len(sr_laptop) > 0:
        lt_top = sr_laptop[1]
        lt_left = sr_laptop[0]
        lt_bottom = sr_laptop[3]
        lt_right = sr_laptop[2]

    # Detect Face
    pltimg = plt.imread(path_to_image + image)
    detector = MTCNN()

    faces = detector.detect_faces(pltimg)

    global list_of_bad

    # settype tuple to dict
    if len(faces) > 0:
        setfaces = faces[0]

        for face in faces:
            x, y, width, height = setfaces['box']

        setkeypoint = setfaces['keypoints']
        setle = setkeypoint["left_eye"]
        setre = setkeypoint["right_eye"]
        setn = setkeypoint["nose"]
        setml = setkeypoint["mouth_left"]
        setmr = setkeypoint["mouth_right"]

        if len(sr_screen) > 0 and len(sr_laptop) == 0:
            if abs(setn[0] - x) > abs(setn[0] - (x + width)):
                if setle[0] > x + (width // 2) and setre[0] > x + (width // 2) and abs(x - screen_right) > abs(
                        (x + width) - screen_left):
                    result_face = 1
                    p4_face = "FacingCom on the left side(OK)"
                else:
                    result_face = 0
                    p4_face = "Not FacingCom"
                    # pymsgbox.alert('Not FacingCom!', str(image))

            elif abs(setn[0] - x) < abs(setn[0] - (x + width)):
                if setle[0] < x + (width // 2) and setre[0] < x + (width // 2) and abs(x - screen_right) < abs(
                        (x + width) - screen_left):
                    result_face = 1
                    p4_face = "FacingCom on the Right side(OK)"
                else:
                    result_face = 0
                    p4_face = "Not FacingCom"
                    # pymsgbox.alert('Not FacingCom!', str(image))

        elif len(sr_screen) == 0 and len(sr_laptop) > 0:
            if abs(setn[0] - x) > abs(setn[0] - (x + width)):
                if setle[0] > x + (width // 2) and setre[0] > x + (width // 2) and abs(x - lt_right) > abs(
                        (x + width) - lt_left):
                    result_face = 1
                    p4_face = "FacingCom on the left side(OK)"
                else:
                    result_face = 0
                    p4_face = "Not FacingCom"
                    # pymsgbox.alert('Not FacingCom!', str(image))

            elif abs(setn[0] - x) < abs(setn[0] - (x + width)):
                if setle[0] < x + (width // 2) and setre[0] < x + (width // 2) and abs(x - lt_right) < abs(
                        (x + width) - lt_left):
                    result_face = 1
                    p4_face = "FacingCom on the right side(OK)"
                else:
                    result_face = 0
                    p4_face = "Not FacingCom"
                    # pymsgbox.alert('Not FacingCom!', str(image))
        else:
            if abs(setn[0] - x) < abs(setn[0] - (x + width)):
                if setle[0] < x + (width // 2) and setre[0] < x + (width // 2):
                    result_face = 1
                    p4_face = "Face on the left side(OK)"


            elif abs(setn[0] - x) > abs(setn[0] - (x + width)):
                if setle[0] > x + (width // 2) and setre[0] > x + (width // 2):
                    result_face = 1
                    p4_face = "Face on the right side (OK)"

    else:
        result_face = 0
        p4_face = "Face Not Found"
        # pymsgbox.alert('Face Not Found!', str(image))

    print("Face Direction Analysis :")
    print(p4_face)
    isBadFacing = "Face direction: " + p4_face + "\n"
    list_of_bad.append(isBadFacing)
    print("------------------------------------------------------------------------------")

    array_face = {"Setfaces": setfaces, "P4_face": p4_face, "Result_face": result_face}
    return array_face


def position_proportion(image, dl_h, dl_w, segmentation):
    # read people image
    peoimg = cv2.imread(path_to_image + image, cv2.IMREAD_COLOR)

    # dl_h = Draw Line Height, dl_w = Draw Line width = dl_h*dl_w = 20*20
    imgheight = peoimg.shape[0]
    imgwidth = peoimg.shape[1]

    h = peoimg.shape[0] // dl_h
    w = peoimg.shape[1] // dl_w

    result_peo_phase = 0
    result_screen_phase = 0
    result_lt_phase = 0
    result_proport = 0

    p5_peo_phase = ""
    p5_screen_phase = ""
    p5_lt_phase = ""
    p5_proport = ""

    # print("From segmentation", segmentation)

    sr_people = segmentation['People']
    if len(sr_people) > 0:
        p_top = sr_people[1]
        p_left = sr_people[0]
        p_bottom = sr_people[3]
        p_right = sr_people[2]

    sr_screen = segmentation['Screen']
    if len(sr_screen) > 0:
        screen_top = sr_screen[1]
        screen_left = sr_screen[0]
        screen_bottom = sr_screen[3]
        screen_right = sr_screen[2]

    sr_laptop = segmentation['Laptop']
    if len(sr_laptop) > 0:
        lt_top = sr_laptop[1]
        lt_left = sr_laptop[0]
        lt_bottom = sr_laptop[3]
        lt_right = sr_laptop[2]

    global list_of_bad

    if len(sr_people) == 0:
        result_peo_phase = 0
        p5_peo_phase = ""
        # pymsgbox.alert('People Not Found!', str(image))

    elif len(sr_screen) == 0 and len(sr_laptop) == 0:
        result_screen_phase = 0
        p5_screen_phase = ""
        # pymsgbox.alert('Screen Not Found!', str(image))

    if len(sr_people) > 0:
        if 90 <= ((((p_bottom // h) + 1) - ((p_top // h) + 1)) * (((p_right // w) + 1) - ((p_left // w) + 1))) <= 250:
            result_proport = 1
            p5_proport = "Normal (OK)"

        elif 90 > ((((p_bottom // h) + 1) - ((p_top // h) + 1)) * (((p_right // w) + 1) - ((p_left // w) + 1))):
            result_proport = 0
            p5_proport = "Far"
            # pymsgbox.alert('Proportion : Far!', str(image))

        elif 250 < ((((p_bottom // h) + 1) - ((p_top // h) + 1)) * (((p_right // w) + 1) - ((p_left // w) + 1))):
            result_proport = 0
            p5_proport = "Near"
            # pymsgbox.alert('Proportion : Near!', str(image))

        if len(sr_screen) > 0 and len(sr_laptop) == 0:
            if abs(screen_left - p_right) < abs(screen_right - p_left):
                if p_top > 1 and p_right < imgwidth and 1 <= ((p_top // h) + 1) <= 9 and 2 <= (
                        (p_left // w) + 1) <= 7 and 8 <= ((p_right // w) + 1) <= 16:
                    result_peo_phase = 1
                    p5_peo_phase = "OK"
                else:
                    result_peo_phase = 0
                    p5_peo_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : People!', str(image))

                if 3 <= ((screen_top // h) + 1) <= 12 and 10 <= ((screen_bottom // h) + 1) <= 19 and 10 <= (
                        (screen_left // w) + 1) <= 14 and 15 <= ((screen_right // w) + 1) <= 19:
                    result_screen_phase = 1
                    p5_screen_phase = "OK"
                else:
                    result_screen_phase = 0
                    p5_screen_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : Screen!', str(image))

            elif abs(screen_left - p_right) > abs(screen_right - p_left):
                if p_top > 1 and p_right < imgwidth and 1 <= ((p_top // h) + 1) <= 9 and 5 <= (
                        (p_left // w) + 1) <= 13 and 14 <= ((p_right // w) + 1) <= 19:
                    result_peo_phase = 1
                    p5_peo_phase = "OK"
                else:
                    result_peo_phase = 0
                    p5_peo_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : People!', str(image))

                if 3 <= ((screen_top // h) + 1) <= 12 and 10 <= ((screen_bottom // h) + 1) <= 19 and 2 <= (
                        (screen_left // w) + 1) <= 6 and 7 <= ((screen_right // w) + 1) <= 11:
                    result_screen_phase = 1
                    p5_screen_phase = "OK"
                else:
                    result_screen_phase = 0
                    p5_screen_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : Screen!', str(image))

        if len(sr_laptop) > 0 and len(sr_screen) == 0:
            if abs(lt_left - p_right) < abs(lt_right - p_left):
                if p_top > 1 and p_right < imgwidth and 1 <= ((p_top // h) + 1) <= 9 and 2 <= (
                        (p_left // w) + 1) <= 7 and 8 <= ((p_right // w) + 1) <= 16:
                    result_peo_phase = 1
                    p5_peo_phase = "OK"

                else:
                    result_peo_phase = 0
                    p5_peo_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : People!', str(image))

                if 3 <= ((lt_top // h) + 1) <= 12 and 10 <= ((lt_bottom // h) + 1) <= 19 and 10 <= (
                        (lt_left // w) + 1) <= 14 and 15 <= ((lt_right // w) + 1) <= 19:
                    result_lt_phase = 1
                    p5_lt_phase = "OK"

                else:
                    result_lt_phase = 0
                    p5_lt_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : Laptop!', str(image))

            elif abs(lt_left - p_right) > abs(lt_right - p_left):
                if p_top > 1 and p_right < imgwidth and 1 <= ((p_top // h) + 1) <= 9 and 5 <= (
                        (p_left // w) + 1) <= 13 and 14 <= ((p_right // w) + 1) <= 19:
                    result_peo_phase = 1
                    p5_peo_phase = "OK"

                else:
                    result_peo_phase = 0
                    p5_peo_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : People!', str(image))

                if 3 <= ((lt_top // h) + 1) <= 12 and 10 <= ((lt_bottom // h) + 1) <= 19 and 2 <= (
                        (lt_left // w) + 1) <= 6 and 7 <= ((lt_right // w) + 1) <= 11:
                    result_lt_phase = 1
                    p5_lt_phase = "OK"

                else:
                    result_lt_phase = 0
                    p5_lt_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : Laptop!', str(image))

        if len(sr_screen) == 0 and len(sr_laptop) == 0:
            if p_left < abs(p_left - imgwidth) and p_right < abs(p_right - imgwidth):
                if p_top > 1 and p_right < imgwidth and 1 <= ((p_top // h) + 1) <= 9 and 2 <= (
                        (p_left // w) + 1) <= 7 and 8 <= ((p_right // w) + 1) <= 16:
                    result_peo_phase = 1
                    p5_peo_phase = "OK"

                else:
                    result_peo_phase = 0
                    p5_peo_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : People!', str(image))

            else:
                if p_top > 1 and p_right < imgwidth and 1 <= ((p_top // h) + 1) <= 9 and 5 <= (
                        (p_left // w) + 1) <= 13 and 14 <= ((p_right // w) + 1) <= 19:
                    result_peo_phase = 1
                    p5_peo_phase = "OK"

                else:
                    result_peo_phase = 0
                    p5_peo_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : People!', str(image))

        if len(sr_screen) > 0 and len(sr_laptop) > 0:
            if p_left < abs(p_left - imgwidth) and p_right < abs(p_right - imgwidth):
                if p_top > 1 and p_right < imgwidth and 1 <= ((p_top // h) + 1) <= 9 and 2 <= (
                        (p_left // w) + 1) <= 7 and 8 <= ((p_right // w) + 1) <= 16:
                    result_peo_phase = 1
                    p5_peo_phase = "OK"

                else:
                    result_peo_phase = 0
                    p5_peo_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : People!', str(image))

            else:
                if p_top > 1 and p_right < imgwidth and 1 <= ((p_top // h) + 1) <= 9 and 5 <= (
                        (p_left // w) + 1) <= 13 and 14 <= ((p_right // w) + 1) <= 19:
                    result_peo_phase = 1
                    p5_peo_phase = "OK"

                else:
                    result_peo_phase = 0
                    p5_peo_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : People!', str(image))

            result_screen_phase = 0
            p5_screen_phase = ""
            # pymsgbox.alert('Found 2 screen!', str(image))

    else:
        result_proport = 0
        p5_proport = ""
        # pymsgbox.alert('People Not Found!', str(image))

        if len(sr_screen) > 0 and len(sr_laptop) == 0:
            if screen_left < abs(screen_left - imgwidth) and screen_right < abs(screen_right - imgwidth):
                if 3 <= ((screen_top // h) + 1) <= 12 and 10 <= ((screen_bottom // h) + 1) <= 19 and 2 <= (
                        (screen_left // w) + 1) <= 6 and 7 <= ((screen_right // w) + 1) <= 11:
                    result_screen_phase = 1
                    p5_screen_phase = "OK"
                else:
                    result_screen_phase = 0
                    p5_screen_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : Screen!', str(image))

            else:
                if 3 <= ((screen_top // h) + 1) <= 12 and 10 <= ((screen_bottom // h) + 1) <= 19 and 10 <= (
                        (screen_left // w) + 1) <= 14 and 15 <= ((screen_right // w) + 1) <= 19:
                    result_screen_phase = 1
                    p5_screen_phase = "OK"
                else:
                    result_screen_phase = 0
                    p5_screen_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : Screen!', str(image))

        if len(sr_screen) == 0 and len(sr_laptop) > 0:
            if lt_left < abs(lt_left - imgwidth) and lt_right < abs(lt_right - imgwidth):
                if 3 <= ((lt_top // h) + 1) <= 12 and 10 <= ((lt_bottom // h) + 1) <= 19 and 2 <= (
                        (lt_left // w) + 1) <= 6 and 7 <= ((lt_right // w) + 1) <= 11:
                    result_lt_phase = 1
                    p5_lt_phase = "OK"

                else:
                    result_lt_phase = 0
                    p5_lt_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : Laptop!', str(image))

            else:
                if 3 <= ((lt_top // h) + 1) <= 12 and 10 <= ((lt_bottom // h) + 1) <= 19 and 10 <= (
                        (lt_left // w) + 1) <= 14 and 15 <= ((lt_right // w) + 1) <= 19:
                    result_lt_phase = 1
                    p5_lt_phase = "OK"

                else:
                    result_lt_phase = 0
                    p5_lt_phase = "impropriety of phase"
                    # pymsgbox.alert('impropriety of phase : Laptop!', str(image))

        if len(sr_screen) > 0 and len(sr_laptop) > 0:
            result_peo_phase = 0
            p5_peo_phase = ""
            # pymsgbox.alert('Found 2 screen!', str(image))

    print("Human Position and Proportion Analysis :")
    if len(sr_people) > 0:
        print("People : " + p5_peo_phase)

    elif len(sr_people) == 0:
        print(p5_peo_phase)

    if len(sr_screen) > 0 and len(sr_laptop) == 0:
        print("Screen : " + p5_screen_phase)

    elif len(sr_screen) == 0 and len(sr_laptop) > 0:
        print("Laptop : " + p5_lt_phase)

    else:
        print(p5_screen_phase)

    print("Proportion : " + p5_proport)
    isBadProportion = "Proportion: " + p5_proport + "\n"
    list_of_bad.append(isBadProportion)
    print("------------------------------------------------------------------------------")

    array_proportion = {"P5_peo_phase": p5_peo_phase, "P5_screen_phase": p5_screen_phase, "P5_lt_phase": p5_lt_phase,
                        "Result_peo_phase": result_peo_phase, "Result_screen_phase": result_screen_phase,
                        "Result_lt_phase": result_lt_phase, "P5_proport": p5_proport, "Result_proport": result_proport}
    return array_proportion


def kb_m_location(image, dl_h, dl_w, segmentation):
    # read people image
    peoimg = cv2.imread(path_to_image + image, cv2.IMREAD_COLOR)

    # dl_h = Draw Line Height, dl_w = Draw Line width = dl_h*dl_w = 20*20
    imgheight = peoimg.shape[0]
    imgwidth = peoimg.shape[1]

    h = peoimg.shape[0] // dl_h
    w = peoimg.shape[1] // dl_w

    result_kb_phase = 0
    result_m_phase = 0

    p6_kb_phase = ""
    p6_m_phase = ""

    global list_of_bad

    sr_mouse = segmentation['Mouse']
    if len(sr_mouse) > 0:
        m_top = sr_mouse[1]
        m_left = sr_mouse[0]
        m_bottom = sr_mouse[3]
        m_right = sr_mouse[2]

    sr_keyboard = segmentation['Keyboard']
    if len(sr_keyboard) > 0:
        kb_top = sr_keyboard[1]
        kb_left = sr_keyboard[0]
        kb_bottom = sr_keyboard[3]
        kb_right = sr_keyboard[2]

    if len(sr_keyboard) == 0:
        result_kb_phase = 0
        p6_kb_phase = "Keyboard Not Found"
        # pymsgbox.alert('Keyboard Not Found!', str(image))

    elif len(sr_keyboard) > 0:
        if ((kb_bottom // h) + 1) < 20 and ((kb_right // w) + 1) < 20:
            result_kb_phase = 1
            p6_kb_phase = "OK"

        else:
            result_kb_phase = 0
            p6_kb_phase = "impropriety of phase"
            # pymsgbox.alert('impropriety of phase : Keyboard!', str(image))

    if len(sr_mouse) == 0:
        result_m_phase = 0
        p6_m_phase = "Mouse Not Found"
        # pymsgbox.alert('Mouse Not Found!', str(image))

    elif len(sr_mouse) > 0:
        if ((m_bottom // h) + 1) < 20 and ((m_right // w) + 1) < 20:
            result_m_phase = 1
            p6_m_phase = "OK"

        else:
            result_m_phase = 0
            p6_m_phase = "impropriety of phase"
            # pymsgbox.alert('impropriety of phase : Mouse!', str(image))

    print("Keyboard & Mouse Location Analysis :")
    print("Keyboard : " + p6_kb_phase)
    isBadKL = "Keyboard: " + p6_kb_phase
    list_of_bad.append(isBadKL)
    print("Mouse : " + p6_m_phase)
    isBadML = "Mouse: " + p6_m_phase
    list_of_bad.append(isBadML)
    print("------------------------------------------------------------------------------")

    array_kb_m_location = {"Result_kb_phase": result_kb_phase, "Result_m_phase": result_m_phase,
                           "P6_kb_phase": p6_kb_phase, "P6_m_phase": p6_m_phase}
    return array_kb_m_location


def result_calculation(bn__analysis, segmentation, location_h_c, f_direction, proportion, location_kb_m):
    p_totalresult = ""
    totalresult = 0

    sr_screen = segmentation['Screen']
    sr_laptop = segmentation['Laptop']

    result_brightness = bn__analysis["Result_brightness"]

    result_peo_segimg = segmentation["Result_peo_segimg"]
    result_screen_segimg = segmentation["Result_screen_segimg"]

    result_peo_location = location_h_c["Result_peo_location"]
    result_screen_location = location_h_c["Result_screen_location"]
    result_lt_location = location_h_c["Result_lt_location"]

    result_face = f_direction["Result_face"]

    result_peo_phase = proportion["Result_peo_phase"]
    result_screen_phase = proportion["Result_screen_phase"]
    result_lt_phase = proportion["Result_lt_phase"]
    result_proport = proportion["Result_proport"]

    result_kb_phase = location_kb_m["Result_kb_phase"]
    result_m_phase = location_kb_m["Result_m_phase"]

    global isGood
    if len(sr_screen) > 0 and len(sr_laptop) == 0:
        if result_brightness == 1 and result_peo_segimg == 1 and result_screen_segimg == 1 and result_peo_location == 1 and result_screen_location == 1 and result_face == 1 and result_peo_phase == 1 and result_screen_phase == 1 and result_proport == 1:
            p_totalresult = "Good"
            totalresult = 1
            isGood = 1

        else:
            p_totalresult = "Bad"
            totalresult = 0
            isGood = 0

    elif len(sr_screen) == 0 and len(sr_laptop) > 0:
        if result_brightness == 1 and result_peo_segimg == 1 and result_screen_segimg == 1 and result_peo_location == 1 and result_lt_location == 1 and result_face == 1 and result_peo_phase == 1 and result_lt_phase == 1 and result_proport == 1:
            p_totalresult = "Good"
            totalresult = 1
            isGood = 1

        else:
            p_totalresult = "Bad"
            totalresult = 0
            isGood = 0

    else:
        p_totalresult = "Bad"
        totalresult = 0
        isGood = 0

    print("Result Calculation :")
    print(p_totalresult)
    print(isGood)
    print("------------------------------------------------------------------------------")

    array_Result = {"P_totalresult": p_totalresult}
    return array_Result


def print_imwrite(image, dl_h, dl_w, bn__analysis, segmentation, location_h_c, f_direction, proportion, location_kb_m,
                  result_calculation):
    # read people image
    peoimg = cv2.imread(path_to_image + image, cv2.IMREAD_COLOR)

    # dl_h = Draw Line Height, dl_w = Draw Line width = dl_h*dl_w = 20*20
    imgheight = peoimg.shape[0]
    imgwidth = peoimg.shape[1]

    h = peoimg.shape[0] // dl_h
    w = peoimg.shape[1] // dl_w

    # drawline image
    cv2.line(peoimg, (0, imgheight // 2), (imgwidth, imgheight // 2), (238, 203, 173), 1)
    cv2.putText(peoimg, str(dl_h // 2), ((dl_h + (dl_h // 4)), (imgheight // 2) - (dl_h // 2)), cv2.FONT_HERSHEY_PLAIN,
                1, (238, 203, 173), 2, cv2.LINE_AA)

    for hl in range(dl_h // 2):
        # h_result : horizontal result
        h_result = (imgheight // 2) - (h * (((dl_h // 2) - 1) - hl))
        cv2.line(peoimg, (0, h_result), (imgwidth, h_result), (238, 203, 173), 1)
        cv2.putText(peoimg, str(hl + 1), ((dl_h + (dl_h // 4)), h_result - (dl_h // 2)), cv2.FONT_HERSHEY_PLAIN, 1,
                    (238, 203, 173), 2, cv2.LINE_AA)

    for hl in range(dl_h // 2):
        # h_result : horizontal result
        h_result = (imgheight // 2) + (h * (hl + 1))
        cv2.line(peoimg, (0, h_result), (imgwidth, h_result), (238, 203, 173), 1)
        cv2.putText(peoimg, str(hl + 11), ((dl_h + (dl_h // 4)), h_result - (dl_h // 2)), cv2.FONT_HERSHEY_PLAIN, 1,
                    (238, 203, 173), 2, cv2.LINE_AA)

    cv2.line(peoimg, (imgwidth // 2, 0), (imgwidth // 2, imgheight), (238, 203, 173), 1)
    cv2.putText(peoimg, str(dl_w // 2), ((imgwidth // 2) - (dl_w * 2), dl_h + (dl_w // 4)), cv2.FONT_HERSHEY_PLAIN, 1,
                (238, 203, 173), 2, cv2.LINE_AA)

    for wl in range(dl_w // 2):
        # v_result : vertical result
        v_result = (imgwidth // 2) - (w * (((dl_w // 2) - 1) - wl))
        cv2.line(peoimg, (v_result, 0), (v_result, imgheight), (238, 203, 173), 1)
        cv2.putText(peoimg, str(wl + 1), (v_result - (dl_w * 2), dl_h + (dl_w // 4)), cv2.FONT_HERSHEY_PLAIN, 1,
                    (238, 203, 173), 2, cv2.LINE_AA)

    for wl in range(dl_w // 2):
        # v_result : vertical result
        v_result = (imgwidth // 2) + (w * (wl + 1))
        cv2.line(peoimg, (v_result, 0), (v_result, imgheight), (238, 203, 173), 1)
        cv2.putText(peoimg, str(wl + 11), (v_result - (dl_w * 2), dl_h + (dl_w // 4)), cv2.FONT_HERSHEY_PLAIN, 1,
                    (238, 203, 173), 2, cv2.LINE_AA)

    sr_people = segmentation['People']
    if len(sr_people) > 0:
        p_top = sr_people[1]
        p_left = sr_people[0]
        p_bottom = sr_people[3]
        p_right = sr_people[2]

    sr_screen = segmentation['Screen']
    if len(sr_screen) > 0:
        screen_top = sr_screen[1]
        screen_left = sr_screen[0]
        screen_bottom = sr_screen[3]
        screen_right = sr_screen[2]

    sr_laptop = segmentation['Laptop']
    if len(sr_laptop) > 0:
        lt_top = sr_laptop[1]
        lt_left = sr_laptop[0]
        lt_bottom = sr_laptop[3]
        lt_right = sr_laptop[2]

    sr_mouse = segmentation['Mouse']
    if len(sr_mouse) > 0:
        m_top = sr_mouse[1]
        m_left = sr_mouse[0]
        m_bottom = sr_mouse[3]
        m_right = sr_mouse[2]

    sr_keyboard = segmentation['Keyboard']
    if len(sr_keyboard) > 0:
        kb_top = sr_keyboard[1]
        kb_left = sr_keyboard[0]
        kb_bottom = sr_keyboard[3]
        kb_right = sr_keyboard[2]

    # maskpoint segimg
    if len(sr_people) > 0:
        cv2.rectangle(peoimg, (p_left, p_top), (p_right, p_bottom), (210, 105, 30), 2)
        cv2.putText(peoimg, 'Person', (p_left, p_top - dl_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (139, 69, 19), 2,
                    cv2.LINE_AA)

    if len(sr_screen) > 0:
        cv2.rectangle(peoimg, (screen_left, screen_top), (screen_right, screen_bottom), (208, 224, 64), 2)
        cv2.putText(peoimg, 'screen', (screen_left, screen_top - dl_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 178, 32), 2,
                    cv2.LINE_AA)

    if len(sr_laptop) > 0:
        cv2.rectangle(peoimg, (lt_left, lt_top), (lt_right, lt_bottom), (208, 224, 64), 2)
        cv2.putText(peoimg, 'Laptop', (lt_left, lt_top - dl_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 178, 32), 2,
                    cv2.LINE_AA)

    if len(sr_mouse) > 0:
        cv2.rectangle(peoimg, (m_left, m_top), (m_right, m_bottom), (136, 196, 221), 2)
        cv2.putText(peoimg, 'Mouse', (m_left, m_top - dl_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (153, 204, 255), 2,
                    cv2.LINE_AA)

    if len(sr_keyboard) > 0:
        cv2.rectangle(peoimg, (kb_left, kb_top), (kb_right, kb_bottom), (136, 196, 221), 2)
        cv2.putText(peoimg, 'Keyboard', (kb_left, kb_top - dl_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (153, 204, 255), 2,
                    cv2.LINE_AA)

    setfaces = {}
    setfaces = f_direction["Setfaces"]

    if len(setfaces) > 0:
        x, y, width, height = setfaces['box']

        setkeypoint = setfaces['keypoints']
        setle = setkeypoint["left_eye"]
        setre = setkeypoint["right_eye"]
        setn = setkeypoint["nose"]
        setml = setkeypoint["mouth_left"]
        setmr = setkeypoint["mouth_right"]

        # maskpoint Face
        cv2.rectangle(peoimg, (x, y), (x + width, y + height), (128, 128, 240), 2)
        cv2.putText(peoimg, 'Face', (x, y - (dl_h // 4)), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)

        cv2.drawMarker(peoimg, (setle[0], setle[1]), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 8, 2)
        cv2.putText(peoimg, 'LE', (setle[0] - ((dl_h // 2) * 3), setle[1] + (dl_h // 4)), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 255), 1, cv2.LINE_AA)

        cv2.drawMarker(peoimg, (setre[0], setre[1]), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 8, 2)
        cv2.putText(peoimg, 'RE', (setre[0] + (dl_h // 2), setre[1] + (dl_h // 4)), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 0), 1, cv2.LINE_AA)

        cv2.drawMarker(peoimg, (setn[0], setn[1]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 2)
        cv2.putText(peoimg, 'N', (setn[0] - dl_h, setn[1] + (dl_h // 4)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1,
                    cv2.LINE_AA)

        cv2.drawMarker(peoimg, (setml[0], setml[1]), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 2)
        cv2.putText(peoimg, 'ML', (setml[0] - ((dl_h // 2) * 3), setml[1] + (dl_h // 4)), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 255), 1, cv2.LINE_AA)

        cv2.drawMarker(peoimg, (setmr[0], setmr[1]), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 8, 2)
        cv2.putText(peoimg, 'MR', (setmr[0] + (dl_h // 2), setmr[1] + (dl_h // 4)), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 0), 1, cv2.LINE_AA)

    p1_brightness = bn__analysis["P1_brightness"]
    result_brightness = bn__analysis["Result_brightness"]

    p2_peo_segimg = segmentation["P2_peo_segimg"]
    result_peo_segimg = segmentation["Result_peo_segimg"]
    p2_screen_segimg = segmentation["P2_screen_segimg"]
    result_screen_segimg = segmentation["Result_screen_segimg"]

    p3_peo_location = location_h_c["P3_peo_location"]
    result_peo_location = location_h_c["Result_peo_location"]
    p3_screen_location = location_h_c["P3_screen_location"]
    result_screen_location = location_h_c["Result_screen_location"]
    p3_lt_location = location_h_c["P3_lt_location"]
    result_lt_location = location_h_c["Result_lt_location"]

    p4_face = f_direction["P4_face"]
    result_face = f_direction["Result_face"]

    p5_peo_phase = proportion["P5_peo_phase"]
    result_peo_phase = proportion["Result_peo_phase"]
    p5_screen_phase = proportion["P5_screen_phase"]
    result_screen_phase = proportion["Result_screen_phase"]
    p5_lt_phase = proportion["P5_lt_phase"]
    result_lt_phase = proportion["Result_lt_phase"]
    p5_proport = proportion["P5_proport"]
    result_proport = proportion["Result_proport"]

    p6_kb_phase = location_kb_m["P6_kb_phase"]
    result_kb_phase = location_kb_m["Result_kb_phase"]
    p6_m_phase = location_kb_m["P6_m_phase"]
    result_m_phase = location_kb_m["Result_m_phase"]

    p_totalresult = result_calculation["P_totalresult"]

    # print for img
    if len(sr_people) > 0 and len(sr_screen) > 0 and len(sr_laptop) == 0:
        if abs(screen_left - p_right) < abs(screen_right - p_left):
            cv2.rectangle(peoimg, (890, 110), (1190, 230), (255, 255, 255), 140)

            cv2.putText(peoimg, "1. Brightness      : " + str(p1_brightness), (870, 60), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "2. People Segimg   : " + str(p2_peo_segimg), (870, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Segimg   : " + str(p2_screen_segimg), (870, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "3. People Location : " + str(p3_peo_location), (870, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Location : " + str(p3_screen_location), (870, 140), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "4. Face Direction  : " + str(p4_face), (870, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "5. People Phase    : " + str(p5_peo_phase), (870, 180), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Phase    : " + str(p5_screen_phase), (870, 200), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Proportion      : " + str(p5_proport), (870, 220), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "6. Keyboard Phase  : " + str(p6_kb_phase), (870, 240), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Mouse Phase     : " + str(p6_m_phase), (870, 260), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "Total Result       : " + str(p_totalresult), (870, 290), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)

        elif abs(screen_left - p_right) > abs(screen_right - p_left):
            cv2.rectangle(peoimg, (190, 110), (490, 230), (255, 255, 255), 140)

            cv2.putText(peoimg, "1. Brightness      : " + str(p1_brightness), (170, 60), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "2. People Segimg   : " + str(p2_peo_segimg), (170, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Segimg   : " + str(p2_screen_segimg), (170, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "3. People Location : " + str(p3_peo_location), (170, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Location : " + str(p3_screen_location), (170, 140), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "4. Face Direction  : " + str(p4_face), (170, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "5. People Phase    : " + str(p5_peo_phase), (170, 180), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Phase    : " + str(p5_screen_phase), (170, 200), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Proportion      : " + str(p5_proport), (170, 220), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "6. Keyboard Phase  : " + str(p6_kb_phase), (170, 240), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Mouse Phase     : " + str(p6_m_phase), (170, 260), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "Total Result       : " + str(p_totalresult), (170, 290), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)

    elif len(sr_people) > 0 and len(sr_screen) == 0 and len(sr_laptop) > 0:
        if abs(lt_left - p_right) < abs(lt_right - p_left):
            cv2.rectangle(peoimg, (890, 110), (1190, 230), (255, 255, 255), 140)

            cv2.putText(peoimg, "1. Brightness      : " + str(p1_brightness), (870, 60), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "2. People Segimg   : " + str(p2_peo_segimg), (870, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Segimg   : " + str(p2_screen_segimg), (870, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "3. People Location : " + str(p3_peo_location), (870, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Laptop Location : " + str(p3_lt_location), (870, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "4. Face Direction  : " + str(p4_face), (870, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "5. People Phase    : " + str(p5_peo_phase), (870, 180), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Laptop Phase    : " + str(p5_lt_phase), (870, 200), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Proportion      : " + str(p5_proport), (870, 220), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "6. Keyboard Phase  : " + str(p6_kb_phase), (870, 240), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Mouse Phase     : " + str(p6_m_phase), (870, 260), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "Total Result       : " + str(p_totalresult), (870, 290), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)

        elif abs(lt_left - p_right) > abs(lt_right - p_left):
            cv2.rectangle(peoimg, (190, 110), (490, 230), (255, 255, 255), 140)

            cv2.putText(peoimg, "1. Brightness      : " + str(p1_brightness), (170, 60), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "2. People Segimg   : " + str(p2_peo_segimg), (170, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Segimg   : " + str(p2_screen_segimg), (170, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "3. People Location : " + str(p3_peo_location), (170, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Laptop Location : " + str(p3_lt_location), (170, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "4. Face Direction  : " + str(p4_face), (170, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "5. People Phase    : " + str(p5_peo_phase), (170, 180), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Laptop Phase    : " + str(p5_lt_phase), (170, 200), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Proportion      : " + str(p5_proport), (170, 220), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "6. Keyboard Phase  : " + str(p6_kb_phase), (170, 240), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Mouse Phase     : " + str(p6_m_phase), (170, 260), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "Total Result       : " + str(p_totalresult), (170, 290), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)

    elif len(sr_people) > 0 and len(sr_screen) == 0 and len(sr_laptop) == 0:
        if p_left < abs(p_left - imgwidth) and p_right < abs(p_right - imgwidth):
            cv2.rectangle(peoimg, (890, 110), (1190, 230), (255, 255, 255), 140)

            cv2.putText(peoimg, "1. Brightness      : " + str(p1_brightness), (870, 60), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "2. People Segimg   : " + str(p2_peo_segimg), (870, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Segimg   : " + str(p2_screen_segimg), (870, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "3. People Location : " + str(p3_peo_location), (870, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Location : " + str(p3_screen_location), (870, 140), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "4. Face Direction  : " + str(p4_face), (870, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "5. People Phase    : " + str(p5_peo_phase), (870, 180), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Phase    : " + str(p5_screen_phase), (870, 200), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Proportion      : " + str(p5_proport), (870, 220), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "6. Keyboard Phase  : " + str(p6_kb_phase), (870, 240), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Mouse Phase     : " + str(p6_m_phase), (870, 260), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "Total Result       : " + str(p_totalresult), (870, 290), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)

        else:
            cv2.rectangle(peoimg, (190, 110), (490, 230), (255, 255, 255), 140)

            cv2.putText(peoimg, "1. Brightness      : " + str(p1_brightness), (170, 60), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "2. People Segimg   : " + str(p2_peo_segimg), (170, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Segimg   : " + str(p2_screen_segimg), (170, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "3. People Location : " + str(p3_peo_location), (170, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Location : " + str(p3_screen_location), (170, 140), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "4. Face Direction  : " + str(p4_face), (170, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "5. People Phase    : " + str(p5_peo_phase), (170, 180), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Phase    : " + str(p5_screen_phase), (170, 200), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Proportion      : " + str(p5_proport), (170, 220), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "6. Keyboard Phase  : " + str(p6_kb_phase), (170, 240), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Mouse Phase     : " + str(p6_m_phase), (170, 260), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "Total Result       : " + str(p_totalresult), (170, 290), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)

    elif len(sr_people) == 0 and len(sr_screen) > 0 and len(sr_laptop) == 0:
        if screen_left < abs(screen_right - imgwidth):
            cv2.rectangle(peoimg, (890, 110), (1190, 230), (255, 255, 255), 140)

            cv2.putText(peoimg, "1. Brightness      : " + str(p1_brightness), (870, 60), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "2. People Segimg   : " + str(p2_peo_segimg), (870, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Segimg   : " + str(p2_screen_segimg), (870, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "3. People Location : " + str(p3_peo_location), (870, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Location : " + str(p3_screen_location), (870, 140), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "4. Face Direction  : " + str(p4_face), (870, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "5. People Phase    : " + str(p5_peo_phase), (870, 180), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Phase    : " + str(p5_screen_phase), (870, 200), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Proportion      : " + str(p5_proport), (870, 220), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "6. Keyboard Phase  : " + str(p6_kb_phase), (870, 240), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Mouse Phase     : " + str(p6_m_phase), (870, 260), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "Total Result       : " + str(p_totalresult), (870, 290), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)

        elif screen_left > abs(screen_right - imgwidth):
            cv2.rectangle(peoimg, (190, 110), (490, 230), (255, 255, 255), 140)

            cv2.putText(peoimg, "1. Brightness      : " + str(p1_brightness), (170, 60), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "2. People Segimg   : " + str(p2_peo_segimg), (170, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Segimg   : " + str(p2_screen_segimg), (170, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "3. People Location : " + str(p3_peo_location), (170, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Location : " + str(p3_screen_location), (170, 140), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "4. Face Direction  : " + str(p4_face), (170, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "5. People Phase    : " + str(p5_peo_phase), (170, 180), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Phase    : " + str(p5_screen_phase), (170, 200), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Proportion      : " + str(p5_proport), (170, 220), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "6. Keyboard Phase  : " + str(p6_kb_phase), (170, 240), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Mouse Phase     : " + str(p6_m_phase), (170, 260), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "Total Result       : " + str(p_totalresult), (170, 290), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)

    elif len(sr_people) == 0 and len(sr_screen) == 0 and len(sr_laptop) > 0:
        if lt_left < abs(lt_right - imgwidth):
            cv2.rectangle(peoimg, (890, 110), (1190, 230), (255, 255, 255), 140)

            cv2.putText(peoimg, "1. Brightness      : " + str(p1_brightness), (870, 60), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "2. People Segimg   : " + str(p2_peo_segimg), (870, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Segimg   : " + str(p2_screen_segimg), (870, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "3. People Location : " + str(p3_peo_location), (870, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Laptop Location : " + str(p3_lt_location), (870, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "4. Face Direction  : " + str(p4_face), (870, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "5. People Phase    : " + str(p5_peo_phase), (870, 180), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Laptop Phase    : " + str(p5_lt_phase), (870, 200), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Proportion      : " + str(p5_proport), (870, 220), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "6. Keyboard Phase  : " + str(p6_kb_phase), (870, 240), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Mouse Phase     : " + str(p6_m_phase), (870, 260), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "Total Result       : " + str(p_totalresult), (870, 290), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)

        elif lt_left > abs(lt_right - imgwidth):
            cv2.rectangle(peoimg, (190, 110), (490, 230), (255, 255, 255), 140)

            cv2.putText(peoimg, "1. Brightness      : " + str(p1_brightness), (170, 60), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "2. People Segimg   : " + str(p2_peo_segimg), (170, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Screen Segimg   : " + str(p2_screen_segimg), (170, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "3. People Location : " + str(p3_peo_location), (170, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Laptop Location : " + str(p3_lt_location), (170, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "4. Face Direction  : " + str(p4_face), (170, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "5. People Phase    : " + str(p5_peo_phase), (170, 180), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Laptop Phase    : " + str(p5_lt_phase), (170, 200), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Proportion      : " + str(p5_proport), (170, 220), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "6. Keyboard Phase  : " + str(p6_kb_phase), (170, 240), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "    Mouse Phase     : " + str(p6_m_phase), (170, 260), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(peoimg, "Total Result       : " + str(p_totalresult), (170, 290), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 0), 1, cv2.LINE_AA)

    global path
    filename = 'new_anls' + str(image)
    path = os.path.join(UPLOAD_FOLDER, filename)

    cv2.imwrite(path, peoimg)

    print_alert = {}

    if len(sr_screen) == 0 and len(sr_laptop) > 0:
        print_alert = {"P1_bn": "1. Brightness : " + p1_brightness, "P2_peo": "2. People Detection : " + p2_peo_segimg,
                       "P2_screen": "   Screen Detection : " + p2_screen_segimg,
                       "P3_peo": "3. People Location : " + p3_peo_location,
                       "P3_lt": "   Laptop Location : " + p3_lt_location, "P4_face": "4. Face : " + p4_face,
                       "P5_peo": "5. People Phase : " + p5_peo_phase, "P5_lt": "   Laptop Phase : " + p5_lt_phase,
                       "P5_proport": "   Proport : " + p5_proport, "P6_kb": "6. Keyboard Phase : " + p6_kb_phase,
                       "P6_m": "   Mouse Phase : " + p6_m_phase}

    else:
        print_alert = {"P1_bn": "1. Brightness : " + p1_brightness, "P2_peo": "2. People Detection : " + p2_peo_segimg,
                       "P2_screen": "   Screen Detection : " + p2_screen_segimg,
                       "P3_peo": "3. People Location : " + p3_peo_location,
                       "P3_screen": "   Screen Location : " + p3_screen_location, "P4_face": "4. Face : " + p4_face,
                       "P5_peo": "5. People Phase : " + p5_peo_phase,
                       "P5_screen": "   Screen Phase : " + p5_screen_phase, "P5_proport": "   Proport : " + p5_proport,
                       "P6_kb": "6. Keyboard Phase : " + p6_kb_phase, "P6_m": "   Mouse Phase : " + p6_m_phase}

    if result_brightness == 0 and len(p1_brightness) > 0:
        print_alert["P1_bn"] = "1. Brightness : " + p1_brightness + "!"

    if result_peo_segimg == 0 and len(p2_peo_segimg) > 0:
        print_alert["P2_peo"] = "2. People Detection : " + p2_peo_segimg + "!"
    if result_screen_segimg == 0 and len(p2_screen_segimg) > 0:
        print_alert["P2_screen"] = "   Screen Detection : " + p2_screen_segimg + "!"

    if result_peo_location == 0 and len(p3_peo_location) > 0:
        print_alert["P3_peo"] = "3. People Location : " + p3_peo_location + "!"
    if result_screen_location == 0 and len(p3_screen_location) > 0:
        print_alert["P3_screen"] = "   Screen Location : " + p3_screen_location + "!"
    if result_lt_location == 0 and len(p3_lt_location) > 0:
        print_alert["P3_lt"] = "   Laptop Location : " + p3_lt_location + "!"

    if result_face == 0 and len(p4_face) > 0:
        print_alert["P4_face"] = "4. Face : " + p4_face + "!"

    if result_peo_phase == 0 and len(p5_peo_phase) > 0:
        print_alert["P5_peo"] = "5. People Phase : " + p5_peo_phase + "!"
    if result_screen_phase == 0 and len(p5_screen_phase) > 0:
        print_alert["P5_screen"] = "   Screen Phase : " + p5_screen_phase + "!"
    if result_lt_phase == 0 and len(p5_lt_phase) > 0:
        print_alert["P5_lt"] = "   Laptop Phase : " + p5_lt_phase + "!"
    if result_proport == 0 and len(p5_proport) > 0:
        print_alert["P5_proport"] = "   Proportion : " + p5_proport + "!"

    if result_kb_phase == 0 and len(p6_kb_phase) > 0:
        print_alert["P6_kb"] = "6. Keyboard Phase : " + p6_kb_phase + "!"
    if result_m_phase == 0 and len(p6_m_phase) > 0:
        print_alert["P6_m"] = "   Mouse Phase : " + p6_m_phase + "!"

    print_alert_all = {}

    if len(sr_screen) == 0 and len(sr_laptop) > 0:
        print_alert_all = print_alert["P1_bn"] + "\n" + "\n" + print_alert["P2_peo"] + "\n" + print_alert[
            "P2_screen"] + "\n" + "\n" + print_alert["P3_peo"] + "\n" + print_alert["P3_lt"] + "\n" + "\n" + \
                          print_alert["P4_face"] + "\n" + "\n" + print_alert["P5_peo"] + "\n" + print_alert[
                              "P5_lt"] + "\n" + print_alert["P5_proport"] + "\n" + "\n" + print_alert["P6_kb"] + "\n" + \
                          print_alert["P6_m"]
    else:
        print_alert_all = print_alert["P1_bn"] + "\n" + "\n" + print_alert["P2_peo"] + "\n" + print_alert[
            "P2_screen"] + "\n" + "\n" + print_alert["P3_peo"] + "\n" + print_alert["P3_screen"] + "\n" + "\n" + \
                          print_alert["P4_face"] + "\n" + "\n" + print_alert["P5_peo"] + "\n" + print_alert[
                              "P5_screen"] + "\n" + print_alert["P5_proport"] + "\n" + "\n" + print_alert[
                              "P6_kb"] + "\n" + print_alert["P6_m"]

    # pymsgbox.alert(print_alert_all, str(image))


def all_seg(image, dl_h, dl_w, level, percent, numdiff):
    bn__analysis = brightness_analysis(image, dl_h, dl_w, level, percent, numdiff)
    segmentation = ins_segimg(image, dl_h, dl_w)
    location_h_c = human_com_location(image, dl_h, dl_w, segmentation)
    f_direction = face_direction(image, dl_h, dl_w, segmentation)
    proportion = position_proportion(image, dl_h, dl_w, segmentation)
    location_kb_m = kb_m_location(image, dl_h, dl_w, segmentation)

    r_calculation = result_calculation(bn__analysis, segmentation, location_h_c, f_direction, proportion, location_kb_m)

    p_imwrite = print_imwrite(image, dl_h, dl_w, bn__analysis, segmentation, location_h_c, f_direction, proportion,
                              location_kb_m, r_calculation)


def array_img(image, dl_h, dl_w, level, percent, numdiff):
    for i in range(len(image)):
        print(image[i])
        all_seg(image[i], dl_h, dl_w, level, percent, numdiff)

# image = (['normal7.jpg'])
# # print(type(image))
# start_time = time.time()
# array_img(image, 20, 20, 32, 0.05, 4)
# print("Time : %s Seconds" % (time.time() - start_time))


@app.route('/process_data', methods=['POST'])
def process_data():

    #file = request.json.get('image', None)
    if 'image' not in request.files:
        return jsonify({'error': 'No image provide'}), 400

    name = request.form.get('name', '')
    if name == '':
        return jsonify({'error': 'No Name provided'}), 400

    id = request.form.get('id', '')
    if id == '':
        return jsonify({'error': 'No ID provided'}), 400

    email = request.form.get('email', '')
    if email == '':
        return jsonify({'error': 'No ID provided'}), 400

    #test
    image_file = request.files['image'] #(['normal7'])
    if image_file.filename == '':
        return jsonify({'error': 'Empty image file'}), 400

    filename = secure_filename(image_file.filename)
    unique_idenfier = str(uuid4())
    file_extension = os.path.splitext(filename)[1]
    new_filename = f"{unique_idenfier}{file_extension}"

    image_path = os.path.join(UPLOAD_FOLDER, new_filename)
    image_file.save(image_path)

    try:

        image = ([new_filename])
        start_time = time.time()
        array_img(image, 20, 20, 32, 0.05, 4)
        print("Time : %s Seconds" % (time.time() - start_time))

        stid = int(id)

        global isGood
        isPass = ''
        if isGood == 1:
            isPass = 'Pass'
        else:
            isPass = 'Not Pass'

        global path
        sql = ("INSERT INTO app_rec (id, name, picture, result, picture_result, email) "
               "VALUES (%s, %s, %s, %s, %s, %s) "
               "ON DUPLICATE KEY UPDATE "
               "name=VALUES(name), picture=VALUES(picture), result=VALUES(result), picture_result=VALUES(picture_result), email=VALUES(email)")
        # print("Connection", mysql.connection)
        connection = mysql.connector.connect(**db_config)
        cur = connection.cursor()

        cur.execute(sql, (stid, name, image_path, isPass, path, email))
        connection.commit()

        cur.close()
        connection.close()


        # conn = sqlite3.connect('app_database.db')
        # cursor = conn.cursor()
        #
        # query = "INSERT INTO app_record (date, name, picture, isPass) VALUES (?, ?, ?, ?)"
        # cursor.execute(query, (date, name, image_path, isPass))
        # conn.commit()
        # conn.close()

        return jsonify({'status': "ok", 'message ': "Success! Thank you for waiting"}), 200

    except Exception as e:
        print('error', e)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 800

@app.route('/get_result', methods=['GET'])
def get_result():
    global isGood
    global list_of_bad
    # print(isGood)
    # print(list_of_bad)
    response_data = {'status': "ok", 'message ': "Success! Thank you for waiting", 'score': isGood, 'list': list_of_bad}
    list_of_bad = []
    isGood = 1

    return jsonify(response_data), 200

@app.route('/get_email', methods=['GET'])
def get_email():
    email = request.args.get('email')

    connection = mysql.connector.connect(**db_config)
    cur = connection.cursor()

    try:
        cur.execute("SELECT * FROM app_rec WHERE email = %s", (email, ))
        result = cur.fetchone()
        if result is not None and result[0] > 0:
            email_exist = True
        else:
            email_exist = False

        return jsonify({'status': "ok", 'message ': "Success! Thank you for waiting", "email": email_exist}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 800

    finally:
        cur.close()
        connection.close()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

