import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import time
from tensorflow.keras.models import load_model


def get_face_detector(modelFile = r"C:/Users/Acer/OneDrive/Desktop/Project/res10_300x300_ssd_iter_140000.caffemodel",
                      configFile = r"C:/Users/Acer/OneDrive/Desktop/Project/deploy.prototxt.txt"):

    modelFile = r"C:/Users/Acer/OneDrive/Desktop/Project/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = r"C:/Users/Acer/OneDrive/Desktop/Project/deploy.prototxt.txt"
    model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def sort_array(np_array):
    return np_array[2]


def find_faces(img, model):
    
    No_person = 1
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    
    res = list(np.squeeze(res))
    res.sort(key = sort_array, reverse = True)
    
    faces = []
    for i in res[:No_person]: 
        box = i[3:7] * np.array([w, h, w, h])
        (x, y, x1, y1) = box.astype("int")
        faces.append([x, y, x1, y1])
    return faces

def get_landmark_model(saved_model='C:/Users/Acer/OneDrive/Desktop/Project/pose_model'):

    model = load_model(saved_model)
    return model

def get_square_box(box):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        return box
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]

def move_box(box, offset):
    """Move the box to direction specified by vector offset"""
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]
    return [left_x, top_y, right_x, bottom_y]

def detect_marks(img, model, face):

    offset_y = int(abs((face[3] - face[1]) * 0.1))
    box_moved = move_box(face, [0, offset_y])
    facebox = get_square_box(box_moved)
    
    face_img = img[facebox[1]: facebox[3],
                     facebox[0]: facebox[2]]
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # # Actual detection.
    predictions = model.signatures["predict"](
        tf.constant([face_img], dtype=tf.uint8))

    # Convert predictions to landmarks.
    marks = np.array(predictions['output']).flatten()[:136]
    marks = np.reshape(marks, (-1, 2))
    
    marks *= (facebox[2] - facebox[0])
    marks[:, 0] += facebox[0]
    marks[:, 1] += facebox[1]
    marks = marks.astype(np.uint)

    return marks, facebox

def draw_marks(image, mark, color=(0, 255, 0)):
    for mark in marks:
        cv2.circle(image, (mark[0], mark[1]), 2, color, -1, cv2.LINE_AA)


face_model = get_face_detector()
landmark_model = get_landmark_model()
prev_frame_time = 0
new_frame_time = 0

cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    cv2.imshow("original", img)
    rects = find_faces(img, face_model)
    result = img.copy()
    
    for rect in rects:
        marks, facebox = detect_marks(img, landmark_model, rect)
        draw_marks(img, marks)
        #cv2.rectangle(img, (facebox[0], facebox[1]), (facebox[2], facebox[3]), (0, 0, 255), 2)
        
        dst_pts = np.array([ marks[1], marks[2], marks[3], marks[4], marks[5], marks[6], marks[7], marks[8], marks[9], marks[10], marks[11], marks[12], marks[13], marks[14], marks[15], marks[29]], dtype = "float32")
    cv2.imshow("landmark", img)
    with  open("C:/Users/Acer/OneDrive/Desktop/Project/Masks/labels/green1-mask.csv", "r+") as f:
        row = f.readlines()
        src_pts = []
        for r in row:
            r = r.split(',')
            src_pts.append([r[1],r[2]])
        
    src_pts = np.array(src_pts, dtype = "float32")
    
    mask_img = cv2.imread("C:/Users/Acer/OneDrive/Desktop/Project/Masks/images/green1-mask.png")

    M, _ = cv2.findHomography(src_pts, dst_pts)

    # transformed masked image
    transformed_mask = cv2.warpPerspective(mask_img, M,(img.shape[1], img.shape[0]), None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,)



    new_frame_time = time.time()

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            a,b,c = transformed_mask[i][j]
            if a>0 and b>0 and c>0:
                img[i][j][0] = a
                img[i][j][1] = b
                img[i][j][2] = c
  
    # puting the FPS count on the frame
    #cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    
    cv2.imshow("image",img)
    cv2.imshow('mask', transformed_mask)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()