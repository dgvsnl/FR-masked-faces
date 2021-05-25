import cv2
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tensorflow.keras.models import load_model

modelFile = "C:/Users/Acer/OneDrive/Desktop/Project/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "C:/Users/Acer/OneDrive/Desktop/Project/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)  

def sort_array(np_array):
    return np_array[2]


def face_detector(img, No_person = 1, disp_box = False):
    
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    
    face_list = []
    faces = list(np.squeeze(faces)) 
    faces.sort(key = sort_array, reverse = True)
    
    for i in faces[:No_person]:
        
        box = i[3:7] * np.array([w, h, w, h])
        x, y, x1, y1 = (box.astype("int"))     # x, y, x1, y1
        face_list.append((x, y, x1, y1))
        if disp_box:
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    full_image = img
        
    return face_list, full_image

        
LD = load_model("C:/Users/Acer/all_lossses_weighst_extended.h5")
def landmark_detector(img):
    
    x_h, y_h = img.shape[1]//160, img.shape[0]//160
    
    img = cv2.resize(img, (160, 160))
    p = LD.predict(np.array([img]))
    marks = p[0][0]
    mask = p[5][0]
    
    for i in range(5):
        marks[i] = marks[i]*x_h
    for i in range(5,10):
        marks[i] = marks[i]*y_h
                
        
    print(mask)
    if mask[0] > mask[1]:
        mask = "Mask"
    else:
        mask = "No Mask"
       
    return marks, mask

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


facenet_model = load_model('C:/Users/Acer/OneDrive/Desktop/Project/facenet_keras.h5')
def get_embedding(face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = facenet_model.predict(sample)
    return yhat[0]

def cosine_dis(a,b):
    return dot(a, b)/(norm(a)*norm(b))

emd_data = np.load('C:/Users/Acer/OneDrive/Desktop/Project/g5-facedataset.npz', allow_pickle = True)['arr_0']
emd_data_cropped = np.load('C:/Users/Acer/OneDrive/Desktop/Project/g5-facedataset-cropped.npz', allow_pickle = True)['arr_0']
people = ['Nilay', 'Dhruv', 'Shreyansh', 'Krushan', 'Hiren', 'Rinkal', 'Maharishi', 'Sanket', 'Keyur', 'Ajitesh', 'Jay', 'Ruchita', 'Bhavyesh']
    
if __name__ == "__main__":
    
    vid = cv2.VideoCapture(0)
    while True:

        ret, img = vid.read()
        
        faces, img = face_detector(img)
        try:
            for f in faces:
                sq_box = get_square_box(f)
                # sq_box[0] -= 40
                # sq_box[1] -= 50
                # sq_box[2] += 40
                #sq_box[3] += 40
                
                if sq_box[0] <= 0:
                    sq_box[0] = 0
                if sq_box[1] <= 0:
                    sq_box[1] = 0
                if sq_box[2] > img.shape[1]:
                    sq_box[2] = img.shape[1] - 1
                if sq_box[3] > img.shape[0]:
                    sq_box[3] = img.shape[0] - 1
                
                # mark, mask  = landmark_detector(img)
                cv2.rectangle(img, (sq_box[0], sq_box[1]), (sq_box[2], sq_box[3]), (0, 0, 255), 2)
                mark, mask  = landmark_detector(img[sq_box[1]:sq_box[3], sq_box[0]:sq_box[2]])
                
                for m in range(5):
                    cv2.circle(img, (int(mark[0 + m] + sq_box[0]), int(mark[5 + m] + sq_box[1])), 2, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(img, mask, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        
            emd = get_embedding(img[f[1]:f[3], f[0]:f[2]])
            max_emd = 0
            
            if mask == "No mask":
                for itr,e in enumerate(emd_data):
                    cos_dis = cosine_dis(e, emd)
                    if max_emd < cos_dis:
                        max_emd = max(max_emd, cos_dis)
                        recg_emd = itr
            else:
                for itr,e in enumerate(emd_data_cropped):
                    cos_dis = cosine_dis(e, emd)
                    if max_emd < cos_dis:
                        max_emd = max(max_emd, cos_dis)
                        recg_emd = itr
                        
        except:
            continue
                
        recognized_person = people[recg_emd]
        #print("Recognized Person is: ", recognized_person)
        
        cv2.imshow("face", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    vid.release()
    cv2.destroyAllWindows()